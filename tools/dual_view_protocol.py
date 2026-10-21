"""Independent Run B warm-up, fixed samples and calibration contracts.

The importable validation/hash helpers use only the standard library.  Torch
and the model/data modules are loaded only by GPU commands.  No command edits
old calibration files, automatically picks weights, or resumes warm-up.
"""

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "instance-lower-full-v1"
TERMS = ("L_pen", "L_full", "centroid", "logsigma", "shape", "L_H")
MODULES = ("vit", "decoder", "fusion", "head")
COEF_NAMES = ("centroid", "logsigma", "shape")
# Fixed-seed CUDA bf16 repeats had identical four-loss forward values but
# ratio drift up to 6.25e-4 relative. A 1e-4 check rejected numerical noise.
# The protocol target remains 0.5; only its finite remeasurement band changes.
RATIO_RTOL = 1e-3
RATIO_ATOL = 1e-6


def check_gradient_ratio(actual, target, name, minimum=False):
    """Validate the target at the observed CUDA/bf16 numerical precision."""
    if actual is None:
        raise ValueError(f"missing gradient ratio: {name}")
    _positive(actual, name)
    if minimum and actual >= target:
        return
    if not math.isclose(actual, target, rel_tol=RATIO_RTOL,
                        abs_tol=RATIO_ATOL):
        raise ValueError(
            f"remeasured {name} does not match {target}: {actual}"
        )


def save_calibration_stage(output, stage, records, hashes, weights=None):
    """Persist measured matrices before any final acceptance can fail."""
    path = f"{output}.{stage}.json"
    write_new_json(path, {
        "schema": SCHEMA, "purpose": "training_calibration_measurements",
        "stage": stage, "status": "measured" if records else "not_needed",
        "train_mode": True, "autocast": "bfloat16", **hashes,
        "weights": weights, "measurements": records,
    })
    return path


def sha256_file(path):
    """Hash bytes without putting deployment-specific paths in the digest."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def hash_files(paths):
    """Hash sorted file names and bytes; duplicate names are rejected."""
    files = sorted((Path(path) for path in paths), key=lambda p: p.name)
    if not files or len({p.name for p in files}) != len(files):
        raise ValueError("file list is empty or contains duplicate basenames")
    digest = hashlib.sha256()
    for path in files:
        name = path.name.encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def cache_hash(directory):
    return hash_files(
        [Path(directory) / "index.json", Path(directory) / "metadata.json"]
    )


def read_json(value):
    return (
        value
        if isinstance(value, dict)
        else json.loads(Path(value).read_text(encoding="utf-8"))
    )


def write_new_json(path, value):
    """Do not silently replace an existing scientific record."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _positive(value, name):
    if (
        isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"{name} must be finite and strictly positive")


def validate_calibration(value, expected_hashes=None):
    """Reject old J coefficients and calibration for another dataset/model.

    ``expected_hashes`` must be freshly computed by the caller, rather than
    copied out of this JSON.  The final checkpoint is independently checked by
    ``preflight`` and by the main training loader.
    """
    result = read_json(value)
    if result.get("schema") != SCHEMA or result.get("purpose") != "training":
        raise ValueError(
            "a new instance/lower/full TRAINING calibration is required"
        )
    if result.get("status") != "validated":
        raise ValueError("calibration has not passed the measurement checks")
    for name in ("lambda_full", "w_j", "w_h"):
        _positive(result[name], name)
    if set(result.get("coef", {})) != set(COEF_NAMES):
        raise ValueError(
            "training J must contain exactly three new coefficients"
        )
    for name, number in result["coef"].items():
        _positive(number, name)
    required = (
        "train_manifest_sha256",
        "target_cache_sha256",
        "sample_manifest_sha256",
        "warm_checkpoint_sha256",
        "parent_checkpoint_sha256",
        "fixed_pair_sha256",
        "val_manifest_sha256",
        "eval_coef_sha256",
        "s_baseline_sha256",
    )
    for name in required:
        digest = result.get(name, "")
        if len(digest) != 64 or any(
            c not in "0123456789abcdef" for c in digest
        ):
            raise ValueError(f"missing or invalid calibration hash: {name}")
    if result.get("edge_weight") != 0:
        raise ValueError("initial Run B calibration must use edge weight 0")
    if result.get("norm_rule") != "ratio_of_mean_batch_norms":
        raise ValueError("norms must be composed per batch before averaging")
    for name, digest in (expected_hashes or {}).items():
        if result.get(name) != digest:
            raise ValueError(f"stale or mismatched calibration: {name}")
    return result


def gram_norm(gram, vector):
    value = sum(
        a * b * gram[i][j]
        for i, a in enumerate(vector)
        for j, b in enumerate(vector)
    )
    if not math.isfinite(value) or value < -1e-7:
        raise ValueError("gradient Gram matrix is nonfinite or indefinite")
    return math.sqrt(max(0.0, value))


def norm_stats(grams, vector):
    values = [gram_norm(gram, vector) for gram in grams]
    if not values:
        raise ValueError("no gradient batches")
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return {"mean": mean, "std": math.sqrt(variance), "values": values}


def cosine_stats(grams, left, right):
    values = []
    for gram in grams:
        denominator = gram_norm(gram, left) * gram_norm(gram, right)
        inner = sum(
            a * b * gram[i][j]
            for i, a in enumerate(left)
            for j, b in enumerate(right)
        )
        values.append(
            max(-1.0, min(1.0, inner / denominator))
            if denominator > 0
            else None
        )
    finite = [value for value in values if value is not None]
    return {
        "values": values,
        "mean": sum(finite) / len(finite) if finite else None,
        "negative_fraction": sum(value < 0 for value in finite) / len(finite)
        if finite
        else None,
    }


def derive_weights(records, lambda_full=0.5):
    """Fit weights using JT batches and actual sums, including cancellation."""
    jt = [record for record in records if record["group"] == "jt"]
    decoder = [record["grams"]["decoder"] for record in jt]
    vit = [record["grams"]["vit"] for record in jt]
    pen = [1, 0, 0, 0, 0, 0]
    full = [0, 1, 0, 0, 0, 0]
    pen_norm = norm_stats(decoder, pen)["mean"]
    full_norm = norm_stats(decoder, full)["mean"]
    _positive(pen_norm, "decoder pen gradient")
    _positive(full_norm, "decoder full gradient")
    before = lambda_full * full_norm / pen_norm
    if before < 0.5:
        lambda_full = 0.5 * pen_norm / full_norm
    unit = [[int(index == item) for index in range(6)] for item in (2, 3, 4)]
    j_norms = [norm_stats(vit, vector)["mean"] for vector in unit]
    for name, number in zip(COEF_NAMES, j_norms):
        _positive(number, f"ViT new J {name} gradient")
    coefficients = [j_norms[2] / value for value in j_norms]
    j = [0, 0, *coefficients, 0]
    base = [1, lambda_full, 0, 0, 0, 0]
    head = [0, 0, 0, 0, 0, 1]
    base_norm = norm_stats(vit, base)["mean"]
    j_norm = norm_stats(vit, j)["mean"]
    h_norm = norm_stats(vit, head)["mean"]
    _positive(base_norm, "composed base ViT gradient")
    _positive(j_norm, "composed J ViT gradient")
    _positive(h_norm, "head ViT gradient")
    w_j = 0.5 * base_norm / j_norm
    w_h = 0.5 * w_j * j_norm / h_norm
    return {
        "lambda_full": lambda_full,
        "w_j": w_j,
        "w_h": w_h,
        "coef": dict(zip(COEF_NAMES, coefficients)),
        "lambda_initial_ratio": before,
        "lambda_decoder_ratio": lambda_full * full_norm / pen_norm,
        "base_j_cosine": cosine_stats(vit, base, j),
        "j_h_cosine": cosine_stats(vit, j, head),
    }


def weight_summary(records, weights):
    """Report each group/module, without normalizing BF-only zero J away."""
    vectors = {
        "pen": [1, 0, 0, 0, 0, 0],
        "weighted_full": [0, weights["lambda_full"], 0, 0, 0, 0],
        "base": [1, weights["lambda_full"], 0, 0, 0, 0],
        "weighted_j": [
            0,
            0,
            *[weights["w_j"] * weights["coef"][key] for key in COEF_NAMES],
            0,
        ],
        "weighted_h": [0, 0, 0, 0, 0, weights["w_h"]],
    }
    result = {}
    direct_vectors = {
        "pen": [1, 0, 0, 0],
        "weighted_full": [0, weights["lambda_full"], 0, 0],
        "base": [1, weights["lambda_full"], 0, 0],
        "weighted_j": [0, 0, weights["w_j"], 0],
        "weighted_h": [0, 0, 0, weights["w_h"]],
    }
    for group in sorted({record["group"] for record in records}):
        result[group] = {}
        for module in MODULES:
            selected = [r for r in records if r["group"] == group]
            direct = all("objective_grams" in r for r in selected)
            grams = [
                r["objective_grams" if direct else "grams"][module]
                for r in selected
            ]
            used_vectors = direct_vectors if direct else vectors
            stats = {
                name: norm_stats(grams, vec)
                for name, vec in used_vectors.items()
            }
            base = stats["base"]["mean"]
            j = stats["weighted_j"]["mean"]
            result[group][module] = {
                "norms": stats,
                "j_base_ratio": j / base if base > 0 else None,
                "h_j_ratio": stats["weighted_h"]["mean"] / j
                if j > 0
                else None,
                "base_j_cosine": cosine_stats(
                    grams, used_vectors["base"], used_vectors["weighted_j"]
                ),
                "j_h_cosine": cosine_stats(
                    grams,
                    used_vectors["weighted_j"],
                    used_vectors["weighted_h"],
                ),
            }
    return result


def runtime_imports():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import numpy as np
    import torch

    from data.dual_dataset import DualDataset
    from util.masking_generator import MaskingGenerator

    return torch, np, DualDataset, MaskingGenerator


def seed_all(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataset(args):
    _, _, dataset_class, masking_class = runtime_imports()
    return dataset_class(
        args.data_path,
        args.json_path,
        args.target_cache,
        masking_class((56, 28), num_masking_patches=784, max_num_patches=392),
        is_train=True,
        num_mask_annotations_bf=11,
        mask_coverage_threshold=0.1,
        jieti_pool=224,
        jieti_valid_ink=200,
    )


def load_dual_model(checkpoint, device, seed=0, allow_new=False):
    import torch

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    import models_train as models

    seed_all(seed)
    model = models.vit_base_patch16_input896x448_win_dec64_8glb_sl1(
        dual_view=True
    )
    saved = torch.load(checkpoint, map_location="cpu")
    state = saved["model"]
    message = model.load_state_dict(state, strict=False)
    allowed = (
        "dual_head.",
        "dual_fusion.",
        "dual_jieti_loss.",
        "dual_head_loss.",
    )
    missing = [
        name
        for name in message.missing_keys
        if not (allow_new and name.startswith(allowed))
    ]
    if missing or message.unexpected_keys:
        raise ValueError(f"checkpoint keys do not match: {message}")
    model.style_weight = 14.73
    model.fusion_lambda = 0.5
    model.edge_warmup_epochs = 10
    model.loss_warmup_duration = 8
    model.edge_weight_final = 0.2
    model.enable_dual_losses(0.5, 1.0, 1.0, 1.0, 1.0, 1.0)
    for name, parameter in model.named_parameters():
        if name.startswith(("discriminator.", "vgg_loss.")):
            parameter.requires_grad_(False)
    return model.to(device), saved


def apply_weights(model, weights):
    model.enable_dual_losses(
        weights["lambda_full"],
        weights["w_j"],
        weights["w_h"],
        *[weights["coef"][key] for key in COEF_NAMES],
    )
    if weights.get('image_detail_loss') == 'tip' and 'tip_final_weight' in weights:
        model.enable_tip(weights['tip_final_weight'])


def fixed_batch(dataset, rows, device):
    import torch

    batch = torch.utils.data.default_collate(
        [
            dataset.get_pair(
                row["target_index"],
                row["reference_index"],
                row["augmentation_seed"],
            )
            for row in rows
        ]
    )
    return {
        key: val.to(device) if torch.is_tensor(val) else val
        for key, val in batch.items()
    }


def forward_terms(model, batch, epoch=0):
    import torch

    device_type = batch["image"].device.type
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
        result = model(
            batch["image"].to(torch.bfloat16),
            batch["target"].to(torch.bfloat16),
            batch["mask"].flatten(1).to(torch.bfloat16),
            batch["valid"].to(torch.bfloat16),
            epoch=epoch,
            no_gan=True,
            dual_targets=batch,
        )
    extra = result[-1]
    if not isinstance(extra, dict) or any(key not in extra for key in TERMS):
        raise ValueError("dual model did not expose differentiable loss terms")
    if any(not torch.isfinite(extra[key]).all() for key in TERMS):
        raise ValueError("nonfinite dual loss")
    return extra


def _writer(record):
    for key in ("calligrapher", "writer", "author", "style_id"):
        if key in record:
            return str(record[key])
    raise ValueError("dual records must explicitly identify the writer")


def _pool(record):
    return (
        "dual"
        if record["has_bf"] and record["has_jt"]
        else "bf_only"
        if record["has_bf"]
        else "jt_only"
    )


def choose_fixed_rows(records, weights, indices, count, rng,
                      reference_candidates=None):
    """Draw distinct targets with D weights and uniform within-pool refs."""
    available = list(indices)
    count = min(count, len(available))
    groups = {}
    for index, record in enumerate(records):
        groups.setdefault((_writer(record), _pool(record)), []).append(index)
    rows = []
    for _ in range(count):
        index = rng.choices(available, [float(weights[i]) for i in available])[
            0
        ]
        available.remove(index)
        record = records[index]
        candidates = (
            reference_candidates[index] if reference_candidates is not None
            else groups[(_writer(record), _pool(record))]
        )
        reference = rng.choice(candidates)
        rows.append(
            {
                "target_index": index,
                "reference_index": reference,
                "writer": _writer(record),
                "pool": _pool(record),
                "target_path": record["target_path"],
                "reference_path": records[reference]["target_path"],
                "augmentation_seed": rng.randrange(2**31),
                "forward_seed": rng.randrange(2**31),
            }
        )
    return rows


def create_samples(args):
    if not 64 <= args.targets <= 128:
        raise ValueError(
            "JT calibration requires 64 through 128 fixed targets"
        )
    dataset = build_dataset(args)
    jt = [i for i, row in enumerate(dataset.pairs) if row["has_jt"]]
    bf = [
        i
        for i, row in enumerate(dataset.pairs)
        if row["has_bf"] and not row["has_jt"]
    ]
    if len(jt) < args.targets or not bf:
        raise ValueError(
            "insufficient JT targets or no BF-only diagnostic pool"
        )
    rng = random.Random(args.seed)
    all_rows = range(len(dataset))
    references = dataset.reference_candidates
    groups = {
        "jt": choose_fixed_rows(
            dataset.pairs, dataset.weights, jt, args.targets, rng, references
        ),
        "bf_only": choose_fixed_rows(
            dataset.pairs, dataset.weights, bf, args.diagnostic_targets, rng,
            references,
        ),
        "mixed": choose_fixed_rows(
            dataset.pairs,
            dataset.weights,
            all_rows,
            args.diagnostic_targets,
            rng,
            references,
        ),
    }
    if {row["pool"] for row in groups["jt"]} != {"dual", "jt_only"}:
        raise ValueError(
            "fixed JT targets must cover dual and JT-only pools; "
            "use another seed"
        )
    if {row["pool"] for row in groups["mixed"]} != {
        "dual",
        "bf_only",
        "jt_only",
    }:
        raise ValueError(
            "mixed probe must cover all three pools; increase diagnostics"
        )
    shares = {}
    for index in jt:
        name = f"{_writer(dataset.pairs[index])}:{_pool(dataset.pairs[index])}"
        shares[name] = shares.get(name, 0) + float(dataset.weights[index])
    total = sum(shares.values())
    result = {
        "schema": SCHEMA,
        "purpose": "fixed_gradient_samples",
        "seed": args.seed,
        "batch_size": args.batch_size,
        "sampling": (
            "D weights without replacement; same-pool uniform reference"
        ),
        "jt_pool_expected_share": {
            key: v / total for key, v in shares.items()
        },
        "train_manifest_sha256": hash_files(args.json_path),
        "target_cache_sha256": cache_hash(args.target_cache),
        "groups": groups,
        "reference_policy": dataset.reference_policy,
    }
    write_new_json(args.output, result)
    print(
        json.dumps(
            {
                "output": args.output,
                "counts": {key: len(rows) for key, rows in groups.items()},
            }
        )
    )


def check_samples(value, args):
    manifest = read_json(value)
    if manifest.get("schema") != SCHEMA:
        raise ValueError("old sample manifest")
    if manifest["train_manifest_sha256"] != hash_files(args.json_path):
        raise ValueError(
            "sample manifest does not match current training inputs"
        )
    if manifest["target_cache_sha256"] != cache_hash(args.target_cache):
        raise ValueError(
            "sample manifest does not match current instance cache"
        )
    policy = read_json(Path(args.target_cache) / "metadata.json").get(
        "reference_policy"
    )
    if manifest.get("reference_policy") != policy:
        raise ValueError("Probe reference policy differs from target cache")
    if not 64 <= len(manifest["groups"]["jt"]) <= 128:
        raise ValueError(
            "sample manifest must contain 64 through 128 JT targets"
        )
    if not manifest["groups"]["bf_only"] or not manifest["groups"]["mixed"]:
        raise ValueError("BF-only and mixed probes are required")
    return manifest


def check_head(model, dataset, samples, device):
    """Repeat fixed train/bf16 checks with identical augmentation/dropout."""
    import torch

    rows = samples["groups"]["jt"]
    values, predictions, existence = [], [], []
    size = samples["batch_size"]
    model.train()
    with torch.no_grad():
        for start in range(0, len(rows), size):
            current = rows[start : start + size]
            seed_all(current[0]["forward_seed"])
            extra = forward_terms(model, fixed_batch(dataset, current, device))
            loss = extra.get(
                "head_occupancy_loss", extra.get("head_occupancy")
            )
            if loss is None:
                raise ValueError(
                    "head occupancy BCE/Dice loss must be exposed"
                )
            values.append((float(loss), len(current)))
            predictions.append(
                extra["head_occupancy_logits"].sigmoid().float()
            )
            existence.append(extra["head_existence_logits"].sigmoid().float())
    occupancy = torch.cat(predictions)
    active = torch.cat(existence) >= 0.5
    binary = occupancy >= 0.5
    present = (binary & active[:, :, None, None]).flatten(1).any(1)
    full = (binary & active[:, :, None, None]).any(1).flatten(1).all(1)
    mean_loss = sum(value * n for value, n in values) / sum(
        n for _, n in values
    )
    return {
        "occupancy_loss": mean_loss,
        "mean_occupancy_probability": float(occupancy.mean()),
        "empty_glyph_fraction": float((~present).float().mean()),
        "full_glyph_fraction": float(full.float().mean()),
        "collapsed": bool((~present).all() or full.all()),
    }


def warmup(args):
    """Use a separate head-only optimizer lifetime before formal DDP."""
    import torch

    if Path(args.output).exists() or Path(args.output + ".json").exists():
        raise FileExistsError(args.output)
    if args.device.split(":")[0] != "cuda":
        raise ValueError(
            "full-model warm-up requires CUDA bf16; CPU is unit-test only"
        )
    if args.batch_size <= 0 or 64 % args.batch_size:
        raise ValueError(
            "single-process warm-up batch size must divide effective batch 64"
        )
    dataset = build_dataset(args)
    samples = check_samples(args.samples, args)
    model, _ = load_dual_model(
        args.checkpoint, args.device, args.seed, allow_new=True
    )
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("dual_head."))
    groups = [
        {
            "params": [p for p in model.dual_head.parameters() if p.ndim > 1],
            "weight_decay": 0.1,
        },
        {
            "params": [p for p in model.dual_head.parameters() if p.ndim <= 1],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(groups, lr=1e-3, betas=(0.9, 0.999))
    baseline = check_head(model, dataset, samples, args.device)
    checks, updates = [], 0
    accum = 64 // args.batch_size
    seed_all(args.seed)
    for epoch in range(2):
        generator = torch.Generator().manual_seed(args.seed + epoch)
        # Full effective batches only; the drop count is recorded.
        count = len(dataset) // 64 * 64
        if count == 0:
            raise ValueError("warm-up dataset has fewer than 64 samples")
        sampler = torch.utils.data.WeightedRandomSampler(
            dataset.weights, count, replacement=True, generator=generator
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0
        )
        optimizer.zero_grad(set_to_none=True)
        model.train()
        for step, batch in enumerate(loader):
            batch = {
                key: val.to(args.device) if torch.is_tensor(val) else val
                for key, val in batch.items()
            }
            extra = forward_terms(model, batch)
            (extra["L_H"] / accum).backward()
            if (step + 1) % accum == 0:
                norm = torch.nn.utils.clip_grad_norm_(
                    model.dual_head.parameters(), 3.0
                )
                if not torch.isfinite(norm):
                    raise ValueError("nonfinite head warm-up gradient")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
        check = check_head(model, dataset, samples, args.device)
        check["epoch"] = epoch + 1
        checks.append(check)
        print(json.dumps(check), flush=True)
        if (
            check["occupancy_loss"] < baseline["occupancy_loss"]
            and not check["collapsed"]
        ):
            break
    passed = (
        checks[-1]["occupancy_loss"] < baseline["occupancy_loss"]
        and not checks[-1]["collapsed"]
    )
    metadata = {
        "schema": SCHEMA,
        "purpose": "head_warmup",
        "status": "validated" if passed else "diagnostic_required",
        "parent_checkpoint_sha256": sha256_file(args.checkpoint),
        "sample_manifest_sha256": sha256_file(args.samples),
        "train_manifest_sha256": hash_files(args.json_path),
        "target_cache_sha256": cache_hash(args.target_cache),
        "seed": args.seed,
        "updates": updates,
        "epochs": len(checks),
        "effective_batch": 64,
        "baseline": baseline,
        "checks": checks,
        "optimizer": {
            "lr": 1e-3,
            "betas": [0.9, 0.999],
            "weight_decay": 0.1,
            "clip_grad": 3.0,
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "dual_warmup": metadata,
            "dual_view": True,
            "epoch": -1,
            "dual_view_config": {
                "lambda_full": 0.5,
                "w_j": 1.0,
                "w_h": 1.0,
                "coef": dict.fromkeys(COEF_NAMES, 1.0),
                "pool": 224,
                "soft_fg": "linear",
                "pred_mass_ratio": 0.1,
            },
        },
        args.output,
    )
    write_new_json(args.output + ".json", metadata)
    if not passed:
        raise RuntimeError(
            "head failed to improve or collapsed after two epochs; "
            "diagnose before calibration"
        )


def current_hashes(args):
    return {
        "train_manifest_sha256": hash_files(args.json_path),
        "target_cache_sha256": cache_hash(args.target_cache),
        "sample_manifest_sha256": sha256_file(args.samples),
        "warm_checkpoint_sha256": sha256_file(args.checkpoint),
        "parent_checkpoint_sha256": sha256_file(args.parent_checkpoint),
        "fixed_pair_sha256": sha256_file(args.fixed_pair_path),
        "val_manifest_sha256": hash_files(args.val_json_path),
        "eval_coef_sha256": sha256_file(args.eval_coef),
        "s_baseline_sha256": sha256_file(args.s_baseline),
    }


def check_eval_inputs(args, hashes):
    from tools.eval_s_baseline import validate_eval_coefficients

    coef, baseline = read_json(args.eval_coef), read_json(args.s_baseline)
    validate_eval_coefficients(
        coef,
        {
            key: hashes[key]
            for key in (
                "fixed_pair_sha256",
                "val_manifest_sha256",
                "target_cache_sha256",
            )
        },
    )
    for value, purpose in ((coef, "evaluation"), (baseline, "s_baseline")):
        if value.get("schema") != SCHEMA or value.get("purpose") != purpose:
            raise ValueError(f"new {purpose} record required")
        for key in (
            "fixed_pair_sha256",
            "val_manifest_sha256",
            "target_cache_sha256",
        ):
            if value.get(key) != hashes[key]:
                raise ValueError(f"{purpose} inputs mismatch: {key}")
    if baseline.get("eval_coef_sha256") != hashes["eval_coef_sha256"]:
        raise ValueError(
            "S denominator was computed with other evaluation coefficients"
        )
    if coef.get("source_epoch") != 46 or baseline.get("source_epoch") != 50:
        raise ValueError(
            "evaluation coefficients require ctrl_g12 best ep46; "
            "denominator requires baseline ep50"
        )
    if baseline.get("source_role") != "baseline_vggfix_ep50":
        raise ValueError("S denominator must come from baseline vggfix ep50")
    digest = baseline.get("source_checkpoint_sha256", "")
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("S denominator source checkpoint hash is missing")
    reconciliation = baseline.get("legacy_l1_reconciliation", {})
    if reconciliation.get("status") != "passed":
        raise ValueError(
            "old-val L1 reconciliation must pass "
            "before a new S denominator is used"
        )
    for key in ("L1_JT", "L1_BF", "J"):
        _positive(baseline[key], f"S denominator {key}")


def calibrate(args):
    from tools.dual_gradient_probe import measure_batches

    stages = ("initial", "lambda", "inner_j", "final")
    destinations = [args.output] + [f"{args.output}.{s}.json" for s in stages]
    if any(Path(path).exists() for path in destinations):
        raise FileExistsError("calibration output/stages must be new files")
    dataset = build_dataset(args)
    samples = check_samples(args.samples, args)
    hashes = current_hashes(args)
    check_eval_inputs(args, hashes)
    model, checkpoint = load_dual_model(
        args.checkpoint, args.device, args.seed
    )
    if getattr(args, 'image_detail_loss', 'edge') == 'tip':
        # Both details have zero weight at epoch 0. Still measure the actual
        # new graph, preserving the original sequential calibration protocol.
        model.enable_tip(0.0)
        model.edge_weight_final = 0.0
    warm = checkpoint.get("dual_warmup", {})
    if (
        warm.get("status") != "validated"
        or warm.get("parent_checkpoint_sha256")
        != hashes["parent_checkpoint_sha256"]
    ):
        raise ValueError(
            "validated head warm-up from the supplied ckpt14 is required"
        )
    for name in (
        "sample_manifest_sha256",
        "train_manifest_sha256",
        "target_cache_sha256",
    ):
        if warm.get(name) != hashes[name]:
            raise ValueError(f"warm-up used different inputs: {name}")
    first = measure_batches(model, dataset, samples, args.device, epoch=0)
    stage_paths = {
        "initial": save_calibration_stage(
            args.output, "initial", first, hashes,
        ),
    }
    weights = derive_weights(first)
    lambda_measurements = []
    if weights["lambda_full"] > 0.5:
        lambda_stage = {
            **weights,
            "w_j": 1,
            "w_h": 1,
            "coef": dict.fromkeys(COEF_NAMES, 1),
        }
        apply_weights(model, lambda_stage)
        lambda_measurements = measure_batches(
            model, dataset, samples, args.device, epoch=0, groups=("jt",)
        )
        checked = weight_summary(lambda_measurements, lambda_stage)
        decoder_norms = checked["jt"]["decoder"]["norms"]
        measured_ratio = (
            decoder_norms["weighted_full"]["mean"]
            / decoder_norms["pen"]["mean"]
        )
        stage_paths["lambda"] = save_calibration_stage(
            args.output, "lambda", lambda_measurements, hashes, lambda_stage,
        )
        check_gradient_ratio(measured_ratio, 0.5, "lambda_decoder_ratio")
    else:
        stage_paths["lambda"] = save_calibration_stage(
            args.output, "lambda", [], hashes, weights,
        )
    # Set freshly derived inner coefficients before measuring the actual J
    # gradient for the outer weights. This also tests the composed loss graph.
    inner_stage = {**weights, "w_j": 1, "w_h": 1}
    apply_weights(model, inner_stage)
    j_measurements = measure_batches(
        model, dataset, samples, args.device, epoch=0, groups=("jt",)
    )
    stage_paths["inner_j"] = save_calibration_stage(
        args.output, "inner_j", j_measurements, hashes, inner_stage,
    )
    actual = weight_summary(j_measurements, inner_stage)["jt"]["vit"]["norms"]
    for name in ("base", "weighted_j", "weighted_h"):
        _positive(actual[name]["mean"], f"actual composed {name} gradient")
    weights["w_j"] = (
        0.5 * actual["base"]["mean"] / actual["weighted_j"]["mean"]
    )
    weights["w_h"] = (
        0.5
        * weights["w_j"]
        * actual["weighted_j"]["mean"]
        / actual["weighted_h"]["mean"]
    )
    # Independently remeasure the final fixed outer and inner weights.
    apply_weights(model, weights)
    second = measure_batches(model, dataset, samples, args.device, epoch=0)
    stage_paths["final"] = save_calibration_stage(
        args.output, "final", second, hashes, weights,
    )
    summaries = weight_summary(second, weights)
    ratio = (
        summaries["jt"]["decoder"]["norms"]["weighted_full"]["mean"]
        / summaries["jt"]["decoder"]["norms"]["pen"]["mean"]
    )
    check_gradient_ratio(ratio, 0.5, "decoder_full_pen_ratio", minimum=True)
    for ratio_name, target in (("j_base_ratio", 0.5), ("h_j_ratio", 0.5)):
        actual = summaries["jt"]["vit"][ratio_name]
        check_gradient_ratio(actual, target, ratio_name)
    result = {
        "schema": SCHEMA,
        "purpose": "training",
        "status": "validated",
        "edge_weight": 0,
        "image_detail_loss": getattr(args, 'image_detail_loss', 'edge'),
        "train_mode": True,
        "autocast": "bfloat16",
        "norm_rule": "ratio_of_mean_batch_norms",
        "j_calibration_space": "shared_vit_parameters",
        "j_coef_rule": "shape_mean_norm_reference",
        "remeasurement_precision": {
            "relative_tolerance": RATIO_RTOL,
            "absolute_tolerance": RATIO_ATOL,
            "target_j_base_ratio": 0.5, "target_h_j_ratio": 0.5,
            "evidence": "fixed-seed CUDA/bf16 backward repeats; forward exact",
        },
        "measurement_stage_files": stage_paths,
        **weights,
        **hashes,
        "summary": summaries,
        "initial_measurements": first,
        "lambda_remeasured": lambda_measurements,
        "j_coef_measurements": j_measurements,
        "remeasured": second,
    }
    write_new_json(args.output, result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("lambda_full", "w_j", "w_h", "coef")
            },
            indent=2,
        )
    )


def preflight(args):
    hashes = current_hashes(args)
    value = validate_calibration(args.calibration, hashes)
    check_samples(args.samples, args)
    check_eval_inputs(args, hashes)
    sidecar = read_json(args.checkpoint + ".json")
    if (
        sidecar.get("status") != "validated"
        or sidecar.get("parent_checkpoint_sha256")
        != hashes["parent_checkpoint_sha256"]
    ):
        raise ValueError(
            "warm-up did not pass or was initialized from another checkpoint"
        )
    print(
        json.dumps({"status": "preflight_passed", "schema": value["schema"]})
    )


def validate_training_inputs(args):
    """Pure-stdlib preflight after distributed initialization in main."""
    required = (
        "dual_calibration_json",
        "dual_eval_calibration_json",
        "dual_target_cache",
        "dual_sample_manifest",
        "finetune",
        "dual_parent_checkpoint",
        "fixed_pair_path",
        "s_baseline_path",
    )
    for name in required:
        if not getattr(args, name, None):
            raise ValueError(f"Run B requires --{name}")
    checks = {
        "no_gan": True,
        "freeze_encoder": False,
        "fusion_lambda": 0.5,
        "half_mask_ratio": 0.0,
        "jieti_pool": 224,
        "jieti_k_max": 4,
        "jieti_alpha_jt": 1.0,
        "jieti_soft_fg": "linear",
        "jieti_pred_mass_ratio": 0.1,
        "jieti_valid_ink": 200,
        "num_mask_annotations_bf": 11,
        "mask_coverage_threshold": 0.1,
        "style_weight": 14.73,
        "clip_grad": 3.0,
        "drop_path": 0.1,
        "layer_decay": 0.8,
        "augmentation_policy": "finetune",
        "seed": 0,
        "model": "vit_base_patch16_input896x448_win_dec64_8glb_sl1",
        "num_mask_patches": 784,
        "max_mask_patches_per_block": 392,
        "edge_warmup_epochs": 10,
        "loss_warmup_duration": 8,
        "edge_weight_final": 0.0 if getattr(args, 'tip_weight', None) is not None
        else 0.2,
        "warmup_epochs": 10,
        "lr": 0.00195699,
        "min_lr": 0.0,
        "save_best": True,
        "save_best_only": False,
        "save_freq": 10,
    }
    for name, expected in checks.items():
        if getattr(args, name, None) != expected:
            raise ValueError(f"Run B requires {name}={expected}")
    if list(args.input_size) != [896, 448]:
        raise ValueError("Run B requires input_size 896 448")
    effective = (
        args.batch_size * args.accum_iter * getattr(args, "world_size", 1)
    )
    if effective != 64:
        raise ValueError(f"Run B requires effective batch 64, got {effective}")
    if getattr(args, "auto_resume", False):
        raise ValueError(
            "Run B must start from the validated warm-up, not auto-resume"
        )
    if getattr(args, "resume", "") or getattr(args, "enable_deepspeed", False):
        raise ValueError("Run B uses a fresh Native DDP optimizer, no resume")
    output = Path(args.output_dir)
    if any(output.glob("checkpoint*")) or any(
        (output / name).exists() for name in ("log.txt", "hparams.json")
    ):
        raise ValueError(
            "Run B output contains previous results; choose a new directory"
        )
    blind_path = getattr(args, "dual_blind_manifest", None)
    if not blind_path:
        raise ValueError(
            "Run B requires a fixed --dual_blind_manifest before training"
        )
    blind = read_json(blind_path)
    if (
        blind.get("schema") != SCHEMA
        or blind.get("purpose") != "fixed_blind_manifest"
    ):
        raise ValueError("invalid blind manifest")
    if blind.get("fixed_pair_sha256") != sha256_file(args.fixed_pair_path):
        raise ValueError("blind manifest belongs to another validation set")
    args.dual_blind_manifest_sha256 = sha256_file(blind_path)
    args.dual_reference_policy = read_json(
        Path(args.dual_target_cache) / "metadata.json"
    ).get("reference_policy")
    mapped = argparse.Namespace(
        json_path=args.json_path,
        val_json_path=args.val_json_path,
        target_cache=args.dual_target_cache,
        samples=args.dual_sample_manifest,
        checkpoint=args.finetune,
        parent_checkpoint=args.dual_parent_checkpoint,
        fixed_pair_path=args.fixed_pair_path,
        eval_coef=args.dual_eval_calibration_json,
        s_baseline=args.s_baseline_path,
    )
    hashes = current_hashes(mapped)
    value = validate_calibration(args.dual_calibration_json, hashes)
    if getattr(args, 'tip_weight', None) is not None:
        from util.tip_calibration import validate_tip_calibration

        validate_tip_calibration(
            args.tip_calibration_json, args.tip_weight,
            args.dual_calibration_json, hashes,
        )
    elif (getattr(args, 'tip_calibration_json', None)
          or value.get('image_detail_loss') == 'tip'):
        raise ValueError('Tip calibration requires explicit --tip_weight')
    check_samples(args.dual_sample_manifest, mapped)
    check_eval_inputs(mapped, hashes)
    metadata = read_json(args.finetune + ".json")
    if metadata.get("status") != "validated":
        raise ValueError(
            "head warm-up failed or its validation record is missing"
        )
    for name in (
        "parent_checkpoint_sha256",
        "sample_manifest_sha256",
        "train_manifest_sha256",
        "target_cache_sha256",
    ):
        if metadata.get(name) != hashes[name]:
            raise ValueError(f"warm-up metadata mismatch: {name}")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("samples", "warmup", "calibrate", "preflight"):
        sub = commands.add_parser(command)
        sub.add_argument("--data-path", required=True)
        sub.add_argument("--json-path", nargs="+", required=True)
        sub.add_argument("--target-cache", required=True)
        sub.add_argument("--seed", type=int, default=0)
        if command == "samples":
            sub.add_argument("--targets", type=int, default=96)
            sub.add_argument("--diagnostic-targets", type=int, default=32)
            sub.add_argument("--batch-size", type=int, default=2)
        else:
            sub.add_argument("--samples", required=True)
            sub.add_argument("--checkpoint", required=True)
            sub.add_argument("--device", default="cuda")
        if command == "warmup":
            sub.add_argument("--batch-size", type=int, default=2)
        if command == 'calibrate':
            sub.add_argument('--image-detail-loss', choices=('edge', 'tip'),
                             default='edge')
        if command in ("calibrate", "preflight"):
            sub.add_argument("--parent-checkpoint", required=True)
            sub.add_argument("--val-json-path", nargs="+", required=True)
            sub.add_argument("--fixed-pair-path", required=True)
            sub.add_argument("--eval-coef", required=True)
            sub.add_argument("--s-baseline", required=True)
        if command == "preflight":
            sub.add_argument("--calibration", required=True)
        else:
            sub.add_argument("--output", required=True)
    args = parser.parse_args()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    {
        "samples": create_samples,
        "warmup": warmup,
        "calibrate": calibrate,
        "preflight": preflight,
    }[args.command](args)


if __name__ == "__main__":
    main()
