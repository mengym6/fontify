"""Evaluate legacy J or the shared run B instance/lower/full S contract.

Legacy CLI remains supported. New denominators require a passed legacy L1
reconciliation, the clean 240 fixed pairs, and the ctrl_g12 evaluation coefficients.
Formal checkpoint loading is strict for both architectures.
"""

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DUAL_SCHEMA = "instance-lower-full-v1"
CLEAN_FIXED_MD5 = "09894e6abdca5231169057ad53d5deec"
TERMS = ("centroid", "logsigma", "shape")
S_TERMS = ("L1_JT", "L1_BF", "J")
EVAL_HASH_KEYS = (
    "fixed_pair_sha256", "val_manifest_sha256", "target_cache_sha256",
    "eval_coef_sha256",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def eval_metadata(args):
    """Bind evaluation to versioned lists, fixed pairing and audited raw cache."""
    from tools.dual_view_protocol import hash_files

    fixed = Path(args.fixed_pair_path)
    data = fixed.read_bytes()
    if hashlib.md5(data).hexdigest() != CLEAN_FIXED_MD5:
        raise ValueError("run B evaluation requires the approved clean fixed pairs")
    if len(json.loads(data)) != 240:
        raise ValueError("run B evaluation requires exactly 240 fixed pairs")
    cache = Path(args.dual_target_cache)
    cache_meta = json.loads((cache / "metadata.json").read_text())
    if cache_meta.get("definition") != DUAL_SCHEMA:
        raise ValueError("Target cache has the wrong J definition")
    return {
        "fixed_pair_sha256": file_sha256(fixed),
        "val_manifest_sha256": hash_files(args.val_json_path),
        "target_cache_sha256": hash_files(
            [cache / "index.json", cache / "metadata.json"]
        ),
    }


def validate_eval_coefficients(payload, metadata):
    """Reject training/legacy coefficients and stale evaluation target hashes."""
    if payload.get("schema") != DUAL_SCHEMA or payload.get("purpose") != "evaluation":
        raise ValueError("Expected instance-lower-full-v1 evaluation coefficients")
    if payload.get("status") != "validated":
        raise ValueError("Evaluation coefficients have not passed calibration")
    if (
        payload.get("source_role") != "ctrl_g12_best"
        or payload.get("source_epoch") != 46
    ):
        raise ValueError("Evaluation coefficients must come from ctrl_g12 best ep46")
    for key, value in metadata.items():
        if payload.get(key) != value:
            raise ValueError(f"Evaluation coefficient metadata mismatch: {key}")
    digest = payload.get("source_checkpoint_sha256", "")
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("Evaluation coefficient checkpoint hash is missing")
    coef = payload.get("coef", {})
    if set(coef) != set(TERMS) or any(
        isinstance(coef[name], bool)
        or not math.isfinite(float(coef[name]))
        or float(coef[name]) <= 0
        for name in TERMS
    ):
        raise ValueError("J coefficients must be three positive finite scalars")
    return {name: float(coef[name]) for name in TERMS}


def absolute_s_score(result, denominator):
    """Normalize three cohort means; J is averaged over JT samples only."""
    ratios = {}
    for key in S_TERMS:
        base, value = denominator.get(key), result.get(key)
        if (
            isinstance(base, bool) or not isinstance(base, (int, float))
            or not math.isfinite(base) or base <= 0
        ):
            raise ValueError(f"S denominator {key} must be positive and finite")
        if (
            isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0
        ):
            raise ValueError(f"S numerator {key} must be nonnegative and finite")
        ratios[key] = float(value) / float(base)
    return sum(ratios.values()), ratios


def validate_s_baseline(payload, metadata):
    """Reject old/stale denominators before adding an absolute new S score."""
    if payload.get("schema") != DUAL_SCHEMA or payload.get("purpose") != "s_baseline":
        raise ValueError("Expected instance-lower-full-v1 S denominator")
    for key in EVAL_HASH_KEYS:
        if not metadata.get(key) or payload.get(key) != metadata[key]:
            raise ValueError(f"S denominator metadata mismatch: {key}")
    if (
        payload.get("source_role") != "baseline_vggfix_ep50"
        or payload.get("source_epoch") != 50
    ):
        raise ValueError("S denominator must come from baseline vggfix ep50")
    digest = payload.get("source_checkpoint_sha256", "")
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("S denominator source checkpoint hash is missing")
    if payload.get("legacy_l1_reconciliation", {}).get("status") != "passed":
        raise ValueError("S denominator requires passed old-val L1 reconciliation")
    absolute_s_score(payload, payload)


def reconcile_legacy_l1(result, reference, atol=1e-7, rtol=1e-6):
    """Check old-val reconstruction only; old/new J are never reconciled."""
    for key in ("L1_JT", "L1_BF"):
        left, right = float(result[key]), float(reference[key])
        if not math.isfinite(left) or not math.isfinite(right):
            raise ValueError(f"Nonfinite legacy L1: {key}")
        if not math.isclose(left, right, abs_tol=atol, rel_tol=rtol):
            raise ValueError(
                f"Legacy L1 reconciliation failed: {key} {left} != {right}"
            )
    for key in ("n_jt", "n_bf"):
        if key in reference and result[key] != reference[key]:
            raise ValueError(f"Legacy L1 sample-count mismatch: {key}")
    return {
        "passed": True,
        "status": "passed",
        "atol": atol,
        "rtol": rtol,
        "actual": {key: result[key] for key in ("L1_JT", "L1_BF")},
        "reference": {key: reference[key] for key in ("L1_JT", "L1_BF")},
    }


def build_val_transform(identity=False):
    import data.pair_transforms as transforms

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    geometry = [transforms.PadToSquare(fill=255)]
    if not identity:
        geometry.append(
            transforms.RandomResizedCrop(448, scale=(0.9999, 1.0), interpolation=3)
        )
    return transforms.Compose(geometry + [transforms.ToTensor(), normalize])


def load_model(checkpoint_path):
    import torch

    from util.checkpoint import build_checkpoint_model, checkpoint_args

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_checkpoint_model(checkpoint)
    model._checkpoint_epoch = checkpoint.get("epoch")
    model._checkpoint_sha256 = file_sha256(checkpoint_path)
    model._checkpoint_source = str(checkpoint_path)
    model._checkpoint_args = checkpoint_args(checkpoint)
    return model


def require_source_model(model, role):
    """Verify the saved training identity before labeling a scientific source."""
    expected = {
        "ctrl_g12_best": ("finetune_jieti_ctrl_g12", 46),
        "baseline_vggfix_ep50": ("finetune_no_gan_nojt_no_freeze_baseline_vggfix", 50),
    }
    directory, epoch = expected[role]
    saved_directory = Path(model._checkpoint_args.get("output_dir", "")).name
    if saved_directory != directory or model._checkpoint_epoch != epoch:
        raise ValueError(f"Wrong checkpoint training identity for {role}")
    if getattr(model, "dual_view", False):
        raise ValueError(f"{role} must use the legacy architecture")
    if (
        role == "ctrl_g12_best"
        and Path(model._checkpoint_source).name != "checkpoint-best.pth"
    ):
        raise ValueError("Evaluation calibration requires checkpoint-best.pth")


def build_dataset(args, definition=None, identity=False):
    from util.masking_generator import MaskingGenerator

    masking = MaskingGenerator((56, 28), num_masking_patches=784, max_num_patches=392)
    definition = definition or getattr(args, "j_definition", "legacy")
    if definition == DUAL_SCHEMA:
        from data.dual_dataset import DualDataset

        if not args.dual_target_cache:
            raise ValueError("Instance J evaluation requires --dual_target_cache")
        return DualDataset(
            args.data_path,
            args.val_json_path,
            args.dual_target_cache,
            masking,
            is_train=False,
            fixed_pair_path=args.fixed_pair_path,
        )
    from data.pairdataset import PairDataset

    return PairDataset(
        args.data_path,
        args.val_json_path,
        transform=build_val_transform(identity),
        masked_position_generator=masking,
        use_two_pairs=True,
        half_mask_ratio=1.0,
        semantic_mask_dir=args.semantic_mask_dir,
        num_mask_annotations_jt=1,
        num_mask_annotations_bf=11,
        mask_coverage_threshold=0.1,
        return_jieti=True,
        fixed_pair_path=args.fixed_pair_path,
    )


def unpack_sample(sample):
    """Normalize containers without forwarding annotations to the model."""
    if isinstance(sample, dict):
        return (
            sample["image"],
            sample["target"],
            sample["mask"],
            sample["valid"],
            sample["voro"],
            sample["valid_parts"],
            sample["has_jt"],
        )
    return sample


def make_jieti_loss(definition, coef):
    if definition == DUAL_SCHEMA:
        from util.dual_loss import LowerJietiLoss as Loss
    else:
        from util.jieti_loss import JietiLoss as Loss
    return Loss(
        w_centroid=coef["centroid"],
        w_logsigma=coef["logsigma"],
        w_shape=coef["shape"],
    )


def evaluate(
    model, dataset, jmod, device, definition="legacy", collect_predictions=None
):
    """Use identical per-item reconstruction for legacy and dual models."""
    import numpy as np
    import torch

    scores = {"JT": [], "BF": [], "J": []}
    rows = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            image, target, mask, valid, voro, valid_parts, is_jt = unpack_sample(
                dataset[idx]
            )
            images, targets = image[None].to(device), target[None].to(device)
            bmp = torch.as_tensor(mask, device=device).bool().flatten()[None]
            jt = bool(torch.as_tensor(is_jt).item())
            if (
                not bmp[:, bmp.shape[1] // 2 :].all()
                or bmp[:, : bmp.shape[1] // 2].any()
            ):
                raise ValueError(f"Sample {idx}: evaluation requires lower-full mask")
            # Annotations enter only the metric after prediction.
            pred = model.predict_full(images, targets, bmp)
            mask_px = bmp[:, :, None].repeat(1, 1, model.patch_size**2 * 3).float()
            mask_px = model.unpatchify(mask_px)
            area = (
                mask_px * valid[None].to(device)
                if definition == DUAL_SCHEMA
                else mask_px
            )
            l1 = ((pred - targets).abs() * area).sum() / (area.sum() + 1e-2)
            if not torch.isfinite(l1):
                raise FloatingPointError(f"Nonfinite validation L1 for sample {idx}")
            kind = "JT" if jt else "BF"
            scores[kind].append(float(l1))
            reference = dataset.pairs[idx]
            query = dataset._fixed_pairs[reference["target_path"]]
            row = {
                "idx": idx,
                "index": idx,
                "kind": kind,
                "is_jt": jt,
                "target_path": query["target_path"],
                "ref_path": reference["target_path"],
                "type": query["type"],
                "L1": float(l1),
                "J": None,
            }
            if jt and jmod is not None:
                composite = pred * mask_px + targets * (1 - mask_px)
                j_value, parts, _ = jmod(
                    composite,
                    targets,
                    voro[None].to(device),
                    valid_parts[None].to(device),
                    torch.tensor([jt], device=device),
                )
                if not torch.isfinite(j_value) or any(
                    not torch.isfinite(parts[name]) for name in TERMS
                ):
                    raise FloatingPointError(f"Nonfinite validation J for sample {idx}")
                scores["J"].append(float(j_value))
                row.update(
                    J=float(j_value), **{name: float(parts[name]) for name in TERMS}
                )
            rows.append(row)
            if collect_predictions is not None:
                collect_predictions(idx, pred, targets)
    if not scores["JT"] or not scores["BF"]:
        raise ValueError("Fixed validation must contain both JT and BF items")
    result = {
        "L1_JT": float(np.mean(scores["JT"])) if scores["JT"] else float("nan"),
        "L1_BF": float(np.mean(scores["BF"])) if scores["BF"] else float("nan"),
        "J": float(np.mean(scores["J"])) if scores["J"] else float("nan"),
        "n_jt": len(scores["JT"]),
        "n_bf": len(scores["BF"]),
    }
    return result, rows


def get_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--val_json_path", nargs="+", required=True)
    parser.add_argument("--semantic_mask_dir", default=None)
    parser.add_argument("--fixed_pair_path", required=True)
    parser.add_argument("--calibration_json", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_json")
    parser.add_argument(
        "--j_definition", choices=["legacy", DUAL_SCHEMA], default="legacy"
    )
    parser.add_argument("--dual_target_cache")
    parser.add_argument("--as_dual_denominator", action="store_true")
    parser.add_argument(
        "--s_baseline_path",
        help="New S denominator; adds absolute S and three normalized cohort means",
    )
    parser.add_argument("--legacy_val_json_path", nargs="+")
    parser.add_argument("--legacy_fixed_pair_path")
    parser.add_argument("--legacy_l1_reference_json")
    parser.add_argument("--l1_atol", type=float, default=1e-7)
    parser.add_argument("--l1_rtol", type=float, default=1e-6)
    return parser


def main():
    args = get_parser().parse_args()
    if args.s_baseline_path and (
        args.j_definition != DUAL_SCHEMA or args.as_dual_denominator
    ):
        raise ValueError("--s_baseline_path requires new-J evaluation, not calibration")
    import numpy as np
    import torch

    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not args.output_json:
        if args.as_dual_denominator:
            args.output_json = "models/dual_view/s_baseline_dual.json"
        elif args.j_definition == DUAL_SCHEMA:
            args.output_json = "models/dual_view/evaluation.json"
        else:
            args.output_json = "models/jieti_search/s_baseline.json"
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    payload = json.loads(Path(args.calibration_json).read_text())
    metadata = eval_metadata(args) if args.j_definition == DUAL_SCHEMA else {}
    coef = (
        validate_eval_coefficients(payload, metadata) if metadata else payload["coef"]
    )
    denominator = None
    if args.s_baseline_path:
        denominator = json.loads(Path(args.s_baseline_path).read_text())
        validate_s_baseline(denominator, {
            **metadata, "eval_coef_sha256": file_sha256(args.calibration_json),
        })
    model = load_model(args.checkpoint).to(args.device).eval()
    reconciliation = None
    if args.as_dual_denominator:
        if (
            args.j_definition != DUAL_SCHEMA
            or model._checkpoint_epoch != 50
            or getattr(model, "dual_view", False)
        ):
            raise ValueError("New S denominator requires baseline ep50 and instance J")
        require_source_model(model, "baseline_vggfix_ep50")
        if not all(
            (
                args.legacy_val_json_path,
                args.legacy_fixed_pair_path,
                args.legacy_l1_reference_json,
            )
        ):
            raise ValueError(
                "New S denominator requires old-val L1 reconciliation inputs"
            )
        old_args = argparse.Namespace(**vars(args))
        old_args.val_json_path = args.legacy_val_json_path
        old_args.fixed_pair_path = args.legacy_fixed_pair_path
        old_result, _ = evaluate(
            model, build_dataset(old_args, "legacy"), None, args.device
        )
        reference = json.loads(Path(args.legacy_l1_reference_json).read_text())
        reconciliation = reconcile_legacy_l1(
            old_result, reference, args.l1_atol, args.l1_rtol
        )
        reconciliation.update(
            fixed_pair_sha256=file_sha256(args.legacy_fixed_pair_path),
            reference_sha256=file_sha256(args.legacy_l1_reference_json),
        )
    dataset = build_dataset(args)
    jmod = make_jieti_loss(args.j_definition, coef).to(args.device)
    result, rows = evaluate(model, dataset, jmod, args.device, args.j_definition)
    if metadata:
        if len(rows) != 240 or not all(
            math.isfinite(result[key]) for key in ("L1_JT", "L1_BF", "J")
        ):
            raise ValueError("Incomplete or nonfinite clean evaluation")
        result.update(
            schema=DUAL_SCHEMA,
            purpose="s_baseline" if args.as_dual_denominator else "evaluation",
            eval_coef_sha256=file_sha256(args.calibration_json),
            source_checkpoint_sha256=model._checkpoint_sha256,
            source_checkpoint=model._checkpoint_source,
            source_epoch=model._checkpoint_epoch,
            seed=args.seed,
            **metadata,
        )
        if reconciliation is not None:
            if any(result[key] <= 0 for key in ("L1_JT", "L1_BF", "J")):
                raise ValueError("S denominator values must be strictly positive")
            result.update(
                source_role="baseline_vggfix_ep50",
                legacy_l1_reconciliation=reconciliation,
            )
            result["S"], result["S_components"] = absolute_s_score(result, result)
            result["s_baseline_reference"] = "self; first new S denominator"
        elif denominator is not None:
            result["S"], result["S_components"] = absolute_s_score(result, denominator)
            result["s_baseline_sha256"] = file_sha256(args.s_baseline_path)
            result["s_baseline_path"] = str(Path(args.s_baseline_path).resolve())
        result["per_item"] = rows
    out = Path(args.output_json)
    if metadata and out.exists():
        raise FileExistsError(f"Refusing to overwrite new-metric result: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "per_item"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
