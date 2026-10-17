"""Versioned runc runner for structure-first generation and BF reconstruction.

Every invocation owns a fresh output directory. Training requires an explicit
JSON configuration; probe coefficients are suggestions, never an acceptance of
the coarse image or authorization to start the refinement stage.
"""

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCHEMA = "structure-first-bifa-runtime-v1"
BASELINE_SHA256 = (
    "57ac8276e3468ad92ce2aff033736128b33862a4f922618c51910717166ae852"
)
WEIGHT_KEYS = (
    "lambda_low", "lambda_j0", "lambda_h", "lambda_img1", "lambda_bf",
    "lambda_j1", "gram", "edge", "edge_local",
)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """Create one immutable scientific record, rejecting nonfinite scalars."""
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def validate_config(config, command):
    """Validate research boundaries before allocating a model or a GPU."""
    config = deepcopy(config)
    if config.get("schema") != SCHEMA or config.get("experiment") != "runc":
        raise ValueError(f"config requires schema={SCHEMA}, experiment=runc")
    if config.get("stage") not in ("coarse", "refine"):
        raise ValueError("stage must be coarse or refine")
    if not config.get("checkpoint"):
        raise ValueError("checkpoint is required")
    jcoef = config.get("j_coefficients", {})
    if (set(jcoef) != {"centroid", "logsigma", "shape"}
            or any(isinstance(value, bool) or not math.isfinite(float(value))
                   or value <= 0 for value in jcoef.values())):
        raise ValueError(
            "explicit positive new J centroid/logsigma/shape coefficients required"
        )
    weights = config.get("weights", {})
    if set(weights) != set(WEIGHT_KEYS):
        raise ValueError(f"explicit weights must contain exactly {WEIGHT_KEYS}")
    if any(isinstance(value, bool) or not math.isfinite(float(value))
           or float(value) < 0 for value in weights.values()):
        raise ValueError("weights must be finite and nonnegative")
    active = ("lambda_low", "lambda_j0", "lambda_h") if (
        config["stage"] == "coarse"
    ) else ("lambda_img1", "lambda_bf", "lambda_j1")
    if not any(weights[key] > 0 for key in active):
        raise ValueError("the selected stage needs a positive objective")
    lowpass = config.get("lowpass", {})
    if set(lowpass) != {"kernel_size", "sigma", "boundary"}:
        raise ValueError("explicit lowpass kernel_size/sigma/boundary required")
    kernel = lowpass["kernel_size"]
    if (not isinstance(kernel, int) or kernel < 1 or kernel % 2 != 1
            or not math.isfinite(float(lowpass["sigma"]))
            or lowpass["sigma"] <= 0
            or lowpass["boundary"] not in ("reflect", "replicate")):
        raise ValueError("invalid lowpass configuration")
    train = config.get("training", {})
    if command == "train":
        required = {"steps", "batch_size", "accum_iter", "lr", "weight_decay",
                    "seed", "save_every", "amp", "freeze_vit"}
        if not required.issubset(train):
            raise ValueError(f"training requires explicit fields: {required}")
        for key in ("steps", "batch_size", "accum_iter", "save_every"):
            if type(train[key]) is not int or train[key] < 1:
                raise ValueError(f"training.{key} must be a positive integer")
        workers = train.get("data_workers", 0)
        if type(workers) is not int or workers < 0:
            raise ValueError("training.data_workers must be a nonnegative integer")
        if not train["freeze_vit"]:
            raise ValueError("initial runc protocol requires frozen stage-I ViT")
        if (not math.isfinite(float(train["lr"])) or train["lr"] <= 0
                or not math.isfinite(float(train["weight_decay"]))
                or train["weight_decay"] < 0):
            raise ValueError("invalid optimizer settings")
        require_lowpass_acceptance(config)
        require_training_authorization(config)
        if config["stage"] == "refine":
            require_coarse_acceptance(config)
    return config


def require_lowpass_acceptance(config):
    """Require recorded lowpass review by the user or a delegated agent."""
    path = config.get("lowpass_acceptance_json")
    if not path:
        raise ValueError("training requires lowpass_acceptance_json")
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if (record.get("schema") != SCHEMA or not acceptance_confirmed(record)
            or record.get("lowpass") != config["lowpass"]):
        raise ValueError(
            "lowpass acceptance must record user confirmation and exact settings"
        )
    return record


def acceptance_confirmed(record):
    """Distinguish human review from explicitly delegated agent acceptance."""
    if record.get("user_confirmed") is True and record.get("confirmation_reference"):
        return True
    paths = record.get("evidence_paths", [])
    return bool(
        record.get("actor") == "codex" and record.get("status") == "accepted"
        and record.get("delegated_authority_reference") and paths
        and all(Path(path).exists() for path in paths)
    )


def require_coarse_acceptance(config):
    """Bind human or delegated coarse acceptance to the exact stage-I weights."""
    path = config.get("coarse_acceptance_json")
    if not path:
        raise ValueError("refine training requires coarse_acceptance_json")
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if (record.get("schema") != SCHEMA or not acceptance_confirmed(record)
            or record.get("checkpoint_sha256") != sha256(config["checkpoint"])):
        raise ValueError("coarse acceptance must record user confirmation and SHA")
    return record


def training_authorization_digest(config):
    """Bind formal authorization to exact parameters, data paths and parent."""
    specification = {
        key: config.get(key) for key in (
            "experiment", "stage", "model", "weights", "lowpass", "training", "data",
            "j_coefficients",
        )
    }
    specification["checkpoint_sha256"] = sha256(config["checkpoint"])
    payload = json.dumps(specification, sort_keys=True, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def require_training_authorization(config):
    """Accept current inline user authority or a bound authorization record."""
    delegated = config.get("training_authorization", {})
    if delegated.get("authorized") is True and delegated.get("reference"):
        return delegated
    path = config.get("formal_training_authorization_json")
    if not path:
        raise ValueError("train requires formal_training_authorization_json")
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if (record.get("schema") != SCHEMA
            or record.get("training_authorized") is not True
            or record.get("user_confirmed") is not True
            or not record.get("confirmation_reference")
            or record.get("specification_sha256") != (
                training_authorization_digest(config)
            )):
        raise ValueError(
            "formal training authorization does not match this configuration"
        )
    return record


def prepare_output(path, rank=0):
    """Never reuse a run directory, including a directory with no checkpoint."""
    target = Path(path)
    if rank == 0:
        target.mkdir(parents=True, exist_ok=False)
    return target


def _distributed(device_name):
    import torch
    import torch.distributed as dist

    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(device_name or (f"cuda:{local}" if (
        torch.cuda.is_available()
    ) else "cpu"))
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return device, rank, world


def to_device(batch, device):
    import torch

    return {
        key: value.to(device) if torch.is_tensor(value) else (
            to_device(value, device) if isinstance(value, dict) else value
        ) for key, value in batch.items()
    }


def metadata(config):
    from tools.dual_view_protocol import cache_hash, hash_files

    data = config.get("data", {})
    result = {
        "schema": SCHEMA, "experiment": "runc",
        "parent_checkpoint_sha256": sha256(config["checkpoint"]),
        "baseline_checkpoint_sha256": BASELINE_SHA256,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()),
        "mask_distribution_limitation": (
            "train: existing 11 BF annotation layers; infer: predicted auto "
            "patches. Semantic granularity and coverage mismatch remains."
        ),
    }
    if data.get("train_json"):
        result["train_manifest_sha256"] = hash_files(data["train_json"])
    if data.get("cache"):
        result["target_cache_sha256"] = cache_hash(data["cache"])
    for key in ("fixed_pairs", "eval_coef", "s_baseline"):
        if data.get(key):
            result[f"{key}_sha256"] = sha256(data[key])
    if data.get("val_json"):
        result["val_manifest_sha256"] = hash_files(data["val_json"])
    return result


def build_dataset(config, validation=False):
    from data.dual_dataset import DualDataset
    from data.structure_first_dataset import StructureFirstDataset
    from util.masking_generator import MaskingGenerator

    data = config["data"]
    common = dict(
        root=data["root"], target_cache_dir=data["cache"],
        masked_position_generator=MaskingGenerator(
            (56, 28), num_masking_patches=784, max_num_patches=392
        ), num_mask_annotations_bf=11, mask_coverage_threshold=0.1,
        jieti_pool=224, jieti_valid_ink=200,
    )
    if validation:
        return DualDataset(
            json_path_list=data["val_json"], is_train=False,
            fixed_pair_path=data["fixed_pairs"], **common,
        )
    return StructureFirstDataset(
        json_path_list=data["train_json"], stage=config["stage"],
        is_train=True, **common,
    )


def load_model(config, device):
    """Baseline migration is SHA gated; every new checkpoint loads strictly."""
    import torch

    from tools.dual_view_protocol import seed_all
    from util.checkpoint import build_checkpoint_model, checkpoint_args
    from util.structure_first import enable_structure_first, set_structure_stage

    seed_all(config.get("training", {}).get("seed", 0))
    saved = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
    args = checkpoint_args(saved)
    if (saved.get("g_scale_config") or saved.get("g_control_config")
            or any(key.startswith(("g_scale.", "g_control.", "tip_"))
                   for key in saved["model"])
            or args.get("tip_weight") is not None):
        raise ValueError("Tip/G Scale/control checkpoints are not runc parents")
    if saved.get("structure_first_config"):
        provenance = saved.get("structure_first_metadata", {})
        if (saved.get("experiment") != "runc"
                or provenance.get("baseline_checkpoint_sha256") != BASELINE_SHA256):
            raise ValueError("new runc checkpoint lacks the sealed baseline lineage")
        model = build_checkpoint_model(saved)
        if any(saved["structure_first_config"].get(key) != value
               for key, value in config.get("model", {}).items()
               if key != "stage"):
            raise ValueError("new checkpoint model config cannot be overwritten")
    else:
        if sha256(config["checkpoint"]) != BASELINE_SHA256:
            raise ValueError("runc initialization requires the sealed Run B Fix SHA")
        model = build_checkpoint_model(saved)
        enable_structure_first(model, config.get("model", {}))
    set_structure_stage(model, config["stage"])
    # These are new-target coefficients, explicitly chosen in this config.
    # The parent Run B outer loss weights never enter structure_losses().
    for name, value in config["j_coefficients"].items():
        setattr(model.dual_jieti_loss, f"w_{name}", float(value))
        setattr(model, f"jieti_w_{name}", float(value))
    model.to(device)
    return model, saved


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def training_mode(model, stage):
    """Reapply stage freezing after recursive train(), especially drop_path."""
    from util.structure_first import set_structure_stage

    model.train()
    set_structure_stage(unwrap(model), stage)


def assert_optimizer(model, optimizer, stage):
    raw = unwrap(model)
    wanted = {id(p) for p in raw.parameters() if p.requires_grad}
    actual = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if actual != wanted:
        raise RuntimeError("optimizer and the current stage trainable set disagree")
    if stage == "refine":
        unexpected = [name for name, p in raw.named_parameters()
                      if p.requires_grad and not name.startswith("bf_refiner.")]
        if unexpected:
            raise RuntimeError(f"stage II unfrozen stage-I parameters: {unexpected}")


def clear_unlabelled_head_grads(model, jt_count, distributed=False):
    """Prevent AdamW momentum/decay updates on globally BF-only windows."""
    import torch
    import torch.distributed as dist

    raw = unwrap(model)
    count = torch.as_tensor(jt_count, device=next(raw.parameters()).device).clone()
    if distributed:
        dist.all_reduce(count)
    if int(count.item()) == 0:
        for name, parameter in raw.named_parameters():
            if name.startswith("dual_head."):
                parameter.grad = None
    return int(count.item())


def call_model(model, batch, stage, inference=False, force_refine=False):
    """Training enters the DDP forward; inference never passes annotations."""
    from util.structure_first import structure_forward

    raw = unwrap(model)
    target = batch.get("input_target")
    if target is None:
        target = batch["target"].clone()
        target[:, :, target.shape[-2] // 2:] = 0
    if inference:
        execution = "infer" if force_refine or stage == "refine" else "coarse"
        return structure_forward(raw, batch["image"], target, stage=execution)
    return model(
        batch["image"], target, batch["mask"].flatten(1), batch["valid"],
        no_gan=True, dual_targets=batch,
    )


def loss_for(model, outputs, batch, config):
    from util.structure_first_loss import structure_losses

    return structure_losses(
        unwrap(model), outputs, batch, config["weights"], config["stage"],
        lowpass=config["lowpass"],
    )


def scalar_losses(losses):
    import torch

    return {key: float(value.detach().float()) for key, value in losses.items()
            if torch.is_tensor(value) and value.numel() == 1}


def reduce_training_logs(logs, device, world):
    """Sum counts globally; average mean losses across equal-sized ranks."""
    import torch
    import torch.distributed as dist

    names = sorted(logs[0])
    values = [sum(row[name] for row in logs) / (
        1 if name.startswith("count_") else len(logs)
    ) for name in names]
    values = torch.tensor(values, device=device, dtype=torch.float64)
    if world > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {name: float(values[index]) / (
        1 if name.startswith("count_") else world
    ) for index, name in enumerate(names)}


def parameter_grams(terms, named_parameters):
    """Measure each raw term with actual trainable parameters, allowing zeros."""
    import torch

    named_parameters = list(named_parameters)
    parameters = [p for _, p in named_parameters]
    gradients = []
    for value in terms.values():
        gradients.append(torch.autograd.grad(
            value, parameters, retain_graph=True, allow_unused=True
        ) if value.requires_grad else [None] * len(parameters))
    names = list(terms)
    grams = [[0.0] * len(names) for _ in names]
    for i in range(len(names)):
        for j in range(i + 1):
            inner = sum(float((a.float() * b.float()).sum())
                        for a, b in zip(gradients[i], gradients[j])
                        if a is not None and b is not None)
            grams[i][j] = grams[j][i] = inner
    norms = {name: math.sqrt(max(0.0, grams[i][i]))
             for i, name in enumerate(names)}
    cosine = {}
    for i, left in enumerate(names):
        for j, right in enumerate(names[:i]):
            denom = norms[left] * norms[right]
            cosine[f"{left}:{right}"] = max(-1.0, min(1.0, grams[i][j] / denom)) if (
                denom > 0
            ) else None
    return {"terms": names, "gram": grams, "norms": norms, "cosine": cosine}


def collate_indices(dataset, indices, device, seed=0):
    import torch

    rows = [dataset.get_pair(index, augmentation_seed=seed + pos)
            for pos, index in enumerate(indices)]
    return to_device(torch.utils.data.default_collate(rows), device)


def pool_indices(dataset):
    result = {"dual": [], "bf_only": [], "jt_only": []}
    for index, record in enumerate(dataset.pairs):
        pool = "dual" if record["has_jt"] and record["has_bf"] else (
            "bf_only" if record["has_bf"] else "jt_only"
        )
        result[pool].append(index)
    return result


def save_visual(path, outputs, batch, lowpass=None):
    """Show only generated conditions, with GT explicitly in comparison panels."""
    import numpy as np
    import torch
    from PIL import Image

    height = batch["image"].shape[-2] // 2

    def rgb(value):
        value = value.detach().float().cpu()[0, :, -height:]
        value = value * torch.tensor(STD)[:, None, None]
        value = value + torch.tensor(MEAN)[:, None, None]
        return (value.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")

    panels = [rgb(batch["image"]), rgb(batch["target"]), rgb(outputs["p0"])]
    labels = ["source", "GT-comparison-only", "P0"]
    if "p1" in outputs and outputs["p1"] is not None:
        panels.append(rgb(outputs["p1"]))
        labels.append("P1")
    if lowpass:
        from util.structure_first_loss import lowpass_target

        filtered = lowpass_target(batch["target"][:, :, -height:], lowpass)
        panels.append(rgb(filtered))
        labels.append("LPF-GT-loss-target-only")
    mask = outputs.get("mask", outputs.get("bf_patch_mask"))
    if mask is not None:
        mask = mask.detach().float()
        while mask.ndim > 2:
            mask = mask[0]
        mask = mask[-28:]
        mask = torch.nn.functional.interpolate(
            mask[None, None], size=(height, height), mode="nearest"
        )[0, 0].cpu().numpy()
        panels.append(np.repeat((mask[..., None] * 255).astype("uint8"), 3, 2))
        labels.append("actual-reconstruction-mask")
    head = outputs.get("head", {})
    if "occupancy_logits" in head:
        occupancy = head["occupancy_logits"][0].float().sigmoid().sum(0).clamp(0, 1)
        occupancy = torch.nn.functional.interpolate(
            occupancy[None, None], size=(height, height), mode="nearest"
        )[0, 0].detach().cpu().numpy()
        panels.append(np.repeat((occupancy[..., None] * 255).astype("uint8"), 3, 2))
        labels.append("predicted-S-occupancy")
    Image.fromarray(np.concatenate(panels, axis=1)).save(path)
    write_json(str(path) + ".json", {"panels": labels, "lowpass": lowpass})


def head_metrics(head, batch, details):
    """Measure matched predicted S separately from the generated image's J."""
    import torch

    assignment = details["assignment"]
    occupancy = batch["instance_occupancy"]
    target = occupancy.gather(
        1, assignment[:, :, None, None].expand_as(occupancy)
    )
    probability = head["occupancy_logits"].float().sigmoid()
    matched = details["matched_slots"] & batch["has_jt"][:, None].bool()
    intersection = (probability * target).sum((-2, -1))
    union = (probability + target - probability * target).sum((-2, -1))
    iou = intersection / union.clamp_min(1e-6)
    classes = batch["instance_classes"].gather(1, assignment)
    correct = head["class_logits"].argmax(-1).eq(classes)
    height, width = probability.shape[-2:]
    xcoord = torch.linspace(0, 1, width, device=probability.device)[None, None, None]
    ycoord = torch.linspace(0, 1, height, device=probability.device)
    ycoord = ycoord[None, None, :, None]

    def moments(value):
        mass = value.sum((-2, -1)).clamp_min(1e-6)
        cx = (value * xcoord).sum((-2, -1)) / mass
        cy = (value * ycoord).sum((-2, -1)) / mass
        sx = (value * (xcoord - cx[..., None, None]).square()).sum((-2, -1))
        sy = (value * (ycoord - cy[..., None, None]).square()).sum((-2, -1))
        sx, sy = (sx / mass).clamp_min(1e-6).sqrt(), (sy / mass).clamp_min(1e-6).sqrt()
        return cx, cy, sx, sy

    px, py, psx, psy = moments(probability)
    tx, ty, tsx, tsy = moments(target)
    centroid = (px - tx).abs() + (py - ty).abs()
    scale = (psx.log() - tsx.log()).abs() + (psy.log() - tsy.log()).abs()
    return {
        "head_soft_iou_sum": float(iou[matched].sum()),
        "head_role_correct_sum": float(correct[matched].sum()),
        "head_centroid_sum": float(centroid[matched].sum()),
        "head_logsigma_sum": float(scale[matched].sum()),
        "head_instance_count": int(matched.sum()),
    }


def snapshot(model, saved, config, provenance, optimizer, step, output, world=1):
    import torch

    raw = unwrap(model)
    dual_config = deepcopy(saved.get("dual_view_config") or {})
    dual_config["coef"] = dict(config["j_coefficients"])
    settings = config["training"]
    effective_batch = world * settings["batch_size"] * settings["accum_iter"]
    payload = {
        "model": raw.state_dict(), "args": saved.get("args"),
        "args_role": "parent args retained only for architecture reconstruction",
        "dual_view_config": dual_config,
        "structure_first_config": raw.structure_first_config,
        "structure_first_runtime": config,
        "structure_first_metadata": provenance,
        "optimizer": optimizer.state_dict(), "step": step,
        "optimizer_step": step, "effective_batch": effective_batch,
        "processed_target_count": step * effective_batch,
        "planned_target_count": settings["steps"] * effective_batch,
        "epoch": None, "parent_epoch": saved.get("epoch"),
        "parent_optimizer_step": saved.get("optimizer_step"), "experiment": "runc",
    }
    path = Path(output) / f"checkpoint-step-{step:06d}.pth"
    with path.open("xb") as stream:
        torch.save(payload, stream)
    write_json(str(path) + ".sha256.json", {"sha256": sha256(path), "step": step})


def audit(config, output):
    dataset = build_dataset(config)
    pools = pool_indices(dataset)
    write_json(output / "audit.json", {
        "schema": SCHEMA, "experiment": "runc", "metadata": metadata(config),
        "pool_counts": {name: len(indices) for name, indices in pools.items()},
        "training_mask_layers": 11, "coverage_threshold": 0.1,
        "stage": config["stage"], "records": dataset.pairs,
    })


def balanced_probe_order(pairs, indices, seed=0):
    """Cycle through authors before revisiting an author, without replacement."""
    grouped = {}
    for index in indices:
        author = pairs[index].get("calligrapher")
        if not author:
            raise ValueError("Author-balanced probes require calligrapher metadata")
        grouped.setdefault(author, []).append(index)
    generator = random.Random(seed)
    for group in grouped.values():
        generator.shuffle(group)
    order = []
    authors = sorted(grouped)
    while any(grouped.values()):
        for author in authors:
            if grouped[author]:
                order.append(grouped[author].pop())
    return order


def probe(config, model, dataset, output, device, batches):
    """A multi-batch report measures raw terms; it does not select final weights."""
    import torch

    if batches < 2:
        raise ValueError("gradient probe requires at least two real batches")
    training_mode(model, config["stage"])
    amp = bool(config.get("training", {}).get("amp", False))
    pools = pool_indices(dataset)
    if config.get("probe_balance_authors", False):
        pools = {
            name: balanced_probe_order(dataset.pairs, indices,
                                       config["training"]["seed"])
            for name, indices in pools.items()
        }
    records = []
    active = [name for name, indices in pools.items() if indices]
    batch_size = config.get("probe_batch_size", 1)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("probe_batch_size must be a positive integer")
    scenarios = []
    for number in range(batches):
        pool = active[number % len(active)]
        offset = number // len(active) * batch_size
        indices = [pools[pool][(offset + index) % len(pools[pool])]
                   for index in range(batch_size)]
        scenarios.append((pool, indices))
    if config.get("probe_include_mixed", False):
        scenarios.append(("mixed", [
            pools[pool][index % len(pools[pool])]
            for index, pool in enumerate(active)
        ]))
    for number, (pool, indices) in enumerate(scenarios):
        batch = collate_indices(dataset, indices, device, seed=number)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            outputs = call_model(model, batch, config["stage"])
            losses = loss_for(model, outputs, batch, config)
        names = ("raw_low", "raw_j0", "raw_head") if (
            config["stage"] == "coarse"
        ) else ("raw_img_smooth", "raw_img_gram", "raw_img_edge",
                "raw_bf_smooth", "raw_bf_edge", "raw_j1")
        raw_terms = {key: losses[key] for key in
                     (*names, "j_centroid", "j_logsigma", "j_shape")}
        raw_terms = {key: value for key, value in raw_terms.items()
                     if torch.is_tensor(value) and value.numel() == 1}
        groups = {
            "all_trainable": [(name, p) for name, p in model.named_parameters()
                              if p.requires_grad],
            "head": [(name, p) for name, p in model.named_parameters()
                     if p.requires_grad and name.startswith("dual_head.")],
            "refiner": [(name, p) for name, p in model.named_parameters()
                        if p.requires_grad and name.startswith("bf_refiner.")],
        }
        measurements = {name: parameter_grams(raw_terms, parameters)
                        for name, parameters in groups.items() if parameters}
        records.append({"batch": number, "pool": pool, "indices": indices,
                        "batch_size": len(indices),
                        "sample_has_jt": batch["has_jt"].cpu().tolist(),
                        "sample_has_bf": batch["has_bf"].cpu().tolist(),
                        "losses": scalar_losses(losses), "gradients": measurements,
                        "target_path": batch.get("target_path"),
                        "reference_path": batch.get("reference_path")})
        if number < 3:
            save_visual(output / f"probe-{number:03d}.png", outputs, batch,
                        config["lowpass"])
    term_norms = {}
    for row in records:
        for name, norm in row["gradients"]["all_trainable"]["norms"].items():
            if norm > 0:
                term_norms.setdefault(name, []).append(norm)
    medians = {name: float(torch.tensor(values).median())
               for name, values in term_norms.items()}
    suggestions = {name: 1.0 / value for name, value in medians.items()}
    write_json(output / "gradient-probe.json", {
        "schema": SCHEMA, "experiment": "runc", "status": "measured",
        "measurements": records, "median_nonzero_raw_parameter_norm": medians,
        "inverse_norm_candidates": suggestions,
        "candidate_status": (
            "diagnostic only; no target ratio or user selection assumed"
        ),
        "configured_weights": config["weights"], "metadata": metadata(config),
        "precision": {
            "forward_autocast": amp,
            "forward_dtype": "bfloat16" if amp else "float32",
            "gradient_inner_product_dtype": "float32",
            "device_type": device.type,
        },
        "probe_batch_size": batch_size,
        "probe_include_mixed": config.get("probe_include_mixed", False),
        "probe_balance_authors": config.get("probe_balance_authors", False),
    })


class FixedDrawSamples:
    """Fetch sealed draws with the same per-microbatch augmentation seeds."""

    def __init__(self, dataset, draws, micro, world, rank, seed):
        self.dataset = dataset
        self.draws = draws
        self.micro, self.world, self.rank, self.seed = micro, world, rank, seed

    def __len__(self):
        return len(self.draws)

    def __getitem__(self, position):
        offset, within = divmod(position, self.micro)
        augmentation_seed = (
            self.seed + offset * self.micro * self.world + self.rank + within
        )
        return self.dataset.get_pair(
            self.draws[position], augmentation_seed=augmentation_seed,
        )


def train(config, model, saved, dataset, output, device, rank, world):
    import torch
    import torch.distributed as dist

    settings = config["training"]
    training_mode(model, config["stage"])
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=settings["lr"], weight_decay=settings["weight_decay"],
    )
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True, broadcast_buffers=False,
        )
    provenance = metadata(config)
    generator = torch.Generator().manual_seed(settings["seed"])
    total = settings["steps"] * settings["accum_iter"] * settings["batch_size"]
    draws = torch.multinomial(torch.tensor(dataset.weights, dtype=torch.double),
                             total * world, replacement=True, generator=generator)
    draws = draws[rank::world].tolist()
    micro = settings["batch_size"]
    accumulation = settings["accum_iter"]
    workers = settings.get("data_workers", 0)
    batches = None
    if workers:
        loader = torch.utils.data.DataLoader(
            FixedDrawSamples(dataset, draws, micro, world, rank, settings["seed"]),
            batch_size=micro, shuffle=False, num_workers=workers,
            pin_memory=device.type == "cuda", persistent_workers=True,
            prefetch_factor=2, multiprocessing_context="spawn",
            generator=torch.Generator().manual_seed(settings["seed"]),
        )
        batches = iter(loader)
    for step in range(1, settings["steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        jt_count = 0
        logs = []
        for inner in range(accumulation):
            offset = ((step - 1) * accumulation + inner) * micro
            if batches is None:
                batch = collate_indices(dataset, draws[offset:offset + micro], device,
                                        settings["seed"] + offset * world + rank)
            else:
                batch = to_device(next(batches), device)
            jt_count += int(batch["has_jt"].sum())
            sync = model.no_sync() if (
                world > 1 and inner < accumulation - 1
            ) else nullcontext()
            with sync:
                with torch.autocast(device.type, dtype=torch.bfloat16,
                                    enabled=bool(settings["amp"])):
                    outputs = call_model(model, batch, config["stage"])
                    losses = loss_for(model, outputs, batch, config)
                    loss = losses["total"] / accumulation
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite training objective")
                loss.backward()
            logs.append(scalar_losses(losses))
        global_jt = clear_unlabelled_head_grads(model, jt_count, world > 1)
        assert_optimizer(model, optimizer, config["stage"])
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in unwrap(model).parameters() if p.requires_grad],
            settings.get("clip_grad", 3.0), error_if_nonfinite=True,
        )
        optimizer.step()
        global_logs = reduce_training_logs(logs, device, world)
        if rank == 0:
            row = {"step": step, "global_jt_count": global_jt,
                   "gradient_preclip": float(norm),
                   "global_batch": world * micro * accumulation,
                   "loss_reduction": "global means; count_* global sums",
                   "losses": global_logs}
            if "bf_mask_stats" in batch:
                row["bf_mask_stats_rank0_local_last_microbatch"] = {
                    key: value.detach().cpu().tolist()
                    for key, value in batch["bf_mask_stats"].items()
                }
            with (output / "train.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if (step == 1 or step % settings["save_every"] == 0
                    or step == settings["steps"]):
                snapshot(model, saved, config, provenance, optimizer, step, output,
                         world=world)
                save_visual(output / f"step-{step:06d}.png", outputs, batch,
                            config["lowpass"])
            print(json.dumps(row, allow_nan=False), flush=True)
    if world > 1:
        dist.barrier()


def evaluate(config, model, output, device):
    """Clean fixed validation keeps original JT/BF cohort and per-JT J count."""
    import torch

    from tools.eval_s_baseline import (
        absolute_s_score,
        eval_metadata,
        make_jieti_loss,
        validate_eval_coefficients,
        validate_s_baseline,
    )
    from util.dual_loss import InstanceHeadLoss

    data = config["data"]
    args = argparse.Namespace(
        fixed_pair_path=data["fixed_pairs"], dual_target_cache=data["cache"],
        val_json_path=data["val_json"],
    )
    hashes = eval_metadata(args)
    coefficients = json.loads(Path(data["eval_coef"]).read_text())
    coef = validate_eval_coefficients(coefficients, hashes)
    hashes["eval_coef_sha256"] = sha256(data["eval_coef"])
    baseline = json.loads(Path(data["s_baseline"]).read_text())
    validate_s_baseline(baseline, hashes)
    jmod = make_jieti_loss("instance-lower-full-v1", coef).to(device)
    hmod = InstanceHeadLoss().to(device)
    dataset = build_dataset(config, validation=True)
    sums = {name: 0.0 for name in (
        "L1_JT", "L1_BF", "J", "centroid", "logsigma", "shape", "head"
    )}
    counts = {name: 0 for name in sums}
    rows = []
    head_sums = {name: 0.0 for name in ("head_soft_iou_sum", "head_role_correct_sum",
                                      "head_centroid_sum", "head_logsigma_sum")}
    head_count = 0
    model.eval()
    with torch.no_grad():
        for index in range(len(dataset)):
            batch = collate_indices(dataset, [index], device)
            outputs = call_model(model, batch, config["stage"], inference=True)
            pred = outputs["p0"] if config["stage"] == "coarse" else outputs["p1"]
            height = pred.shape[-2] // 2
            area = batch["valid"][:, :, height:]
            difference = (pred[:, :, height:] - batch["target"][:, :, height:]).abs()
            l1 = (difference * area).sum() / (area.sum() + 1e-2)
            cohort = "L1_JT" if bool(batch["has_jt"][0]) else "L1_BF"
            sums[cohort] += float(l1)
            counts[cohort] += 1
            row = {"index": index, "cohort": cohort, "L1": float(l1),
                   "target_path": batch["target_path"][0]}
            if bool(batch["has_jt"][0]):
                value, parts, _ = jmod(pred, batch["target"], batch["voro"],
                                       batch["valid_parts"], batch["has_jt"])
                head_value, details = hmod(
                    outputs["head"], batch["instance_occupancy"],
                    batch["instance_valid"], batch["instance_classes"], batch["has_jt"],
                )
                measurements = {"J": value, "head": head_value}
                measurements.update({name: parts[name] for name in
                                     ("centroid", "logsigma", "shape")})
                for name, item in measurements.items():
                    sums[name] += float(item)
                    counts[name] += 1
                    row[name] = float(item)
                row["matched_slots"] = int(details["matched_slots"].sum())
                matched_metrics = head_metrics(outputs["head"], batch, details)
                head_count += matched_metrics["head_instance_count"]
                for name in head_sums:
                    head_sums[name] += matched_metrics[name]
                row.update(matched_metrics)
            rows.append(row)
            if index < 8:
                save_visual(output / f"val-{index:03d}.png", outputs, batch,
                            config["lowpass"])
    means = {name: sums[name] / counts[name] if counts[name] else None for name in sums}
    score, ratios = absolute_s_score(means, baseline)
    write_json(output / "evaluation.json", {
        "schema": SCHEMA, "experiment": "runc", "stage": config["stage"],
        "metadata": metadata(config), "sum": sums, "count": counts,
        "mean": means, "S": score, "ratios": ratios, "rows": rows,
        "head_instance_sum": head_sums, "head_instance_count": head_count,
        "head_instance_mean": {name.removesuffix("_sum"): value / head_count
                               if head_count else None
                               for name, value in head_sums.items()},
        "validation_limitation": (
            "clean val has been reused for model selection; "
            "not an independent generalization test"
        ),
    })


def infer(config, model, args, output, device):
    """Deployment uses only source glyph and visible reference, with no labels."""
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    def image(path):
        with Image.open(path) as opened:
            rgb = opened.convert("RGB")
        side = max(rgb.size)
        rgb = ImageOps.pad(rgb, (side, side), color="white")
        rgb = rgb.resize((448, 448), Image.Resampling.BICUBIC)
        value = torch.tensor(np.array(rgb).copy()).permute(2, 0, 1).float() / 255
        mean = torch.tensor(MEAN)[:, None, None]
        std = torch.tensor(STD)[:, None, None]
        return (value - mean) / std

    if not all((args.source, args.reference, args.reference_source)):
        raise ValueError("infer requires --source --reference --reference-source")
    source, reference, reference_source = map(image, (
        args.source, args.reference, args.reference_source,
    ))
    target = torch.cat((reference, torch.zeros_like(reference)), 1)[None].to(device)
    inputs = torch.cat((reference_source, source), 1)[None].to(device)
    batch = {"image": inputs, "target": target, "input_target": target}
    model.eval()
    with torch.no_grad():
        first = call_model(model, batch, config["stage"], inference=True,
                           force_refine=True)
        second = call_model(model, batch, config["stage"], inference=True,
                            force_refine=True)
    if not torch.equal(first["p1"], second["p1"]):
        raise RuntimeError("deployment inference is not deterministic")
    save_visual(output / "inference.png", first, batch)
    for name in ("p0", "p1"):
        value = first[name][0, :, 448:].detach().float().cpu()
        value = (value * torch.tensor(STD)[:, None, None]
                 + torch.tensor(MEAN)[:, None, None]).clamp(0, 1)
        array = (value.permute(1, 2, 0).numpy() * 255).astype("uint8")
        Image.fromarray(array).save(output / f"{name}.png")
    write_json(output / "inference.json", {
        "schema": SCHEMA, "experiment": "runc", "deterministic_equal": True,
        "input_sha256": {
            "source": sha256(args.source), "reference": sha256(args.reference),
            "reference_source": sha256(args.reference_source),
        },
        "metadata": metadata(config), "annotation_inputs": [],
    })


def smoke(config, model, dataset, output, device, rank, world):
    """Exercise real mixed pools and AdamW windows, including DDP when launched."""
    import torch

    from util.structure_first import set_structure_stage

    pools = pool_indices(dataset)
    amp = bool(config.get("training", {}).get("amp", False))
    results = []
    scenarios = [(name, [[indices[rank % len(indices)]]])
                 for name, indices in pools.items() if indices]
    jt = pools["jt_only"] or pools["dual"]
    bf = pools["bf_only"] or pools["dual"]
    if jt and bf:
        scenarios.append(("mixed_batch", [[bf[0], jt[0]]]))
        scenarios.append(("mixed_accumulation", [[bf[0]], [jt[0]]]))
        if world > 1:
            cross = bf[0] if rank % 2 == 0 else jt[0]
            scenarios.append(("mixed_across_ranks", [[cross]]))
    training_mode(model, config["stage"])
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-5,
    )
    wrapped = model
    if world > 1:
        wrapped = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True, broadcast_buffers=False,
        )
    for name, micro_batches in scenarios:
        optimizer.zero_grad(set_to_none=True)
        jt_count = 0
        before = {key: p.detach().clone() for key, p in model.named_parameters()
                  if key.startswith("dual_head.")}
        for inner, indices in enumerate(micro_batches):
            batch = collate_indices(dataset, indices, device, rank + inner)
            jt_count += int(batch["has_jt"].sum())
            sync = wrapped.no_sync() if (
                world > 1 and inner < len(micro_batches) - 1
            ) else nullcontext()
            with sync:
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
                    outputs = call_model(wrapped, batch, config["stage"])
                    losses = loss_for(model, outputs, batch, config)
                (losses["total"] / len(micro_batches)).backward()
        head_norm = sum(
            float(p.grad.float().square().sum())
            for key, p in model.named_parameters()
            if key.startswith("dual_head.") and p.grad is not None
        ) ** 0.5
        refiner_norm = sum(
            float(p.grad.float().square().sum())
            for key, p in model.named_parameters()
            if key.startswith("bf_refiner.") and p.grad is not None
        ) ** 0.5
        if config["stage"] == "refine":
            if not math.isfinite(refiner_norm) or refiner_norm <= 0:
                raise RuntimeError("stage II has no finite refiner gradient")
            if any(p.grad is not None for key, p in model.named_parameters()
                   if not key.startswith("bf_refiner.")):
                raise RuntimeError("stage II produced stage-I gradients")
        global_jt = clear_unlabelled_head_grads(model, jt_count, world > 1)
        assert_optimizer(model, optimizer, config["stage"])
        optimizer.step()
        unchanged = all(torch.equal(before[key], p)
                        for key, p in model.named_parameters() if key in before)
        if name == "bf_only" and (head_norm != 0 or not unchanged):
            raise RuntimeError("BF-only head changed through gradients/AdamW")
        results.append({"pool": name, "rank": rank, "head_grad_norm": head_norm,
                        "refiner_grad_norm": refiner_norm,
                        "head_unchanged": unchanged, "global_jt": global_jt,
                        "losses": scalar_losses(losses)})
        optimizer.zero_grad(set_to_none=True)
    model.eval()
    indices = next(indices for indices in pools.values() if indices)
    batch = collate_indices(dataset, indices[:1], device)
    with torch.no_grad():
        first = call_model(model, batch, config["stage"], inference=True)
        altered = dict(batch)
        altered["target"] = batch["target"].clone()
        lower = batch["target"].shape[-2] // 2
        altered["target"][:, :, lower:] = torch.randn_like(
            altered["target"][:, :, lower:]
        )
        if "input_target" in batch:
            altered["input_target"] = batch["input_target"].clone()
            altered["input_target"][:, :, lower:] = altered["target"][:, :, lower:]
        second = call_model(model, altered, config["stage"], inference=True)
        repeat = call_model(model, batch, config["stage"], inference=True)
    prediction_key = "p0" if config["stage"] == "coarse" else "p1"
    if not torch.equal(first[prediction_key], second[prediction_key]):
        raise RuntimeError("GT perturbation changed inference")
    if not torch.equal(first[prediction_key], repeat[prediction_key]):
        raise RuntimeError("inference is not deterministic")
    if rank == 0:
        save_visual(output / "smoke-inference.png", first, batch)
    write_json(output / f"smoke-rank-{rank}.json", {
        "schema": SCHEMA, "experiment": "runc", "world_size": world,
        "checks": results, "gt_perturbation_equal": True,
        "deterministic_equal": True, "metadata": metadata(config),
        "precision": {
            "forward_autocast": amp,
            "forward_dtype": "bfloat16" if amp else "float32",
            "determinism_inference_dtype": "float32",
            "device_type": device.type,
        },
        "ddp": {
            "enabled": world > 1, "find_unused_parameters": True,
            "static_graph": False, "broadcast_buffers": False,
        },
    })
    set_structure_stage(model, config["stage"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=(
        "audit", "smoke", "probe", "train", "evaluate", "infer"
    ))
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    parser.add_argument("--probe-batches", type=int, default=6)
    parser.add_argument("--source")
    parser.add_argument("--reference")
    parser.add_argument("--reference-source")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    config = validate_config(config, args.command)
    device, rank, world = _distributed(args.device)
    if world > 1 and args.command not in ("train", "smoke"):
        raise ValueError("only train/smoke support torchrun; evaluation runs unsharded")
    output = prepare_output(args.output, rank)
    if world > 1:
        import torch.distributed as dist
        dist.barrier()
    if rank == 0:
        write_json(output / "config.json", config)
        write_json(output / "metadata.json", metadata(config))
    if args.command == "audit":
        audit(config, output)
        return
    model, saved = load_model(config, device)
    if args.command == "infer":
        infer(config, model, args, output, device)
    elif args.command == "evaluate":
        evaluate(config, model, output, device)
    else:
        dataset = build_dataset(config)
        if args.command == "probe":
            probe(config, model, dataset, output, device, args.probe_batches)
        elif args.command == "smoke":
            smoke(config, model, dataset, output, device, rank, world)
        else:
            train(config, model, saved, dataset, output, device, rank, world)
    if world > 1:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
