"""Independent, resumable runtime for JT-only Run C v2 stage I.

This runner never imports old Run C loss weights or its Run B checkpoint
lineage. Supervision is passed only to the loss function. The deployment
wrapper accepts source/reference images and a sanitized target canvas.
"""

import copy
import csv
import hashlib
import html
import json
import math
import os
import random
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "runc-v2-runtime-v1"
REFERENCE_POLICY = "distinct-original-reference-v1"
STAGES = ("ia0", "ia1", "ib", "ic")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
ARCHITECTURE = "vit_base_patch16_input896x448_win_dec64_8glb_sl1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value, exclusive=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x" if exclusive else "w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def validate_config(config, command="train"):
    """Reject ambiguous lineage, old objectives and BF stage-II training."""
    config = copy.deepcopy(config)
    if config.get("schema") != SCHEMA or config.get("experiment") != "runc_v2":
        raise ValueError(f"Expected schema={SCHEMA}, experiment=runc_v2")
    if config.get("stage") not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}; BF training is excluded")
    if not config.get("checkpoint"):
        raise ValueError("checkpoint is required")
    if not config.get("initial_checkpoint_sha256"):
        raise ValueError("verified initial_checkpoint_sha256 is required")
    if config.get("data", {}).get("reference_policy") != REFERENCE_POLICY:
        raise ValueError(f"data.reference_policy must be {REFERENCE_POLICY}")
    if any(key in config for key in ("lowpass", "j_coefficients")):
        raise ValueError("Run C v1 loss configuration cannot enter v2")
    if config.get("model", {}).get("unfreeze_blocks", [2, 5, 8, 11]) != [2, 5, 8, 11]:
        if not config.get("unfreeze_rationale"):
            raise ValueError("nondefault ViT unfreeze needs an explicit rationale")
    settings = config.setdefault("training", {})
    for key in ("seed",):
        if type(settings.get(key)) is not int or settings[key] < 0:
            raise ValueError(f"training.{key} must be an explicit nonnegative int")
    for name, value in config.get("weights", {}).items():
        if isinstance(value, bool) or not math.isfinite(float(value)) or value < 0:
            raise ValueError(f"invalid weight {name}")
    if command in ("train", "smoke"):
        required = ("steps", "batch_size", "accum_iter", "save_every")
        for key in required:
            if type(settings.get(key)) is not int or settings[key] < 1:
                raise ValueError(f"training.{key} must be a positive integer")
        for key in ("lr_head", "lr_vit", "lr_coarse", "clip_grad"):
            value = settings.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"training.{key} must be explicit and positive")
        if settings["lr_vit"] >= settings["lr_head"]:
            raise ValueError("pretrained ViT LR must be lower than new Head LR")
        if type(settings.get("amp")) is not bool:
            raise ValueError("training.amp must be explicit bool")
        if not isinstance(settings.get("weight_decay"), (int, float)):
            raise ValueError("training.weight_decay must be explicit")
        if settings["weight_decay"] < 0:
            raise ValueError("weight_decay must be nonnegative")
    if command == "train" and not config.get("calibration_record"):
        raise ValueError("formal training requires a measured calibration record")
    if command == "train":
        validate_calibration(config)
        validate_stage_gate(config)
        config["_formal_training"] = True
    return config


def validate_stage_gate(config):
    """Require evidence for the technical IA→IB or IB→IC transition."""
    stage = config["stage"]
    if stage not in ("ib", "ic"):
        return None
    path = Path(config.get("stage_gate_record", ""))
    if not path.is_file():
        raise ValueError("formal IB/IC requires a main-agent stage_gate_record")
    record = json.loads(path.read_text(encoding="utf-8"))
    _finite_tree(record)
    if record.get("reference_policy") != REFERENCE_POLICY:
        raise ValueError("stage gate lacks clean reference-policy lineage")
    expected = "ia1" if stage == "ib" else "ib"
    if (
        record.get("status") != "technical_gate_passed"
        or record.get("source_stage") != expected
        or record.get("stage") != stage
    ):
        raise ValueError("stage gate is not a passed preceding-stage technical review")
    checkpoint_sha = stage_start_binding(config)["stage_start_checkpoint_sha256"]
    if (
        checkpoint_sha == config["initial_checkpoint_sha256"]
        or record.get("checkpoint_sha256") != checkpoint_sha
    ):
        raise ValueError("IB/IC cannot use a fresh or different checkpoint")
    if not record.get("rationale") or not record.get("supporting_evaluations"):
        raise ValueError("stage transition requires learning evidence and rationale")
    for item in record["supporting_evaluations"]:
        if sha256_file(item["path"]) != item["sha256"]:
            raise ValueError("stage gate evaluation evidence SHA mismatch")
        evidence = json.loads(Path(item["path"]).read_text(encoding="utf-8"))
        if evidence.get("metadata", {}).get("reference_policy") != REFERENCE_POLICY:
            raise ValueError("stage gate evaluation lacks clean reference policy")
    return {"sha256": sha256_file(path), "record": record}


def _finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("calibration contains a nonfinite number")
    if isinstance(value, dict):
        for child in value.values():
            _finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            _finite_tree(child)


def stage_start_binding(config):
    """Bind a resume to the original measured start saved by the same stage."""
    current = Path(config["checkpoint"])
    current_sha = sha256_file(current)
    explicit = config.get("stage_start_checkpoint")
    if not config.get("resume_optimizer", False):
        start = Path(explicit) if explicit else current
        start_sha = sha256_file(start)
        if start_sha != current_sha:
            raise ValueError("fresh stage_start_checkpoint differs from its checkpoint")
        return {
            "stage_start_checkpoint": str(start),
            "stage_start_checkpoint_sha256": start_sha,
        }

    import torch

    if not explicit:
        raise ValueError("optimizer resume requires explicit stage_start_checkpoint")
    saved = torch.load(current, map_location="cpu", weights_only=False, mmap=True)
    assert_clean_checkpoint_lineage(saved)
    if (
        saved.get("experiment") != "runc_v2"
        or saved.get("stage") != config["stage"]
        or saved.get("optimizer") is None
    ):
        raise ValueError("optimizer resume requires a same-stage safe v2 snapshot")
    previous = saved["runc_v2_runtime"]
    metadata = saved["runc_v2_metadata"]
    if previous.get("stage") != config["stage"]:
        raise ValueError("saved runtime stage differs from optimizer resume stage")
    if metadata.get("initial_checkpoint_sha256") != config["initial_checkpoint_sha256"]:
        raise ValueError("optimizer resume changed initial checkpoint lineage")
    if previous.get("weights") != config.get("weights"):
        raise ValueError("optimizer resume changed selected loss weights")
    previous_start = previous.get("stage_start_checkpoint")
    if not previous_start:
        if previous.get("resume_optimizer"):
            raise ValueError("saved resume lost its original stage-start checkpoint")
        previous_start = previous["checkpoint"]
    expected_sha = metadata.get("stage_start_checkpoint_sha256")
    if not expected_sha and not previous.get("resume_optimizer"):
        expected_sha = metadata.get("checkpoint_sha256")
    if (
        not expected_sha
        or sha256_file(previous_start) != expected_sha
        or sha256_file(explicit) != expected_sha
    ):
        raise ValueError("optimizer resume stage-start checkpoint SHA mismatch")
    expected_calibration = metadata.get("calibration", {}).get("calibration_sha256")
    if (
        not expected_calibration
        or sha256_file(previous["calibration_record"]) != expected_calibration
        or sha256_file(config["calibration_record"]) != expected_calibration
    ):
        raise ValueError("optimizer resume changed its saved calibration file SHA")
    return {
        "stage_start_checkpoint": str(explicit),
        "stage_start_checkpoint_sha256": expected_sha,
        "resumed_from_checkpoint_sha256": current_sha,
    }


def validate_calibration(config):
    """Bind selected weights to a real, finite, nine-source gradient probe."""
    path = Path(config["calibration_record"])
    record = json.loads(path.read_text(encoding="utf-8"))
    _finite_tree(record)
    if record.get("status") != "selected_from_measured_gradients":
        raise ValueError("calibration requires selected_from_measured_gradients status")
    policy = config.get("data", {}).get("reference_policy")
    if policy != REFERENCE_POLICY or record.get("reference_policy") != policy:
        raise ValueError("calibration reference policy is absent or unsafe")
    if record.get("stage") != config["stage"]:
        raise ValueError("calibration stage differs from training stage")
    if record.get("initial_checkpoint_sha256") != config["initial_checkpoint_sha256"]:
        raise ValueError("calibration initial checkpoint differs from ckpt14")
    if record.get("selected_weights") != config.get("weights"):
        raise ValueError("training weights differ from measured calibration selection")
    if not record.get("rationale"):
        raise ValueError(
            "gradient-derived weight selection needs an explicit rationale"
        )
    source = Path(record["source_probe"])
    if sha256_file(source) != record.get("source_probe_sha256"):
        raise ValueError("calibration source probe SHA mismatch")
    probe_record = json.loads(source.read_text(encoding="utf-8"))
    _finite_tree(probe_record)
    if probe_record.get("metadata", {}).get("reference_policy") != policy:
        raise ValueError("calibration probe lacks a safe distinct-original policy")
    for row in probe_record.get("rows", []):
        assert_distinct_references(row)
    if (
        probe_record.get("status") != "measured"
        or probe_record.get("stage") != config["stage"]
    ):
        raise ValueError("calibration source is not a measured same-stage probe")
    if len(set(probe_record.get("authors", []))) != 9:
        raise ValueError("calibration probe does not cover nine sources")
    if (
        probe_record["metadata"].get("initial_checkpoint_sha256")
        != config["initial_checkpoint_sha256"]
    ):
        raise ValueError("probe initial checkpoint differs from training")
    binding = stage_start_binding(config)
    if (
        probe_record["metadata"].get("checkpoint_sha256")
        != binding["stage_start_checkpoint_sha256"]
    ):
        raise ValueError("probe was measured at a different training start checkpoint")
    if probe_record.get("amp") != config["training"]["amp"]:
        raise ValueError("probe AMP precision differs from formal training")
    for key, value in config.get("model", {}).items():
        if key != "stage" and probe_record.get("model_config", {}).get(key) != value:
            raise ValueError(f"probe model configuration differs: {key}")
    if not probe_record.get("rows") or not probe_record.get("median_raw_norm"):
        raise ValueError("empty gradient measurements cannot calibrate training")
    if not any(value > 0 for value in probe_record["median_raw_norm"].values()):
        raise ValueError("all calibration gradients are zero")
    required = []
    if config["stage"] in ("ia0", "ia1", "ic"):
        required.extend(("head", "head_logsigma", "head_shape"))
    if config["stage"] in ("ib", "ic"):
        required.extend(("density", "occupancy_bce", "j_logsigma", "j_shape"))
    rows = probe_record["rows"]
    minimum = min(2, len(rows))
    for term in required:
        positive = sum(
            row.get("gradients", {})
            .get("all_trainable", {})
            .get("norms", {})
            .get(term, 0)
            > 0
            for row in rows
        )
        if positive < minimum:
            raise ValueError(
                f"required {term} gradient is absent in multiple real batches"
            )
    for term, eligibility in (
        ("head_centroid", "head_relative_eligible_parts"),
        ("j_centroid", "coarse_relative_eligible_parts"),
    ):
        eligible = [row for row in rows if row.get(eligibility, 0) >= 2]
        if eligible and term in probe_record["median_raw_norm"]:
            positive = sum(
                row["gradients"]["all_trainable"]["norms"].get(term, 0) > 0
                for row in eligible
            )
            if positive < min(2, len(eligible)):
                raise ValueError(
                    f"{term} has no gradient in eligible multi-part batches"
                )
    return {
        "calibration_sha256": sha256_file(path),
        "source_probe_sha256": record["source_probe_sha256"],
        "selected_weights": record["selected_weights"],
        "rationale": record["rationale"],
        "reference_policy": policy,
        **binding,
    }


def seed_all(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def distributed_device(device_name=None):
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(
        device_name or (f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return device, rank, world


def backend_record(device):
    import torch

    return {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def configure_backend(config):
    import torch

    backend = config.get("backend", {})
    torch.backends.cuda.matmul.allow_tf32 = bool(backend.get("matmul_tf32", False))
    torch.backends.cudnn.allow_tf32 = bool(backend.get("cudnn_tf32", False))
    torch.backends.cudnn.benchmark = bool(backend.get("cudnn_benchmark", False))


def source_record(config):
    """Record actual checked-out source and dirty tracked file hashes."""

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    tracked = git("diff", "--name-only", "HEAD").splitlines()
    data = config["data"]
    paths = []
    for name in ("train_json", "val_json"):
        paths.extend(data.get(name, []))
    for name in ("fixed_pairs", "jt_manifest"):
        if data.get(name):
            paths.append(data[name])
    calibration = (
        validate_calibration(config) if config.get("calibration_record") else None
    )
    stage_start_sha = (
        calibration["stage_start_checkpoint_sha256"]
        if calibration
        else sha256_file(config["checkpoint"])
    )
    return {
        "source_sha": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "tracked_dirty": bool(tracked),
        "tracked_dirty_sha256": {
            path: sha256_file(ROOT / path)
            for path in tracked
            if (ROOT / path).is_file()
        },
        "checkpoint_sha256": sha256_file(config["checkpoint"]),
        "initial_checkpoint_sha256": config["initial_checkpoint_sha256"],
        "reference_policy": config["data"]["reference_policy"],
        "stage_start_checkpoint_sha256": stage_start_sha,
        "data_file_sha256": {str(path): sha256_file(path) for path in paths},
        "calibration": calibration,
        "stage_gate": validate_stage_gate(config)
        if config.get("_formal_training")
        else None,
    }


def tensor_sha256(value):
    import torch

    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def state_sha256(model, frozen_only=False):
    """Hash exact parameter bytes; mode buffers are audited separately."""
    digest = hashlib.sha256()
    count = 0
    for name, parameter in sorted(model.named_parameters()):
        if frozen_only and parameter.requires_grad:
            continue
        digest.update(name.encode())
        digest.update(bytes.fromhex(tensor_sha256(parameter)))
        count += parameter.numel()
    return {"sha256": digest.hexdigest(), "parameter_count": count}


def load_model(config, device):
    """Only ckpt14 may initialize stage I; v2 snapshots reload strictly."""
    import torch

    if config.get("data", {}).get("reference_policy") != REFERENCE_POLICY:
        raise ValueError("model loading requires the safe reference policy")

    import models_train
    from util.checkpoint import checkpoint_args
    from util.runc_v2 import (
        enable_runc_v2,
        restore_runc_v2,
        set_runc_v2_stage,
    )

    seed_all(config["training"]["seed"])
    saved = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
    actual_sha = sha256_file(config["checkpoint"])
    is_v2 = bool(saved.get("runc_v2_config"))
    if is_v2:
        assert_clean_checkpoint_lineage(saved)
    architecture = config.get("architecture", ARCHITECTURE)
    args = checkpoint_args(saved)
    if args.get("model", architecture) != architecture:
        raise ValueError("checkpoint/model architecture mismatch")
    model = getattr(models_train, architecture)(dual_view=False)
    if is_v2:
        if saved.get("experiment") != "runc_v2":
            raise ValueError("v2 checkpoint lacks explicit experiment lineage")
        if config.get("_formal_training") and config["stage"] in ("ib", "ic"):
            expected = "ia1" if config["stage"] == "ib" else "ib"
            if saved.get("stage") not in (expected, config["stage"]):
                raise ValueError("formal training skipped the preceding v2 stage")
        provenance = saved.get("runc_v2_metadata", {})
        if (
            provenance.get("initial_checkpoint_sha256")
            != config["initial_checkpoint_sha256"]
        ):
            raise ValueError("v2 checkpoint initial ckpt14 lineage changed")
        model_config = saved["runc_v2_config"]
        for key, value in config.get("model", {}).items():
            if key != "stage" and model_config.get(key) != value:
                raise ValueError(f"v2 architecture override forbidden: {key}")
        restore_runc_v2(model, saved)
        loading = {"mode": "strict_v2_resume", "checkpoint_sha256": actual_sha}
    else:
        if actual_sha != config["initial_checkpoint_sha256"]:
            raise ValueError(
                "fresh v2 initialization must use verified checkpoint14 SHA"
            )
        forbidden = (
            "dual_head.",
            "dual_fusion.",
            "structure_adapter.",
            "g_scale.",
            "g_control.",
            "tip_",
            "runc_v2_",
        )
        if any(key.startswith(forbidden) for key in saved["model"]):
            raise ValueError("Run B/Run C v1 weights cannot initialize Run C v2")
        expected = model.state_dict()
        compatible = {}
        incompatible = []
        unexpected = []
        for key, value in saved["model"].items():
            if key not in expected:
                unexpected.append(key)
            elif value.shape != expected[key].shape:
                incompatible.append(
                    {
                        "name": key,
                        "saved": list(value.shape),
                        "model": list(expected[key].shape),
                    }
                )
            else:
                compatible[key] = value
        missing = sorted(set(expected) - set(compatible))
        inactive = ("vgg_loss.", "discriminator.", "decoder_embed.", "decoder_pred.")
        active_missing = [key for key in missing if not key.startswith(inactive)]
        if active_missing or unexpected:
            raise ValueError(
                f"unsafe checkpoint14 migration: missing={active_missing}, "
                f"unexpected={unexpected}, incompatible={incompatible}"
            )
        merged = dict(expected)
        merged.update(compatible)
        model.load_state_dict(merged, strict=True)
        legacy_sha = state_sha256(model)
        enable_runc_v2(model, config.get("model", {}))
        loading = {
            "mode": "checkpoint14_compatible_only",
            "checkpoint_sha256": actual_sha,
            "loaded_keys": sorted(compatible),
            "new_legacy_inactive_keys": missing,
            "incompatible": incompatible,
            "unexpected": unexpected,
            "legacy_state_before_new_modules": legacy_sha,
            "new_module_keys": sorted(set(model.state_dict()) - set(expected)),
            "new_module_initialization_seed": config["training"]["seed"],
        }
    set_runc_v2_stage(model, config["stage"])
    model.to(device)
    return model, saved, loading


def build_dataset(config, validation=False):
    from data.runc_v2_dataset import build_runc_v2_dataset

    dataset = build_runc_v2_dataset(config, validation=validation)
    assert_dataset_reference_policy(dataset)
    return dataset


def assert_dataset_reference_policy(dataset):
    """Fail before sampling when the dataset lacks the audited policy."""
    if getattr(dataset, "reference_policy", None) != REFERENCE_POLICY:
        raise ValueError("dataset lacks the distinct-original reference policy")


def assert_clean_checkpoint_lineage(saved):
    """Reject old run1 adapted weights even if their ckpt14 SHA matches."""
    metadata_policy = saved.get("runc_v2_metadata", {}).get("reference_policy")
    runtime_policy = (
        saved.get("runc_v2_runtime", {}).get("data", {}).get("reference_policy")
    )
    if metadata_policy != REFERENCE_POLICY or runtime_policy != REFERENCE_POLICY:
        raise ValueError("adapted checkpoint lacks clean reference-policy lineage")


def assert_distinct_references(batch):
    """Reject upper-reference GT leakage using sealed originals and targets."""
    fields = (
        "query_original_identity",
        "reference_original_identity",
        "query_original_sha256",
        "reference_original_sha256",
        "query_target_sha256",
        "reference_target_sha256",
    )
    values = {}
    for name in fields:
        value = batch.get(name)
        items = [value] if isinstance(value, str) else value
        if (
            not isinstance(items, (list, tuple))
            or not items
            or any(not isinstance(item, str) or not item.strip() for item in items)
        ):
            raise ValueError(f"missing or invalid reference-safety metadata: {name}")
        values[name] = list(items)
    count = len(values[fields[0]])
    if any(len(items) != count for items in values.values()):
        raise ValueError("reference identity/hash metadata cardinality mismatch")
    policy = batch.get("reference_policy")
    policies = [policy] * count if isinstance(policy, str) else policy
    if (
        not isinstance(policies, (list, tuple))
        or len(policies) != count
        or any(item != REFERENCE_POLICY for item in policies)
    ):
        raise ValueError("batch lacks the distinct-original reference policy")
    image = batch.get("image")
    if image is not None:
        expected = image.shape[0] if image.ndim == 4 else 1
        if expected != count:
            raise ValueError("reference metadata differs from actual image batch size")
    for name in ("target_path", "reference_path"):
        paths = batch.get(name)
        if paths is not None:
            paths = [paths] if isinstance(paths, str) else paths
            if not isinstance(paths, (list, tuple)) or len(paths) != count:
                raise ValueError(
                    "reference path/identity metadata cardinality mismatch"
                )
    for index in range(count):
        for name in fields[2:]:
            value = values[name][index]
            if len(value) != 64 or any(
                char not in "0123456789abcdef" for char in value
            ):
                raise ValueError(f"reference image SHA256 is invalid: {name}")
        for query, reference in zip(fields[::2], fields[1::2]):
            if values[query][index] == values[reference][index]:
                raise ValueError(f"self-reference GT leakage: identical {query}")
        for prefix in ("query", "reference"):
            identity = values[f"{prefix}_original_identity"][index]
            original_sha = values[f"{prefix}_original_sha256"][index]
            if identity.rsplit(":", 1)[-1] != original_sha:
                raise ValueError("original identity and sealed SHA256 disagree")
    return {"reference_policy": REFERENCE_POLICY, **values}


def to_device(batch, device):
    import torch

    return {
        key: value.to(device, non_blocking=True)
        if torch.is_tensor(value)
        else (to_device(value, device) if isinstance(value, dict) else value)
        for key, value in batch.items()
    }


def collate_indices(dataset, indices, device, seed=0):
    import torch

    assert_dataset_reference_policy(dataset)
    rows = [
        dataset.get_pair(index, augmentation_seed=seed + within)
        for within, index in enumerate(indices)
    ]
    for row in rows:
        assert_distinct_references(row)
    batch = torch.utils.data.default_collate(rows)
    assert_distinct_references(batch)
    return to_device(batch, device)


def safe_target(batch):
    """Sanitize at the final call boundary even if a dataset regresses."""
    target = batch.get("input_target", batch["target"]).clone()
    target[:, :, target.shape[-2] // 2 :] = 0
    return target


def wrap_forward(model):
    import torch

    from util.runc_v2 import runc_v2_forward

    class DeploymentForward(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, image, target, render=True):
            return runc_v2_forward(self.base, image, target, render=render)

    return DeploymentForward(model)


def call_model(wrapped, batch, render=True):
    assert_distinct_references(batch)
    return wrapped(batch["image"], safe_target(batch), render=render)


def training_mode(model, stage):
    from util.runc_v2 import set_runc_v2_stage

    model.train()
    set_runc_v2_stage(model, stage)


def loss_for(outputs, batch, config):
    from util.runc_v2_loss import runc_v2_losses

    return runc_v2_losses(outputs, batch, config.get("weights"), config["stage"])


def scalar_losses(losses):
    import torch

    return {
        name: float(value.detach().float())
        for name, value in losses.items()
        if torch.is_tensor(value) and value.numel() == 1
    }


def parameter_group(name):
    if name.startswith("bf_refiner."):
        return "refiner"
    if name.startswith("dual_head."):
        return "head"
    if name.startswith(("blocks.", "norm.")):
        return "vit"
    return "coarse"


def build_optimizer(model, settings):
    import torch

    groups = defaultdict(list)
    names = defaultdict(list)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = parameter_group(name)
            groups[group].append(parameter)
            names[group].append(name)
    optimizer = torch.optim.AdamW(
        [
            {"params": parameters, "lr": settings[f"lr_{name}"], "name": name}
            for name, parameters in sorted(groups.items())
        ],
        weight_decay=settings["weight_decay"],
        betas=(0.9, 0.999),
    )
    assert_optimizer(model, optimizer)
    return optimizer, dict(names)


def assert_optimizer(model, optimizer):
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    values = [id(p) for group in optimizer.param_groups for p in group["params"]]
    if set(values) != expected or len(values) != len(set(values)):
        raise RuntimeError("optimizer parameter membership differs from requires_grad")


def grouped_gradient_norms(model):
    import torch

    squares = defaultdict(float)
    used = Counter()
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if not parameter.requires_grad:
            raise RuntimeError(f"frozen parameter has gradient: {name}")
        grad = parameter.grad.detach().float()
        if not bool(torch.isfinite(grad).all()):
            raise FloatingPointError(f"nonfinite gradient: {name}")
        group = parameter_group(name)
        squares[group] += float(grad.square().sum())
        used[group] += 1
    return {
        name: {"norm": math.sqrt(value), "used_tensors": used[name]}
        for name, value in squares.items()
    }


def parameter_grams(terms, named_parameters):
    """Measure actual leaf-loss parameter gradients, their cosine and Gram."""
    import torch

    named_parameters = list(named_parameters)
    parameters = [p for _, p in named_parameters]
    gradients = []
    for value in terms.values():
        gradients.append(
            torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
            if value.requires_grad
            else [None] * len(parameters)
        )
    names = list(terms)
    gram = [[0.0] * len(names) for _ in names]
    for i in range(len(names)):
        for j in range(i + 1):
            inner = 0.0
            for left, right in zip(gradients[i], gradients[j]):
                if left is not None and right is not None:
                    if not bool(
                        torch.isfinite(left).all() & torch.isfinite(right).all()
                    ):
                        raise FloatingPointError("nonfinite calibration gradient")
                    inner += float((left.float() * right.float()).sum())
            gram[i][j] = gram[j][i] = inner
    norms = {name: math.sqrt(max(0.0, gram[i][i])) for i, name in enumerate(names)}
    cosine = {}
    for i, left in enumerate(names):
        for j, right in enumerate(names[:i]):
            denominator = norms[left] * norms[right]
            cosine[f"{left}:{right}"] = (
                max(-1.0, min(1.0, gram[i][j] / denominator))
                if denominator > 0
                else None
            )
    return {
        "terms": names,
        "gram": gram,
        "norms": norms,
        "cosine": cosine,
        "parameter_tensors": len(parameters),
    }


def author_probe_indices(dataset, batch_size, repeats=1):
    """Cycle all observed sources; source coverage is a checked output."""
    pools = defaultdict(list)
    for index, row in enumerate(dataset.pairs):
        pools[row.get("calligrapher", row.get("author", "unknown"))].append(index)
    result = []
    for repeat in range(repeats):
        for author, indices in sorted(pools.items()):
            result.append(
                (
                    author,
                    [
                        indices[(repeat * batch_size + offset) % len(indices)]
                        for offset in range(batch_size)
                    ],
                )
            )
    return result


def probe(config, model, dataset, output, device):
    import torch

    assert_dataset_reference_policy(dataset)
    training_mode(model, config["stage"])
    wrapped = wrap_forward(model)
    cases = author_probe_indices(
        dataset, config.get("probe_batch_size", 1), config.get("probe_repeats", 1)
    )
    if len(set(author for author, _ in cases)) != 9:
        raise ValueError("formal gradient calibration requires all nine sources")
    records = []
    for number, (author, indices) in enumerate(cases):
        batch = collate_indices(dataset, indices, device, seed=number + 1701)
        with torch.autocast(
            device.type,
            dtype=torch.bfloat16,
            enabled=config["training"].get("amp", False),
        ):
            outputs = call_model(wrapped, batch, render=config["stage"] in ("ib", "ic"))
            losses = loss_for(outputs, batch, config)
        terms = losses["probe_terms"]
        groups = {
            "all_trainable": [
                (name, p) for name, p in model.named_parameters() if p.requires_grad
            ]
        }
        for name in ("head", "vit", "coarse"):
            groups[name] = [
                (key, p)
                for key, p in model.named_parameters()
                if p.requires_grad and parameter_group(key) == name
            ]
        measurements = {
            name: parameter_grams(terms, parameters)
            for name, parameters in groups.items()
            if parameters
        }
        model.zero_grad(set_to_none=True)
        losses["total"].backward()
        before = grouped_gradient_norms(model)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            config["training"].get("clip_grad", 3.0),
            error_if_nonfinite=True,
        )
        after = grouped_gradient_norms(model)
        records.append(
            {
                "batch": number,
                "author": author,
                "indices": indices,
                "target_path": batch["target_path"],
                "reference_path": batch["reference_path"],
                **assert_distinct_references(batch),
                "losses": scalar_losses(losses),
                "gradients": measurements,
                "weighted_preclip": before,
                "weighted_postclip": after,
                "head_relative_eligible_parts": int(
                    losses["head_geometry"]["relative_valid"].sum()
                ),
                "coarse_relative_eligible_parts": int(
                    losses["coarse_geometry"]["relative_valid"].sum()
                )
                if "coarse_geometry" in losses
                else 0,
            }
        )
        model.zero_grad(set_to_none=True)
        print(json.dumps({"probe_batch": number, "author": author}), flush=True)
    norms = defaultdict(list)
    for row in records:
        for name, norm in row["gradients"]["all_trainable"]["norms"].items():
            norms[name].append(norm)
    report = {
        "schema": SCHEMA,
        "stage": config["stage"],
        "status": "measured",
        "metadata": source_record(config),
        "backend": backend_record(device),
        "amp": config["training"].get("amp", False),
        "rows": records,
        "configured_weights": config.get("weights", {}),
        "model_config": model.runc_v2_config,
        "median_raw_norm": {
            name: statistics.median(values) for name, values in norms.items()
        },
        "authors": sorted(set(author for author, _ in cases)),
        "interpretation": "measured gradients; no final budget or quality claim",
    }
    write_json(Path(output) / "gradient_probe.json", report)
    return report


def quantiles(values):
    """Finite numeric summaries, using a deterministic linear quantile."""
    values = sorted(float(value) for value in values)
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "p95": None}
    if not all(math.isfinite(value) for value in values):
        raise ValueError("nonfinite metric cannot be summarized")

    def percentile(q):
        position = q * (len(values) - 1)
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        return values[lower] + (values[upper] - values[lower]) * (position - lower)

    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(0.5),
        "p90": percentile(0.9),
        "p95": percentile(0.95),
    }


def summarize_rows(rows, metric_names):
    """Pair and unique-target means use distinct, explicit denominators."""
    by_unique = defaultdict(list)
    by_author = defaultdict(list)
    by_author_character = defaultdict(list)
    for row in rows:
        by_unique[row["identity"]].append(row)
        by_author[row["author"]].append(row)
        by_author_character[(row["author"], row["character"])].append(row)

    def summary(items):
        return {
            name: quantiles([row[name] for row in items if name in row])
            for name in metric_names
        }

    def grouped_means(groups):
        return [
            {
                name: statistics.fmean(row[name] for row in items if name in row)
                for name in metric_names
                if any(name in row for row in items)
            }
            for items in groups.values()
        ]

    return {
        "pair_count": len(rows),
        "unique_target_count": len(by_unique),
        "author_character_count": len(by_author_character),
        "pairs": summary(rows),
        "unique_targets": summary(grouped_means(by_unique)),
        "author_character": summary(grouped_means(by_author_character)),
        "by_author": {
            name: {"pairs": len(items), "metrics": summary(items)}
            for name, items in sorted(by_author.items())
        },
    }


def _batch_text(batch, key, default="unknown"):
    value = batch.get(key, [default])
    return str(value[0] if isinstance(value, (list, tuple)) else value)


def _head_instance_metrics(outputs, batch, losses):
    """Same Hungarian assignment as the training loss, never threshold occupancy."""
    import torch

    from util.runc_v2_loss import head_metric_sums

    head = outputs["head"]
    details = losses["head_parts"]
    canonical = head_metric_sums(head, batch, losses)
    assignment = details["assignment"]
    matched = details["matched_slots"] & batch["has_jt"][:, None].bool()
    p = head["occupancy_logits"].float().sigmoid()
    target = (
        batch["instance_occupancy"]
        .gather(1, assignment[:, :, None, None].expand_as(p))
        .float()
    )
    intersection = (p * target).sum((-2, -1))
    union = (p + target - p * target).sum((-2, -1))
    classes = batch["instance_classes"].gather(1, assignment)
    role = head["class_logits"].argmax(-1).eq(classes).float()
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        head["occupancy_logits"].float(), target, reduction="none"
    ).mean((-2, -1))
    dice = 1 - (2 * intersection + 1e-6) / (
        p.sum((-2, -1)) + target.sum((-2, -1)) + 1e-6
    )
    values = {
        "head_soft_iou": intersection / union.clamp_min(1e-6),
        "head_role_accuracy": role,
        "head_bce": bce,
        "head_dice": dice,
        "head_existence_recall": (head["existence_logits"] > 0).float(),
    }
    result = {name: value[matched].cpu().tolist() for name, value in values.items()}
    for name in (
        "logsigma_x",
        "logsigma_y",
        "centroid_absolute",
        "centroid_relative",
        "shape",
    ):
        mask = canonical["relative_valid"] if name == "centroid_relative" else matched
        result[f"head_{name}"] = canonical["per_instance"][name][mask].cpu().tolist()
    result["head_logsigma"] = [
        a + b for a, b in zip(result["head_logsigma_x"], result["head_logsigma_y"])
    ]
    return result


def _coarse_instance_metrics(outputs, batch, losses, config):
    """Report model/oracle on one original 448-pixel geometric coordinate grid.

    The existing audited GT Voronoi has 224 cells and is expanded nearest to
    448. Prediction, original GT and upsampled oracle all use that same map.
    These metrics are separate from training's directly built 112 partition.
    """
    import torch

    from util.runc_v2_loss import CoarseJietiLoss

    size = batch["image"].shape[-1]
    mean = batch["target"].new_tensor(MEAN)[None, :, None, None]
    std = batch["target"].new_tensor(STD)[None, :, None, None]

    def ink(normalized):
        rgb = normalized.float()[:, :, -size:] * std + mean
        gray = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
        return (1 - gray).clamp(0, 1)

    partition = torch.nn.functional.interpolate(
        batch["voro"][:, None].float(), size=(size, size), mode="nearest"
    )[:, 0].long()
    criterion = CoarseJietiLoss(pool=size, slots=4).to(batch["image"].device)
    target = ink(batch["target"])
    prediction = ink(outputs["p0_lower"])
    oracle = torch.nn.functional.interpolate(
        batch["density_target"].float(),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )
    original = criterion(
        prediction, target, partition, batch["valid_parts"], batch["has_jt"]
    )
    floor = criterion(oracle, target, partition, batch["valid_parts"], batch["has_jt"])
    result = {}
    for prefix, geometry in (
        ("coarse", losses["coarse_geometry"]),
        ("p0_original448", original),
        ("oracle_original448", floor),
    ):
        for name in (
            "logsigma_x",
            "logsigma_y",
            "centroid_absolute",
            "centroid_relative",
            "shape",
        ):
            mask = (
                geometry["relative_valid"]
                if name == "centroid_relative"
                else geometry["valid"]
            )
            result[f"{prefix}_{name}"] = geometry[name][mask].cpu().tolist()
        result[f"{prefix}_shield"] = [float(geometry["shield_count"])]
        reductions = geometry["reductions"]
        weights = config.get("weights", {})
        value = (
            weights.get("j_centroid", 1) * reductions["centroid_relative"]["mean"]
            + weights.get("j_logsigma", 1)
            * (reductions["logsigma_x"]["mean"] + reductions["logsigma_y"]["mean"])
            + weights.get("j_shape", 1) * reductions["shape"]["mean"]
        )
        result[f"{prefix}_J_per_character"] = [float(value)]
    return result


def save_visual(path, outputs, batch):
    """A labeled strip keeps deployment predictions separate from GT panels."""
    import numpy as np
    import torch
    from PIL import Image, ImageDraw

    size = batch["image"].shape[-1]
    mean = torch.tensor(MEAN)[:, None, None]
    std = torch.tensor(STD)[:, None, None]

    def rgb(value, normalized=True):
        value = value.detach().float().cpu()[0, :, -size:]
        if value.shape[0] == 1:
            value = value.repeat(3, 1, 1)
        if normalized:
            value = value * std + mean
        if value.shape[-2:] != (size, size):
            value = torch.nn.functional.interpolate(
                value[None], size=(size, size), mode="nearest"
            )[0]
        return (value.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype("uint8")

    labels = ["source", "GT comparison only", "JT GT occupancy", "S predicted"]
    head = outputs["head"]
    s_pred = head["occupancy_logits"].float().sigmoid()
    exists = head["existence_logits"].float().sigmoid()
    s_pred = (s_pred * exists[..., None, None]).sum(1, keepdim=True).clamp(0, 1)
    s_gt = batch["instance_occupancy"].sum(1, keepdim=True).clamp(0, 1)
    panels = [
        rgb(batch["image"]),
        rgb(batch["target"]),
        rgb(1 - s_gt, False),
        rgb(1 - s_pred, False),
    ]
    for key, label in (
        ("density_target", "D_GT oracle"),
        ("occupancy_target", "O_GT oracle"),
    ):
        if key in batch:
            labels.append(label)
            panels.append(rgb(1 - batch[key], False))
    if "density" in outputs:
        labels.extend(["D0 generated", "O0 predicted", "P0 generated"])
        panels.extend(
            [
                rgb(1 - outputs["density"], False),
                rgb(1 - outputs["occupancy_logits"].sigmoid(), False),
                rgb(outputs.get("p0_lower", outputs["p0"])),
            ]
        )
    array = np.concatenate(panels, axis=1)
    canvas = Image.new("RGB", (array.shape[1], size + 30), "white")
    canvas.paste(Image.fromarray(array), (0, 30))
    draw = ImageDraw.Draw(canvas)
    for index, label in enumerate(labels):
        draw.text((index * size + 4, 5), label, fill="black")
    canvas.save(path)
    return labels


def evaluate(
    config,
    model,
    dataset,
    output,
    device,
    limit=None,
    indices=None,
    cohort_name="fixed_historical_val",
):
    """Fixed 101 JT pairs are primary; per-instance and unique-target coexist."""
    import torch

    assert_dataset_reference_policy(dataset)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    render = config["stage"] in ("ib", "ic")
    model.eval()
    wrapped = wrap_forward(model)
    rows = []
    instances = defaultdict(list)
    author_instances = defaultdict(lambda: defaultdict(list))
    instance_rows = []
    visuals = []
    required = {"無", "容", "惟", "雲", "崇"}
    seen_authors = set()
    for index in range(len(dataset)) if indices is None else indices:
        if limit is not None and len(rows) >= limit:
            break
        batch = collate_indices(dataset, [index], device, seed=0)
        if not bool(batch["has_jt"][0]):
            continue
        with torch.no_grad():
            outputs = call_model(wrapped, batch, render=render)
            losses = loss_for(outputs, batch, config)
            imetrics = _head_instance_metrics(outputs, batch, losses)
            if render:
                imetrics.update(
                    _coarse_instance_metrics(outputs, batch, losses, config)
                )
        scalar = scalar_losses(losses)
        path = _batch_text(batch, "target_path")
        identity = _batch_text(batch, "original_identity", path)
        author = _batch_text(batch, "author", path.split("/")[0])
        character = _batch_text(batch, "character", Path(path).stem)
        row = {
            "index": index,
            "identity": identity,
            "author": author,
            "character": character,
            "target_path": path,
            "reference_path": _batch_text(batch, "reference_path"),
            "instance_count": len(imetrics["head_soft_iou"]),
        }
        row.update(scalar)
        for name, values in imetrics.items():
            instances[name].extend(values)
            author_instances[author][name].extend(values)
            if values:
                row[name] = statistics.fmean(values)
        rows.append(row)
        instance_rows.append(
            {
                "identity": identity,
                "author": author,
                "character": character,
                "metrics": imetrics,
            }
        )
        if len(visuals) < 12 or character in required or author not in seen_authors:
            filename = f"jt-{index:03d}.png"
            labels = save_visual(output / filename, outputs, batch)
            visuals.append(
                {
                    "file": filename,
                    "target": path,
                    "author": author,
                    "character": character,
                    "panels": labels,
                }
            )
            required.discard(character)
            seen_authors.add(author)
    metrics = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float))
            and key not in ("index", "instance_count")
        }
    )
    summary = summarize_rows(rows, metrics)
    report = {
        "schema": SCHEMA,
        "stage": config["stage"],
        "metadata": source_record(config),
        "backend": backend_record(device),
        "rows": rows,
        "summary": summary,
        "cohort": cohort_name,
        "evaluated_parameter_sha256": state_sha256(model),
        "instance_rows": instance_rows,
        "instance_count": len(instances["head_soft_iou"]),
        "instance_metrics": {
            name: quantiles(values) for name, values in instances.items()
        },
        "by_author_instance_metrics": {
            author: {name: quantiles(values) for name, values in metrics.items()}
            for author, metrics in sorted(author_instances.items())
        },
        "missing_required_characters": sorted(required),
        "visuals": visuals,
        "precision": "native FP32 forward; model eval; TF32 per backend record",
        "original_geometry_coordinate": (
            "448 original GT/P0/oracle; common GT Voronoi nearest-expanded "
            "from audited 224; training coarse partition is directly built at112"
        ),
        "validation_limitation": (
            "historical reused validation; not independent generalization"
        ),
    }
    if (
        limit is None
        and indices is None
        and cohort_name == "fixed_historical_val"
        and config.get("data", {}).get("enforce_fixed_val_counts", True)
    ):
        observed = (len(rows), report["instance_count"], summary["unique_target_count"])
        if observed != (101, 216, 77):
            raise ValueError(
                f"fixed-val denominator changed: {observed}, expected 101/216/77"
            )
    write_json(output / "evaluation.json", report)
    if rows:
        fields = [
            "index",
            "identity",
            "author",
            "character",
            "target_path",
            "reference_path",
            "instance_count",
            *metrics,
        ]
        with (output / "metrics.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    gallery = "\n".join(
        f"<figure><figcaption>{html.escape(item['author'] + ' / ' + item['character'])}"
        f'</figcaption><img src="{item["file"]}" style="max-width:100%"></figure>'
        for item in visuals
    )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Run C v2 JT comparison</title>'
        "<p>GT panels are comparison only; reused historical validation.</p>" + gallery,
        encoding="utf-8",
    )
    return report


def _rng_state():
    import numpy as np
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def _restore_rng(state):
    import numpy as np
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def snapshot(model, optimizer, config, metadata, output, step, rank, world, counts):
    import torch
    import torch.distributed as dist

    local_rng = _rng_state()
    rng = [None] * world
    if world > 1:
        dist.all_gather_object(rng, local_rng)
    else:
        rng[0] = local_rng
    if rank != 0:
        return
    settings = config["training"]
    payload = {
        "experiment": "runc_v2",
        "model": model.state_dict(),
        "runc_v2_config": model.runc_v2_config,
        "runc_v2_refiner_config": getattr(model, "runc_v2_refiner_config", None),
        "runc_v2_runtime": config,
        "runc_v2_metadata": metadata,
        "optimizer": optimizer.state_dict(),
        "optimizer_step": step,
        "stage": config["stage"],
        "rng_states": rng,
        "world_size": world,
        "effective_batch": settings["batch_size"] * settings["accum_iter"] * world,
        "sample_counts_rank0": dict(counts),
        "args": {"model": config.get("architecture", ARCHITECTURE)},
    }
    path = Path(output) / f"checkpoint-step-{step:06d}.pth"
    with path.open("xb") as stream:
        torch.save(payload, stream)
    write_json(
        str(path) + ".sha256.json",
        {
            "sha256": sha256_file(path),
            "optimizer_step": step,
            "initial_checkpoint_sha256": config["initial_checkpoint_sha256"],
        },
    )


def _reduce_logs(logs, device, world):
    import torch
    import torch.distributed as dist

    names = sorted(logs[0])
    values = torch.tensor(
        [
            sum(row[name] for row in logs)
            if name.startswith("count_")
            else statistics.fmean(row[name] for row in logs)
            for name in names
        ],
        device=device,
        dtype=torch.float64,
    )
    if world > 1:
        dist.all_reduce(values)
    return {
        name: float(values[index]) / (1 if name.startswith("count_") else world)
        for index, name in enumerate(names)
    }


def train(config, model, saved, dataset, output, device, rank, world, smoke=False):
    """Complete accumulation windows, real AdamW state, exact frozen-state check."""
    import torch
    import torch.distributed as dist

    assert_dataset_reference_policy(dataset)
    settings = config["training"]
    training_mode(model, config["stage"])
    optimizer, parameter_names = build_optimizer(model, settings)
    provenance = source_record(config)
    start = 0
    if config.get("resume_optimizer", False):
        if saved.get("stage") != config["stage"] or not saved.get("optimizer"):
            raise ValueError("optimizer resume requires the same stage snapshot")
        if saved.get("world_size") != world:
            raise ValueError("exact resume requires unchanged world size")
        previous = saved["runc_v2_runtime"]["training"]
        for key in (
            "seed",
            "batch_size",
            "accum_iter",
            "amp",
            "lr_head",
            "lr_vit",
            "lr_coarse",
            "weight_decay",
            "warmup_steps",
        ):
            if previous.get(key) != settings.get(key):
                raise ValueError(f"exact optimizer resume changed training.{key}")
        optimizer.load_state_dict(saved["optimizer"])
        start = saved["optimizer_step"]
        _restore_rng(saved["rng_states"][rank])
    frozen_before = state_sha256(model, frozen_only=True)
    wrapped = wrap_forward(model)
    if world > 1:
        wrapped = torch.nn.parallel.DistributedDataParallel(
            wrapped,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,
            broadcast_buffers=False,
        )
    effective = settings["batch_size"] * settings["accum_iter"] * world
    generator = torch.Generator().manual_seed(settings["seed"])
    draws = torch.multinomial(
        torch.as_tensor(dataset.weights, dtype=torch.float64),
        settings["steps"] * effective,
        replacement=True,
        generator=generator,
    ).tolist()
    counts = Counter(saved.get("sample_counts_rank0", {}) if start else {})
    if rank == 0:
        write_json(Path(output) / "optimizer_groups.json", parameter_names)
        write_json(
            Path(output) / "metadata.json",
            {
                **provenance,
                "backend": backend_record(device),
                "world_size": world,
                "effective_batch": effective,
                "config": config,
            },
        )
    render = config["stage"] in ("ib", "ic")
    if start == 0 and settings.get("save_initial", True):
        snapshot(model, optimizer, config, provenance, output, 0, rank, world, counts)
        if not smoke and settings.get("evaluate_every_save", False):
            if rank == 0:
                evaluate_training_tracks(config, model, dataset, output, device, 0)
                training_mode(model, config["stage"])
            if world > 1:
                dist.barrier()
    for step in range(start + 1, settings["steps"] + 1):
        tick = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        logs = []
        reference_trace = []
        warmup = settings.get("warmup_steps", 0)
        fraction = min(1.0, step / warmup) if warmup else 1.0
        for group in optimizer.param_groups:
            group["lr"] = settings[f"lr_{group['name']}"] * fraction
        for inner in range(settings["accum_iter"]):
            offset = (step - 1) * settings["accum_iter"] + inner
            offset *= settings["batch_size"] * world
            indices = draws[offset : offset + settings["batch_size"] * world]
            indices = indices[rank::world]
            batch = collate_indices(
                dataset, indices, device, seed=settings["seed"] + offset + rank
            )
            if not bool(batch["has_jt"].all()):
                raise ValueError("BF-only query reached Run C v2 stage I")
            reference_trace.append(assert_distinct_references(batch))
            counts.update(batch.get("original_identity", batch["target_path"]))
            sync = (
                wrapped.no_sync()
                if (world > 1 and inner < settings["accum_iter"] - 1)
                else nullcontext()
            )
            with sync:
                with torch.autocast(
                    device.type, dtype=torch.bfloat16, enabled=settings["amp"]
                ):
                    outputs = call_model(wrapped, batch, render=render)
                    losses = loss_for(outputs, batch, config)
                    loss = losses["total"] / settings["accum_iter"]
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite objective; stage stopped")
                loss.backward()
            logs.append(scalar_losses(losses))
        assert_optimizer(model, optimizer)
        before = grouped_gradient_norms(model)
        if not any(value["norm"] > 0 for value in before.values()):
            raise RuntimeError("no nonzero stage parameter gradient")
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            settings["clip_grad"],
            error_if_nonfinite=True,
        )
        after = grouped_gradient_norms(model)
        optimizer.step()
        row = {
            "optimizer_step": step,
            "elapsed_seconds": time.monotonic() - tick,
            "losses": _reduce_logs(logs, device, world),
            "gradient_preclip": float(norm),
            "gradient_groups_preclip": before,
            "gradient_groups_postclip": after,
            "effective_batch": effective,
            "reference_policy": REFERENCE_POLICY,
            "reference_identities_rank0": reference_trace,
            "lr": {group["name"]: group["lr"] for group in optimizer.param_groups},
            "peak_cuda_allocated": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
        }
        if rank == 0:
            with (Path(output) / "train.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            print(json.dumps(row, allow_nan=False), flush=True)
        should_save = (
            step == 1 or step % settings["save_every"] == 0 or step == settings["steps"]
        )
        if should_save:
            snapshot(
                model, optimizer, config, provenance, output, step, rank, world, counts
            )
            if rank == 0:
                save_visual(Path(output) / f"step-{step:06d}.png", outputs, batch)
        if world > 1:
            dist.barrier()
        if should_save and not smoke and settings.get("evaluate_every_save", False):
            if rank == 0:
                evaluate_training_tracks(config, model, dataset, output, device, step)
                training_mode(model, config["stage"])
            if world > 1:
                dist.barrier()
    frozen_after = state_sha256(model, frozen_only=True)
    if frozen_before != frozen_after:
        raise RuntimeError("frozen stage parameters changed after AdamW")
    if rank == 0:
        write_json(
            Path(output) / "freeze_sha.json",
            {
                "before": frozen_before,
                "after": frozen_after,
                "exact_unchanged": True,
                "optimizer_steps": settings["steps"] - start,
                "world_size": world,
                "sample_counts_rank0": dict(counts),
            },
        )
    return {
        "optimizer_steps": settings["steps"] - start,
        "frozen_exact_unchanged": True,
        "world_size": world,
    }


def evaluate_training_tracks(config, model, dataset, output, device, step):
    """Keep fixed nine-source train targets and the historical val separate."""
    selected = [
        index
        for _, indices in author_probe_indices(dataset, batch_size=2)
        for index in indices
    ]
    evaluate(
        config,
        model,
        dataset,
        Path(output) / f"train-evaluation-step-{step:06d}",
        device,
        indices=selected,
        cohort_name="fixed_nine_source_train18",
    )
    evaluate(
        config,
        model,
        build_dataset(config, validation=True),
        Path(output) / f"evaluation-step-{step:06d}",
        device,
    )


def smoke(config, model, saved, dataset, output, device, rank, world):
    """Actual optimizer/AMP/DDP steps plus deployment GT replacement invariance."""
    import torch

    result = train(
        config, model, saved, dataset, output, device, rank, world, smoke=True
    )
    batch = collate_indices(dataset, [rank % len(dataset)], device, seed=9021)
    altered = dict(batch)
    altered["target"] = torch.randn_like(batch["target"])
    half = batch["target"].shape[-2] // 2
    altered["target"][:, :, :half] = batch["target"][:, :, :half]
    altered.pop("input_target", None)
    render = config["stage"] in ("ib", "ic")
    wrapped = wrap_forward(model)
    model.eval()
    with torch.no_grad():
        normal = call_model(wrapped, batch, render=render)
        changed = call_model(wrapped, altered, render=render)
        repeated = call_model(wrapped, batch, render=render)
    comparisons = {}
    for name in ("occupancy_logits", "existence_logits", "class_logits"):
        left, right = normal["head"][name], changed["head"][name]
        repeat = repeated["head"][name]
        if not torch.equal(left, right) or not torch.equal(left, repeat):
            raise RuntimeError(f"GT leak/nondeterminism in predicted S: {name}")
        comparisons[name] = {"gt_replacement_equal": True, "repeat_equal": True}
    for name in ("density", "p0") if render else ():
        if not torch.equal(normal[name], changed[name]):
            raise RuntimeError(f"GT leak in {name}")
        comparisons[name] = {"gt_replacement_equal": True}
    result.update(
        {
            "schema": SCHEMA,
            "stage": config["stage"],
            "checks": comparisons,
            "backend": backend_record(device),
            "amp": config["training"]["amp"],
            "scope": "runtime correctness only; no training benefit claim",
        }
    )
    if render:
        result["condition"] = condition_response(model, batch)
    if config.get("smoke_accum_equivalence", False):
        mixed = collate_indices(
            dataset, [0, min(1, len(dataset) - 1)], device, seed=9021
        )
        result["accum_equivalence"] = accumulation_equivalence(config, model, mixed)
    write_json(Path(output) / f"smoke-rank-{rank}.json", result)
    return result


def condition_response(model, batch):
    """Fix real F/Q and perturb only S; numerical effect is not quality evidence."""
    import torch
    import torch.nn.functional as functional

    from util.runc_v2 import render_runc_v2, runc_v2_forward

    assert_distinct_references(batch)
    model.eval()
    with torch.no_grad():
        result = runc_v2_forward(
            model, batch["image"], safe_target(batch), render=True, return_features=True
        )
    lower = result["lower_features"].detach()
    reference = result["reference_features"].detach()
    head = {key: value.detach() for key, value in result["head"].items()}
    keys = ("occupancy_logits", "existence_logits", "class_logits")
    structure = {key: head[key].detach().clone().requires_grad_(True) for key in keys}
    generated = render_runc_v2(model, lower, reference, head, structure=structure)
    coefficients = torch.linspace(
        0.1, 1.0, generated["density"].numel(), device=lower.device
    ).reshape_as(generated["density"])
    gradients = torch.autograd.grad(
        (generated["density"] * coefficients).mean(),
        list(structure.values()),
        allow_unused=True,
    )
    gradient_norm = {
        name: float(value.float().norm()) if value is not None else 0.0
        for name, value in zip(keys, gradients)
    }
    if not any(value > 0 for value in gradient_norm.values()):
        raise RuntimeError("generated density has no gradient to predicted S")
    if not all(math.isfinite(value) for value in gradient_norm.values()):
        raise FloatingPointError("nonfinite S condition gradient")
    variants = {}
    baseline = {key: head[key].detach() for key in keys}
    absent = {key: value.clone() for key, value in baseline.items()}
    absent["occupancy_logits"].fill_(-30)
    absent["existence_logits"].fill_(-30)
    absent["class_logits"].zero_()
    variants["no_s"] = absent
    shifted = {key: value.clone() for key, value in baseline.items()}
    shifted["occupancy_logits"] = torch.roll(
        shifted["occupancy_logits"], shifts=3, dims=-1
    )
    variants["shift_x_three_tokens"] = shifted
    stretched = {key: value.clone() for key, value in baseline.items()}
    logits = stretched["occupancy_logits"]
    theta = logits.new_tensor([[0.75, 0, 0], [0, 1, 0]])[None].expand(
        logits.shape[0], -1, -1
    )
    grid = functional.affine_grid(theta, logits.shape, align_corners=False)
    stretched["occupancy_logits"] = functional.grid_sample(
        logits,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    variants["stretch_x"] = stretched
    with torch.no_grad():
        original = render_runc_v2(model, lower, reference, head, structure=baseline)
        responses = {}
        for name, value in variants.items():
            changed = render_runc_v2(model, lower, reference, head, structure=value)
            difference = (original["density"] - changed["density"]).abs()
            responses[name] = {
                "mean_abs_density_delta": float(difference.mean()),
                "max_abs_density_delta": float(difference.max()),
            }
    if responses["no_s"]["max_abs_density_delta"] == 0:
        raise RuntimeError("normal and no-S are identical; conditioning is ineffective")
    return {
        "fixed": "actual F_full, Q and reference features",
        "s_gradient_norm": gradient_norm,
        "responses": responses,
        "interpretation": (
            "effective S path; correct geometric response needs visual/metric review"
        ),
    }


def accumulation_equivalence(config, model, batch):
    """Compare real full-batch and equal microbatch gradients in native FP32.

    Stochastic drop probabilities are temporarily zero so RNG draws do not
    confound the reduction identity. This checks the actual loss and backbone
    path; the ordinary training smoke separately exercises AMP and DDP.
    """
    import torch

    count = batch["image"].shape[0]
    if count < 2:
        raise ValueError("accumulation equivalence needs at least two actual samples")
    stochastic = []
    for module in model.modules():
        for attribute in ("drop_prob", "p"):
            if hasattr(module, attribute) and isinstance(
                getattr(module, attribute), (int, float)
            ):
                if attribute == "p" and not isinstance(module, torch.nn.Dropout):
                    continue
                stochastic.append((module, attribute, getattr(module, attribute)))
                setattr(module, attribute, 0.0)
    render = config["stage"] in ("ib", "ic")
    wrapped = wrap_forward(model)
    parameters = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    try:
        model.zero_grad(set_to_none=True)
        losses = loss_for(call_model(wrapped, batch, render=render), batch, config)
        losses["total"].backward()
        full_loss = float(losses["total"].detach())
        reference = {
            name: p.grad.detach().float().cpu().clone() if p.grad is not None else None
            for name, p in parameters
        }
        del losses
        model.zero_grad(set_to_none=True)
        micro_loss = 0.0
        for index in range(count):
            part = {
                key: value[index : index + 1]
                if (
                    torch.is_tensor(value)
                    and value.ndim
                    and value.shape[0] == count
                    or isinstance(value, (list, tuple))
                    and len(value) == count
                )
                else value
                for key, value in batch.items()
            }
            losses = loss_for(call_model(wrapped, part, render=render), part, config)
            micro_loss += float(losses["total"].detach()) / count
            (losses["total"] / count).backward()
        difference_square = 0.0
        reference_square = 0.0
        max_absolute = 0.0
        for name, parameter in parameters:
            expected = reference[name]
            actual = (
                parameter.grad.detach().float().cpu()
                if parameter.grad is not None
                else None
            )
            if (expected is None) != (actual is None):
                raise RuntimeError(f"accumulation changed gradient membership: {name}")
            if expected is None:
                continue
            difference = (expected - actual).abs()
            difference_square += float(difference.square().sum())
            reference_square += float(expected.square().sum())
            max_absolute = max(max_absolute, float(difference.max()))
        relative = math.sqrt(difference_square / max(reference_square, 1e-30))
        tolerance = float(config.get("accum_relative_tolerance", 1e-4))
        if not math.isfinite(relative) or relative > tolerance:
            raise RuntimeError(
                f"accumulation gradient relative error {relative} > {tolerance}"
            )
        return {
            "full_batch_size": count,
            "micro_batch_size": 1,
            "full_loss": full_loss,
            "mean_micro_loss": micro_loss,
            "gradient_relative_l2": relative,
            "gradient_max_absolute": max_absolute,
            "relative_tolerance": tolerance,
            "precision": "native FP32; stochastic drop disabled only for this identity",
        }
    finally:
        model.zero_grad(set_to_none=True)
        for module, attribute, value in stochastic:
            setattr(module, attribute, value)


def stage2_interface_smoke(config, model, output, device):
    """One diagnostic BF update; no BF long-training entry or quality claim."""
    import torch
    import torch.nn.functional as functional

    from data.runc_v2_dataset import build_runc_v2_refine_dataset
    from util.dual_loss import DualJietiLoss
    from util.runc_v2 import (
        enable_runc_v2_refiner,
        runc_v2_refine_forward,
        set_runc_v2_stage,
    )

    bf_config = copy.deepcopy(config)
    bf_config["stage"] = "refine"
    dataset = build_runc_v2_refine_dataset(bf_config, validation=False)
    assert_dataset_reference_policy(dataset)
    pools = {"dual": [], "bf_only": []}
    for index, row in enumerate(dataset.pairs):
        pools["dual" if row["has_jt"] else "bf_only"].append(index)
    if {name: len(indices) for name, indices in pools.items()} != {
        "dual": 466,
        "bf_only": 352,
    }:
        raise ValueError("stage-II interface requires 466 dual + 352 BF-only")
    if not hasattr(model, "bf_refiner"):
        enable_runc_v2_refiner(model)
    set_runc_v2_stage(model, "refine")
    model.train()
    frozen_before = state_sha256(model, frozen_only=True)
    indices = [pools["dual"][0], pools["bf_only"][0]]
    batch = collate_indices(dataset, indices, device, seed=811)
    optimizer = torch.optim.AdamW(model.bf_refiner.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    amp = config["training"].get("amp", False)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
        outputs = runc_v2_refine_forward(
            model,
            batch["image"],
            safe_target(batch),
            bf_mask=batch["bf_patch_mask"],
        )
        half = batch["image"].shape[-2] // 2
        prediction = outputs["p1"][:, :, half:]
        target = batch["target"][:, :, half:]
        per_pixel = functional.smooth_l1_loss(
            prediction,
            target,
            reduction="none",
        )
        image_loss = per_pixel.mean()
        area = batch["bf_loss_mask"]
        local_loss = (
            (per_pixel * area).sum((1, 2, 3)) / area.sum((1, 2, 3)).clamp_min(1e-6)
        ).mean()
        j_value, j_parts, _ = DualJietiLoss().to(device)(
            outputs["p1"],
            batch["target"],
            batch["voro"],
            batch["valid_parts"],
            batch["has_jt"],
        )
        total = image_loss + local_loss + j_value
    if outputs["p0"].requires_grad or outputs["head"]["occupancy_logits"].requires_grad:
        raise RuntimeError("stage-II must stop stage-I P0/Head gradients")
    if float(j_parts["per_sample"][1]) != 0:
        raise RuntimeError("BF-only sample received J(P1) supervision")
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("nonfinite stage-II interface loss")
    total.backward()
    gradient = grouped_gradient_norms(model)
    if not any(value["norm"] > 0 for value in gradient.values()):
        raise RuntimeError("stage-II Refiner received no effective gradient")
    if any(
        p.requires_grad
        for name, p in model.named_parameters()
        if not name.startswith("bf_refiner.")
    ):
        raise RuntimeError("stage-II optimizer exposed a stage-I parameter")
    optimizer.step()
    frozen_after = state_sha256(model, frozen_only=True)
    if frozen_before != frozen_after:
        raise RuntimeError("stage-I bytes changed during BF interface smoke")
    model.eval()
    altered = dict(batch)
    altered["target"] = batch["target"].clone()
    altered["target"][:, :, half:] = torch.randn_like(target)
    altered.pop("input_target", None)
    with torch.no_grad():
        normal = runc_v2_refine_forward(
            model,
            batch["image"],
            safe_target(batch),
            bf_mask=batch["bf_patch_mask"],
        )
        changed = runc_v2_refine_forward(
            model,
            altered["image"],
            safe_target(altered),
            bf_mask=batch["bf_patch_mask"],
        )
        automatic = runc_v2_refine_forward(
            model,
            batch["image"],
            safe_target(batch),
            inference=True,
        )
    if not torch.equal(normal["p1"], changed["p1"]):
        raise RuntimeError("BF prediction changed when hidden GT was replaced")
    path = Path(output) / "refiner-interface-only.pth"
    state = {
        "schema": "runc-v2-refiner-interface-only-v1",
        "bf_refiner": model.bf_refiner.state_dict(),
        "runc_v2_refiner_config": model.runc_v2_refiner_config,
        "stage_one_checkpoint_sha256": sha256_file(config["checkpoint"]),
        "purpose": "one-update interface smoke only; not a trained Refiner",
    }
    with path.open("xb") as stream:
        torch.save(state, stream)
    restored = torch.load(path, map_location="cpu", weights_only=False)
    model.bf_refiner.load_state_dict(restored["bf_refiner"], strict=True)
    for name, parameter in model.bf_refiner.state_dict().items():
        expected = restored["bf_refiner"][name]
        if parameter.dtype != expected.dtype or not torch.equal(
            parameter.cpu(), expected
        ):
            raise RuntimeError("Refiner strict restoration changed bytes/dtype")
    report = {
        "schema": SCHEMA,
        "scope": "stage-II one-update interface smoke only",
        "stage_one_accepted": False,
        "formal_bf_training_started": False,
        "bf_pool_counts": {name: len(values) for name, values in pools.items()},
        "bf_annotation_layers": 11,
        "aggregate_bf_types": False,
        "source_stage_one_checkpoint_sha256": sha256_file(config["checkpoint"]),
        "metadata": source_record(config),
        "backend": backend_record(device),
        "amp": amp,
        "loss": float(total.detach()),
        "gradient": gradient,
        "frozen_before": frozen_before,
        "frozen_after": frozen_after,
        "stage_one_exact_unchanged": True,
        "hidden_gt_replacement_equal": True,
        "bf_only_j_zero": True,
        "strict_refiner_restore": True,
        "shape": {key: list(normal[key].shape) for key in ("p0", "p1", "masked_p0")},
        "dtype": {key: str(normal[key].dtype) for key in ("p0", "p1", "masked_p0")},
        "automatic_mask_shape": list(automatic["mask"].shape),
        "mask_distribution_limitation": (
            "training eleven BF annotations; inference automatic block B; "
            "distribution mismatch intentionally remains"
        ),
        "checkpoint_sha256": sha256_file(path),
    }
    write_json(Path(output) / "stage2_interface_smoke.json", report)
    return report
