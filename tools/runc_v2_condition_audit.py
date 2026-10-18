"""Audit predicted-S control with fixed real F/Q/reference after stage I-B.

This is read-only model evaluation. Occupancy perturbations operate on sigmoid
probabilities with zero padding, then clamp/logit. Existence and roles stay
fixed. GT is used only in the displayed comparison panels.
"""

import argparse
import copy
import csv
import hashlib
import html
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import runc_v2_runtime as runtime  # noqa: E402

SCHEMA = "runc-v2-fixed-feature-condition-audit-v1"
REQUIRED_CHARACTERS = ("無", "容", "惟", "雲", "崇")
PROBABILITY_EPS = 1e-6
MOMENT_EPS = 1e-8
GAIN_EPS = 1e-6
TRAINING_START_FIELDS = (
    "calibration_record", "stage_gate_record", "stage_start_checkpoint",
    "stage_start_checkpoint_sha256", "resume_optimizer", "_formal_training",
)


def evaluation_config(config):
    """Drop stage-start bindings when evaluating a later saved snapshot."""
    result = copy.deepcopy(config)
    removed = [name for name in TRAINING_START_FIELDS if name in result]
    for name in removed:
        del result[name]
    result["backend"] = {
        "matmul_tf32": False, "cudnn_tf32": False, "cudnn_benchmark": False,
    }
    result["training"]["amp"] = False
    return runtime.validate_config(result, command="evaluate"), removed


@dataclass(frozen=True)
class Variant:
    """Use grid-pixel shifts and scale about each predicted slot center."""

    name: str
    kind: str = "baseline"
    axis: str = ""
    scale_x: float = 1.0
    scale_y: float = 1.0
    shift_x: float = 0.0
    shift_y: float = 0.0


def variants():
    """A fixed, symmetric set of scale/translation probes plus no-S."""
    result = [Variant("normal"), Variant("noS", "remove")]
    for axis in ("x", "y"):
        for scale in (1.25, 0.8):
            result.append(
                Variant(
                    f"scale_{axis}_{scale:g}",
                    "scale",
                    axis,
                    **{f"scale_{axis}": scale},
                )
            )
        for shift in (2.0, -2.0):
            result.append(
                Variant(
                    f"shift_{axis}_{shift:+g}",
                    "shift",
                    axis,
                    **{f"shift_{axis}": shift},
                )
            )
    return result


def spatial_moments(probabilities, eps=MOMENT_EPS):
    """Differentiable mass/geometry on normalized pixel centers in [0, 1].

    Leading dimensions are retained. Mass is a pixel sum; mass_fraction and
    normalized centroid/sigma allow comparing the 28 and 112 grids. Empty
    Empty fields are invalid; finite epsilon values are placeholders.
    """
    if probabilities.ndim < 2:
        raise ValueError("Spatial moments need at least two image dimensions")
    p = probabilities.float()
    height, width = p.shape[-2:]
    x = (torch.arange(width, device=p.device, dtype=p.dtype) + 0.5) / width
    y = (torch.arange(height, device=p.device, dtype=p.dtype) + 0.5) / height
    mass = p.sum((-2, -1))
    denominator = mass.clamp_min(eps)
    cx = (p * x).sum((-2, -1)) / denominator
    cy = (p * y[:, None]).sum((-2, -1)) / denominator
    variance_x = (p * (x - cx[..., None, None]).square()).sum(
        (-2, -1)
    ) / denominator
    variance_y = (p * (y[:, None] - cy[..., None, None]).square()).sum(
        (-2, -1)
    ) / denominator
    sigma_x = (variance_x.clamp_min(0) + eps).sqrt()
    sigma_y = (variance_y.clamp_min(0) + eps).sqrt()
    return {
        "mass": mass,
        "mass_fraction": mass / (height * width),
        "centroid_x": cx,
        "centroid_y": cy,
        "sigma_x": sigma_x,
        "sigma_y": sigma_y,
        "logsigma_x": sigma_x.log(),
        "logsigma_y": sigma_y.log(),
        "valid": mass > eps,
    }


def warp_probabilities(
    probabilities, scale_x=1.0, scale_y=1.0, shift_x=0.0, shift_y=0.0
):
    """Warp each predicted slot about its own centroid; outside is zero.

    ``shift_x/y`` are 28-grid pixels in production, positive right/down.
    Input and output are probabilities, never logits. Inverse sampling uses
    align_corners=False with the exact pixel-center coordinate conversion.
    """
    if probabilities.ndim != 4:
        raise ValueError("Expected B x slots x H x W probabilities")
    if not torch.isfinite(probabilities).all():
        raise ValueError("Occupancy probabilities must be finite")
    if torch.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Warp input must be probabilities in [0, 1]")
    if (
        not all(
            math.isfinite(value)
            for value in (scale_x, scale_y, shift_x, shift_y)
        )
        or scale_x <= 0
        or scale_y <= 0
    ):
        raise ValueError("Scales must be finite and positive; shifts finite")
    p = probabilities.float()
    batch, slots, height, width = p.shape
    geometry = spatial_moments(p)
    cx = geometry["centroid_x"] * width - 0.5
    cy = geometry["centroid_y"] * height - 0.5
    yy, xx = torch.meshgrid(
        torch.arange(height, device=p.device, dtype=p.dtype),
        torch.arange(width, device=p.device, dtype=p.dtype),
        indexing="ij",
    )
    source_x = (
        cx[..., None, None] + (xx - shift_x - cx[..., None, None]) / scale_x
    )
    source_y = (
        cy[..., None, None] + (yy - shift_y - cy[..., None, None]) / scale_y
    )
    grid = torch.stack(
        (
            2 * (source_x + 0.5) / width - 1,
            2 * (source_y + 0.5) / height - 1,
        ),
        dim=-1,
    ).reshape(batch * slots, height, width, 2)
    return F.grid_sample(
        p.reshape(batch * slots, 1, height, width),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).reshape_as(p)


def perturb_structure(head, variant, eps=PROBABILITY_EPS):
    """Return only S; Q/F cannot be replaced through this interface."""
    p = head["occupancy_logits"].float().sigmoid()
    if variant.kind == "remove":
        changed = torch.zeros_like(p)
    else:
        changed = warp_probabilities(
            p,
            variant.scale_x,
            variant.scale_y,
            variant.shift_x,
            variant.shift_y,
        )
    # Zero padding in logit space would create sigmoid(0)=0.5 fake occupancy.
    logits = torch.logit(changed.clamp(eps, 1 - eps))
    if variant.kind == "baseline":
        logits = head["occupancy_logits"].detach().clone().float()
    return {
        "occupancy_logits": logits,
        "existence_logits": head["existence_logits"],
        "class_logits": head["class_logits"],
    }


def structure_gates(structure):
    return (
        structure["occupancy_logits"].sigmoid()
        * structure["existence_logits"].sigmoid()[:, :, None, None]
    )


def scalar_geometry(field):
    """Serialize one field; normalized units are explicit in report policy."""
    result = spatial_moments(field)
    if result["mass"].numel() != 1:
        raise ValueError("Scalar geometry requires one spatial field")
    return {
        name: bool(value.item()) if name == "valid" else float(value.item())
        for name, value in result.items()
    }


def response_gain(
    s_base, s_changed, d_base, d_changed, variant, denominator_eps=GAIN_EPS
):
    """Output/input response in common normalized units, not raw grid pixels.

    Scale gain is d(log sigma_D)/d(log sigma_S); shift gain is normalized
    centroid_D delta / normalized centroid_S delta. Near-zero S changes do
    not get an invented finite gain. No-S and baseline have no direction test.
    """
    if variant.kind == "scale":
        metric = f"logsigma_{variant.axis}"
        intended = math.log(getattr(variant, f"scale_{variant.axis}"))
    elif variant.kind == "shift":
        metric = f"centroid_{variant.axis}"
        intended = getattr(variant, f"shift_{variant.axis}")
    else:
        return {
            "primary_metric": None,
            "input_delta": None,
            "output_delta": None,
            "gain": None,
            "direction_matches_input": None,
            "input_matches_requested_direction": None,
        }
    input_delta = s_changed[metric] - s_base[metric]
    output_delta = d_changed[metric] - d_base[metric]
    usable = abs(input_delta) > denominator_eps
    return {
        "primary_metric": metric,
        "input_delta": input_delta,
        "output_delta": output_delta,
        "gain": output_delta / input_delta if usable else None,
        "direction_matches_input": (
            output_delta * input_delta > 0 if usable else None
        ),
        "input_matches_requested_direction": (
            input_delta * intended > 0 if usable else None
        ),
    }


def summarize_responses(rows):
    """Aggregate defined gains without claiming control acceptance."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["variant"]].append(row)
    result = {}
    for name, values in groups.items():
        gains = [row["gain"] for row in values if row["gain"] is not None]
        directions = [
            row["direction_matches_input"]
            for row in values
            if row["direction_matches_input"] is not None
        ]
        requested = [
            row["input_matches_requested_direction"]
            for row in values
            if row["input_matches_requested_direction"] is not None
        ]
        result[name] = {
            "samples": len(values),
            "defined_gain_count": len(gains),
            "median_gain": statistics.median(gains) if gains else None,
            "min_gain": min(gains) if gains else None,
            "max_gain": max(gains) if gains else None,
            "direction_matches_input": sum(directions),
            "direction_denominator": len(directions),
            "input_matches_requested_direction": sum(requested),
            "requested_direction_denominator": len(requested),
            "median_density_l1_delta": statistics.median(
                row["density_l1_delta"] for row in values
            ),
            "exactly_equal_output_count": sum(
                row["density_max_delta"] == 0 for row in values
            ),
        }
    return result


def select_fixed_samples(dataset, minimum=12):
    """Deterministically cover required glyphs and all nine JT authors."""
    if minimum < 12 or dataset.is_train:
        raise ValueError("Use at least twelve fixed validation JT queries")
    eligible = []
    seen = set()
    for index, upper in enumerate(dataset.pairs):
        pair = dataset.base_dataset._fixed_pairs[upper["target_path"]]
        if not pair["has_jt"]:
            continue
        identity = dataset._record_identities[pair["target_path"]][
            "original_identity"
        ]
        if identity in seen:
            continue
        seen.add(identity)
        eligible.append(
            {
                "index": index,
                "author": pair["calligrapher"],
                "character": pair["character"],
                "target_path": pair["target_path"],
                "reference_path": upper["target_path"],
            }
        )
    authors = sorted({row["author"] for row in eligible})
    if len(authors) != 9:
        raise ValueError(
            f"Expected nine fixed-val JT authors, found {authors}"
        )
    missing = set(REQUIRED_CHARACTERS) - {row["character"] for row in eligible}
    if missing:
        raise ValueError(
            f"Required fixed JT glyphs unavailable: {sorted(missing)}"
        )
    selected, chosen, covered = [], set(), set()

    def add(row):
        if row["index"] not in chosen:
            selected.append(row)
            chosen.add(row["index"])
            covered.add(row["author"])

    for character in REQUIRED_CHARACTERS:
        candidates = [row for row in eligible if row["character"] == character]
        add(
            min(
                candidates,
                key=lambda row: (
                    row["author"] in covered,
                    row["index"],
                ),
            )
        )
    for author in authors:
        if author not in covered:
            add(next(row for row in eligible if row["author"] == author))
    for row in eligible:
        if len(selected) >= minimum:
            break
        add(row)
    if len(selected) < minimum:
        raise ValueError("Insufficient distinct fixed JT targets")
    return selected


def gradient_probe(model, lower, reference, head):
    """Measure three true renderer derivatives with F/Q/reference detached."""
    from util.runc_v2 import render_runc_v2

    logits = head["occupancy_logits"].detach().clone().requires_grad_(True)
    structure = {
        name: head[name]
        for name in (
            "occupancy_logits",
            "existence_logits",
            "class_logits",
        )
    }
    structure["occupancy_logits"] = logits
    output = render_runc_v2(model, lower, reference, head, structure)
    geometry = spatial_moments(output["density"])
    scalars = {
        "density_mean": output["density"].mean(),
        "density_logsigma_x": geometry["logsigma_x"].mean(),
        "density_logsigma_y": geometry["logsigma_y"].mean(),
    }
    result = {}
    for index, (name, value) in enumerate(scalars.items()):
        gradient = torch.autograd.grad(
            value,
            logits,
            retain_graph=index < len(scalars) - 1,
        )[0]
        finite = bool(torch.isfinite(gradient).all())
        result[name] = {
            "finite": finite,
            "nonzero_elements": int(gradient.count_nonzero()),
            "elements": gradient.numel(),
            "l2_norm": float(gradient.norm()) if finite else None,
            "max_abs": float(gradient.abs().max()) if finite else None,
            "scalar": float(value.detach()) if torch.isfinite(value) else None,
        }
    return result


def audit_sample(model, batch):
    """Cache real deployment features once; only predicted S varies."""
    from util.runc_v2 import (
        render_runc_v2,
        runc_v2_forward,
        state_tensor_digest,
    )

    with torch.no_grad():
        original = runc_v2_forward(
            model,
            batch["image"],
            runtime.safe_target(batch),
            render=True,
            return_features=True,
        )
    head = {
        name: value.detach().clone().float()
        for name, value in original["head"].items()
    }
    lower = original["lower_features"].detach().clone().float()
    reference = original["reference_features"].detach().clone().float()
    cache = {
        "lower_F": lower,
        "reference_F": reference,
        **{f"head_{name}": value for name, value in head.items()},
    }
    before = state_tensor_digest(cache)
    baseline_structure = perturb_structure(head, variants()[0])
    with torch.no_grad():
        cached_baseline = render_runc_v2(
            model, lower, reference, head, baseline_structure,
        )
    normal = cached_baseline["density"].detach().float()
    replay_delta = float((normal - original["density"].float()).abs().max())
    baseline_gates = structure_gates(baseline_structure)
    s_base = scalar_geometry(baseline_gates.mean(1))
    d_base = scalar_geometry(normal)
    rows, visuals = [], []
    mean = normal.new_tensor(runtime.MEAN)[None, :, None, None]
    std = normal.new_tensor(runtime.STD)[None, :, None, None]
    for variant in variants():
        with torch.no_grad():
            structure = perturb_structure(head, variant)
            output = render_runc_v2(model, lower, reference, head, structure)
            if not torch.isfinite(output["density"]).all():
                raise RuntimeError(f"Nonfinite density for {variant.name}")
            if not torch.equal(
                structure["class_logits"], head["class_logits"]
            ):
                raise RuntimeError("Role predictions changed during S warp")
            if not torch.equal(
                structure["existence_logits"], head["existence_logits"]
            ):
                raise RuntimeError(
                    "Existence predictions changed during S warp"
                )
            gates = structure_gates(structure)
            s_geometry = scalar_geometry(gates.mean(1))
            d_geometry = scalar_geometry(output["density"])
            delta = (output["density"] - normal).abs()
            slots = spatial_moments(gates)
            row = {
                "variant": variant.name,
                "transform": asdict(variant),
                "s_geometry": s_geometry,
                "density_geometry": d_geometry,
                "s_slot_geometry": {
                    name: value[0].cpu().tolist()
                    for name, value in slots.items()
                },
                "density_l1_delta": float(delta.mean()),
                "density_max_delta": float(delta.max()),
                "density_relative_mass_change": (
                    d_geometry["mass_fraction"]
                    / max(
                        d_base["mass_fraction"],
                        MOMENT_EPS,
                    )
                    - 1
                ),
                "s_relative_mass_change": (
                    s_geometry["mass_fraction"]
                    / max(
                        s_base["mass_fraction"],
                        MOMENT_EPS,
                    )
                    - 1
                ),
                **response_gain(
                    s_base, s_geometry, d_base, d_geometry, variant
                ),
            }
            rows.append(row)
            visuals.append(
                {
                    "variant": variant.name,
                    "gates": gates.mean(1)[0].cpu(),
                    "density": output["density"][0, 0].cpu(),
                    "p0_rgb": (output["p0_lower"] * std + mean)[0].cpu(),
                    "delta": delta[0, 0].cpu(),
                }
            )
    gradients = gradient_probe(model, lower, reference, head)
    after = state_tensor_digest(cache)
    return {
        "rows": rows,
        "gradients": gradients,
        "feature_cache_sha256_before": before,
        "feature_cache_sha256_after": after,
        "feature_cache_unchanged": before == after,
        "cached_replay_density_max_delta": replay_delta,
    }, visuals


def save_visual(path, batch, visuals):
    """Show control, density, P0 and amplified change side by side."""
    from PIL import Image, ImageDraw

    size, label = 224, 24

    def gray(field, invert=False):
        pixels = (1 - field if invert else field).clamp(0, 1)
        return Image.fromarray((pixels.numpy() * 255).round().astype("uint8"))

    def rgb(tensor):
        return Image.fromarray(
            (
                tensor.clamp(0, 1)
                .permute(
                    1,
                    2,
                    0,
                )
                .numpy()
                * 255
            )
            .round()
            .astype("uint8")
        )

    def denormalize(tensor):
        mean = tensor.new_tensor(runtime.MEAN)[:, None, None]
        std = tensor.new_tensor(runtime.STD)[:, None, None]
        return tensor * std + mean

    target = batch["target"][0].detach().cpu()
    source = batch["image"][0].detach().cpu()
    half = target.shape[-2] // 2
    canvas = Image.new(
        "RGB", (4 * size, (len(visuals) + 1) * (size + label)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    top = [
        ("GT query: display only", rgb(denormalize(target[:, half:]))),
        ("Visible reference", rgb(denormalize(target[:, :half]))),
        ("Computer source", rgb(denormalize(source[:, half:]))),
    ]
    for column, (title, panel) in enumerate(top):
        canvas.paste(panel.resize((size, size)), (column * size, label))
        draw.text((column * size + 4, 4), title, fill="black")
    for index, item in enumerate(visuals, start=1):
        panels = [
            ("S gates (1=white)", gray(item["gates"])),
            ("D0 (nearest)", gray(item["density"], invert=True)),
            ("P0 (RGB)", rgb(item["p0_rgb"])),
            ("abs delta D0 x5", gray((item["delta"] * 5).clamp(0, 1))),
        ]
        top_y = index * (size + label)
        for column, (title, panel) in enumerate(panels):
            resample = (
                Image.Resampling.NEAREST
                if column < 2
                else (Image.Resampling.BILINEAR)
            )
            canvas.paste(
                panel.resize((size, size), resample),
                (column * size, top_y + label),
            )
            draw.text(
                (column * size + 4, top_y + 4),
                f"{item['variant']} | {title}",
                fill="black",
            )
    canvas.save(path, format="PNG")


def write_csv(path, rows):
    fields = sorted({name for row in rows for name in row})
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_reports(output, report):
    runtime.write_json(output / "condition_audit.json", report)
    rows, gradients = [], []
    for sample in report["samples"]:
        identity = {
            name: sample[name]
            for name in (
                "index",
                "author",
                "character",
                "target_path",
                "reference_path",
            )
        }
        for row in sample["rows"]:
            flat = {
                **identity,
                **{
                    key: value
                    for key, value in row.items()
                    if not isinstance(value, dict)
                },
            }
            for prefix, key in (
                ("s", "s_geometry"),
                ("density", "density_geometry"),
            ):
                flat.update(
                    {
                        f"{prefix}_{name}": value
                        for name, value in row[key].items()
                    }
                )
            flat.update(
                {
                    f"transform_{name}": value
                    for name, value in row["transform"].items()
                }
            )
            rows.append(flat)
        for name, gradient in sample["gradients"].items():
            gradients.append({**identity, "scalar_name": name, **gradient})
    write_csv(output / "condition_metrics.csv", rows)
    write_csv(output / "gradient_metrics.csv", gradients)
    lines = [
        "# Run C v2 固定特征条件响应审计",
        "",
        f"状态：`{report['status']}`。只测模型数值作用，不替代结构人工验收。",
        "",
        "真实部署前向只运行一次并缓存 F/Q/参考特征；后续仅修改预测 S 的 occupancy。",
        "存在性和角色固定，GT 只进入并列图。输出始终为模型生成 density/P0。",
        "",
        "概率域 bilinear warp、零 padding，之后 clamp/logit；尺度围绕每槽预测质心；",
        "平移单位为 28 网格的 pixel，正方向为右/下。noS 为 epsilon 概率，保留存在性。",
        "几何采用 [0,1] pixel-center 坐标；gain 使用同单位质心差或 logσ 差。",
        "",
        "| 变换 | gain 样本数 | gain 中位数 | 方向一致 | density L1 改变量 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in report["summary"].items():
        gain = (
            "undefined"
            if row["median_gain"] is None
            else (f"{row['median_gain']:.6g}")
        )
        lines.append(
            f"| {name} | {row['defined_gain_count']} | {gain} | "
            f"{row['direction_matches_input']}/"
            f"{row['direction_denominator']} | "
            f"{row['median_density_l1_delta']:.6g} |"
        )
    lines.extend(
        [
            "",
            "`condition_audit.json` 记录每个 S/密度的质量、质心、横纵尺度、每槽统计、",
            "梯度和真实特征缓存 SHA；CSV 可逐字比较。",
            "边界裁切会改变实际输入尺度或质量，",
            "因此还记录请求方向与实际 S 方向是否一致，不用请求倍率冒充实际变换。",
            "",
            "数值非零、有限梯度仅证明通路有效；不能证明几何变化正确、生成质量改善或泛化。",
            "固定 Q 只隔离显式 S 通路，且这些变换可能落在训练分布之外。",
            "",
            f"完整 state SHA 不变：`{report['state_unchanged']}`。",
            "技术失败：`"
            f"{json.dumps(report['technical_failures'], ensure_ascii=False)}`"
            "。",
            "",
            "并列图见 [HTML](visuals/index.html)。",
            "",
        ]
    )
    (output / "condition_response.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    cards = []
    for sample in report["samples"]:
        title = html.escape(f"{sample['author']} / {sample['character']}")
        image = html.escape(sample["visual_file"])
        cards.append(f"<section><h2>{title}</h2><img src='{image}'></section>")
    page = (
        "<!doctype html><meta charset='utf-8'><title>Run C v2 S audit</title>"
        "<style>body{font-family:sans-serif;margin:20px}img{width:100%;"
        "max-width:1200px}section{margin-bottom:40px}</style>"
        "<h1>固定真实 F/Q，预测 S 条件响应</h1>"
        "<p>GT 只显示于对照。数值变化不自动代表正确的结构响应。</p>"
        + "".join(cards)
    )
    (output / "visuals" / "index.html").write_text(page, encoding="utf-8")


def run(config, output, device, minimum=12):
    """Evaluate an IB/IC checkpoint without any optimizer update."""
    from util.runc_v2 import set_runc_v2_stage, state_tensor_digest

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "visuals").mkdir()
    config, removed_bindings = evaluation_config(config)
    runtime.configure_backend(config)
    torch.set_float32_matmul_precision("highest")
    model, saved, loading = runtime.load_model(config, device)
    if saved.get("stage") not in ("ib", "ic"):
        raise ValueError(
            "Condition acceptance audit requires an IB/IC snapshot"
        )
    if any(
        value.is_floating_point() and value.dtype != torch.float32
        for value in model.state_dict().values()
    ):
        raise ValueError(
            "Condition audit requires native FP32 saved model state"
        )
    if (model.runc_v2_config["grid_size"] != 28
            or model.runc_v2_config["density_size"] != 112):
        raise ValueError("Condition audit requires the real 28/112 grids")
    model.eval()
    set_runc_v2_stage(model, "infer")
    before = state_tensor_digest(model.state_dict())
    dataset = runtime.build_dataset(config, validation=True)
    selection = select_fixed_samples(dataset, minimum)
    runtime.write_json(output / "selected_samples.json", selection)
    runtime.write_json(output / "audit_config.json", config)
    samples, failures = [], []
    for selected in selection:
        batch = runtime.collate_indices(
            dataset,
            [selected["index"]],
            device,
            seed=0,
        )
        try:
            with torch.autocast(device.type, enabled=False):
                sample, visuals = audit_sample(model, batch)
            sample.update(selected)
            sample["query_original_identity"] = batch[
                "query_original_identity"
            ][0]
            sample["reference_original_identity"] = batch[
                "reference_original_identity"
            ][0]
            sample["visual_file"] = f"sample_{selected['index']:03d}.png"
            save_visual(
                output / "visuals" / sample["visual_file"], batch, visuals
            )
            if not sample["feature_cache_unchanged"]:
                failures.append(
                    f"index {selected['index']}: feature cache changed"
                )
            for name, value in sample["gradients"].items():
                if not value["finite"] or value["nonzero_elements"] == 0:
                    failures.append(
                        f"index {selected['index']}: invalid {name} gradient"
                    )
            no_s = next(
                row for row in sample["rows"] if row["variant"] == "noS"
            )
            if no_s["density_max_delta"] == 0:
                failures.append(
                    f"index {selected['index']}: noS output exactly equal"
                )
            samples.append(sample)
        except Exception as error:
            failures.append(
                f"index {selected['index']}: {type(error).__name__}: {error}"
            )
            break
    after = state_tensor_digest(model.state_dict())
    if before != after:
        failures.append(
            "Full model state changed during read-only condition audit"
        )
    tracked = subprocess.check_output(
        ["git", "diff", "--name-only", "HEAD"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    report = {
        "schema": SCHEMA,
        "status": "technical_failure"
        if failures
        else ("measured_pending_manual_geometric_review"),
        "metadata": {
            "source_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=ROOT,
                text=True,
            ).strip(),
            "tracked_dirty_sha256": {
                name: runtime.sha256_file(ROOT / name)
                for name in tracked
                if (ROOT / name).is_file()
            },
            "script_sha256": runtime.sha256_file(__file__),
            "initial_checkpoint_sha256": config["initial_checkpoint_sha256"],
            "current_checkpoint_sha256": runtime.sha256_file(
                config["checkpoint"]
            ),
            "checkpoint_source": saved.get("runc_v2_metadata", {}),
            "training_start_bindings_removed": removed_bindings,
            "checkpoint_stage": saved.get("stage"),
            "loading": loading,
            "reference_policy": config["data"]["reference_policy"],
            "seed": config["training"]["seed"],
            "backend": runtime.backend_record(device),
            "precision": "native FP32; autocast off; TF32 off; eval mode",
            "selection_sha256": hashlib.sha256(
                json.dumps(
                    selection,
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
        },
        "policy": {
            "query_grid": [28, 28],
            "density_grid": [112, 112],
            "warp_domain": "sigmoid occupancy probability; zero padding",
            "scale_center": "each predicted slot's own soft centroid",
            "shift_units": "28-grid pixels; positive right/down",
            "moment_units": "normalized pixel centers in [0,1]",
            "s_field": "mean of four existence-weighted occupancy gates",
            "roles_and_existence": "fixed real predictions",
            "feature_cache": "real F/Q/reference; detached and SHA verified",
            "noS": "occupancy epsilon probability; existence/classes fixed",
            "probability_eps": PROBABILITY_EPS,
            "moment_eps": MOMENT_EPS,
            "gain_denominator_eps": GAIN_EPS,
            "gt_usage": "fixed JT cohort selection and comparison image only",
            "acceptance": "numerical effect does not imply correct geometry",
        },
        "state_sha256_before": before,
        "state_sha256_after": after,
        "state_unchanged": before == after,
        "technical_failures": failures,
        "samples": samples,
        "summary": summarize_responses(
            [row for sample in samples for row in sample["rows"]]
        ),
    }
    write_reports(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--device", default=("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    parser.add_argument("--minimum-samples", type=int, default=12)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.checkpoint:
        config["checkpoint"] = args.checkpoint
    report = run(
        config, args.output, torch.device(args.device), args.minimum_samples
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": args.output,
                "samples": len(report["samples"]),
                "technical_failures": report["technical_failures"],
            },
            ensure_ascii=False,
        )
    )
    return 2 if report["technical_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
