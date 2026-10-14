"""Small, reproducible image-only CalliPhase perception experiment."""

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter
from torch import nn
from torchvision.models import vgg19

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from util.calli_labels import LabelRenderer, class_names, label_loss  # noqa: E402


def save_json(path, value):
    """Write an inspectable UTF-8 result."""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2))


def image_tensor(path):
    """Match label centering, then resize the complete image to 224."""
    im = Image.open(path).convert("RGB")
    side = max(im.size)
    canvas = Image.new("RGB", (side, side), "white")
    canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    im = canvas.resize((224, 224), Image.Resampling.BICUBIC)
    return torch.from_numpy(np.array(im)).permute(2, 0, 1).float() / 255


class Extractor(nn.Module):
    """Frozen ImageNet VGG19; preserve spatial correspondence."""

    def __init__(self, weights):
        super().__init__()
        net = vgg19(weights=None)
        net.load_state_dict(torch.load(weights, map_location="cpu"))
        self.features = net.features[:23]
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        )
        self.requires_grad_(False)

    def forward(self, x):
        x = (x - self.mean) / self.std
        levels = []
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in (17, 22):
                levels.append(F.adaptive_avg_pool2d(x, 28))
        return torch.cat(levels, 1)


class PerceptionHead(nn.Module):
    """Small shared spatial adapter with task-specific output channels."""

    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(768, 64, 1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=2, dilation=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.out = nn.Conv2d(64, 70, 1)
        nn.init.constant_(self.out.bias, -2)

    def forward(self, x):
        return self.out(self.body(x))


def make_manifest(root, output):
    """Group by character and decoded-pixel identity before splitting."""
    records = []
    seen_paths = set()
    for split in ("train", "val"):
        for path in sorted((root / f"{split}_json_new").glob("*.json")):
            for row in json.loads(path.read_text()):
                if row["target_path"] in seen_paths:
                    continue
                seen_paths.add(row["target_path"])
                row = dict(row)
                row["character"] = Path(row["image_path"]).stem
                row["author"] = row["font_folder"][:-2]
                im = Image.open(root / row["target_path"]).convert("RGB")
                row["pixel_sha256"] = hashlib.sha256(
                    str(im.size).encode() + im.tobytes()
                ).hexdigest()
                row["original_split"] = split
                records.append(row)
    chars = sorted({r["character"] for r in records})
    parent = {c: c for c in chars}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c

    hashes = {}
    for row in records:
        c = row["character"]
        key = row["pixel_sha256"]
        if key in hashes:
            parent[find(c)] = find(hashes[key])
        hashes[key] = c
    groups = sorted({find(c) for c in chars})
    random.Random(20261005).shuffle(groups)
    n = len(groups)
    split_of = {
        g: "train" if i < int(0.7 * n) else "dev" if i < int(0.85 * n) else "test"
        for i, g in enumerate(groups)
    }
    for row in records:
        row["split"] = split_of[find(row["character"])]
    for key in ("character", "pixel_sha256", "target_path"):
        sets = [
            {r[key] for r in records if r["split"] == s}
            for s in ("train", "dev", "test")
        ]
        assert all(not (sets[i] & sets[j]) for i in range(3) for j in range(i))
    audit = {
        "n": len(records),
        "characters": len(chars),
        "groups": n,
        "counts": dict(Counter((r["split"] + "_" + r["type"][-2:]) for r in records)),
        "overlap_character_pixel_path": 0,
        "limitation": "Exact pixels and character grouped; no exhaustive near-duplicate audit.",
    }
    save_json(output / "manifest.json", records)
    save_json(output / "audit.json", audit)
    print("AUDIT", audit, flush=True)
    return records


def build_cache(records, root, extractor, output, device):
    renderer = LabelRenderer(str(root))
    images, targets, kinds = [], [], []
    for row in records:
        images.append(image_tensor(root / row["target_path"]))
        y, kind = renderer.render(row)
        targets.append(torch.from_numpy(y))
        kinds.append(kind)
    images = torch.stack(images)
    targets = torch.stack(targets)
    kinds = torch.tensor(kinds)
    feats = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for start in range(0, len(images), 4):
            feats.append(extractor(images[start : start + 4].to(device)).half().cpu())
            if start % 200 == 0:
                print("CACHE", start, len(images), flush=True)
    features = torch.cat(feats)
    train = torch.tensor([i for i, r in enumerate(records) if r["split"] == "train"])
    rms = torch.zeros(768)
    for ids in train.split(16):
        rms += features[ids].float().square().mean((2, 3)).sum(0)
    rms = (rms / len(train)).sqrt().clamp_min(0.01)[None, :, None, None]
    features.div_(rms.half())
    cache = dict(
        images=images, targets=targets, kinds=kinds, features=features, rms=rms
    )
    torch.save(cache, output / "cache.pt")
    return cache


def scores(prob, target, kinds):
    """Per-image scores; absent-class predictions affect all-channel IoU."""
    result = {}
    for kind, name, sl in ((0, "BF", slice(0, 50)), (1, "JT", slice(50, 70))):
        keep = kinds == kind
        p, y = prob[keep, sl], target[keep, sl]
        present = y.sum((2, 3)) > 0
        inter = (p * y).sum((2, 3))
        dice = 2 * inter / (p.sum((2, 3)) + y.sum((2, 3))).clamp_min(1e-8)
        sample = (dice * present).sum(1) / present.sum(1).clamp_min(1)
        binary_p, binary_y = p > 0.5, y > 0.1
        binary_dice = (
            2
            * (binary_p & binary_y).sum((2, 3))
            / (binary_p.sum((2, 3)) + binary_y.sum((2, 3))).clamp_min(1)
        )
        sample_binary = (binary_dice * present).sum(1) / present.sum(1).clamp_min(1)
        union = (p + y - p * y).sum((1, 2, 3))
        sample_iou = inter.sum(1) / union.clamp_min(1e-8)
        result[name] = {
            "n": int(keep.sum()),
            "soft_dice": sample.mean().item(),
            "all_channel_soft_iou": sample_iou.mean().item(),
            "binary_dice_at_0.5": sample_binary.mean().item(),
            "per_image_soft_dice": sample.tolist(),
        }
    return result


def compact(result):
    return {
        k: {m: v for m, v in r.items() if not m.startswith("per_")}
        for k, r in result.items()
    }


def predict(model, cache, ids, device):
    preds = []
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for part in ids.split(16):
            preds.append(
                model(cache["features"][part].to(device)).sigmoid().float().cpu()
            )
    return torch.cat(preds)


def evaluate(model, cache, ids, device):
    return scores(
        predict(model, cache, ids, device), cache["targets"][ids], cache["kinds"][ids]
    )


def fit(seed, cache, splits, args, output):
    torch.manual_seed(seed)
    rng = torch.Generator().manual_seed(seed)
    model = PerceptionHead().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()
    best = -1
    history = []
    for epoch in range(args.epochs):
        model.train()
        order = splits["train"][torch.randperm(len(splits["train"]), generator=rng)]
        losses = []
        for ids in order.split(16):
            x = cache["features"][ids].to(args.device)
            y = cache["targets"][ids].to(args.device)
            kinds = cache["kinds"][ids].to(args.device)
            # Transform features and labels together; zero is mean-free padded feature.
            dx, dy = torch.randint(-2, 3, (2,), generator=rng).tolist()
            x, y = shift(x, dx, dy), shift(y, dx, dy)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = label_loss(model(x), y, kinds)
            assert torch.isfinite(loss)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        val = evaluate(model, cache, splits["dev"], args.device)
        score = np.mean([val[k]["soft_dice"] for k in ("BF", "JT")])
        entry = dict(
            seed=seed, epoch=epoch + 1, loss=float(np.mean(losses)), dev=compact(val)
        )
        history.append(entry)
        print("EPOCH", json.dumps(entry), flush=True)
        if score > best:
            best = score
            torch.save(
                dict(model=model.state_dict(), epoch=epoch + 1, seed=seed),
                output / f"seed{seed}_best.pt",
            )
    save_json(output / f"seed{seed}_history.json", history)
    ckpt = torch.load(output / f"seed{seed}_best.pt", map_location=args.device)
    model.load_state_dict(ckpt["model"])
    result = {s: evaluate(model, cache, ids, args.device) for s, ids in splits.items()}
    result["best_epoch"] = ckpt["epoch"]
    # Within-task random permutation preserves the BF/JT marginal distribution.
    ids = splits["test"]
    permuted = ids.clone()
    for kind in (0, 1):
        positions = torch.where(cache["kinds"][ids] == kind)[0]
        p = positions[torch.randperm(len(positions), generator=rng)]
        permuted[positions] = ids[p]
    result["shuffled_input_test"] = scores(
        predict(model, cache, permuted, args.device),
        cache["targets"][ids],
        cache["kinds"][ids],
    )
    save_json(output / f"seed{seed}_results.json", result)
    return model, result


def shift(x, dx, dy, fill=0):
    """Integer translation without wraparound."""
    out = torch.full_like(x, fill)
    h, w = x.shape[-2:]
    out[..., max(0, dy) : min(h, h + dy), max(0, dx) : min(w, w + dx)] = x[
        ..., max(0, -dy) : min(h, h - dy), max(0, -dx) : min(w, w - dx)
    ]
    return out


def baselines(cache, splits, records):
    train, test = splits["train"], splits["test"]
    global_pred, author_pred = [], []
    for i in test.tolist():
        kind = cache["kinds"][i]
        pool = train[cache["kinds"][train] == kind]
        global_pred.append(cache["targets"][pool].mean(0))
        author_pool = [
            j for j in pool.tolist() if records[j]["author"] == records[i]["author"]
        ]
        author_pred.append(cache["targets"][author_pool].mean(0))
    return {
        "global_mean": scores(
            torch.stack(global_pred), cache["targets"][test], cache["kinds"][test]
        ),
        "author_mean_oracle": scores(
            torch.stack(author_pred), cache["targets"][test], cache["kinds"][test]
        ),
    }


def distort(x, name, level, y=None):
    """Controlled errors at 224 px; part shift uses held-out GT only for testing."""
    if name == "translation":
        return shift(x, level, 0, fill=1)
    if name == "thickening":
        return -F.max_pool2d(-x, 2 * level + 1, 1, level)
    if name == "blur":
        im = Image.fromarray(
            (x[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        )
        im = im.filter(ImageFilter.GaussianBlur(level))
        return torch.from_numpy(np.array(im)).permute(2, 0, 1)[None].to(x) / 255
    if name == "part_translation":
        active = torch.where(y[50:].sum((1, 2)) > 0)[0]
        channel = int(active[0]) + 50
        mask = (
            F.interpolate(
                y[None, channel : channel + 1].to(x), (224, 224), mode="nearest"
            )
            > 0.1
        )
        ink = (1 - x) * mask
        remaining = (1 - x) * (~mask)
        return 1 - torch.maximum(remaining, shift(ink, level, 0))
    raise ValueError(name)


def sensitivity(model, extractor, cache, splits, records, args, output):
    """Measure distortion response and dependence on actual image position."""
    model.eval().requires_grad_(False)
    rms = cache["rms"].to(args.device)
    test = splits["test"]
    selected = torch.cat([test[cache["kinds"][test] == k][:24] for k in (0, 1)])
    rows = []
    shifted_prob, shifted_target, shifted_kinds = [], [], []
    with torch.no_grad():
        for i in selected.tolist():
            x = cache["images"][i : i + 1].to(args.device)
            f0 = extractor(x) / rms
            h0 = model.body(f0)
            for name, levels in (
                ("translation", [2, 4, 8]),
                ("blur", [0.5, 1, 2]),
                ("thickening", [1, 2, 3]),
                ("part_translation", [2, 4, 8]),
            ):
                if name == "part_translation" and cache["kinds"][i] != 1:
                    continue
                distances = []
                for level in levels:
                    damaged = distort(x, name, level, cache["targets"][i])
                    f1 = extractor(damaged) / rms
                    h1 = model.body(f1)
                    distances.append(
                        dict(
                            level=level,
                            pixel=(damaged - x).abs().mean().item(),
                            vgg=(f1 - f0).abs().mean().item(),
                            trained=(h1 - h0).abs().mean().item(),
                        )
                    )
                rows.append(
                    dict(
                        index=i,
                        character=records[i]["character"],
                        kind=int(cache["kinds"][i]),
                        error=name,
                        distances=distances,
                    )
                )
            shifted = shift(x, 16, 0, fill=1)
            shifted_prob.append(model(extractor(shifted) / rms).sigmoid().cpu())
            shifted_target.append(shift(cache["targets"][i : i + 1], 2, 0))
            shifted_kinds.append(cache["kinds"][i : i + 1])
    aggregate = {}
    for name in sorted({r["error"] for r in rows}):
        matching = [r for r in rows if r["error"] == name]
        aggregate[name] = {}
        for metric in ("pixel", "vgg", "trained"):
            values = np.array([[d[metric] for d in r["distances"]] for r in matching])
            aggregate[name][metric] = dict(
                n=len(values),
                means=values.mean(0).tolist(),
                strictly_monotonic_fraction=float(
                    (np.diff(values, axis=1) > 0).all(1).mean()
                ),
            )
    # Actual gradient through the frozen image network: finite, nonzero, useful descent.
    i = int(selected[0])
    clean = cache["images"][i : i + 1].to(args.device)
    with torch.no_grad():
        ref = model.body(extractor(clean) / rms).detach()
    damaged = distort(clean, "blur", 1).detach().requires_grad_()
    distance = (model.body(extractor(damaged) / rms) - ref).square().mean()
    (grad,) = torch.autograd.grad(distance, damaged)
    steps = []
    for step in (0.0001, 0.001, 0.01):
        direction = grad / grad.square().mean().sqrt().clamp_min(1e-9)
        candidate = (damaged.detach() - step * direction).clamp(0, 1)
        with torch.no_grad():
            after = (
                (model.body(extractor(candidate) / rms) - ref).square().mean().item()
            )
        steps.append(dict(step=step, distance_after=after))
    result = dict(
        response=aggregate,
        translated_image=compact(
            scores(
                torch.cat(shifted_prob),
                torch.cat(shifted_target),
                torch.cat(shifted_kinds),
            )
        ),
        clean_same_subset=compact(evaluate(model, cache, selected, args.device)),
        gradient_check=dict(
            finite=bool(torch.isfinite(grad).all()),
            norm=grad.norm().item(),
            distance_before=distance.item(),
            steps=steps,
        ),
        items=rows,
    )
    save_json(output / "sensitivity.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--vgg-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(0.10)
    torch.backends.cudnn.benchmark = False
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    save_json(
        output / "config.json",
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )
    started = time.time()
    records = make_manifest(args.data_root, output)
    extractor = Extractor(args.vgg_weights).to(args.device).eval()
    if (output / "cache.pt").exists():
        cache = torch.load(output / "cache.pt", map_location="cpu")
        assert len(cache["kinds"]) == len(records)
    else:
        cache = build_cache(records, args.data_root, extractor, output, args.device)
    splits = {
        s: torch.tensor([i for i, r in enumerate(records) if r["split"] == s])
        for s in ("train", "dev", "test")
    }
    base = baselines(cache, splits, records)
    save_json(output / "baselines.json", base)
    results = {}
    for seed in args.seeds:
        model, result = fit(seed, cache, splits, args, output)
        results[str(seed)] = {
            s: compact(result[s]) for s in (*splits, "shuffled_input_test")
        }
        results[str(seed)]["best_epoch"] = result["best_epoch"]
        if seed == 0:
            sensitivity(model, extractor, cache, splits, records, args, output)
        del model
        torch.cuda.empty_cache()
    summary = dict(
        results=results,
        baselines={k: compact(v) for k, v in base.items()},
        elapsed_seconds=time.time() - started,
        peak_gpu_allocated_mb=torch.cuda.max_memory_allocated() / 1024**2,
        parameters=sum(p.numel() for p in PerceptionHead().parameters()),
        classes=list(class_names()),
    )
    save_json(output / "summary.json", summary)
    print("DONE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
