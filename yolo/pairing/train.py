"""Explicit paired-region training entry. Image pairs alone are not supervision."""

import argparse
import json
import hashlib
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from .data import PairDataset, collate_pairs, read_manifest, validate_pairs
from .model import PairRegions, load_pair_checkpoint
from .loss import PairLoss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--check-only", action="store_true")
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Continue last.pt with optimizer; epochs is total target",
    )
    p.add_argument("--tensorboard-dir", default=None)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--unfreeze-encoder", action="store_true")
    p.add_argument("--allow-pseudo", action="store_true")
    a = p.parse_args()
    document = read_manifest(a.manifest)
    # Validate supervision separately on train and val. Never select by unknown targets.
    report = {}
    for split in ["train", "val"]:
        part = dict(
            document, pairs=[r for r in document["pairs"] if r["split"] == split]
        )
        report[split] = {
            "image_pairs": len(part["pairs"]),
            "supervised_pairs": validate_pairs(part, allow_pseudo=a.allow_pseudo),
        }
    if a.check_only:
        print(
            json.dumps(
                {
                    "data": report,
                    "training_ready": all(
                        r["supervised_pairs"] for r in report.values()
                    ),
                    "initialization": "baseline encoder+neck+prototype; new joint-query heads",
                },
                indent=2,
            )
        )
        return
    for split in ["train", "val"]:
        if not report[split]["supervised_pairs"]:
            raise ValueError(
                split
                + " lacks paired region supervision; annotate generated side first"
            )
    if a.epochs < 1 or a.batch < 1 or not 0 < a.lr <= 1:
        raise ValueError("Invalid training budget or learning rate")
    output = Path(a.output)
    saved_resume = None
    start_epoch = 0
    if a.resume:
        if a.resume.resolve() != (output / "last.pt").resolve():
            raise ValueError("Resume must use this run last.pt")
        saved_resume = torch.load(a.resume, map_location="cpu")
        old_config = json.loads((output / "config.json").read_text())
        for key in ("batch", "size", "lr", "seed", "unfreeze_encoder"):
            if getattr(a, key) != old_config[key]:
                raise ValueError("Resume configuration changed: " + key)
        if (
            saved_resume.get("manifest_sha256")
            != hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest()
        ):
            raise ValueError("Resume dataset changed")
        start_epoch = saved_resume["epoch"]
        if not saved_resume.get("optimizer") or a.epochs <= start_epoch:
            raise ValueError("Resume needs optimizer and a larger total epoch budget")
        history = [
            json.loads(x) for x in (output / "metrics.jsonl").read_text().splitlines()
        ]
        if history[-1]["epoch"] != start_epoch:
            raise ValueError("Checkpoint/history epoch mismatch")
    elif output.exists():
        raise FileExistsError("Use a new output directory")
    torch.manual_seed(a.seed)
    model = (
        load_pair_checkpoint(a.resume, a.device)
        if a.resume
        else PairRegions.from_baseline(a.baseline).to(a.device)
    )
    model.freeze_encoder(not a.unfreeze_encoder)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0005
    )
    if saved_resume:
        optimizer.load_state_dict(saved_resume["optimizer"])
        if "rng_state" in saved_resume:
            torch.set_rng_state(saved_resume["rng_state"])
            if torch.cuda.is_available() and saved_resume.get("cuda_rng_state"):
                torch.cuda.set_rng_state_all(saved_resume["cuda_rng_state"])
    criterion = PairLoss()
    loaders = {
        s: DataLoader(
            PairDataset(a.manifest, s, a.size, a.allow_pseudo),
            batch_size=a.batch,
            shuffle=s == "train",
            collate_fn=collate_pairs,
        )
        for s in ["train", "val"]
    }
    output.mkdir(parents=True, exist_ok=bool(a.resume))
    config_name = f"resume_from_{start_epoch:03d}.json" if a.resume else "config.json"
    (output / config_name).write_text(json.dumps(vars(a), indent=2, default=str))
    from torch.utils.tensorboard import SummaryWriter
    from .visualization import paired_preview

    writer = SummaryWriter(
        a.tensorboard_dir or str(output.parent / "tensorboard" / output.name)
    )
    manifest_sha = hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest()
    best = min(r["val"] for r in history) if saved_resume else float("inf")
    if a.resume:
        print(
            f"Resumed epoch {start_epoch}; optimizer restored; historical best {best}; target {a.epochs}",
            flush=True,
        )
    for epoch in range(start_epoch, a.epochs):
        metrics = {}
        for split, loader in loaders.items():
            model.train(split == "train")
            total, n = 0.0, 0
            components = {}
            preview_index = 0
            for images, targets in loader:
                images = images.to(a.device)
                with torch.set_grad_enabled(split == "train"):
                    predictions = model(images)
                    loss, parts = criterion(predictions, targets)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite loss")
                    if split == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                for key, value in parts.items():
                    components[key] = components.get(key, 0.0) + value * len(images)
                if split == "val":
                    for item in range(len(images)):
                        writer.add_image(
                            f"epoch:{epoch + 1:03d}/validation",
                            paired_preview(images, targets, predictions, item),
                            preview_index,
                            dataformats="HWC",
                        )
                        preview_index += 1
                total += float(loss.detach()) * len(images)
                n += len(images)
            metrics[split] = total / n
            writer.add_scalar(f"loss/{split}_total", metrics[split], epoch + 1)
            for key, value in components.items():
                tag = (
                    key.split("/")[0] + f"/loss/{split}"
                    if "/" in key
                    else f"components/{split}/{key}"
                )
                writer.add_scalar(tag, value / n, epoch + 1)
        saved = {
            "schema": "calli-pair-regions-v1",
            "manifest_sha256": manifest_sha,
            "model": model.state_dict(),
            "architecture": model.architecture,
            "config": model.config,
            "baseline_sha256": model.baseline_sha256,
            "epoch": epoch + 1,
            "optimizer": optimizer.state_dict(),
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "metrics": metrics,
            "selection": "minimum paired validation loss, not YOLO mAP",
        }
        torch.save(saved, output / "last.pt")
        if metrics["val"] < best:
            best = metrics["val"]
            torch.save(saved, output / "best.pt")
        with (output / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"epoch": epoch + 1, **metrics}) + "\n")
        writer.flush()
        print(epoch + 1, metrics, flush=True)
        if (output / "STOP").exists():
            break

    writer.close()


if __name__ == "__main__":
    main()
