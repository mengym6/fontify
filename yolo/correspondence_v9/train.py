"""V9 trainer. Does not change v8 runs, datasets, optimizer states or checkpoints."""

import argparse
from collections import Counter
import hashlib, json, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from .data import CorrespondenceDataset, read_data, collate
from .model import GTCorrespondence, load_checkpoint
from .loss import CorrespondenceLoss
from .evaluate import evaluate
from .visualization import preview


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ["manifest", "init", "output"]:
        p.add_argument("--" + k, required=True)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--allow-weak-labels", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--size", type=int, default=320)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--unfreeze-encoder", action="store_true")
    p.add_argument("--resume", type=Path)
    p.add_argument("--tensorboard-dir")
    a = p.parse_args()
    data = read_data(a.manifest)
    counts = {
        s: dict(
            Counter(
                "matchable"
                if g["matchable"] is True
                else "unmatchable"
                if g["matchable"] is False
                else "unknown"
                for r in data["pairs"]
                if r["split"] == s
                for g in r["regions"]
            )
        )
        for s in ["train", "val"]
    }
    if a.check_only:
        print(
            json.dumps(
                {
                    "schema": "v9",
                    "counts": counts,
                    "reference_type": data.get("reference_type"),
                    "rejection_validation_available": counts["val"].get(
                        "unmatchable", 0
                    )
                    > 0,
                },
                indent=2,
            )
        )
        return
    if data.get("reference_type") == "reviewed_weak" and not a.allow_weak_labels:
        raise ValueError("Reviewed weak references require --allow-weak-labels")
    if a.epochs < 1 or a.batch < 1 or not 0 < a.lr <= 1:
        raise ValueError("Invalid training parameters")
    if any(
        not sum(v for k, v in counts[s].items() if k != "unknown")
        for s in ["train", "val"]
    ):
        raise ValueError("Both splits require known region supervision")
    digest = hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest()
    output = Path(a.output)
    start = 0
    old = None
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    if a.resume:
        if a.resume.resolve() != (output / "last.pt").resolve():
            raise ValueError("Resume this run last.pt only")
        old = torch.load(a.resume, map_location="cpu")
        config = json.loads((output / "config.json").read_text())
        if old["manifest_sha256"] != digest:
            raise ValueError("Resume data changed")
        for key in ["batch", "size", "lr", "seed", "unfreeze_encoder"]:
            if config[key] != getattr(a, key):
                raise ValueError("Resume config changed: " + key)
        start = old["epoch"]
        history = [
            json.loads(x) for x in (output / "metrics.jsonl").read_text().splitlines()
        ]
        if history[-1]["epoch"] != start or a.epochs <= start:
            raise ValueError("Resume epoch/history mismatch")
        model = load_checkpoint(a.resume, a.device)
    else:
        if output.exists():
            raise FileExistsError("New v9 output directory required")
        model = GTCorrespondence.from_init(a.init).to(a.device)
    model.freeze_encoder(not a.unfreeze_encoder)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0005
    )
    if old:
        optimizer.load_state_dict(old["optimizer"])
        torch.set_rng_state(old["rng_state"])
        if str(a.device).startswith("cuda") and old.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(old["cuda_rng_state"])
    criterion = CorrespondenceLoss()
    datasets = {
        s: CorrespondenceDataset(a.manifest, s, a.size) for s in ["train", "val"]
    }
    loaders = {
        s: DataLoader(ds, batch_size=a.batch, shuffle=s == "train", collate_fn=collate)
        for s, ds in datasets.items()
    }
    output.mkdir(parents=True, exist_ok=bool(a.resume))
    (output / (f"resume_{start}.json" if old else "config.json")).write_text(
        json.dumps(vars(a), indent=2, default=str)
    )
    writer = SummaryWriter(
        a.tensorboard_dir or str(output.parent / "tensorboard" / output.name)
    )
    best = min(r["val"] for r in history) if old else float("inf")
    try:
        for epoch in range(start, a.epochs):
            metrics = {}
            for split, loader in loaders.items():
                model.train(split == "train")
                total = 0.0
                n = 0
                image_index = 0
                for images, targets in loader:
                    images = images.to(a.device)
                    with torch.set_grad_enabled(split == "train"):
                        predictions = model(
                            images,
                            [t["gt_masks"] for t in targets],
                            [t["groups"] for t in targets],
                        )
                        loss, parts = criterion(predictions, targets)
                        if not torch.isfinite(loss):
                            raise RuntimeError("Nonfinite loss")
                        if split == "train":
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            optimizer.step()
                    total += float(loss.detach()) * len(images)
                    n += len(images)
                    if split == "val":
                        for i, t in enumerate(targets):
                            writer.add_image(
                                f"epoch:{epoch + 1:03}/validation",
                                preview(images[i], t, predictions[i]),
                                image_index,
                                dataformats="HWC",
                            )
                            image_index += 1
                metrics[split] = total / n
                writer.add_scalar("loss/" + split, metrics[split], epoch + 1)
            report = evaluate(model, datasets["val"], a.device)
            for group, values in report["groups"].items():
                for key, value in values.items():
                    if value is not None:
                        writer.add_scalar(group + "/" + key, value, epoch + 1)
            saved = {
                "schema": "calli-gt2gen-v9",
                "model": model.state_dict(),
                "architecture": model.architecture,
                "config": model.config,
                "init_sha256": model.init_sha256,
                "manifest_sha256": digest,
                "epoch": epoch + 1,
                "optimizer": optimizer.state_dict(),
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all()
                if str(a.device).startswith("cuda")
                else None,
                "metrics": metrics,
                "reference_type": data.get("reference_type"),
                "selection": "minimum v9 validation loss; no rejection claim without support",
            }
            torch.save(saved, output / "last.pt")
            if metrics["val"] < best:
                best = metrics["val"]
                torch.save(saved, output / "best.pt")
            with (output / "metrics.jsonl").open("a") as f:
                f.write(
                    json.dumps(
                        {"epoch": epoch + 1, **metrics, "evaluation": report["groups"]}
                    )
                    + "\n"
                )
            (output / "evaluation.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
            )
            writer.flush()
            print(epoch + 1, metrics, report["groups"]["all"], flush=True)
            if (output / "STOP").exists():
                break
    finally:
        writer.close()


if __name__ == "__main__":
    main()
