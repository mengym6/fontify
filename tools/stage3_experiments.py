"""Emit reviewable stage-3 commands; never infer which experiment won."""

import argparse
import json
from pathlib import Path


def experiments(phase, structure, detail, coefficients):
    """Return only the current controlled sweep, with duplicate configs removed."""
    if phase == "internal":
        if not coefficients:
            raise ValueError("internal comparison requires --coefficients")
        candidates = [
            ("equal", 0.2, 0.2, "off", None),
            ("calibrated", 0.2, 0.2, "off", coefficients),
        ]
    elif phase == "structure":
        candidates = [
            (f"structure-{w}", w, detail, "off", coefficients)
            for w in (0, 0.2, 0.5)
        ]
    elif phase == "detail":
        candidates = [
            (f"detail-{w}", structure, w, "off", coefficients)
            for w in (0, 0.2, 0.5)
        ]
    elif phase == "style":
        candidates = [
            (mode, structure, detail, mode, coefficients)
            for mode in ("off", "reference", "constant")
        ]
    else:
        candidates = [("reference", structure, detail, "reference", coefficients)]
        if structure > 0:
            candidates.extend(
                (
                    f"structure-x{factor}",
                    structure * factor,
                    detail,
                    "reference",
                    coefficients,
                )
                for factor in (0.5, 2)
            )
        if detail > 0:
            candidates.extend(
                (
                    f"detail-x{factor}",
                    structure,
                    detail * factor,
                    "reference",
                    coefficients,
                )
                for factor in (0.5, 2)
            )
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=["internal", "structure", "detail", "style", "local"]
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--semantic-mask-dir", required=True)
    parser.add_argument("--coefficients")
    parser.add_argument("--structure-weight", type=float, default=0.2)
    parser.add_argument("--detail-weight", type=float, default=0.2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    commands = []
    for name, structure, detail, mode, coefficients in experiments(
        args.phase, args.structure_weight, args.detail_weight, args.coefficients
    ):
        for seed in args.seeds:
            destination = str(
                Path(args.output_root) / args.phase / f"{name}-seed-{seed}"
            )
            command = [
                "torchrun",
                "--standalone",
                "--nproc_per_node=2",
                "tools/stage3.py",
                "train",
                "--vgg-input-mode",
                "legacy",
                "--manifest",
                args.manifest,
                "--data-root",
                args.data_root,
                "--checkpoint",
                args.checkpoint,
                "--semantic-mask-dir",
                args.semantic_mask_dir,
                "--style-mode",
                mode,
                "--structure-weight",
                str(structure),
                "--detail-weight",
                str(detail),
                "--seed",
                str(seed),
                "--output",
                destination,
            ]
            if coefficients:
                command.extend(["--coefficients", coefficients])
            commands.append(
                {
                    "name": name,
                    "seed": seed,
                    "argv": command,
                    "reuse_only_if_same_run_config": True,
                }
            )
    Path(args.output_json).write_text(json.dumps(commands, indent=2) + "\n")


if __name__ == "__main__":
    main()
