"""CLI for the independent Run C v2 runtime (no BF long-training command)."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import runc_v2_runtime as runtime  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("audit", "probe", "smoke", "train", "evaluate", "stage2-smoke"),
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config = runtime.validate_config(
        json.loads(Path(args.config).read_text()), args.command
    )
    output = Path(args.output)
    if args.command == "audit":
        output.mkdir(parents=True, exist_ok=True)
        runtime.write_json(
            output / "runtime_audit.json",
            {
                "schema": runtime.SCHEMA,
                "config": config,
                "metadata": runtime.source_record(config),
            },
        )
        return
    runtime.configure_backend(config)
    device, rank, world = runtime.distributed_device(args.device)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        runtime.write_json(output / "config.json", config)
    if world > 1:
        import torch.distributed as dist

        dist.barrier()
    model, saved, loading = runtime.load_model(config, device)
    if rank == 0:
        runtime.write_json(output / "checkpoint_loading.json", loading)
    if args.command == "stage2-smoke":
        if world != 1:
            raise ValueError("stage-II interface check uses one process")
        runtime.stage2_interface_smoke(config, model, output, device)
    elif args.command == "evaluate":
        if world != 1:
            raise ValueError("fixed validation is serial; use one process")
        dataset = runtime.build_dataset(config, validation=True)
        runtime.evaluate(config, model, dataset, output, device, limit=args.limit)
    else:
        dataset = runtime.build_dataset(config)
        if args.command == "probe":
            if world != 1:
                raise ValueError("calibration uses one process for parameter Grams")
            runtime.probe(config, model, dataset, output, device)
        elif args.command == "smoke":
            runtime.smoke(config, model, saved, dataset, output, device, rank, world)
        else:
            runtime.train(config, model, saved, dataset, output, device, rank, world)
    if world > 1:
        import torch.distributed as dist

        dist.destroy_process_group()


if __name__ == "__main__":
    main()
