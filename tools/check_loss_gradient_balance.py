#!/usr/bin/env python

"""Compare weighted gradient norms of the loss terms used by stage 2."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from check_loss_gradients import build_transform, configure_model, load_model
from data.pairdataset import PairDataset
from util.masking_generator import MaskingGenerator


def freeze_like_training(model, freeze_blocks):
    model.discriminator.requires_grad_(False)
    model.patch_embed.requires_grad_(False)
    model.mask_token.requires_grad_(False)
    model.segment_token_x.requires_grad_(False)
    model.segment_token_y.requires_grad_(False)
    if model.pos_embed is not None:
        model.pos_embed.requires_grad_(False)
    for block in model.blocks[:freeze_blocks]:
        block.requires_grad_(False)


def grad_norm(loss, parameters):
    grads = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        create_graph=False,
        allow_unused=True,
    )
    finite_grads = [grad for grad in grads if grad is not None]
    if not finite_grads:
        return 0.0
    return float(
        torch.sqrt(sum(grad.square().sum() for grad in finite_grads)).item()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--json_path", nargs="+", required=True)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--freeze_blocks", type=int, default=9)
    parser.add_argument("--detail_gradient_ratio", type=float, default=0.1)
    parser.add_argument("--detail_per_sample_normalize", action="store_true")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset = PairDataset(
        args.data_path,
        args.json_path,
        transform=build_transform(),
        masked_position_generator=MaskingGenerator(
            (56, 28), num_masking_patches=784, max_num_patches=392
        ),
        use_two_pairs=True,
        half_mask_ratio=0.0,
        semantic_only_epochs=0,
    )

    samples = [
        dataset[(args.sample_index + offset) % len(dataset)]
        for offset in range(args.num_samples)
    ]
    images = torch.stack([sample[0] for sample in samples]).to(args.device)
    targets = torch.stack([sample[1] for sample in samples]).to(args.device)
    valids = torch.stack([sample[3] for sample in samples]).to(args.device)
    masks = np.stack([sample[2] for sample in samples])
    bool_masked_pos = torch.from_numpy(masks).bool().flatten(1).to(args.device)

    model = load_model(args.checkpoint)
    configure_model(model, args)
    model.detail_gradient_ratio = args.detail_gradient_ratio
    freeze_like_training(model, args.freeze_blocks)
    model.to(args.device)
    model.eval()

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    latent = model.forward_encoder(images, targets, bool_masked_pos)
    pred = model.forward_decoder(latent)
    model.forward_loss(
        images,
        pred,
        targets,
        bool_masked_pos,
        valids.clone(),
        epoch=35,
        no_gan=True,
        keep_loss_graph=True,
    )

    rows = []
    for name, loss in model.last_loss_graph.items():
        value = float(loss.detach().item())
        norm = grad_norm(loss, trainable_params)
        rows.append({"name": name, "value": value, "grad_norm": norm})

    weighted_names = {
        "recon_weighted",
        "style_weighted",
        "edge_weighted",
        "structure_weighted",
        "detail_weighted",
    }
    weighted_sum = sum(
        row["grad_norm"] for row in rows if row["name"] in weighted_names
    )
    print(
        f"num_samples={args.num_samples} trainable_params="
        f"{sum(p.numel() for p in trainable_params)} "
        f"per_sample={args.detail_per_sample_normalize} "
        f"gradient_ratio={args.detail_gradient_ratio}"
    )
    print(f"{'term':28s} {'value':>12s} {'grad_norm':>12s} {'share%':>8s}")
    for row in rows:
        share = (
            100.0 * row["grad_norm"] / weighted_sum
            if weighted_sum > 0 and row["name"] in weighted_names
            else None
        )
        share_text = f"{share:.3f}" if share is not None else "-"
        print(
            f"{row['name']:28s} {row['value']:12.6g} "
            f"{row['grad_norm']:12.6g} {share_text:>8s}"
        )

    output = {
        "num_samples": args.num_samples,
        "trainable_params": sum(p.numel() for p in trainable_params),
        "per_sample": args.detail_per_sample_normalize,
        "gradient_ratio": args.detail_gradient_ratio,
        "weighted_grad_norm_sum": weighted_sum,
        "rows": rows,
    }
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(output, indent=2))
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
