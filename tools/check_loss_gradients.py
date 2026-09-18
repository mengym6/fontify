#!/usr/bin/env python

"""Check that structure and detail sub-losses backpropagate to `pred`."""

import argparse
import random

import numpy as np
import torch

import data.pair_transforms as pair_transforms
import models_train
from data.pairdataset import PairDataset
from util.masking_generator import MaskingGenerator


def build_transform():
    normalize = pair_transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    return pair_transforms.Compose([
        pair_transforms.PadToSquare(fill=255),
        pair_transforms.RandomResizedCrop(
            448, scale=(0.9999, 1.0), interpolation=3
        ),
        pair_transforms.ToTensor(),
        normalize,
    ])


def load_model(checkpoint_path):
    model = models_train.vit_base_patch16_input896x448_win_dec64_8glb_sl1()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")["model"]
    state_dict = model.state_dict()
    for key in ("decoder_embed.weight", "decoder_embed.bias", "mask_token"):
        if key in checkpoint and checkpoint[key].shape != state_dict[key].shape:
            del checkpoint[key]
    model.load_state_dict(checkpoint, strict=False)
    return model


def configure_model(model, args):
    model.semantic_only_epochs = 0
    model.adv_warmup_epochs = 8
    model.edge_warmup_epochs = 8
    model.loss_warmup_duration = 8
    model.adv_weight_final = 0.4
    model.edge_weight_final = 0.3
    model.structure_loss_weight = 0.05
    model.structure_warmup_epochs = 6
    model.structure_warmup_duration = 6
    model.detail_loss_weight = 0.05
    model.detail_warmup_epochs = 4
    model.detail_warmup_duration = 4
    model.detail_kernel_size = 5
    model.detail_sigma = 1.0
    model.detail_gradient_ratio = 0.1
    model.detail_per_sample_normalize = args.detail_per_sample_normalize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--json_path", nargs="+", required=True)
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--detail_per_sample_normalize", action="store_true")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--min_grad_norm", type=float, default=1e-9)
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
    image, target, mask, valid = dataset[args.sample_index]

    model = load_model(args.checkpoint)
    configure_model(model, args)
    model.to(args.device)
    model.eval()

    image = image.unsqueeze(0).to(args.device)
    target = target.unsqueeze(0).to(args.device)
    valid = valid.unsqueeze(0).to(args.device)
    bool_masked_pos = torch.from_numpy(mask).unsqueeze(0).bool().flatten(1).to(args.device)

    latent = model.forward_encoder(image, target, bool_masked_pos)
    pred = model.forward_decoder(latent)
    model.forward_loss(
        image,
        pred,
        target,
        bool_masked_pos,
        valid.clone(),
        epoch=35,
        no_gan=True,
        keep_loss_graph=True,
    )

    failures = []
    print(f"sample_index={args.sample_index} "
          f"detail_per_sample_normalize={args.detail_per_sample_normalize}")
    for name, loss in model.last_loss_graph.items():
        grad = torch.autograd.grad(
            loss,
            pred,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )[0]
        if grad is None:
            failures.append(name)
            print(f"{name:24s} grad: None")
            continue
        norm = float(grad.detach().norm().item())
        finite = bool(torch.isfinite(grad).all().item())
        print(f"{name:24s} grad_norm={norm:.8f} finite={finite}")
        if not finite or norm <= args.min_grad_norm:
            failures.append(name)

    if failures:
        raise SystemExit(f"gradient check failed for: {', '.join(sorted(failures))}")
    print("gradient check passed")


if __name__ == "__main__":
    main()
