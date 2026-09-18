import math
import os
import sys
import csv
from collections import defaultdict
from typing import Iterable

import torch
import torch.nn.functional as F
import util.misc as misc
import util.lr_sched as lr_sched

import numpy as np
#import wandb

import time


_GRADIENT_GROUP_PREFIXES = (
    "patch_embed",
    "pos_embed",
    "mask_token",
    "segment_token_x",
    "segment_token_y",
    "norm",
    "decoder_embed",
    "decoder_pred",
    "discriminator",
    "vgg_loss",
)


def _gradient_group_name(name):
    if name.startswith("blocks."):
        parts = name.split(".")
        if len(parts) >= 2 and parts[1].isdigit():
            return f"blocks.{parts[1]}"
    for prefix in _GRADIENT_GROUP_PREFIXES:
        if name == prefix or name.startswith(prefix + "."):
            return prefix
    return "other"


def _collect_gradient_stats(model):
    raw_model = model.module if hasattr(model, "module") else model
    groups = defaultdict(
        lambda: {
            "params": 0,
            "elements": 0,
            "trainable": 0,
            "with_grad": 0,
            "no_grad": 0,
            "frozen": 0,
            "nonfinite": 0,
            "grad_elements": 0,
            "sum_sq": 0.0,
            "max": 0.0,
        }
    )

    for name, param in raw_model.named_parameters():
        group = _gradient_group_name(name)
        stats = groups[group]
        stats["params"] += 1
        stats["elements"] += param.numel()
        if not param.requires_grad:
            stats["frozen"] += 1
            continue
        stats["trainable"] += 1
        grad = param.grad
        if grad is None:
            stats["no_grad"] += 1
            continue
        finite = torch.isfinite(grad)
        if not bool(finite.all()):
            stats["nonfinite"] += 1
        grad = torch.where(finite, grad, torch.zeros_like(grad))
        stats["with_grad"] += 1
        stats["grad_elements"] += grad.numel()
        stats["sum_sq"] += float(grad.square().sum().item())
        stats["max"] = max(stats["max"], float(grad.abs().max().item()))

    rows = []
    computed_norm = 0.0
    for group, stats in groups.items():
        grad_norm = math.sqrt(stats["sum_sq"])
        grad_mean = math.sqrt(stats["sum_sq"] / stats["grad_elements"]) if stats["grad_elements"] else 0.0
        computed_norm += stats["sum_sq"]
        rows.append((group, grad_norm, grad_mean, stats))
    computed_norm = math.sqrt(computed_norm)
    return rows, computed_norm


class GradientMonitor:
    """Export grouped gradient statistics to a CSV in the model output directory."""

    CSV_FIELDS = [
        "epoch",
        "data_iter_step",
        "update_idx",
        "rank",
        "reported_grad_norm",
        "pre_clip_grad_norm",
        "post_clip_grad_norm",
        "grad_clip_ratio",
        "computed_grad_norm",
        "group",
        "params",
        "elements",
        "trainable",
        "with_grad",
        "no_grad",
        "frozen",
        "nonfinite",
        "grad_elements",
        "grad_norm",
        "grad_mean",
        "grad_max",
        "norm_ratio",
    ]

    def __init__(self, output_dir, interval, rank=0):
        self.enabled = bool(output_dir and interval and interval > 0 and rank == 0)
        self.interval = max(1, int(interval))
        self.path = os.path.join(output_dir, "gradient_log.csv") if output_dir else None
        self.file = None
        self.writer = None
        if self.enabled:
            is_new = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
            self.file = open(self.path, "a", encoding="utf-8", newline="")
            self.writer = csv.DictWriter(self.file, fieldnames=self.CSV_FIELDS)
            if is_new:
                self.writer.writeheader()
                self.file.flush()

    def log(
        self,
        epoch,
        data_iter_step,
        update_idx,
        model,
        reported_grad_norm=None,
        pre_clip_grad_norm=None,
        post_clip_grad_norm=None,
    ):
        if not self.enabled or update_idx % self.interval != 0:
            return

        def as_float(value):
            if value is None:
                return float("nan")
            if torch.is_tensor(value):
                return float(value.detach().item())
            return float(value)

        reported_grad_norm = as_float(reported_grad_norm)
        pre_clip_grad_norm = as_float(pre_clip_grad_norm)
        post_clip_grad_norm = as_float(post_clip_grad_norm)
        grad_clip_ratio = (
            post_clip_grad_norm / pre_clip_grad_norm
            if pre_clip_grad_norm > 0
            else 0.0
        )
        rows, computed_norm = _collect_gradient_stats(model)
        print(
            f"[grad] epoch={epoch} step={data_iter_step} update={update_idx} "
            f"pre_clip={pre_clip_grad_norm:.6f} post_clip={post_clip_grad_norm:.6f} "
            f"clip_ratio={grad_clip_ratio:.6f} "
            f"computed={computed_norm:.6f}"
        )
        for group, grad_norm, grad_mean, stats in rows:
            if stats["frozen"] == stats["params"]:
                state = f"frozen params={stats['params']}"
            elif stats["with_grad"] == 0:
                state = f"no-grad trainable={stats['trainable']}"
            else:
                state = (
                    f"norm={grad_norm:.6f} mean={grad_mean:.6f} "
                    f"max={stats['max']:.6f} nonfinite={stats['nonfinite']}"
                )
            print(f"[grad]   {group:16s} {state}")
            self.writer.writerow(
                {
                    "epoch": epoch,
                    "data_iter_step": data_iter_step,
                    "update_idx": update_idx,
                    "rank": 0,
                    "reported_grad_norm": reported_grad_norm,
                    "pre_clip_grad_norm": pre_clip_grad_norm,
                    "post_clip_grad_norm": post_clip_grad_norm,
                    "grad_clip_ratio": grad_clip_ratio,
                    "computed_grad_norm": computed_norm,
                    "group": group,
                    "params": stats["params"],
                    "elements": stats["elements"],
                    "trainable": stats["trainable"],
                    "with_grad": stats["with_grad"],
                    "no_grad": stats["no_grad"],
                    "frozen": stats["frozen"],
                    "nonfinite": stats["nonfinite"],
                    "grad_elements": stats["grad_elements"],
                    "grad_norm": grad_norm,
                    "grad_mean": grad_mean,
                    "grad_max": stats["max"],
                    "norm_ratio": grad_norm / computed_norm if computed_norm > 0 else 0.0,
                }
            )
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


#检索关键词tb调整画图数量
def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    loss_scale = None
    if hasattr(optimizer, 'loss_scale'):
        loss_scale = optimizer.loss_scale
    elif hasattr(optimizer, 'cur_scale'):
        loss_scale = optimizer.cur_scale
    return loss_scale, optimizer._global_grad_norm
    # return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    global_rank=None,
                    args=None,
                    optimizer_d=None,
                    gradient_monitor=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter
    num_updates = 0

    optimizer.zero_grad()


    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    # wandb_images = []
    tensorboard_images = []
    for data_iter_step, (samples, targets, bool_masked_pos, valid) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True, dtype=torch.bfloat16)
        targets = targets.to(device, non_blocking=True, dtype=torch.bfloat16)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True, dtype=torch.bfloat16)
        valid = valid.to(device, non_blocking=True, dtype=torch.bfloat16)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss, loss_l1l2, loss_vgg, y, mask, pred = model(
                samples, targets, bool_masked_pos=bool_masked_pos,
                valid=valid, epoch=epoch, no_gan=args.no_gan
            )

        if not args.no_gan:
            raw_model = model.module if hasattr(model, "module") else model
            requires_grad_original = {}
            for name, param in raw_model.named_parameters():
                requires_grad_original[name] = param.requires_grad
                if 'discriminator' not in name:
                    param.requires_grad = False

            raw_model.discriminator.requires_grad_(True)
            optimizer_d.zero_grad()

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                real_imgs = raw_model.resize(targets)
                real_output = raw_model.discriminator(real_imgs)
                real_loss = F.binary_cross_entropy_with_logits(real_output, torch.ones_like(real_output))

                fake_imgs = raw_model.resize(pred.detach())
                fake_output = raw_model.discriminator(fake_imgs)
                fake_loss = F.binary_cross_entropy_with_logits(fake_output, torch.zeros_like(fake_output))

                d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_d.step()
            # D gradients are for optimizer_d only. Clear them before the
            # generator backward/update so G gradient logs do not include stale
            # discriminator gradients.
            raw_model.discriminator.zero_grad(set_to_none=True)

            raw_model.discriminator.requires_grad_(False)
            for name, param in raw_model.named_parameters():
                param.requires_grad = requires_grad_original[name]

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        pre_clip_grad_norm = None
        post_clip_grad_norm = None
        if loss_scaler is None:
            loss /= accum_iter
            model.backward(loss)
            model.step()

            # if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
            # grad_norm = None
            loss_scale_value, grad_norm = get_loss_scale_for_deepspeed(model)
        else:
            loss /= accum_iter
            update_grad = (data_iter_step + 1) % accum_iter == 0
            grad_norm = loss_scaler(loss, optimizer, clip_grad=args.clip_grad,
                                    parameters=model.parameters(),
                                    update_grad=update_grad)
            pre_clip_grad_norm = getattr(loss_scaler, "last_unclipped_norm", None)
            post_clip_grad_norm = getattr(loss_scaler, "last_clipped_norm", None)

            if update_grad:
                num_updates += 1
                if gradient_monitor is not None:
                    gradient_monitor.log(
                        epoch,
                        data_iter_step,
                        num_updates,
                        model,
                        grad_norm,
                        pre_clip_grad_norm,
                        post_clip_grad_norm,
                    )
                optimizer.zero_grad()
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()
        #print(f"loss:{loss},grad_norm:{grad_norm}")
        metric_logger.update(loss=loss_value)
        raw_model = model.module if hasattr(model, 'module') else model
        def scalar(value):
            if torch.is_tensor(value):
                return float(value.detach().item())
            return float(value)

        detail_loss = raw_model.last_loss_components.get('detail', torch.tensor(0.0))
        highpass_loss = raw_model.last_loss_components.get('highpass', torch.tensor(0.0))
        gradient_loss = raw_model.last_loss_components.get('gradient', torch.tensor(0.0))
        structure_row_loss = raw_model.last_loss_components.get(
            'structure_row', torch.tensor(0.0)
        )
        structure_col_loss = raw_model.last_loss_components.get(
            'structure_col', torch.tensor(0.0)
        )
        structure_centroid_loss = raw_model.last_loss_components.get(
            'structure_centroid', torch.tensor(0.0)
        )
        structure_area_loss = raw_model.last_loss_components.get(
            'structure_area', torch.tensor(0.0)
        )
        structure_weighted = raw_model.last_loss_components.get(
            'structure_weighted', torch.tensor(0.0)
        )
        detail_weighted = raw_model.last_loss_components.get(
            'detail_weighted', torch.tensor(0.0)
        )
        gradient_contribution = raw_model.last_loss_components.get(
            'gradient_contribution', torch.tensor(0.0)
        )
        detail_region_ratio = raw_model.last_loss_components.get(
            'detail_region_ratio', torch.tensor(0.0)
        )
        valid_ratio = raw_model.last_loss_components.get(
            'valid_ratio', torch.tensor(0.0)
        )
        target_highpass_mean = raw_model.last_loss_components.get(
            'target_highpass_mean', torch.tensor(0.0)
        )
        target_gradient_mean = raw_model.last_loss_components.get(
            'target_gradient_mean', torch.tensor(0.0)
        )
        metric_logger.update(loss_detail=detail_loss.item())
        metric_logger.update(loss_highpass=highpass_loss.item())
        metric_logger.update(loss_gradient=gradient_loss.item())
        metric_logger.update(loss_structure_row=structure_row_loss.item())
        metric_logger.update(loss_structure_col=structure_col_loss.item())
        metric_logger.update(loss_structure_centroid=structure_centroid_loss.item())
        metric_logger.update(loss_structure_area=structure_area_loss.item())
        metric_logger.update(structure_weighted=structure_weighted.item())
        metric_logger.update(detail_weighted=detail_weighted.item())
        metric_logger.update(gradient_contribution=gradient_contribution.item())
        metric_logger.update(detail_region_ratio=detail_region_ratio.item())
        metric_logger.update(valid_ratio=valid_ratio.item())
        metric_logger.update(target_highpass_mean=target_highpass_mean.item())
        metric_logger.update(target_gradient_mean=target_gradient_mean.item())

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        metric_logger.update(loss_scale=loss_scale_value)
        metric_logger.update(grad_norm=grad_norm)
        if pre_clip_grad_norm is not None:
            metric_logger.update(grad_norm_pre_clip=pre_clip_grad_norm)
            if post_clip_grad_norm is not None:
                metric_logger.update(grad_norm_post_clip=post_clip_grad_norm)
                metric_logger.update(grad_clip_ratio=(
                    post_clip_grad_norm / pre_clip_grad_norm
                    if pre_clip_grad_norm > 0
                    else 0.0
                ))

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_l1l2_reduce = misc.all_reduce_mean(loss_l1l2)
        loss_vgg_reduce = misc.all_reduce_mean(loss_vgg)
        structure_reduce = misc.all_reduce_mean(raw_model.last_loss_components['structure'])
        detail_reduce = misc.all_reduce_mean(detail_loss)
        highpass_reduce = misc.all_reduce_mean(highpass_loss)
        gradient_reduce = misc.all_reduce_mean(gradient_loss)
        structure_row_reduce = misc.all_reduce_mean(structure_row_loss)
        structure_col_reduce = misc.all_reduce_mean(structure_col_loss)
        structure_centroid_reduce = misc.all_reduce_mean(structure_centroid_loss)
        structure_area_reduce = misc.all_reduce_mean(structure_area_loss)
        structure_weighted_reduce = misc.all_reduce_mean(structure_weighted)
        detail_weighted_reduce = misc.all_reduce_mean(detail_weighted)
        gradient_contribution_reduce = misc.all_reduce_mean(gradient_contribution)
        detail_region_ratio_reduce = misc.all_reduce_mean(detail_region_ratio)
        valid_ratio_reduce = misc.all_reduce_mean(valid_ratio)
        target_highpass_mean_reduce = misc.all_reduce_mean(target_highpass_mean)
        target_gradient_mean_reduce = misc.all_reduce_mean(target_gradient_mean)
        detail_weight = raw_model.last_loss_components.get('detail_weight', 0.0)

        if log_writer is not None and grad_norm is not None:
            with open(os.path.join(args.output_dir, "log_detail.txt"), mode="a", encoding="utf-8") as f:
                f.write(
                    f"[{time.time()}] Epoch: [{epoch}]  [{data_iter_step}/{len(data_loader)}]  lr: {lr}  loss: {loss}   "
                    f"loss_scale_value: {loss_scale_value}  grad_norm: {grad_norm} "
                    f"detail: {scalar(detail_reduce):.6f} highpass: {scalar(highpass_reduce):.6f} "
                    f"gradient: {scalar(gradient_reduce):.6f} detail_weight: {scalar(detail_weight):.6f} "
                    f"structure: {scalar(structure_reduce):.6f} "
                    f"row: {scalar(structure_row_reduce):.6f} col: {scalar(structure_col_reduce):.6f} "
                    f"centroid: {scalar(structure_centroid_reduce):.6f} "
                    f"area: {scalar(structure_area_reduce):.6f} "
                    f"structure_weighted: {scalar(structure_weighted_reduce):.6f} "
                    f"detail_weighted: {scalar(detail_weighted_reduce):.6f} "
                    f"gradient_contribution: {scalar(gradient_contribution_reduce):.6f} "
                    f"detail_region_ratio: {scalar(detail_region_ratio_reduce):.6f} "
                    f"valid_ratio: {scalar(valid_ratio_reduce):.6f} "
                    f"target_highpass_mean: {scalar(target_highpass_mean_reduce):.6f} "
                    f"target_gradient_mean: {scalar(target_gradient_mean_reduce):.6f} "
                    f"grad_ok: "
                    f"{scalar(raw_model.last_loss_components.get('structure_row_grad_ok', 0)):.0f}"
                    f"{scalar(raw_model.last_loss_components.get('structure_col_grad_ok', 0)):.0f}"
                    f"{scalar(raw_model.last_loss_components.get('structure_centroid_grad_ok', 0)):.0f}"
                    f"{scalar(raw_model.last_loss_components.get('structure_area_grad_ok', 0)):.0f}"
                    f"{scalar(raw_model.last_loss_components.get('highpass_grad_ok', 0)):.0f}"
                    f"{scalar(raw_model.last_loss_components.get('gradient_grad_ok', 0)):.0f} "
                    f"grad_pre_clip: "
                    f"{pre_clip_grad_norm if pre_clip_grad_norm is not None else float('nan'):.6f} "
                    f"grad_post_clip: "
                    f"{post_clip_grad_norm if post_clip_grad_norm is not None else float('nan'):.6f}\n")
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            if pre_clip_grad_norm is not None:
                log_writer.add_scalar('grad_pre_clip', pre_clip_grad_norm, epoch_1000x)
                log_writer.add_scalar(
                    'grad_post_clip',
                    post_clip_grad_norm if post_clip_grad_norm is not None else grad_norm,
                    epoch_1000x,
                )
                log_writer.add_scalar(
                    'grad_clip_ratio',
                    (
                        post_clip_grad_norm / pre_clip_grad_norm
                        if post_clip_grad_norm is not None and pre_clip_grad_norm > 0
                        else 0.0
                    ),
                    epoch_1000x,
                )
            log_writer.add_scalars('train_loss_detail', {
                'loss_l1l2': loss_l1l2_reduce,
                'loss_vgg': loss_vgg_reduce,
                'loss_structure': structure_reduce,
                'loss_detail': detail_reduce,
                'loss_highpass': highpass_reduce,
                'loss_gradient': gradient_reduce,
                'loss_structure_row': structure_row_reduce,
                'loss_structure_col': structure_col_reduce,
                'loss_structure_centroid': structure_centroid_reduce,
                'loss_structure_area': structure_area_reduce,
                'detail_weight': detail_weight,
                'structure_weighted': structure_weighted_reduce,
                'detail_weighted': detail_weighted_reduce,
                'gradient_contribution': gradient_contribution_reduce,
                'detail_region_ratio': detail_region_ratio_reduce,
                'valid_ratio': valid_ratio_reduce,
                'target_highpass_mean': target_highpass_mean_reduce,
                'target_gradient_mean': target_gradient_mean_reduce,
                'structure_row_grad_ok': scalar(
                    raw_model.last_loss_components.get('structure_row_grad_ok', 0)
                ),
                'structure_col_grad_ok': scalar(
                    raw_model.last_loss_components.get('structure_col_grad_ok', 0)
                ),
                'structure_centroid_grad_ok': scalar(
                    raw_model.last_loss_components.get('structure_centroid_grad_ok', 0)
                ),
                'structure_area_grad_ok': scalar(
                    raw_model.last_loss_components.get('structure_area_grad_ok', 0)
                ),
                'highpass_grad_ok': scalar(
                    raw_model.last_loss_components.get('highpass_grad_ok', 0)
                ),
                'gradient_grad_ok': scalar(
                    raw_model.last_loss_components.get('gradient_grad_ok', 0)
                ),
            }, epoch_1000x)


            with torch.no_grad():
                imagenet_mean = np.array([0.485, 0.456, 0.406])
                imagenet_std = np.array([0.229, 0.224, 0.225])
                y = y[[0]]
                y = model.module.unpatchify(y)
                y = torch.einsum('nchw->nhwc', y).detach().cpu()
                mask = mask[[0]]
                mask = mask.detach().float().cpu()
                mask = mask.unsqueeze(-1).repeat(1, 1, model.module.patch_size ** 2 * 3)  # (N, H*W, p*p*3)
                mask = model.module.unpatchify(mask)  # 1 is removing, 0 is keeping
                mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
                x = samples[[0]]
                x = x.detach().float().cpu()
                x = torch.einsum('nchw->nhwc', x)
                tgt = targets[[0]]
                tgt = tgt.detach().float().cpu()
                tgt = torch.einsum('nchw->nhwc', tgt)
                im_masked = tgt * (1 - mask)

                frame = torch.cat((x, im_masked, y, tgt), dim=2)
                frame = frame[0]
                # print(frame.shape)
                frame = torch.clip((frame * imagenet_std + imagenet_mean) * 255, 0, 255).to(torch.uint8)
                #frame = frame[:, :, [2, 1, 0]]
                log_writer.add_image(f'x; im_masked; y; tgt', frame.numpy(), epoch_1000x, dataformats='HWC')

            # if global_rank == 0 and args.log_wandb:
            #     wandb.log({'train_loss': loss_value_reduce, 'lr': lr, 'train_loss_scale': loss_scale_value, 'grad_norm': grad_norm})
            #     if len(tensorboard_images) < 20:
            #         imagenet_mean = np.array([0.485, 0.456, 0.406])
            #         imagenet_std = np.array([0.229, 0.224, 0.225]) 
            #         y = y[[0]]
            #         y = model.module.unpatchify(y)
            #         y = torch.einsum('nchw->nhwc', y).detach().cpu()
            #         mask = mask[[0]]
            #         mask = mask.detach().float().cpu()
            #         mask = mask.unsqueeze(-1).repeat(1, 1, model.module.patch_size**2 *3)  # (N, H*W, p*p*3)
            #         mask = model.module.unpatchify(mask)  # 1 is removing, 0 is keeping
            #         mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
            #         x = samples[[0]]
            #         x = x.detach().float().cpu()
            #         x = torch.einsum('nchw->nhwc', x)
            #         tgt = targets[[0]]
            #         tgt = tgt.detach().float().cpu()
            #         tgt = torch.einsum('nchw->nhwc', tgt)
            #         im_masked = tgt * (1 - mask)
                    
            #         frame = torch.cat((x, im_masked, y, tgt), dim=2)
            #         frame = frame[0]
            #         frame = torch.clip((frame * imagenet_std + imagenet_mean) * 255, 0, 255).int()
            #         wandb_images.append(wandb.Image(frame.numpy(), caption="x; im_masked; y; tgt"))

    # if global_rank == 0 and args.log_wandb and len(wandb_images) > 0:
    #     wandb.log({"Training examples": wandb_images})


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_pt(data_loader, model, device, epoch=None, global_rank=None, args=None, log_writer=None, dataformats=None):
    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'
    # switch to evaluation mode
    model.eval()
    # wandb_images = []
    num_batch = 0
    num_tb_images = 0
    # rank 0 写 TB 比其他 rank 慢一个数量级，间隔写避免拖慢同步导致 NCCL timeout
    tb_save_every = 1
    val_tb_image_limit = getattr(args, "val_tb_image_limit", 0) if args is not None else 0
    val_tb_images_per_batch = max(
        1, getattr(args, "val_tb_images_per_batch", 1) if args is not None else 1
    )
    val_tb_image_freq = max(1, getattr(args, "val_tb_image_freq", 1) if args is not None else 1)
    write_tb_image_this_epoch = True
    if epoch is not None and val_tb_image_freq > 1:
        write_tb_image_this_epoch = (epoch + 1) % val_tb_image_freq == 0
        total_epochs = getattr(args, "epochs", None) if args is not None else None
        if total_epochs is not None:
            write_tb_image_this_epoch = write_tb_image_this_epoch or (epoch + 1 == total_epochs)
    for batch in metric_logger.log_every(data_loader, 10, header):

        samples = batch[0]
        targets = batch[1]
        bool_masked_pos = batch[2]
        valid = batch[3]
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss, loss_l1l2, loss_vgg, y, mask, pred = model(
                samples, targets, bool_masked_pos=bool_masked_pos,
                valid=valid, epoch=epoch, no_gan=args.no_gan
            )

        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_l1l2=loss_l1l2)
        metric_logger.update(loss_vgg=loss_vgg)
        """
            在tensorboard内展示图片nchw->nhwc
        """
        write_tb_image = (
            log_writer is not None
            and write_tb_image_this_epoch
            and num_batch % tb_save_every == 0
            and (val_tb_image_limit <= 0 or num_tb_images < val_tb_image_limit)
        )
        if write_tb_image:
            imagenet_mean = np.array([0.485, 0.456, 0.406])
            imagenet_std = np.array([0.229, 0.224, 0.225])
            batch_show_count = min(
                val_tb_images_per_batch,
                samples.shape[0],
                val_tb_image_limit - num_tb_images if val_tb_image_limit > 0 else samples.shape[0],
            )
            batch_show_count = max(0, batch_show_count)
            for image_idx in range(batch_show_count):
                y_show = y[[image_idx]]
                y_show = model.module.unpatchify(y_show)
                y_show = torch.einsum('nchw->nhwc', y_show).detach().cpu()
                mask_show = mask[[image_idx]]
                mask_show = mask_show.detach().float().cpu()
                mask_show = mask_show.unsqueeze(-1).repeat(
                    1, 1, model.module.patch_size ** 2 * 3
                )  # (N, H*W, p*p*3)
                mask_show = model.module.unpatchify(mask_show)  # 1 is removing, 0 is keeping
                mask_show = torch.einsum('nchw->nhwc', mask_show).detach().cpu()
                x_show = samples[[image_idx]]
                x_show = x_show.detach().float().cpu()
                x_show = torch.einsum('nchw->nhwc', x_show)
                tgt_show = targets[[image_idx]]
                tgt_show = tgt_show.detach().float().cpu()
                tgt_show = torch.einsum('nchw->nhwc', tgt_show)
                im_masked_show = tgt_show * (1 - mask_show)

                frame = torch.cat((x_show, im_masked_show, y_show, tgt_show), dim=2)
                frame = frame[0]
                frame = torch.clip(
                    (frame * imagenet_std + imagenet_mean) * 255, 0, 255
                ).to(torch.uint8)
                log_writer.add_image(
                    f'epoch:{epoch} val x; im_masked; y; tgt',
                    frame.numpy(),
                    num_batch * val_tb_images_per_batch + image_idx,
                    dataformats='HWC',
                )
                num_tb_images += 1
        num_batch += 1

        # if global_rank == 0 and args.log_wandb:
        #     imagenet_mean = np.array([0.485, 0.456, 0.406])
        #     imagenet_std = np.array([0.229, 0.224, 0.225])
        #     y = y[[0]]
        #     y = model.module.unpatchify(y)
        #     y = torch.einsum('nchw->nhwc', y).detach().cpu()
        #     mask = mask[[0]]
        #     mask = mask.detach().float().cpu()
        #     mask = mask.unsqueeze(-1).repeat(1, 1, model.module.patch_size**2 *3)  # (N, H*W, p*p*3)
        #     mask = model.module.unpatchify(mask)  # 1 is removing, 0 is keeping
        #     mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
        #     x = samples[[0]]
        #     x = x.detach().float().cpu()
        #     x = torch.einsum('nchw->nhwc', x)
        #     tgt = targets[[0]]
        #     tgt = tgt.detach().float().cpu()
        #     tgt = torch.einsum('nchw->nhwc', tgt)
        #     im_masked = tgt * (1 - mask)

        #     frame = torch.cat((x, im_masked, y, tgt), dim=2)
        #     frame = frame[0]
        #     frame = torch.clip((frame * imagenet_std + imagenet_mean) * 255, 0, 255).int()
        #     wandb_images.append(wandb.Image(frame.numpy(), caption="x; im_masked; y; tgt"))

    metric_logger.synchronize_between_processes()
    print('Val loss {losses.global_avg:.3f}'.format(losses=metric_logger.loss))

    out = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # if global_rank == 0 and args.log_wandb:
    #     wandb.log({**{f'test_{k}': v for k, v in out.items()},'epoch': epoch})
    #     if len(wandb_images) > 0:
    #         wandb.log({"Testing examples": wandb_images[::2][:20]})
    return out
