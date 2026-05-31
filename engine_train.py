import math
import os
import sys
from typing import Iterable

import torch
import torch.nn.functional as F
import util.misc as misc
import util.lr_sched as lr_sched

import numpy as np
#import wandb

import time


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
                    optimizer_d=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

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
                requires_grad_original = {}
                for name, param in model.module.named_parameters():
                    requires_grad_original[name] = param.requires_grad
                    if 'discriminator' not in name:
                        param.requires_grad = False

                model.module.discriminator.requires_grad_(True)
                optimizer_d.zero_grad()

                real_imgs = model.module.resize(targets)
                real_output = model.module.discriminator(real_imgs)
                real_loss = F.binary_cross_entropy_with_logits(real_output, torch.ones_like(real_output))

                fake_imgs = model.module.resize(pred.detach())
                fake_output = model.module.discriminator(fake_imgs)
                fake_loss = F.binary_cross_entropy_with_logits(fake_output, torch.zeros_like(fake_output))

                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()
                optimizer_d.step()

                model.module.discriminator.requires_grad_(False)
                for name, param in model.module.named_parameters():
                    param.requires_grad = requires_grad_original[name]

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

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
            grad_norm = loss_scaler(loss, optimizer, clip_grad=args.clip_grad,
                                    parameters=model.parameters(),
                                    update_grad=(data_iter_step + 1) % accum_iter == 0)

            if (data_iter_step + 1) % accum_iter == 0:
                optimizer.zero_grad()
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()
        #print(f"loss:{loss},grad_norm:{grad_norm}")
        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        metric_logger.update(loss_scale=loss_scale_value)
        metric_logger.update(grad_norm=grad_norm)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        loss_l1l2_reduce = misc.all_reduce_mean(loss_l1l2)
        loss_vgg_reduce = misc.all_reduce_mean(loss_vgg)

        if log_writer is not None and grad_norm is not None:
            with open(os.path.join(args.output_dir, "log_detail.txt"), mode="a", encoding="utf-8") as f:
                f.write(
                    f"[{time.time()}] Epoch: [{epoch}]  [{data_iter_step}/{len(data_loader)}]  lr: {lr}  loss: {loss}   "
                    f"loss_scale_value: {loss_scale_value}  grad_norm: {grad_norm} \n")
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            log_writer.add_scalars('train_loss_detail', {
                'loss_l1l2': loss_l1l2_reduce,
                'loss_vgg': loss_vgg_reduce
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
    # rank 0 写 TB 比其他 rank 慢一个数量级，间隔写避免拖慢同步导致 NCCL timeout
    tb_save_every = 20
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
                valid=valid, epoch=999, no_gan=args.no_gan
            )

        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_l1l2=loss_l1l2)
        metric_logger.update(loss_vgg=loss_vgg)
        """
            在tensorboard内展示图片nchw->nhwc
        """
        if log_writer is not None and num_batch % tb_save_every == 0:
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
            frame = torch.clip((frame * imagenet_std + imagenet_mean) * 255, 0, 255).to(torch.uint8)
            log_writer.add_image(f'epoch:{epoch} val x; im_masked; y; tgt', frame.numpy(), num_batch, dataformats='HWC')
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
