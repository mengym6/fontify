import sys
import os
import warnings
import json

import requests
import argparse

import torch
import torch.nn.functional as F
import numpy as np
import glob
import tqdm

import matplotlib.pyplot as plt
from PIL import Image

sys.path.append('.')
import models_eval

import torch.distributed as dist
import torch.multiprocessing as mp

import random

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def get_args_parser():
    parser = argparse.ArgumentParser('Font_generation', add_help=False)
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt',
                        default='')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='vit_base_patch16_input896x448_win_dec64_8glb_sl1')
    parser.add_argument('--prompt_json', type=str, help='path to prompt json file',
                        default=os.getenv('PROMPT_JSON', 'prompts.json'))
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--num_gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for processing')
    parser.add_argument('--out_dir', type=str, default='', help='Output dir')
    parser.add_argument('--ref_dir', type=str, default='', help='Reference font dir')
    parser.add_argument('--source_dir', type=str, default='', help='Source font dir')
    parser.add_argument('--gen_dir', type=str, default='', help='Dir of the font to be generated')
    return parser.parse_args()


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29555'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def set_seed(seed, rank):
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def prepare_model(chkpt_dir, arch='vit_base_patch16_input896x448_win_dec64_8glb_sl1', rank=0):
    model = getattr(models_eval, arch)()
    checkpoint = torch.load(chkpt_dir, map_location='cuda:{}'.format(rank))
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)

    device = torch.device("cuda:{}".format(rank))
    model.to(device)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    return model


def run_batch_images(imgs, tgts, sizes, model, device):
    x = torch.stack([torch.tensor(img).permute(2, 0, 1) for img in imgs]).float().to(device)
    tgt = torch.stack([torch.tensor(tgt).permute(2, 0, 1) for tgt in tgts]).float().to(device)

    x = x.to(device)
    tgt = tgt.to(device)

    bool_masked_pos = torch.zeros(model.module.patch_embed.num_patches)
    bool_masked_pos[model.module.patch_embed.num_patches // 2:] = 1
    bool_masked_pos = bool_masked_pos.unsqueeze(dim=0).repeat(len(imgs), 1)

    valid = torch.ones_like(tgt)
    with torch.cuda.amp.autocast():
        y, mask, pred = model(x.float().to(device), tgt.float().to(device),
                               bool_masked_pos.to(device), valid.float().to(device))
    y = model.module.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu().numpy()

    outputs = []
    for i, size in enumerate(sizes):
        output = y[i]
        output = output[y.shape[1] // 2:, :, :]
        output = output * imagenet_std + imagenet_mean
        output = F.interpolate(
            torch.tensor(output).permute(2, 0, 1).unsqueeze(0), size=[size[1], size[0]], mode='bicubic').permute(0, 2,
                                                                                                                 3, 1)[
            0].numpy()
        outputs.append(output)
    return outputs


def main(rank, world_size):
    setup(rank, world_size)
    device = torch.device("cuda:{}".format(rank))

    base_seed = 42
    set_seed(base_seed, rank)

    args = get_args_parser()

    # 获取路径
    style_dir = args.ref_dir
    font_dir = args.gen_dir
    img_src_dir = args.source_dir
    out_dir = args.out_dir
    
    # 验证路径存在性
    if not os.path.exists(style_dir):
        print(f"ERROR: Reference directory does not exist: {style_dir}")
        return
    if not os.path.exists(font_dir):
        print(f"ERROR: Generation directory does not exist: {font_dir}")
        return
    if not os.path.exists(img_src_dir):
        print(f"ERROR: Source directory does not exist: {img_src_dir}")
        return
    
    print(f"Reference dir: {style_dir}")
    print(f"Generation dir: {font_dir}")
    print(f"Source dir: {img_src_dir}")
    print(f"Output dir: {out_dir}")

    # 加载prompt列表
    try:
        with open(args.prompt_json, 'r') as f:
            prompt_list = json.load(f)
    except Exception as e:
        print(f"Error loading prompt list: {e}")
        return

    model_fontify = prepare_model(args.ckpt_path, args.model, rank)
    model_fontify.eval()

    font_folders = [os.path.join(font_dir, d) for d in os.listdir(font_dir) if os.path.isdir(os.path.join(font_dir, d))]
    font_folders_split = font_folders[rank::world_size]

    for font_folder in font_folders_split:
        font_name = os.path.basename(font_folder)
        print(f'Rank {rank} processing font: {font_name}')

        out_path_dir = os.path.join(out_dir, font_name)
        os.makedirs(out_path_dir, exist_ok=True)

        # 查找第一个可用的prompt
        selected_prompt = None
        missing_files = []
        for prompt in prompt_list:
            prompt_path = os.path.join(style_dir, font_name, f"{prompt}.png")
            img2_path = os.path.join(img_src_dir, f"{prompt}.png")
            
            # 调试信息：记录缺失的文件
            if not os.path.exists(prompt_path):
                missing_files.append(f"ref: {prompt_path}")
            if not os.path.exists(img2_path):
                missing_files.append(f"src: {img2_path}")
                
            if os.path.exists(prompt_path) and os.path.exists(img2_path):
                selected_prompt = prompt
                break

        if selected_prompt is None:
            print(f"No valid prompt found for {font_name}")
            print(f"Available prompts: {prompt_list[:10]}")  # 显示前10个prompt
            print(f"Missing files for first few prompts:")
            for i, missing in enumerate(missing_files[:6]):
                print(f"  {i+1}. {missing}")
            print(f"Style dir: {style_dir}")
            print(f"Font name: {font_name}")
            print(f"Source dir: {img_src_dir}")
            
            # 检查字体文件夹是否存在
            font_ref_path = os.path.join(style_dir, font_name)
            if not os.path.exists(font_ref_path):
                print(f"ERROR: Font reference folder does not exist: {font_ref_path}")
            else:
                ref_files = os.listdir(font_ref_path)
                print(f"Reference font has {len(ref_files)} files")
                print(f"First few reference files: {ref_files[:5]}")
            continue

        print(f"Selected prompt: {selected_prompt} for font {font_name}(Rank {rank})")

        # 加载选中的prompt图片
        prompt_path = os.path.join(style_dir, font_name, f"{selected_prompt}.png")
        img2_path = os.path.join(img_src_dir, f"{selected_prompt}.png")
        tgt2 = Image.open(prompt_path).convert("RGB")
        tgt2 = tgt2.resize((args.input_size, args.input_size))
        tgt2 = np.array(tgt2) / 255.
        tgt2 = tgt2 - imagenet_mean
        tgt2 = tgt2 / imagenet_std

        img2 = Image.open(img2_path).convert("RGB")
        img2 = img2.resize((args.input_size, args.input_size))
        img2 = np.array(img2) / 255.
        img2 = img2 - imagenet_mean
        img2 = img2 / imagenet_std

        tgt = np.concatenate((tgt2, tgt2), axis=0)

        tgt_img_names = [os.path.basename(path) for path in glob.glob(os.path.join(font_folder, "*.*"))]
        img_path_list = glob.glob(os.path.join(img_src_dir, "*.png")) + glob.glob(os.path.join(img_src_dir, "*.jpg"))
        img_path_list = [img_path for img_path in img_path_list if os.path.basename(img_path) in tgt_img_names]

        for i in range(0, len(img_path_list), args.batch_size):
            batch_paths = img_path_list[i:i + args.batch_size]
            imgs = []
            tgts = []
            sizes = []
            out_paths = []

            for img_path in batch_paths:
                img_name = os.path.basename(img_path)
                if img_name in tgt_img_names:
                    out_path = os.path.join(out_path_dir, img_name)
                    img_org = Image.open(img_path).convert("RGB")
                    size = img_org.size
                    img = img_org.resize((args.input_size, args.input_size))
                    img = np.array(img) / 255.
                    img = img - imagenet_mean
                    img = img / imagenet_std
                    img = np.concatenate((img2, img), axis=0)

                    imgs.append(img)
                    tgts.append(tgt)
                    sizes.append(size)
                    out_paths.append(out_path)

            if imgs:
                outputs = run_batch_images(imgs, tgts, sizes, model_fontify, device)
                for out_path, output in zip(out_paths, outputs):
                    rgb_restored = np.clip(output, 0, 1)
                    output = rgb_restored * 255
                    output = Image.fromarray(output.astype(np.uint8))
                    output.convert('RGB').save(out_path, 'PNG', quality=95)


if __name__ == '__main__':
    args = get_args_parser()
    num_gpus = args.num_gpus
    mp.spawn(main, args=(num_gpus,), nprocs=num_gpus, join=True)