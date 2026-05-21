import sys
import os
import warnings
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import glob
import tqdm
from PIL import Image

sys.path.append('.')
import models_eval

imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


def get_args_parser():
    parser = argparse.ArgumentParser('Font_generation', add_help=False)
    parser.add_argument('--ckpt_path', type=str, help='path to ckpt',
                        default='')
    parser.add_argument('--model', type=str, help='dir to ckpt',
                        default='vit_base_patch16_input896x448_win_dec64_8glb_sl1')
    parser.add_argument('--input_size', type=int, default=448)
    parser.add_argument('--reference_font_dir', type=str, help='directory containing reference font styles',
                        default='font_datasets/reference')
    parser.add_argument('--img_src_dir', type=str, help='directory containing source images',
                        default='fontdata_example/font/train/source/')
    parser.add_argument('--output_dir', type=str, help='directory to save generated images',
                        default='font_datasets/generated_fonts')
    return parser.parse_args()


def prepare_model(chkpt_dir, arch='vit_base_patch16_input896x448_win_dec64_8glb_sl1'):
    model = getattr(models_eval, arch)()
    checkpoint = torch.load(chkpt_dir, map_location='cuda:0')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    print(msg)
    return model


def preprocess_image(img_path, input_size, resize_size):
    # Load and resize the image to 64x64 first
    img = Image.open(img_path).convert("RGB")
    img = img.resize((resize_size, resize_size), Image.BICUBIC)
    # Then resize to the required input size (e.g., 448x448 if needed)
    img = img.resize((input_size, input_size), Image.BICUBIC)
    img = np.array(img) / 255.0
    img = (img - imagenet_mean) / imagenet_std
    return img


def run_one_image(img, tgt, size, model, device):
    x = torch.tensor(img)
    x = x.unsqueeze(dim=0)
    x = torch.einsum('nhwc->nchw', x)

    tgt = torch.tensor(tgt)
    tgt = tgt.unsqueeze(dim=0)
    tgt = torch.einsum('nhwc->nchw', tgt)

    bool_masked_pos = torch.zeros(model.patch_embed.num_patches)
    bool_masked_pos[model.patch_embed.num_patches // 2:] = 1
    bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)

    valid = torch.ones_like(tgt)
    y, mask, pred = model(x.float().to(device), tgt.float().to(device), bool_masked_pos.to(device),
                                               valid.float().to(device))
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()
    x = torch.einsum('nchw->nhwc', x).detach().cpu()

    output = y[0, y.shape[1] // 2:, :, :]
    output = output * imagenet_std + imagenet_mean
    output = F.interpolate(
        output[None, ...].permute(0, 3, 1, 2), size=[size[1], size[0]], mode='bicubic').permute(0, 2, 3, 1)[0]
    return output.numpy()


def run_single_character(char, reference_path, model, device, img_src_dir, output_dir, input_size):
    resize_size = 64  # Resize reference image to 64x64 first
    img2_name = os.path.basename(reference_path)
    img2_path = os.path.join(img_src_dir, img2_name)
    img2 = Image.open(img2_path).convert("RGB")
    img2 = img2.resize((input_size, input_size))
    img2 = np.array(img2) / 255.
    img2 = img2 - imagenet_mean
    img2 = img2 / imagenet_std

    img_path = os.path.join(img_src_dir, f"{char}.png")
    img_org = Image.open(img_path).convert("RGB")
    size = img_org.size
    img = img_org.resize((input_size, input_size))
    img = np.array(img) / 255.
    img = img - imagenet_mean
    img = img / imagenet_std

    img = np.concatenate((img2, img), axis=0)

    tgt2 = Image.open(reference_path).convert("RGB")
    tgt2 = tgt2.resize((64, 64))
    tgt2 = tgt2.resize((input_size, input_size))
    tgt2 = np.array(tgt2) / 255.
    tgt2 = tgt2 - imagenet_mean
    tgt2 = tgt2 / imagenet_std

    tgt = np.concatenate((tgt2, tgt2), axis=0)

    output = run_one_image(img, tgt, size, model, device)
    output = np.clip(output, 0, 1)
    output = output * 255
    output = Image.fromarray(output.astype(np.uint8))

    return output


def main():
    args = get_args_parser()

    ckpt_path = args.ckpt_path
    model_name = args.model
    input_size = args.input_size
    reference_font_dir = args.reference_font_dir
    img_src_dir = args.img_src_dir
    output_dir = args.output_dir

    print(f"Please place square reference style images in {reference_font_dir}. Name each image with its corresponding character, such as '夜.png'.")

    # Check reference font directory
    if not os.path.exists(reference_font_dir):
        raise FileNotFoundError(
            f"Reference font directory {reference_font_dir} not found. Please place square reference style images here.")

    model_fontify = prepare_model(ckpt_path, model_name)
    print('Model loaded.')

    device = torch.device("cuda")
    model_fontify.to(device)
    model_fontify.eval()

    input_text = input("Enter the text you want to generate (e.g., 螳螂捕蝉): ").strip()
    if not input_text:
        print("No text provided. Exiting...")
        return

    reference_paths = glob.glob(os.path.join(reference_font_dir, "*.png")) + glob.glob(
        os.path.join(reference_font_dir, "*.jpg"))

    for reference_path in reference_paths:
        reference_char = os.path.splitext(os.path.basename(reference_path))[0]
        print(f"Processing reference style '{reference_char}'")

        # Generate images for each character and concatenate them
        generated_images = []
        for char in input_text:
            try:
                output = run_single_character(char, reference_path, model_fontify, device, img_src_dir, output_dir,
                                              input_size)
                generated_images.append(output)
            except FileNotFoundError as e:
                print(e)
                continue

        # Concatenate images horizontally
        total_width = sum(img.width for img in generated_images)
        max_height = max(img.height for img in generated_images)
        combined_image = Image.new("RGB", (total_width, max_height))

        current_x = 0
        for img in generated_images:
            combined_image.paste(img, (current_x, 0))
            current_x += img.width

        # Save the combined image
        out_path = os.path.join(output_dir, f"{reference_char}_{input_text}.png")
        combined_image.save(out_path, 'PNG', quality=95)
        print(f"Generated combined image saved to {out_path}")

    print(f"All generated images are saved in {output_dir}")


if __name__ == '__main__':
    main()