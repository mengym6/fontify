"""
convert_to_white_bg_cuda.py — PyTorch GPU 加速版白底黑字提取

依赖安装（AutoDL 服务器）：
    pip install kornia

torch 通常已预装。脚本自动检测 GPU，不可用时降级 CPU。
"""

import cv2
import numpy as np
import os
import shutil
import random
import torch
import torch.nn.functional as F

try:
    import kornia
    HAS_KORNIA = True
except ImportError:
    HAS_KORNIA = False
    print("警告: kornia 未安装, pip install kornia")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_DIR = r"/root/autodl-tmp/new"
INPUT_FOLDER = "images"
OUTPUT_SUFFIX = "_white_bg"

TEST_MODE = False
TEST_SAMPLE_NUM = 5

PARAMS = {
    "upscale_factor": 2,
    "bilateral_kernel": 9,
    "bilateral_sigma_color": 50,
    "bilateral_sigma_space": 50,
    "bg_kernel_ratio": 0.3,
    "contrast_alpha": 2.5,
    "contrast_beta": -30,
    "sigmoid_gain": 8,
    "sigmoid_cutoff": 0,
    "bg_threshold": 210,
    "text_threshold": 50,
    "median_ksize": 7,
    "edge_smooth_sigma": 0.8,
}

# ============================================================
# GPU 工具函数
# ============================================================

def to_tensor(img):
    """numpy灰度图(H,W) → GPU tensor (1,1,H,W) float32 [0,1]"""
    t = torch.from_numpy(img.astype(np.float32) / 255.0)
    return t.unsqueeze(0).unsqueeze(0).to(DEVICE)


def to_numpy(t):
    """GPU tensor (1,1,H,W) → numpy灰度图(H,W) uint8"""
    arr = t.squeeze().cpu().numpy()
    return np.clip(arr * 255, 0, 255).astype(np.uint8)


def gpu_resize(t, size, mode='bilinear'):
    """GPU resize, size=(H,W)"""
    return F.interpolate(t, size=size, mode=mode, align_corners=False)


def gpu_gaussian_blur(t, sigma):
    """GPU 高斯模糊"""
    if HAS_KORNIA and sigma > 0:
        ksize = int(np.ceil(sigma * 3)) * 2 + 1
        return kornia.filters.gaussian_blur2d(t, (ksize, ksize), (sigma, sigma))
    return t


def gpu_morphology_close(t, ksize):
    """GPU 形态学闭运算 — 可分离 max_pool2d 实现，内存高效"""
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    pad = ksize // 2
    # 膨胀（separable: 水平 + 垂直）
    dilated = F.max_pool2d(t, kernel_size=(1, ksize), stride=1, padding=(0, pad))
    dilated = F.max_pool2d(dilated, kernel_size=(ksize, 1), stride=1, padding=(pad, 0))
    # 腐蚀（separable）
    eroded = -F.max_pool2d(-dilated, kernel_size=(1, ksize), stride=1, padding=(0, pad))
    eroded = -F.max_pool2d(-eroded, kernel_size=(ksize, 1), stride=1, padding=(pad, 0))
    return eroded


def gpu_median_blur(t, ksize):
    """GPU 中值滤波"""
    if HAS_KORNIA:
        return kornia.filters.median_blur(t, (ksize, ksize))
    return t

# ============================================================
# 核心处理
# ============================================================

def detect_polarity(gray):
    h, w = gray.shape
    border_size = max(h, w) // 10
    border_pixels = np.concatenate([
        gray[:border_size, :].ravel(), gray[-border_size:, :].ravel(),
        gray[:, :border_size].ravel(), gray[:, -border_size:].ravel(),
    ])
    center = gray[h // 4:3 * h // 4, w // 4:3 * w // 4]
    return border_pixels.mean() < center.mean() - 10


def process_single(img_path, params):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    h_orig, w_orig = img.shape
    scale = params["upscale_factor"]

    # 放大（GPU）
    t = to_tensor(img)
    if scale > 1:
        t = gpu_resize(t, (h_orig * scale, w_orig * scale))

    # 双边滤波（GPU，kornia）
    if HAS_KORNIA:
        ks = params["bilateral_kernel"]
        ks = ks if ks % 2 == 1 else ks + 1
        sc = params["bilateral_sigma_color"] / 255.0
        ss = float(params["bilateral_sigma_space"])
        t = kornia.filters.bilateral_blur(t, (ks, ks), sc, (ss, ss))

    # 下载做极性检测
    img_up = to_numpy(t)
    if detect_polarity(img_up):
        img_up = 255 - img_up
        t = to_tensor(img_up)

    # 背景估计：大核闭运算 + 高斯模糊（GPU）
    h, w = img_up.shape
    ksize = int(min(h, w) * params["bg_kernel_ratio"])
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    t_bg = gpu_morphology_close(t, ksize)
    t_bg = gpu_gaussian_blur(t_bg, ksize / 3.0)
    bg = to_numpy(t_bg)
    # 光照归一化
    img_up = to_numpy(t)
    bg_f = bg.astype(np.float32) + 1e-5
    norm = img_up.astype(np.float32) / bg_f * 128.0
    norm = np.clip(norm, 0, 255).astype(np.uint8)

    # 线性对比度增强
    norm = cv2.convertScaleAbs(norm, alpha=params["contrast_alpha"],
                               beta=params["contrast_beta"])
    contrast_out = norm.copy()

    # 中值滤波（GPU）
    mk = params["median_ksize"]
    if mk > 0:
        mk = mk if mk % 2 == 1 else mk + 1
        t_norm = to_tensor(norm)
        t_norm = gpu_median_blur(t_norm, mk)
        norm = to_numpy(t_norm)

    # Otsu 分割点
    cutoff = params["sigmoid_cutoff"]
    if cutoff == 0:
        cutoff, _ = cv2.threshold(norm, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # sigmoid 映射
    x = norm.astype(np.float32)
    sig = 1.0 / (1.0 + np.exp(-params["sigmoid_gain"] * (x - cutoff) / 255.0))
    result = (sig * 255.0).astype(np.uint8)

    # 背景/前景清理
    result[result > params["bg_threshold"]] = 255
    result[result < params["text_threshold"]] = 0

    # 闭运算填笔画内部白洞（GPU）
    t_ink = to_tensor((result < 128).astype(np.uint8) * 255)
    t_closed = gpu_morphology_close(t_ink, 7)
    ink = to_numpy(t_ink)
    ink_closed = to_numpy(t_closed)
    holes = ink_closed & (~ink)
    result[holes == 255] = 0

    # 边缘抗锯齿（GPU 高斯模糊）
    sigma = params["edge_smooth_sigma"]
    if sigma > 0:
        t_r = to_tensor(result)
        t_r = gpu_gaussian_blur(t_r, sigma)
        result = to_numpy(t_r)

    # 缩回原尺寸
    if scale > 1:
        result = cv2.resize(result, (w_orig, h_orig),
                            interpolation=cv2.INTER_AREA)
        contrast_out = cv2.resize(contrast_out, (w_orig, h_orig),
                                  interpolation=cv2.INTER_AREA)
    return contrast_out, result

# ============================================================
# 主程序
# ============================================================

def main():
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

    input_dirs = []
    for name in sorted(os.listdir(ROOT_DIR)):
        sub = os.path.join(ROOT_DIR, name)
        img_dir = os.path.join(sub, INPUT_FOLDER)
        if os.path.isdir(sub) and os.path.isdir(img_dir):
            input_dirs.append((name, img_dir))

    if not input_dirs:
        print(f"未找到包含 '{INPUT_FOLDER}' 的子文件夹: {ROOT_DIR}")
        return

    print(f"根目录: {ROOT_DIR}")
    print(f"设备: {DEVICE} ({'GPU' if DEVICE.type == 'cuda' else 'CPU'})")
    print(f"kornia: {'可用' if HAS_KORNIA else '不可用'}")
    print(f"发现 {len(input_dirs)} 个待处理文件夹")
    print("=" * 50)

    total_success, total_files = 0, 0

    for folder_name, img_dir in input_dirs:
        parent = os.path.dirname(img_dir)
        output_dir = os.path.join(parent, INPUT_FOLDER + OUTPUT_SUFFIX)
        contrast_dir = os.path.join(parent, INPUT_FOLDER + "_contrast")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(contrast_dir, exist_ok=True)

        files = sorted([f for f in os.listdir(img_dir)
                        if f.lower().endswith(exts)])
        if not files:
            continue
        if TEST_MODE:
            n = min(TEST_SAMPLE_NUM, len(files))
            files = sorted(random.sample(files, n))

        print(f"\n[{folder_name}] {len(files)} 张")
        success = 0
        for i, fname in enumerate(files, 1):
            path = os.path.join(img_dir, fname)
            contrast_img, result = process_single(path, PARAMS)
            if result is not None:
                cv2.imwrite(os.path.join(output_dir, fname), result)
                cv2.imwrite(os.path.join(contrast_dir, fname), contrast_img)
                success += 1
            else:
                print(f"  [跳过] {fname}")
            if i % 50 == 0 or i == len(files):
                print(f"  进度: {i}/{len(files)}")

        print(f"  完成: {success}/{len(files)}")
        total_success += success
        total_files += len(files)

    print("\n" + "=" * 50)
    print(f"全部完成，共处理 {total_success}/{total_files} 张")


if __name__ == "__main__":
    main()
