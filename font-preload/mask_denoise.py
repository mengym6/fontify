import cv2
import numpy as np
import os
import shutil
import random

# ============================================================
# 路径配置
# ============================================================
# new 文件夹根路径，脚本会自动遍历其中每个字体子目录
NEW_DIR = r"/Users/root1/Desktop/Fontify-main/fontdata_example/font/train/new"

# 每个字体子目录中的子文件夹名
CLEAN_SUBDIR = "images_white_bg_v1"   # 干净图（用于生成 mask）
NOISY_SUBDIR = "images_white_bg"      # 噪点图（待处理）

# 输出文件夹自动在字体子目录内创建
OUTPUT_SUFFIX = "_mask_denoised"

# ============================================================
# 运行模式
# ============================================================
TEST_MODE = False          # True=测试模式(随机抽样), False=批量处理全部
TEST_SAMPLE_NUM = 5       # 测试模式下随机抽取的图片数量

# ============================================================
# 可调参数
# ============================================================
PARAMS = {
    # --- 二值化（对图1） ---
    # 0=Otsu自动阈值，>0=手动指定(0~255)
    "bin_threshold": 0,

    # --- 膨胀参数（控制 mask 比实际笔画大多少） ---
    # dilate_kernel_size: 膨胀结构元素大小(奇数)，建议 3~15
    "dilate_kernel_size": 7,
    # dilate_iterations: 膨胀迭代次数，建议 1~5
    "dilate_iterations": 4,
    # kernel_shape: "ellipse" 或 "rect"
    "kernel_shape": "ellipse",

    # --- mask 羽化（控制边缘软硬） ---
    # feather_sigma: 0=硬边缘, 1~3=柔和过渡, 3~5=更宽灰色过渡带
    "feather_sigma": 10.0,

    # --- 填补笔画内部白色小洞 ---
    # hole_max_area: 面积小于此值的白色连通域视为小洞并填充
    #   建议 50~2000。0=关闭
    "hole_max_area": 1500,
    # hole_fill_mode: "erode"=从周围笔画像素扩散, "fixed"=固定灰度值
    "hole_fill_mode": "erode",
    # hole_fill_kernel: erode模式下腐蚀核大小
    "hole_fill_kernel": 5,
    # hole_fill_value: fixed模式下的灰度值
    "hole_fill_value": 0,
}


# ============================================================
# 核心函数
# ============================================================

def setup_output_dir(noisy_dir, suffix):
    """在 NOISY_DIR 同级创建输出目录（已存在则清空重建）"""
    parent = os.path.dirname(noisy_dir)
    name = os.path.basename(noisy_dir)
    output_dir = os.path.join(parent, name + suffix)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    return output_dir


def binarize_clean(gray, threshold):
    """对干净图做二值化，返回前景 mask（文字=255，背景=0）"""
    if threshold == 0:
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(
            gray, threshold, 255, cv2.THRESH_BINARY_INV)
    return binary


def dilate_mask(binary, params):
    """对二值前景做形态学膨胀，扩展出安全边距"""
    ksize = params["dilate_kernel_size"]
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    shape_map = {
        "ellipse": cv2.MORPH_ELLIPSE,
        "rect": cv2.MORPH_RECT,
    }
    shape = shape_map.get(params["kernel_shape"], cv2.MORPH_ELLIPSE)
    kernel = cv2.getStructuringElement(shape, (ksize, ksize))
    return cv2.dilate(binary, kernel, iterations=params["dilate_iterations"])


def feather_mask(mask, sigma):
    """对 mask 做高斯模糊实现边缘羽化，返回 0~1 浮点 mask"""
    mask_f = mask.astype(np.float32) / 255.0
    if sigma > 0:
        ksize = int(np.ceil(sigma * 3)) * 2 + 1
        mask_f = cv2.GaussianBlur(mask_f, (ksize, ksize), sigma)
    return mask_f


def apply_mask(noisy_gray, mask_soft):
    """将羽化 mask 应用到噪点图：mask 外推白，mask 内保留原像素"""
    noisy_f = noisy_gray.astype(np.float32)
    result = noisy_f * mask_soft + 255.0 * (1.0 - mask_soft)
    return np.clip(result, 0, 255).astype(np.uint8)


def fill_small_holes(result, stroke_mask, params):
    """用干净图的 mask 定位笔画内部，将其中亮色小洞填黑（腐蚀后避开边缘）"""
    max_area = params["hole_max_area"]
    if max_area <= 0:
        return result

    # 腐蚀 mask，向内收缩，避免触碰边缘抗锯齿过渡带
    erode_k = params["hole_fill_kernel"]
    erode_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (erode_k, erode_k))
    inner_mask = cv2.erode(stroke_mask, erode_kernel, iterations=1)

    hole_thresh = 50
    holes = (inner_mask == 255) & (result > hole_thresh)
    result[holes] = 0
    return result


def process_single(clean_path, noisy_path, params):
    """处理单对图片，返回去噪结果"""
    clean = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
    noisy = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
    if clean is None or noisy is None:
        return None

    h, w = noisy.shape
    if clean.shape != noisy.shape:
        clean = cv2.resize(clean, (w, h), interpolation=cv2.INTER_AREA)

    binary = binarize_clean(clean, params["bin_threshold"])
    dilated = dilate_mask(binary, params)
    mask_soft = feather_mask(dilated, params["feather_sigma"])
    result = apply_mask(noisy, mask_soft)
    result = fill_small_holes(result, binary, params)
    return result


# ============================================================
# 主程序
# ============================================================

def process_font_dir(clean_dir, noisy_dir, params):
    """处理单个字体目录"""
    output_dir = setup_output_dir(noisy_dir, OUTPUT_SUFFIX)
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

    clean_files = set(f for f in os.listdir(clean_dir)
                      if f.lower().endswith(exts))
    noisy_files = set(f for f in os.listdir(noisy_dir)
                      if f.lower().endswith(exts))
    common = sorted(clean_files & noisy_files)

    if not common:
        print(f"  未找到配对图片，跳过")
        return 0

    if TEST_MODE:
        n = min(TEST_SAMPLE_NUM, len(common))
        files = sorted(random.sample(common, n))
    else:
        files = common

    success = 0
    for i, fname in enumerate(files, 1):
        clean_path = os.path.join(clean_dir, fname)
        noisy_path = os.path.join(noisy_dir, fname)
        result = process_single(clean_path, noisy_path, params)
        if result is not None:
            cv2.imwrite(os.path.join(output_dir, fname), result)
            success += 1
        else:
            print(f"    [跳过] {fname} (读取失败)")
        if i % 50 == 0 or i == len(files):
            print(f"    进度: {i}/{len(files)}")

    return success


def main():
    font_dirs = sorted([
        d for d in os.listdir(NEW_DIR)
        if os.path.isdir(os.path.join(NEW_DIR, d)) and not d.startswith(('_', '.'))
    ])

    mode_str = "测试模式" if TEST_MODE else "批量处理"
    print(f"根目录: {NEW_DIR}")
    print(f"运行模式: {mode_str}")
    print(f"发现 {len(font_dirs)} 个字体目录")
    print("=" * 50)

    total_success = 0
    for idx, font_name in enumerate(font_dirs, 1):
        font_path = os.path.join(NEW_DIR, font_name)
        clean_dir = os.path.join(font_path, CLEAN_SUBDIR)
        noisy_dir = os.path.join(font_path, NOISY_SUBDIR)

        if not os.path.isdir(clean_dir) or not os.path.isdir(noisy_dir):
            print(f"[{idx}/{len(font_dirs)}] {font_name} — 缺少子目录，跳过")
            continue

        print(f"[{idx}/{len(font_dirs)}] {font_name}")
        n = process_font_dir(clean_dir, noisy_dir, PARAMS)
        total_success += n
        print(f"    完成 {n} 张")

    print("=" * 50)
    print(f"全部完成，共处理 {total_success} 张图片")


if __name__ == "__main__":
    main()
