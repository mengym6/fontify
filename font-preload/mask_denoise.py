import cv2
import numpy as np
import os
import shutil
import random

# ============================================================
# 路径配置
# ============================================================
# 图1：干净的白底黑字图（用于生成 mask）
CLEAN_DIR = r"/Users/root1/Desktop/Fontify-main/fontdata_example/font/train/new/颜真卿结体/images_white_bg_v1"
# 图2：有噪点的待处理图（mask 外区域将被推白）
NOISY_DIR = r"/Users/root1/Desktop/Fontify-main/fontdata_example/font/train/new/颜真卿结体/images_white_bg"

# 输出文件夹自动在 NOISY_DIR 同级创建
OUTPUT_SUFFIX = "_mask_denoised"

# ============================================================
# 运行模式
# ============================================================
TEST_MODE = True          # True=测试模式(随机抽样), False=批量处理全部
TEST_SAMPLE_NUM = 2       # 测试模式下随机抽取的图片数量

# ============================================================
# 可调参数
# ============================================================
PARAMS = {
    # --- 二值化（对图1） ---
    # bin_threshold: 图1 二值化阈值。图1 已经很干净，128 通常足够
    #   0=使用 Otsu 自动阈值，>0=手动指定 (0~255)
    #   往小调→更多像素被判为文字前景，往大调→只保留最黑的部分
    "bin_threshold": 0,

    # --- 膨胀参数（核心：控制 mask 比实际笔画大多少） ---
    # dilate_kernel_size: 膨胀结构元素大小(奇数)
    #   决定单次膨胀扩展的像素范围。建议 3~15
    #   太小→笔画边缘灰度过渡带被切掉，太大→噪点也被包进 mask
    "dilate_kernel_size": 7,
    # dilate_iterations: 膨胀迭代次数
    #   总扩展像素 ≈ kernel_radius × iterations。建议 1~5
    "dilate_iterations": 2,
    # kernel_shape: 结构元素形状
    #   "ellipse"=椭圆(更自然), "rect"=矩形(扩展更均匀)
    "kernel_shape": "ellipse",

    # --- mask 羽化（控制 mask 边缘软硬） ---
    # feather_sigma: mask 边缘高斯模糊σ
    #   0=硬边缘(可能有截断感), 1~3=柔和过渡。建议 1.0~2.0
    "feather_sigma": 1.5,

    # --- mask 内加深（将笔画内灰色区域压暗） ---
    # darken_gamma: mask 内像素的 gamma 校正值
    #   公式: output = 255 * (pixel/255)^gamma
    #   >1 中间灰度往黑压，值越大压得越狠；1.0=不变；<1 反而变亮
    #   建议 1.5~3.0。1.5=轻微加深, 2.0=适中, 3.0=很重。0=关闭加深
    "darken_gamma": 1.5,

    # --- 边缘平滑（闭运算填平笔画边缘锯齿） ---
    # smooth_kernel_size: 闭运算核大小(奇数)
    #   填平笔画边缘 1~2px 的锯齿凹陷。越大填平越多但笔画会变粗
    #   建议 3~5。3=轻微, 5=明显。0=关闭
    "smooth_kernel_size": 7,
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
        # Otsu 自动阈值
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
    dilated = cv2.dilate(binary, kernel,
                         iterations=params["dilate_iterations"])
    return dilated


def feather_mask(mask, sigma):
    """对 mask 做高斯模糊实现边缘羽化，返回 0~1 浮点 mask"""
    mask_f = mask.astype(np.float32) / 255.0
    if sigma > 0:
        ksize = int(np.ceil(sigma * 3)) * 2 + 1
        mask_f = cv2.GaussianBlur(mask_f, (ksize, ksize), sigma)
    return mask_f


def darken_ink(gray, mask_binary, gamma):
    """对 mask 内像素做 gamma 校正压暗，保留灰度层次"""
    if gamma == 0 or gamma == 1.0:
        return gray
    result = gray.copy()
    # 只处理 mask 内的像素
    roi = result[mask_binary == 255].astype(np.float32) / 255.0
    roi = np.power(roi, gamma) * 255.0
    result[mask_binary == 255] = roi.astype(np.uint8)
    return result


def apply_mask(noisy_gray, mask_soft):
    """将羽化 mask 应用到噪点图：mask 外推白，mask 内保留原像素"""
    # output = img2 * mask + 255 * (1 - mask)
    noisy_f = noisy_gray.astype(np.float32)
    result = noisy_f * mask_soft + 255.0 * (1.0 - mask_soft)
    return np.clip(result, 0, 255).astype(np.uint8)


def process_single(clean_path, noisy_path, params):
    """处理单对图片，返回去噪结果"""
    # 步骤1：读入两张图（灰度）
    clean = cv2.imread(clean_path, cv2.IMREAD_GRAYSCALE)
    noisy = cv2.imread(noisy_path, cv2.IMREAD_GRAYSCALE)
    if clean is None or noisy is None:
        return None

    # 步骤2：尺寸对齐（将 clean resize 到 noisy 的尺寸）
    h, w = noisy.shape
    if clean.shape != noisy.shape:
        clean = cv2.resize(clean, (w, h), interpolation=cv2.INTER_AREA)

    # 步骤3：图1 二值化，得到前景 mask（文字=255，背景=0）
    binary = binarize_clean(clean, params["bin_threshold"])

    # 步骤4：膨胀，扩展出安全边距
    dilated = dilate_mask(binary, params)

    # 步骤5：mask 内加深，gamma 压暗保留灰度层次
    noisy = darken_ink(noisy, dilated, params["darken_gamma"])

    # 步骤6：mask 羽化，避免硬切割痕迹
    mask_soft = feather_mask(dilated, params["feather_sigma"])

    # 步骤7：应用 mask 到图2，mask 外推白
    result = apply_mask(noisy, mask_soft)

    # 步骤8：闭运算填平笔画边缘锯齿
    sk = params["smooth_kernel_size"]
    if sk > 0:
        sk = sk if sk % 2 == 1 else sk + 1
        k_smooth = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sk, sk))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, k_smooth)

    return result


# ============================================================
# 主程序
# ============================================================

def main():
    output_dir = setup_output_dir(NOISY_DIR, OUTPUT_SUFFIX)
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

    # 取两个目录的交集（同名文件才配对处理）
    clean_files = set(f for f in os.listdir(CLEAN_DIR)
                      if f.lower().endswith(exts))
    noisy_files = set(f for f in os.listdir(NOISY_DIR)
                      if f.lower().endswith(exts))
    common = sorted(clean_files & noisy_files)

    if not common:
        print(f"未找到配对图片")
        print(f"  干净图目录: {CLEAN_DIR}")
        print(f"  噪点图目录: {NOISY_DIR}")
        return

    # 测试模式：随机抽样
    if TEST_MODE:
        n = min(TEST_SAMPLE_NUM, len(common))
        files = sorted(random.sample(common, n))
        mode_str = f"测试模式 (随机 {n} 张)"
    else:
        files = common
        mode_str = "批量处理 (全部)"

    print(f"干净图目录: {CLEAN_DIR}")
    print(f"噪点图目录: {NOISY_DIR}")
    print(f"输出目录:   {output_dir}")
    print(f"运行模式:   {mode_str}")
    print(f"配对文件:   {len(common)} 张, 本次处理: {len(files)} 张")
    if TEST_MODE:
        print(f"抽样文件:   {files}")
    print("-" * 40)

    success = 0
    for i, fname in enumerate(files, 1):
        clean_path = os.path.join(CLEAN_DIR, fname)
        noisy_path = os.path.join(NOISY_DIR, fname)
        result = process_single(clean_path, noisy_path, PARAMS)
        if result is not None:
            cv2.imwrite(os.path.join(output_dir, fname), result)
            success += 1
        else:
            print(f"  [跳过] {fname} (读取失败)")
        if i % 10 == 0 or i == len(files):
            print(f"  进度: {i}/{len(files)}")

    print("-" * 40)
    print(f"完成，成功处理 {success}/{len(files)} 张")


if __name__ == "__main__":
    main()
