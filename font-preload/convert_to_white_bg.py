import cv2
import numpy as np
import os
import shutil
import random
from multiprocessing import Pool, cpu_count

# ============================================================
# 路径配置
# ============================================================
INPUT_DIR = r"/Users/root1/Desktop/Fontify-main/fontdata_example/font/train/new/颜真卿结体/images"

# 输出文件夹自动在 INPUT_DIR 同级创建，名为 <原文件夹名>_white_bg
OUTPUT_SUFFIX = "_white_bg"

# ============================================================
# 运行模式：改这里切换测试/批量
# ============================================================
TEST_MODE = True          # True=测试模式(随机抽样), False=批量处理全部
TEST_SAMPLE_NUM = 5       # 测试模式下随机抽取的图片数量

# ============================================================
# 可调参数
# ============================================================
PARAMS = {
    # --- 预处理 ---
    # upscale_factor: 放大倍数，越大细节越好但越慢。建议 1~3
    "upscale_factor": 2,
    # bilateral_d/sigma: 双边滤波，保边去噪。d 越大越慢，sigma 越大模糊越强
    "bilateral_d": 9,
    "bilateral_sigma_color": 50,
    "bilateral_sigma_space": 50,

    # --- 背景估计与光照归一化 ---
    # bg_kernel_ratio: 核大小=短边×ratio。越大背景越平滑，太小覆盖不了拓片深色边框
    #   建议 0.2~0.4。图像边缘暗区大时调大
    "bg_kernel_ratio": 0.3,

    # --- 线性对比度增强 DB = alpha * DA + beta ---
    # alpha: 斜率, >1 增强对比度。建议 1.5~3.0
    "contrast_alpha": 2.5,
    # beta: 截距, 负值使整体变暗(文字更黑)。建议 -50~0
    "contrast_beta": -30,

    # --- sigmoid 对比度映射（核心参数）---
    # sigmoid_gain: 控制黑白过渡的陡峭程度。越大越接近二值，越小边缘越柔和
    #   建议范围: 8~20。8=很柔和, 12=适中, 20=接近硬二值
    "sigmoid_gain": 8,
    # sigmoid_cutoff: 分割点偏移。0=用 Otsu 自动找, >0 手动指定 (0~255)
    #   往小调→保留更多笔画(但可能多噪点), 往大调→去掉更多(但可能丢细笔画)
    "sigmoid_cutoff": 0,

    # --- 背景清理 ---
    # bg_threshold: sigmoid 输出中灰度值 > 此值的像素直接变纯白。越小去噪越狠
    #   建议 180~230。180=激进去噪, 200=适中, 230=只去最淡的噪点
    "bg_threshold": 210,
    # text_threshold: sigmoid 输出中灰度值 < 此值的像素直接变纯黑。越大字越黑越实
    #   建议 30~100。30=只加深最暗部分, 60=适中, 100=激进加深
    "text_threshold": 50,
    # median_ksize: 中值滤波核大小(奇数)。专杀孤立小噪点，不影响大面积笔画
    #   建议 3~7。3=轻微, 5=适中, 7=激进(可能磨掉极细笔画)。0=关闭
    "median_ksize": 7,

    # --- 边缘抗锯齿 ---
    # edge_smooth_sigma: 缩回原尺寸前的高斯模糊σ，重建边缘灰度过渡带
    #   建议 0.6~1.2。0=关闭。在2x尺寸下操作，缩回后影响约1px
    "edge_smooth_sigma": 0.8,
}

# ============================================================
# 核心函数
# ============================================================

def setup_output_dir(input_dir, suffix):
    """创建输出目录（已存在则清空重建）"""
    parent = os.path.dirname(input_dir)
    name = os.path.basename(input_dir)
    output_dir = os.path.join(parent, name + suffix)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    return output_dir


def estimate_background(gray, params):
    """形态学闭运算估计背景光照"""
    h, w = gray.shape
    ksize = int(min(h, w) * params["bg_kernel_ratio"])
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    bg = cv2.GaussianBlur(bg, (ksize, ksize), 0)
    return bg


def detect_polarity(gray):
    """检测文字极性：返回 True 表示需要反色（原图为亮字暗底）"""
    h, w = gray.shape
    border_size = max(h, w) // 10
    border_pixels = np.concatenate([
        gray[:border_size, :].ravel(),
        gray[-border_size:, :].ravel(),
        gray[:, :border_size].ravel(),
        gray[:, -border_size:].ravel(),
    ])
    center = gray[h // 4:3 * h // 4, w // 4:3 * w // 4]
    return border_pixels.mean() < center.mean() - 10


def sigmoid_map(gray, cutoff, gain):
    """用 sigmoid 曲线把灰度图映射为白底黑字，边缘天然平滑"""
    x = gray.astype(np.float32)
    # sigmoid: 1/(1+exp(-gain*(x-cutoff)/255))  归一化到 0~1
    # x < cutoff → 接近 0 (黑), x > cutoff → 接近 1 (白)
    t = gain * (x - cutoff) / 255.0
    t = np.clip(t, -20, 20)
    sig = 1.0 / (1.0 + np.exp(-t))
    result = (sig * 255.0).astype(np.uint8)
    return result


def process_single(img_path, params):
    """处理单张图片，返回 (对比度增强中间图, 最终白底黑字结果)"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None
    h_orig, w_orig = img.shape

    # 放大处理
    scale = params["upscale_factor"]
    if scale > 1:
        img = cv2.resize(img, (w_orig * scale, h_orig * scale),
                         interpolation=cv2.INTER_CUBIC)

    # 双边滤波保边去噪
    img = cv2.bilateralFilter(img, params["bilateral_d"],
                              params["bilateral_sigma_color"],
                              params["bilateral_sigma_space"])

    # 极性检测：如果是亮字暗底，先反色
    if detect_polarity(img):
        img = 255 - img

    # 背景估计 + 光照归一化（除以背景，消除光照不均）
    bg = estimate_background(img, params)
    bg_f = bg.astype(np.float32) + 1e-5
    norm = img.astype(np.float32) / bg_f * 128.0
    norm = np.clip(norm, 0, 255).astype(np.uint8)

    # 线性对比度增强: DB = alpha * DA + beta
    alpha = params["contrast_alpha"]
    beta = params["contrast_beta"]
    norm = cv2.convertScaleAbs(norm, alpha=alpha, beta=beta)

    # 保存对比度增强中间结果（缩回原尺寸）
    if scale > 1:
        contrast_out = cv2.resize(norm, (w_orig, h_orig), interpolation=cv2.INTER_AREA)
    else:
        contrast_out = norm.copy()

    # 中值滤波前移：此时噪点还是孤立小黑点，中值滤波最有效
    mk = params["median_ksize"]
    if mk > 0:
        mk = mk if mk % 2 == 1 else mk + 1
        norm = cv2.medianBlur(norm, mk)

    # 确定分割点：Otsu 自动 或 手动指定
    cutoff = params["sigmoid_cutoff"]
    if cutoff == 0:
        cutoff, _ = cv2.threshold(norm, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # sigmoid 映射：暗→黑，亮→白，边缘平滑过渡
    result = sigmoid_map(norm, cutoff, params["sigmoid_gain"])

    # 背景清理：大于阈值的灰色噪点直接推到纯白
    bg_thresh = params["bg_threshold"]
    result[result > bg_thresh] = 255

    # 字体加深：小于阈值的灰色像素直接推到纯黑（边缘灰→黑）
    text_thresh = params["text_threshold"]
    result[result < text_thresh] = 0

    # 闭运算只填笔画内部白洞，不覆盖边缘
    ink = (result < 128).astype(np.uint8) * 255
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    ink_closed = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, k_close, iterations=2)
    holes_filled = ink_closed & (~ink)
    result[holes_filled == 255] = 0

    # 边缘抗锯齿：轻度高斯模糊重建灰度过渡带
    sigma = params["edge_smooth_sigma"]
    if sigma > 0:
        ksize = int(np.ceil(sigma * 3)) * 2 + 1
        result = cv2.GaussianBlur(result, (ksize, ksize), sigma)

    # 缩回原尺寸
    if scale > 1:
        result = cv2.resize(result, (w_orig, h_orig),
                            interpolation=cv2.INTER_AREA)
    return contrast_out, result


# ============================================================
# 主程序
# ============================================================

def _worker(args):
    """多进程 worker：处理单张图并写文件，返回 (文件名, 是否成功)"""
    fname, input_dir, output_dir, contrast_dir, params = args
    path = os.path.join(input_dir, fname)
    contrast_img, result = process_single(path, params)
    if result is not None:
        cv2.imwrite(os.path.join(output_dir, fname), result)
        cv2.imwrite(os.path.join(contrast_dir, fname), contrast_img)
        return fname, True
    return fname, False


def main():
    output_dir = setup_output_dir(INPUT_DIR, OUTPUT_SUFFIX)
    contrast_dir = setup_output_dir(INPUT_DIR, "_contrast")
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    files = sorted([f for f in os.listdir(INPUT_DIR)
                    if f.lower().endswith(exts)])

    if not files:
        print(f"未找到图片文件: {INPUT_DIR}")
        return

    # 测试模式：随机抽样
    if TEST_MODE:
        n = min(TEST_SAMPLE_NUM, len(files))
        files = sorted(random.sample(files, n))
        mode_str = f"测试模式 (随机 {n} 张)"
    else:
        mode_str = "批量处理 (全部)"

    num_workers = min(cpu_count(), len(files))
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {output_dir}")
    print(f"对比度中间图: {contrast_dir}")
    print(f"运行模式: {mode_str}")
    print(f"待处理: {len(files)} 张, 进程数: {num_workers}")
    if TEST_MODE:
        print(f"抽样文件: {files}")
    print("-" * 40)

    tasks = [(f, INPUT_DIR, output_dir, contrast_dir, PARAMS)
             for f in files]

    success = 0
    with Pool(num_workers) as pool:
        for i, (fname, ok) in enumerate(
                pool.imap_unordered(_worker, tasks), 1):
            if ok:
                success += 1
            else:
                print(f"  [跳过] {fname} (读取失败)")
            if i % 10 == 0 or i == len(files):
                print(f"  进度: {i}/{len(files)}")

    print("-" * 40)
    print(f"完成，成功处理 {success}/{len(files)} 张")


if __name__ == "__main__":
    main()
