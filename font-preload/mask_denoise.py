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
TEST_SAMPLE_NUM = 5       # 测试模式下随机抽取的图片数量

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
    "darken_gamma": 2.0,

    # --- 边缘平滑（闭运算填平笔画边缘锯齿） ---
    # smooth_kernel_size: 闭运算核大小(奇数)
    #   填平笔画边缘 1~2px 的锯齿凹陷。越大填平越多但笔画会变粗
    #   建议 3~5。3=轻微, 5=明显。0=关闭
    "smooth_kernel_size": 9,

    # --- 轮廓平滑补齐（取锯齿外边缘，高斯平滑轮廓坐标后补齐凹陷） ---
    # contour_smooth_sigma: 轮廓点坐标的一维高斯平滑σ(单位:轮廓点数)
    #   越大→边缘越平滑，拐角越圆。建议 3~8
    #   0=关闭此步骤
    "contour_smooth_sigma": 12,
    # contour_min_area: 忽略面积小于此值的轮廓（过滤噪点碎片）
    #   建议 50~200
    "contour_min_area": 100,
    # contour_fill_mode: 补齐区域灰度值来源
    #   "erode"=从相邻笔画像素扩散(最自然), "fixed"=固定灰度值
    "contour_fill_mode": "erode",
    # contour_fill_kernel: erode模式下灰度腐蚀核大小
    #   建议 3~7
    "contour_fill_kernel": 7,
    # contour_fill_value: fixed模式下的灰度值(0=纯黑)
    "contour_fill_value": 20,

    # --- 填补笔画内部白色小洞（连通域定位，只填小洞不动边缘） ---
    # hole_max_area: 面积小于此值的白色连通域视为小洞并填充
    #   建议 50~500。越大→填的洞越多，太大会把笔画间隙也填上
    #   0=关闭此步骤
    "hole_max_area": 200,
    # hole_fill_mode: 填充灰度来源
    #   "erode"=从周围笔画像素扩散, "fixed"=固定灰度值
    "hole_fill_mode": "erode",
    # hole_fill_kernel: erode模式下腐蚀核大小
    "hole_fill_kernel": 5,
    # hole_fill_value: fixed模式下的灰度值
    "hole_fill_value": 0,

    # --- 灰色像素加深（将笔画内残留灰色压暗） ---
    # gray_darken_gamma: 对笔画区域内灰色像素再做一次gamma加深
    #   只作用于非白(< 250)非黑(> 5)的灰色像素
    #   建议 1.5~3.0。1.0=不变，0=关闭
    "gray_darken_gamma": 2.2,

    # --- 灰色轮廓清除（笔画外围残留灰色推白） ---
    # outline_clean_threshold: 二值化阈值，低于此值的像素视为笔画核心
    #   建议 80~150。越大→保护区越大（保留更多灰度过渡）
    #   0=关闭此步骤
    "outline_clean_threshold": 100,
    # outline_protect_size: 保护区膨胀核大小(奇数)
    #   笔画核心向外扩展的保护范围，保护区内的灰色不清除
    #   建议 3~7。越大→保留越多边缘灰度
    "outline_protect_size": 3,

    # --- 边缘平滑（高斯模糊保留灰度过渡） ---
    # edge_blur_sigma: 对最终结果做轻度高斯模糊，柔化边缘锯齿
    #   保留灰度抗锯齿，不做硬阈值切割
    #   建议 0.5~1.0。0=关闭
    "edge_blur_sigma": 1.0,
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


def _gaussian_kernel_1d(sigma):
    """生成归一化的一维高斯核"""
    radius = int(np.ceil(sigma * 3))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def _smooth_contour_coords(contour, sigma):
    """对闭合轮廓点坐标做环形一维高斯平滑"""
    pts = contour.reshape(-1, 2).astype(np.float64)
    n = len(pts)
    if n < 5:
        return contour

    kernel = _gaussian_kernel_1d(sigma)
    radius = len(kernel) // 2

    # 环形填充（wrap）：首尾相接
    x = np.pad(pts[:, 0], radius, mode='wrap')
    y = np.pad(pts[:, 1], radius, mode='wrap')

    # 一维卷积
    x_smooth = np.convolve(x, kernel, mode='valid')
    y_smooth = np.convolve(y, kernel, mode='valid')

    smoothed = np.stack([x_smooth, y_smooth], axis=1)
    return smoothed.astype(np.int32).reshape(-1, 1, 2)


def smooth_edge_contour(result, binary_mask, params):
    """轮廓坐标高斯平滑：平滑轮廓替换原边缘，凸出削白，凹陷填色"""
    sigma = params["contour_smooth_sigma"]
    if sigma <= 0:
        return result

    min_area = params["contour_min_area"]

    # 提取轮廓（RETR_CCOMP：外轮廓+内轮廓/holes）
    contours, hierarchy = cv2.findContours(
        binary_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

    if not contours or hierarchy is None:
        return result

    # 用平滑轮廓重建 smooth_mask
    smooth_mask = np.zeros_like(binary_mask)
    for i, cnt in enumerate(contours):
        parent = hierarchy[0][i][3]
        is_hole = parent >= 0

        if cv2.contourArea(cnt) < min_area:
            if is_hole:
                cv2.drawContours(smooth_mask, [cnt], -1, 0, cv2.FILLED)
            else:
                cv2.drawContours(smooth_mask, [cnt], -1, 255, cv2.FILLED)
            continue

        smoothed = _smooth_contour_coords(cnt, sigma)
        if is_hole:
            # 内轮廓：挖洞（填0）
            cv2.drawContours(smooth_mask, [smoothed], -1, 0, cv2.FILLED)
        else:
            # 外轮廓：填充前景
            cv2.drawContours(smooth_mask, [smoothed], -1, 255, cv2.FILLED)

    # 补齐区域（凹陷）：smooth_mask有但original没有
    fill_region = cv2.bitwise_and(
        smooth_mask, cv2.bitwise_not(binary_mask))
    # 削除区域（凸出）：original有但smooth_mask没有
    trim_region = cv2.bitwise_and(
        binary_mask, cv2.bitwise_not(smooth_mask))

    # 凹陷处填色
    if cv2.countNonZero(fill_region) > 0:
        if params["contour_fill_mode"] == "erode":
            fk = params["contour_fill_kernel"]
            fk = fk if fk % 2 == 1 else fk + 1
            fill_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (fk, fk))
            eroded = cv2.erode(result, fill_kernel, iterations=1)
            result[fill_region == 255] = eroded[fill_region == 255]
        else:
            result[fill_region == 255] = params["contour_fill_value"]

    # 凸出处推白 + 清除smooth_mask外围残留灰色过渡带
    if cv2.countNonZero(trim_region) > 0:
        result[trim_region == 255] = 255

    # smooth_mask外围一圈内的灰色残留像素也推白
    border_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    smooth_expanded = cv2.dilate(smooth_mask, border_k, iterations=1)
    border_band = cv2.bitwise_and(
        smooth_expanded, cv2.bitwise_not(smooth_mask))
    # 在这个外围带内，非白像素推白
    gray_residue = (border_band == 255) & (result < 250)
    result[gray_residue] = 255

    return result


def fill_small_holes(result, params):
    """连通域分析定位笔画内部白色小洞并填充，不影响外边缘"""
    max_area = params["hole_max_area"]
    if max_area <= 0:
        return result

    # 阈值化得到前景mask（笔画=255）
    _, ink_mask = cv2.threshold(
        result, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 对背景（ink_mask取反）做连通域分析
    bg_mask = cv2.bitwise_not(ink_mask)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        bg_mask, connectivity=8)

    # 找出面积小于阈值的背景连通域（排除label=0即最大背景）
    hole_mask = np.zeros_like(bg_mask)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < max_area:
            hole_mask[labels == i] = 255

    if cv2.countNonZero(hole_mask) == 0:
        return result

    # 膨胀小洞区域，覆盖周围灰色过渡带
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    hole_mask = cv2.dilate(hole_mask, dilate_k, iterations=1)
    # 只扩展到非纯黑区域（不侵入已有笔画）
    hole_mask[result == 0] = 0

    # 填充小洞+过渡带
    if params["hole_fill_mode"] == "erode":
        fk = params["hole_fill_kernel"]
        fk = fk if fk % 2 == 1 else fk + 1
        fill_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (fk, fk))
        eroded = cv2.erode(result, fill_kernel, iterations=1)
        result[hole_mask == 255] = eroded[hole_mask == 255]
    else:
        result[hole_mask == 255] = params["hole_fill_value"]

    return result


def clean_outline_residue(result, params):
    """清除笔画外围残留灰色：灰度 > threshold 的像素全部推白"""
    thresh = params["outline_clean_threshold"]
    if thresh <= 0:
        return result

    result[result > thresh] = 255
    return result


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

    # 步骤8.5：轮廓平滑补齐（取锯齿外边缘，补齐凹陷）
    if params["contour_smooth_sigma"] > 0:
        _, ink_mask = cv2.threshold(
            result, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        result = smooth_edge_contour(result, ink_mask, params)

    # 步骤8.6：填补笔画内部白色小洞
    if params["hole_max_area"] > 0:
        result = fill_small_holes(result, params)

    # 步骤8.7：灰色像素加深
    gamma2 = params["gray_darken_gamma"]
    if gamma2 > 0 and gamma2 != 1.0:
        gray_mask = (result > 5) & (result < 250)
        pixels = result[gray_mask].astype(np.float32) / 255.0
        pixels = np.power(pixels, gamma2) * 255.0
        result[gray_mask] = pixels.astype(np.uint8)

    # 步骤9：轻度高斯模糊柔化边缘，保留灰度抗锯齿
    sigma = params["edge_blur_sigma"]
    if sigma > 0:
        ksize = int(np.ceil(sigma * 3)) * 2 + 1
        result = cv2.GaussianBlur(result, (ksize, ksize), sigma)

    # 步骤10：清除笔画外围残留灰色轮廓线（必须在模糊之后）
    if params["outline_clean_threshold"] > 0:
        result = clean_outline_residue(result, params)

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
