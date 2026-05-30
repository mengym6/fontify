"""
将 CVAT 导出的 COCO JSON 标注渲染为 448×448 单通道二值 mask 图。
每个笔画/结体 polygon 与 text 整字轮廓 RLE 取交集，去除背景溢出。
"""

import json
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_util

# ============================================================
# 模式选择（二选一）
# ============================================================

TEST_MODE = True        # True=测试模式（随机抽样可视化），False=批量处理模式

# ============================================================
# 测试模式配置
# ============================================================

TEST_FONT = "DongqcBF"  # 测试的字体文件夹名，None 则从所有字体中随机选
TEST_CHAR = None         # 指定字符名（不含.png），None 则随机抽取
TEST_COUNT = 3           # 随机测试数量（TEST_CHAR 为 None 时生效）

# ============================================================
# 批量模式配置
# ============================================================

CLEAR_OUTPUT = True      # True=运行前清空输出文件夹再重新生成，False=跳过已存在的文件

# ============================================================
# 路径配置
# ============================================================

# new/ 目录，包含所有字体文件夹（DongqcBF, LiugqJT, ...）
NEW_DIR = Path("/Users/root1/Desktop/Fontify-main/fontdata_example/font/train/new")

# 标注 JSON 所在子目录名
ANNOTATIONS_SUBDIR = "annotations"

# 标注 JSON 文件名
ANNOTATIONS_FILENAME = "instances_default.json"

# 原始图片子目录名（标注坐标基于此目录中的图片尺寸）
IMAGES_SUBDIR = "images"

# 输出 mask 子目录名
OUTPUT_SUBDIR = "semantic_masks"

# ============================================================
# 调参区
# ============================================================

# 输出 mask 尺寸（与训练图一致）
TARGET_SIZE = 448

# mask 输出值：0=不遮盖，255=遮盖
MASK_FG_VALUE = 255
MASK_BG_VALUE = 0


# ============================================================
# 核心函数
# ============================================================


def decode_rle(segmentation, h, w):
    """解码 RLE 格式的 segmentation 为二值 mask"""
    rle = segmentation
    if isinstance(rle['counts'], list):
        rle = mask_util.frPyObjects(rle, h, w)
    return mask_util.decode(rle).astype(np.uint8)


def render_polygon(segmentation, h, w):
    """渲染 polygon 格式的 segmentation 为二值 mask"""
    mask = np.zeros((h, w), dtype=np.uint8)
    for poly in segmentation:
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        pts = pts.astype(np.int32)
        cv2.fillPoly(mask, [pts], 1)
    return mask


def pad_and_resize_mask(mask, target_size):
    """与 pad_and_resize.py 相同的空间变换，但用 NEAREST 插值"""
    h, w = mask.shape[:2]
    max_side = max(h, w)
    # 居中 pad 到正方形（fill=0，即背景）
    square = np.zeros((max_side, max_side), dtype=np.uint8)
    offset_x = (max_side - w) // 2
    offset_y = (max_side - h) // 2
    square[offset_y:offset_y + h, offset_x:offset_x + w] = mask
    # resize 到目标尺寸
    resized = cv2.resize(square, (target_size, target_size),
                         interpolation=cv2.INTER_NEAREST)
    return resized


def extract_stroke_name(category_name):
    """从 category name 提取笔画名，如 '1-横-起笔' → '横'，'撇-收笔' → '撇'"""
    if category_name == 'text':
        return None
    # 去掉末尾阶段（起笔/中笔/收笔）
    base = category_name.rsplit('-', 1)[0]
    # 去掉开头数字前缀
    base = base.lstrip('0123456789-')
    return base if base else None


def render_single_image_mask(image_info, annotations, text_cat_id, categories):
    """为单张图按笔画分组渲染 mask，返回 (K, H, W) 数组，每层一个完整笔画"""
    h = image_info['height']
    w = image_info['width']
    img_id = image_info['id']

    # 构建 category_id → 笔画名 映射
    cat_id_to_stroke = {}
    for cat in categories:
        stroke = extract_stroke_name(cat['name'])
        if stroke:
            cat_id_to_stroke[cat['id']] = stroke

    # 筛选该图的所有标注
    img_anns = [a for a in annotations if a['image_id'] == img_id]
    if not img_anns:
        return None

    # 按笔画名分组，同名笔画的所有标注（含起中收、重复实例）合并为一层
    from collections import defaultdict
    stroke_groups = defaultdict(list)
    for ann in img_anns:
        if ann['category_id'] == text_cat_id:
            continue
        stroke = cat_id_to_stroke.get(ann['category_id'])
        if stroke:
            stroke_groups[stroke].append(ann)

    if not stroke_groups:
        return None

    # 每个笔画名生成一层 mask
    layers = []
    for stroke_name, anns in stroke_groups.items():
        combined = np.zeros((h, w), dtype=np.uint8)
        for ann in anns:
            seg = ann['segmentation']
            if isinstance(seg, dict):
                combined |= decode_rle(seg, h, w)
            else:
                combined |= render_polygon(seg, h, w)
        if combined.any():
            layers.append(combined)

    if not layers:
        return None

    return np.stack(layers, axis=0)  # (K, H, W)


def process_font_dir(font_dir: Path, verbose=False):
    """处理单个字体文件夹，渲染所有图的逐标注 mask 并保存为 .npy"""
    ann_path = font_dir / ANNOTATIONS_SUBDIR / ANNOTATIONS_FILENAME
    if not ann_path.exists():
        if verbose:
            print(f"  [跳过] {font_dir.name}：无标注文件")
        return 0, 0

    with open(ann_path, 'r') as f:
        data = json.load(f)

    categories = data['categories']
    images = data['images']
    annotations = data['annotations']

    # 找 text 类别 ID
    text_cats = [c for c in categories if c['name'] == 'text']
    if not text_cats:
        if verbose:
            print(f"  [跳过] {font_dir.name}：无 text 类别")
        return 0, 0
    text_cat_id = text_cats[0]['id']

    # 输出目录
    output_dir = font_dir / OUTPUT_SUBDIR

    # 清空已有输出
    if CLEAR_OUTPUT and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(exist_ok=True)

    success_count = 0
    skip_count = 0

    for img_info in images:
        file_name = img_info['file_name']
        char_name = Path(file_name).stem
        layers = render_single_image_mask(img_info, annotations, text_cat_id, categories)

        if layers is None:
            skip_count += 1
            if verbose:
                print(f"    [跳过] {file_name}：无有效标注")
            continue

        # 每层独立 pad + resize
        resized_layers = []
        for layer in layers:
            resized = pad_and_resize_mask(layer, TARGET_SIZE)
            resized_layers.append(resized)
        result = np.stack(resized_layers, axis=0)  # (N, 448, 448)

        # 保存为 .npy
        out_path = output_dir / f"{char_name}.npy"
        np.save(str(out_path), result.astype(np.uint8))
        success_count += 1

        if verbose:
            print(f"    {file_name} → {result.shape[0]} 层")

    return success_count, skip_count


# ============================================================
# 测试模式
# ============================================================


def test_single(font_dir: Path, img_info: dict, annotations: list, text_cat_id: int, categories: list):
    """测试单张图的逐标注 mask 渲染，输出合并 mask 和叠加对比图"""
    char_name = Path(img_info['file_name']).stem
    print(f"\n  [{font_dir.name}/{char_name}] 尺寸 {img_info['width']}×{img_info['height']}")

    layers = render_single_image_mask(img_info, annotations, text_cat_id)
    if layers is None:
        print("    无有效标注，跳过")
        return

    print(f"    标注数量：{layers.shape[0]}")

    # 逐层 pad+resize，合并后统计覆盖率
    resized_layers = []
    for layer in layers:
        resized = pad_and_resize_mask(layer, TARGET_SIZE)
        resized_layers.append(resized)
    result = np.stack(resized_layers, axis=0)  # (N, 448, 448)
    combined = np.any(result, axis=0).astype(np.uint8)

    coverage = np.sum(combined > 0) / (TARGET_SIZE ** 2) * 100
    print(f"    全部合并覆盖率：{coverage:.1f}%")

    # 加载原图做叠加对比
    orig_img_path = font_dir / IMAGES_SUBDIR / img_info['file_name']
    overlay = None
    orig_resized = None
    if orig_img_path.exists():
        orig = cv2.imread(str(orig_img_path), cv2.IMREAD_GRAYSCALE)
        h, w = orig.shape[:2]
        max_side = max(h, w)
        square = np.full((max_side, max_side), 255, dtype=np.uint8)
        ox, oy = (max_side - w) // 2, (max_side - h) // 2
        square[oy:oy+h, ox:ox+w] = orig
        orig_resized = cv2.resize(square, (TARGET_SIZE, TARGET_SIZE),
                                  interpolation=cv2.INTER_AREA)
        overlay = cv2.cvtColor(orig_resized, cv2.COLOR_GRAY2BGR)
        red_layer = np.zeros_like(overlay)
        red_layer[:, :, 2] = 255
        mask_bool = combined > 0
        overlay[mask_bool] = cv2.addWeighted(
            overlay, 0.6, red_layer, 0.4, 0)[mask_bool]

    # 保存
    test_output_dir = font_dir / "semantic_masks_test"
    test_output_dir.mkdir(exist_ok=True)

    mask_out = np.where(combined > 0, MASK_FG_VALUE, MASK_BG_VALUE).astype(np.uint8)
    cv2.imwrite(str(test_output_dir / f"{char_name}_mask.png"), mask_out)
    np.save(str(test_output_dir / f"{char_name}.npy"), result.astype(np.uint8))

    if overlay is not None:
        cv2.imwrite(str(test_output_dir / f"{char_name}_overlay.png"), overlay)
    if orig_resized is not None:
        cv2.imwrite(str(test_output_dir / f"{char_name}_orig.png"), orig_resized)

    print(f"    已保存到 {test_output_dir.name}/（含 .npy {result.shape}）")


def test_mode():
    """根据配置随机抽样测试"""
    # 确定要测试的字体文件夹
    if TEST_FONT is not None:
        font_dirs = [NEW_DIR / TEST_FONT]
    else:
        font_dirs = sorted([d for d in NEW_DIR.iterdir()
                            if d.is_dir() and not d.name.startswith(('_', '.'))])

    # 收集所有可用的 (font_dir, img_info, annotations, text_cat_id, categories)
    candidates = []
    for font_dir in font_dirs:
        ann_path = font_dir / ANNOTATIONS_SUBDIR / ANNOTATIONS_FILENAME
        if not ann_path.exists():
            continue
        with open(ann_path, 'r') as f:
            data = json.load(f)
        text_cats = [c for c in data['categories'] if c['name'] == 'text']
        if not text_cats:
            continue
        text_cat_id = text_cats[0]['id']
        for img_info in data['images']:
            candidates.append((font_dir, img_info, data['annotations'], text_cat_id, data['categories']))

    if not candidates:
        print("错误：未找到任何可用标注")
        return

    # 指定字符 or 随机抽样
    if TEST_CHAR is not None:
        target_file = f"{TEST_CHAR}.png"
        selected = [(fd, ii, anns, tid, cats) for fd, ii, anns, tid, cats in candidates
                    if ii['file_name'] == target_file]
        if not selected:
            print(f"错误：找不到 {target_file}")
            available = [ii['file_name'] for _, ii, _, _, _ in candidates[:10]]
            print(f"可用文件（前10）：{available}")
            return
    else:
        count = min(TEST_COUNT, len(candidates))
        selected = random.sample(candidates, count)

    print(f"测试数量：{len(selected)}")
    for font_dir, img_info, annotations, text_cat_id, categories in selected:
        test_single(font_dir, img_info, annotations, text_cat_id, categories)


# ============================================================
# 批量模式
# ============================================================


def batch_mode(verbose=False):
    """遍历 NEW_DIR 下所有字体文件夹，批量渲染语义 mask"""
    font_dirs = sorted([d for d in NEW_DIR.iterdir()
                        if d.is_dir() and not d.name.startswith(('_', '.'))])

    print(f"扫描目录：{NEW_DIR}")
    print(f"找到 {len(font_dirs)} 个字体文件夹")
    print("=" * 50)

    total_success = 0
    total_skip = 0

    for font_dir in font_dirs:
        print(f"\n[{font_dir.name}]")
        success, skip = process_font_dir(font_dir, verbose=verbose)
        total_success += success
        total_skip += skip
        if success > 0 or skip > 0:
            print(f"  完成：{success} 张生成，{skip} 张跳过")

    print("\n" + "=" * 50)
    print(f"全部完成。共生成 {total_success} 张 mask，跳过 {total_skip} 张。")


# ============================================================
# 入口
# ============================================================


if __name__ == "__main__":
    if TEST_MODE:
        print("=== 测试模式 ===")
        test_mode()
    else:
        print("=== 批量处理模式 ===")
        batch_mode(verbose=True)
