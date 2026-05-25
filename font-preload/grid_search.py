import os
import sys
import shutil
import random
import itertools
import time
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np

# 把主程序所在目录加入 path，以便 import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_to_white_bg import process_single, PARAMS as BASE_PARAMS

# ============================================================
# 路径配置
# ============================================================
INPUT_DIR = "/root/autodl-tmp/images"

# 网格搜索输出文件夹（和 INPUT_DIR 同级）
OUTPUT_DIR_NAME = "grid_search_results"

# 测试用图片数量（从输入文件夹随机抽取）
SAMPLE_NUM = 5

# 并行进程数（0=自动使用全部 CPU 核心）
NUM_WORKERS = 0

# ============================================================
# 搜索参数范围（只需改这里）
# ============================================================
SEARCH_SPACE = {
    "clahe_clip": [0, 2.0, 3.0, 4.0],
    "sigmoid_gain": [8, 10, 12, 15, 18],
    "bg_threshold": [130, 150, 170, 190],
    "text_threshold": [60, 80, 100, 130],
    "median_ksize": [0, 3, 5, 7],
}


# ============================================================
# 工具函数
# ============================================================

def make_combo_name(combo):
    """把参数组合转成文件夹名"""
    parts = []
    for k, v in sorted(combo.items()):
        short = k.replace("clahe_clip", "clip") \
                 .replace("sigmoid_gain", "gain") \
                 .replace("bg_threshold", "bgt") \
                 .replace("text_threshold", "txtt") \
                 .replace("median_ksize", "med")
        parts.append(f"{short}{v}")
    return "_".join(parts)


def run_one_combo(args):
    """单个参数组合的处理（供多进程调用）"""
    combo, sample_files, input_dir, output_root = args
    params = dict(BASE_PARAMS)
    params.update(combo)

    folder_name = make_combo_name(combo)
    combo_dir = os.path.join(output_root, folder_name)
    os.makedirs(combo_dir, exist_ok=True)

    for fname in sample_files:
        path = os.path.join(input_dir, fname)
        result = process_single(path, params)
        if result is not None:
            cv2.imwrite(os.path.join(combo_dir, fname), result)

    return folder_name


# ============================================================
# 主程序
# ============================================================

def main():
    # 准备输出根目录
    parent = os.path.dirname(INPUT_DIR)
    output_root = os.path.join(parent, OUTPUT_DIR_NAME)
    if os.path.exists(output_root):
        shutil.rmtree(output_root)
    os.makedirs(output_root)

    # 获取样本图片
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    all_files = sorted([f for f in os.listdir(INPUT_DIR)
                        if f.lower().endswith(exts)])
    if not all_files:
        print(f"未找到图片: {INPUT_DIR}")
        return

    n = min(SAMPLE_NUM, len(all_files))
    sample_files = sorted(random.sample(all_files, n))

    # 生成所有参数组合
    keys = sorted(SEARCH_SPACE.keys())
    values = [SEARCH_SPACE[k] for k in keys]
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    total = len(combos)

    # 确定进程数
    workers = NUM_WORKERS if NUM_WORKERS > 0 else cpu_count()

    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {output_root}")
    print(f"样本图片: {sample_files}")
    print(f"搜索参数: {keys}")
    print(f"总组合数: {total}")
    print(f"并行进程: {workers}")
    print("=" * 50)

    # 构造任务列表
    tasks = [(combo, sample_files, INPUT_DIR, output_root)
             for combo in combos]

    t0 = time.time()
    done = 0
    with Pool(processes=workers) as pool:
        for _ in pool.imap_unordered(run_one_combo, tasks):
            done += 1
            if done % 20 == 0 or done == total:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(f"  [{done}/{total}] 耗时{elapsed:.0f}s "
                      f"剩余~{eta:.0f}s")

    print("=" * 50)
    print(f"网格搜索完成! 共 {total} 组, 耗时 {time.time()-t0:.1f}s")
    print(f"输出目录: {output_root}")


if __name__ == "__main__":
    main()
