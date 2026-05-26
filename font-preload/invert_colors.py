import cv2
import os

# ============================================================
# 路径配置
# ============================================================
INPUT_DIR = r"/Users/root1/Desktop/Fontify-main/fontdata_example/font/train/new/赵孟頫结体/images_white_bg_v1"

# ============================================================
# 主程序
# ============================================================
def main():
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
    files = sorted(f for f in os.listdir(INPUT_DIR) if f.lower().endswith(exts))

    if not files:
        print(f"未找到图片: {INPUT_DIR}")
        return

    print(f"输入目录: {INPUT_DIR}")
    print(f"图片数量: {len(files)}")
    print("-" * 40)

    for i, fname in enumerate(files, 1):
        path = os.path.join(INPUT_DIR, fname)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  [跳过] {fname} (读取失败)")
            continue
        img_inv = 255 - img
        cv2.imwrite(path, img_inv)
        if i % 10 == 0 or i == len(files):
            print(f"  进度: {i}/{len(files)}")

    print("-" * 40)
    print(f"完成，已反转 {len(files)} 张图片（覆盖原文件）")


if __name__ == "__main__":
    main()
