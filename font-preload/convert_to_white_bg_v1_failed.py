import cv2
import numpy as np
import os

input_folder = "/Users/root1/Desktop/Fontify-main/fontdata_example/font/train/new/颜真卿结体/images"

parent_dir = os.path.dirname(input_folder)
folder_name = os.path.basename(input_folder)
output_folder = os.path.join(parent_dir, folder_name + "_white_bg_v1")
os.makedirs(output_folder, exist_ok=True)

exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
files = [f for f in os.listdir(input_folder) if f.lower().endswith(exts)]

for fname in files:
    img = cv2.imread(os.path.join(input_folder, fname), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    h, w = img.shape
    # 中值滤波去石头纹理
    blurred = cv2.medianBlur(img, 19)
    # Otsu 二值化
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 闭运算填补笔画空洞
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
    # 连通域过滤去噪点
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    min_area = h * w * 0.002
    filtered = np.zeros_like(closed)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == i] = 255
    # 开运算去掉边缘小突起和毛刺
    circ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, circ)
    # 高斯模糊不二值化，保留灰度抗锯齿
    smooth = cv2.GaussianBlur(filtered, (7, 7), 0)
    # 反色：白底黑字
    result = 255 - smooth
    cv2.imwrite(os.path.join(output_folder, fname), result)

print(f"完成，共处理 {len(files)} 张图片")
print(f"输出目录: {output_folder}")
