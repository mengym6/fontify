"""
验证图片拼接：模拟 PairDataset 的 transform + 拼接流程，保存结果图片
用法：修改下方 source_path 和 style_path 为你的实际图片路径，运行后查看输出图片
"""
from PIL import Image
import numpy as np
import os

# ===== 修改这里 =====
source_path = "../fontdata_example/font/train/source/茗.png"
style_path = "../fontdata_example/font/train/chinese/JianYu Zhang Excellent Pen Chinese Font-Simplified Chinese Fonts/块.png"
output_path = "./verify_concat_result.png"
input_size = 448
# ====================

def pad_to_square(img, fill=255):
    """模拟 PadToSquare"""
    w, h = img.size
    max_side = max(w, h)
    new_img = Image.new('RGB', (max_side, max_side), (fill, fill, fill))
    paste_x = (max_side - w) // 2
    paste_y = (max_side - h) // 2
    new_img.paste(img, (paste_x, paste_y))
    return new_img

def process_and_concat(source_path, style_path, size):
    """加载两张图，resize 到统一尺寸，竖向拼接"""
    src = Image.open(source_path).convert('RGB')
    sty = Image.open(style_path).convert('RGB')

    print(f"原始尺寸 - source: {src.size}, style: {sty.size}")

    # pad + resize
    src = pad_to_square(src).resize((size, size), Image.BICUBIC)
    sty = pad_to_square(sty).resize((size, size), Image.BICUBIC)

    print(f"resize 后 - source: {src.size}, style: {sty.size}")

    # 竖向拼接（和 _combine_images 一致：上+下）
    src_np = np.array(src)
    sty_np = np.array(sty)
    concat = np.concatenate([src_np, sty_np], axis=0)

    print(f"拼接结果尺寸: {concat.shape} (H, W, C)")
    expected_h = size * 2
    if concat.shape == (expected_h, size, 3):
        print(f"拼接成功! 尺寸符合预期 {expected_h}x{size}")
    else:
        print(f"拼接异常! 预期 ({expected_h}, {size}, 3)")

    return Image.fromarray(concat)

result = process_and_concat(source_path, style_path, input_size)
result.save(output_path)
print(f"结果已保存到: {output_path}")
