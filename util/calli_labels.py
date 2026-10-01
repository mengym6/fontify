"""CalliPhase 语义标签：类别表、区域图渲染与标签损失。

BF 共 50 类（17 种笔画 × 起/中/收笔，点只有起/收笔），JT 共 20 类空间标签。
两者拼成 70 通道：前 50 通道为 BF，后 20 通道为 JT。区域图在 448 分辨率
按与 semantic_masks 相同的居中补白几何渲染，再按 16×16 patch 求覆盖率，
得到与 ViT token 网格一致的 (70, 28, 28) 软标签。
"""

import json
import os

import numpy as np
from PIL import Image, ImageDraw

BF_STROKES = (
    "横", "竖", "撇", "捺", "点", "提", "横折", "横钩", "横折钩", "竖钩",
    "竖提", "竖折", "竖弯钩", "斜钩", "弯钩", "卧钩", "撇折",
)
BF_STAGES = ("起笔", "中笔", "收笔")
BF_CLASSES = tuple(
    f"{stroke}-{stage}"
    for stroke in BF_STROKES
    for stage in BF_STAGES
    if not (stroke == "点" and stage == "中笔")
)
JT_CLASSES = (
    "左", "右", "上", "下", "中", "左上", "左下", "右上", "右下", "上左",
    "上右", "下左", "下右", "下中", "中上", "中下", "中左", "中右", "内", "外",
)
NUM_BF = len(BF_CLASSES)
NUM_JT = len(JT_CLASSES)
NUM_LABELS = NUM_BF + NUM_JT
KIND_BF, KIND_JT = 0, 1
IMAGE_SIZE = 448
PATCH = 16
LABEL_SIZE = IMAGE_SIZE // PATCH

_CHANNEL = {name: i for i, name in enumerate(BF_CLASSES)}
_CHANNEL.update({name: NUM_BF + i for i, name in enumerate(JT_CLASSES)})


def label_kind(pair_type):
    """type 形如 font_<书家>JT / font_<书家>BF。"""
    return KIND_JT if "JT" in pair_type else KIND_BF


def class_names():
    return BF_CLASSES + JT_CLASSES


def normalize_category(name):
    """'1-横-起笔' → '横-起笔'；空间标签不变。"""
    return name.lstrip("0123456789-")


def _decode_rle(segmentation, h, w):
    from pycocotools import mask as mask_util

    rle = segmentation
    if isinstance(rle.get("counts"), list):
        rle = mask_util.frPyObjects(rle, h, w)
    return mask_util.decode(rle).astype(np.uint8)


class LabelRenderer:
    """按样本的 COCO 标注渲染 (70, 28, 28) patch 覆盖率。"""

    def __init__(self, root):
        self.root = root
        self._cache = {}

    def _index(self, ann_path):
        if ann_path not in self._cache:
            with open(ann_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            images = {
                os.path.splitext(os.path.basename(img["file_name"]))[0]: img
                for img in data["images"]
            }
            anns = {}
            for ann in data["annotations"]:
                anns.setdefault(ann["image_id"], []).append(ann)
            names = {c["id"]: c["name"] for c in data["categories"]}
            self._cache[ann_path] = (images, anns, names)
        return self._cache[ann_path]

    def render(self, pair):
        ann_path = os.path.join(self.root, pair["annotation_path"])
        images, anns, names = self._index(ann_path)
        stem = os.path.splitext(os.path.basename(pair["target_path"]))[0]
        info = images[stem]
        h, w = int(info["height"]), int(info["width"])
        side = max(h, w)
        off_x, off_y = (side - w) // 2, (side - h) // 2
        scale = IMAGE_SIZE / side

        layers = {}
        for ann in anns.get(info["id"], []):
            name = names[ann["category_id"]]
            if name == "text":
                continue
            channel = _CHANNEL[normalize_category(name)]
            layer = layers.setdefault(
                channel, Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
            )
            seg = ann["segmentation"]
            if isinstance(seg, dict):
                # RLE 在原图分辨率解码，再按 semantic_masks 的居中补白几何缩放。
                square = Image.new("L", (side, side), 0)
                square.paste(
                    Image.fromarray(_decode_rle(seg, h, w) * 255), (off_x, off_y)
                )
                square = square.resize(
                    (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST
                )
                layer.paste(255, mask=square)
                continue
            polygons = [seg] if seg and isinstance(seg[0], (int, float)) else seg
            draw = ImageDraw.Draw(layer)
            for poly in polygons:
                if len(poly) < 6:
                    continue
                pts = [
                    ((poly[i] + off_x) * scale, (poly[i + 1] + off_y) * scale)
                    for i in range(0, len(poly), 2)
                ]
                draw.polygon(pts, fill=255)

        coverage = np.zeros((NUM_LABELS, LABEL_SIZE, LABEL_SIZE), dtype=np.float32)
        for channel, layer in layers.items():
            binary = (np.asarray(layer) > 0).astype(np.float32)
            coverage[channel] = binary.reshape(
                LABEL_SIZE, PATCH, LABEL_SIZE, PATCH
            ).mean((1, 3))
        return coverage, label_kind(pair["type"])


def task_channel_mask(kinds, device=None):
    """(B,) 的 BF/JT 类型 → (B, 70) 的任务通道掩码。"""
    import torch

    mask = torch.zeros(len(kinds), NUM_LABELS, device=device)
    mask[kinds == KIND_BF, :NUM_BF] = 1
    mask[kinds == KIND_JT, NUM_BF:] = 1
    return mask


def label_loss(logits, target, kinds):
    """按样本类型掩码的 BCE + Dice。

    BCE 作用于该类型的全部通道（缺席类别即全 0 负样本）；Dice 只在出现的
    类别上计算。返回 batch 均值。
    """
    import torch.nn.functional as F

    logits = logits.float()
    target = target.float()
    channel = task_channel_mask(kinds, logits.device)
    bce = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    ).mean((2, 3))
    prob = logits.sigmoid()
    inter = (prob * target).sum((2, 3))
    dice = 1 - (2 * inter + 1) / (prob.sum((2, 3)) + target.sum((2, 3)) + 1)
    present = channel * (target.sum((2, 3)) > 0).float()
    per_sample = (bce * channel).sum(1) / channel.sum(1)
    per_sample = per_sample + (dice * present).sum(1) / present.sum(1).clamp_min(1)
    return per_sample.mean()
