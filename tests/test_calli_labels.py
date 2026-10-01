import json

import pytest
import torch

from util.calli_labels import (
    KIND_BF, KIND_JT, NUM_BF, NUM_JT, NUM_LABELS, LabelRenderer, label_loss,
    task_channel_mask,
)


def write_coco(tmp_path, categories, width, height, annotations):
    data = {
        "images": [{"id": 1, "file_name": "字.png", "width": width, "height": height}],
        "categories": [{"id": i + 1, "name": n} for i, n in enumerate(categories)],
        "annotations": [
            {"id": i, "image_id": 1, "category_id": cat, "segmentation": [poly]}
            for i, (cat, poly) in enumerate(annotations)
        ],
    }
    (tmp_path / "ann.json").write_text(json.dumps(data, ensure_ascii=False))
    return {"annotation_path": "ann.json", "target_path": "x/字.png"}


def test_counts_bf50_jt20():
    assert (NUM_BF, NUM_JT, NUM_LABELS) == (50, 20, 70)


def test_render_uses_center_pad_geometry(tmp_path):
    # 宽 896、高 448 的原图：居中补白到 896×896，再缩到 448，纵向偏移 112 像素。
    pair = write_coco(
        tmp_path, ["text", "1-横-起笔", "左"], 896, 448,
        [(1, [0, 0, 896, 0, 896, 448, 0, 448]),
         (2, [0, 0, 64, 0, 64, 64, 0, 64])],
    )
    pair["type"] = "font_CaoqbBF"
    label, kind = LabelRenderer(str(tmp_path)).render(pair)
    assert kind == KIND_BF and label.shape == (70, 28, 28)
    # '1-横-起笔' 归一化为 '横-起笔'，即第 0 通道；原图 64px → 32px，
    # 落在 448 图的 y∈[112,144]、x∈[0,32]，即 patch 行 7-8、列 0-1（边界多 1 像素）。
    assert label[0, 7:9, 0:2] == pytest.approx(1.0)
    assert label[0].sum() == pytest.approx(4.0, abs=0.3)
    assert label[1:].sum() == 0  # text 不渲染


def test_label_loss_only_uses_task_channels():
    kinds = torch.tensor([KIND_BF, KIND_JT])
    mask = task_channel_mask(kinds)
    assert mask[0, :NUM_BF].all() and not mask[0, NUM_BF:].any()
    assert mask[1, NUM_BF:].all() and not mask[1, :NUM_BF].any()

    target = torch.zeros(2, 70, 28, 28)
    target[0, 3, :4, :4] = 1
    target[1, NUM_BF + 2, 10:, :] = 1
    logits = torch.zeros(2, 70, 28, 28, requires_grad=True)
    loss = label_loss(logits, target, kinds)
    loss.backward()
    grad = logits.grad.abs().sum((2, 3))
    assert (grad[0, NUM_BF:] == 0).all() and (grad[1, :NUM_BF] == 0).all()
    assert grad[0, :NUM_BF].gt(0).all() and grad[1, NUM_BF:].gt(0).all()

    perfect = torch.where(target > 0, 20.0, -20.0)
    assert label_loss(perfect, target, kinds) < 1e-3 < loss.item()
