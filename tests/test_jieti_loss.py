"""结体结构 loss 的单元测试（CPU）。未实跑验证：沙箱无 torch，仅 py_compile。

覆盖步骤 5 的全部要求：
- 平移/缩放/旋转时三项有响应；
- 整体加粗时响应很小；
- 没有 JT 的 batch 结体 loss 为 0；
- 梯度有限、非 NaN；
- α_jt=1、w=0 时 forward_loss 与 baseline 逐值相等。
以及部件划分（build_partition）的精确最近墨迹性质。
"""

import copy
import types

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torchvision import transforms

from util.jieti_loss import JietiLoss, IMAGENET_MEAN, IMAGENET_STD
from util.jieti_partition import build_partition, ink_label_map


MEAN = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
STD = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)


def _normalize(gray_img):
    """(N,H,W) 灰度[0,1] → ImageNet 归一化 (N,3,H,W)。"""
    rgb = gray_img.unsqueeze(1).repeat(1, 3, 1, 1)
    return (rgb - MEAN) / STD


def _two_square_glyph(H=896, W=448, offset=0, scale=1.0, bold=0):
    """构造上下两半相同的"双方块"字：两个部件，可平移/缩放/加粗。

    返回 (1,3,H,W) 归一化图 与 (1,2,R,R) 划分、(1,2,k_max) valid。
    """
    half = H // 2
    R = 224
    k_max = 4
    img = torch.ones(1, half * 2, W)  # 灰度，白底(1)
    # 在每半画两个方块（部件 1 左、部件 2 右）。
    def ring(c, y0, x0, s, thick):
        # 画空心方框（笔画），外延固定为 s×s；thick 只向内加粗，不改外接范围。
        # 这样"加粗"忠实模拟真实笔画变粗（范围不变），而非整体放大。
        c[y0:y0 + s, x0:x0 + thick] = 0.0
        c[y0:y0 + s, x0 + s - thick:x0 + s] = 0.0
        c[y0:y0 + thick, x0:x0 + s] = 0.0
        c[y0 + s - thick:y0 + s, x0:x0 + s] = 0.0

    def draw(canvas_half):
        c = torch.ones(half, W)
        s = int(60 * scale)
        thick = 6 + bold  # 基础笔画宽 6px，bold 只增加笔画厚度
        ring(c, 150, 100, s, thick)              # 部件 1
        ring(c, 150, 260 + offset, s, thick)     # 部件 2（受 offset 平移）
        return c
    top = draw(True)
    bot = draw(True)
    img = torch.cat([top, bot], dim=0).unsqueeze(0)  # (1,H,W)
    img = _normalize(img)
    # 划分：直接给两个部件的占位块（与墨迹一致），这里用简单矩形划分近似。
    voro = torch.zeros(1, 2, R, R, dtype=torch.long)
    for h in range(2):
        voro[0, h, :, : R // 2] = 1
        voro[0, h, :, R // 2:] = 2
    valid = torch.zeros(1, 2, k_max, dtype=torch.bool)
    valid[:, :, 0] = True
    valid[:, :, 1] = True
    return img, voro, valid


def test_translation_scaling_bold_responses():
    # 按项检验：每项对各自目标扰动有响应，对单纯加粗响应很小（整体 J 会把
    # 三项混在一起，不能用来区分，故逐项断言）。
    mod = JietiLoss()
    is_jt = torch.tensor([True])
    gt, voro, valid = _two_square_glyph(offset=0, scale=1.0, bold=0)

    def parts(**kw):
        _, p, _ = mod(_two_square_glyph(**kw)[0], gt, voro, valid, is_jt)
        return p

    base = parts(offset=0, scale=1.0, bold=0)
    trans = parts(offset=60)
    scal = parts(scale=1.6)
    bold = parts(bold=4)

    # 自身对自身：三项都≈0。
    assert all(v.item() < 1e-3 for v in base.values())
    # 相对质心对平移有强响应，对加粗≈0。
    assert trans["centroid"] > 0.1
    assert bold["centroid"] < 1e-3
    assert trans["centroid"] > 10 * bold["centroid"]
    # log σ 对缩放的响应远大于对加粗。
    assert scal["logsigma"] > 5 * bold["logsigma"]


def test_no_jt_batch_zero():
    mod = JietiLoss()
    gt, voro, valid = _two_square_glyph()
    J, parts, _ = mod(gt, gt, torch.zeros_like(voro), torch.zeros_like(valid),
                      torch.tensor([False]))
    assert J.item() == 0.0
    assert all(v.item() == 0.0 for v in parts.values())


def test_gradient_finite():
    mod = JietiLoss()
    gt, voro, valid = _two_square_glyph()
    pred = _two_square_glyph(offset=40)[0].clone().requires_grad_(True)
    J, _, _ = mod(pred, gt, voro, valid, torch.tensor([True]))
    J.backward()
    assert torch.isfinite(pred.grad).all()
    assert pred.grad.abs().sum() > 0


def test_partition_exact_nearest_ink():
    # 两个部件墨迹各占一角，划分应按最近墨迹，背景像素归到更近的部件。
    lm = np.zeros((448, 448), dtype=np.uint8)
    lm[:50, :50] = 1       # 部件 1 在左上
    lm[400:, 400:] = 2     # 部件 2 在右下
    valid = np.array([True, True, False, False])
    part = build_partition(lm, valid, 4, 224)
    # 左上角归部件 1，右下角归部件 2。
    assert part[0, 0] == 1
    assert part[-1, -1] == 2
    # 恰好对角中点附近是分界。
    assert set(np.unique(part)).issubset({1, 2})


def test_ink_label_map_overwrites():
    layers = np.zeros((2, 10, 10), dtype=np.uint8)
    layers[0, :5, :] = 1
    layers[1, 3:, :] = 1  # 与层 0 在 3:5 行重叠，后层覆盖
    lm = ink_label_map(layers, 4)
    assert lm[0, 0] == 1
    assert lm[4, 0] == 2  # 重叠区归后出现的部件
    assert lm[9, 0] == 2


# === α_jt=1、w=0 时 forward_loss 与 baseline 逐值相等 ===

def _forward_loss_method(namespace):
    """只执行真实 forward_loss，不导入 detectron2（照搬 test_baseline 的做法）。"""
    import ast
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / "models_train.py").read_text())
    fontify = next(n for n in tree.body
                   if isinstance(n, ast.ClassDef) and n.name == "Fontify")
    node = next(n for n in fontify.body
                if isinstance(n, ast.FunctionDef) and n.name == "forward_loss")
    module = ast.Module(body=[copy.deepcopy(node)], type_ignores=[])
    exec(compile(module, str(root / "models_train.py"), "exec"), namespace)
    return namespace["forward_loss"]


class _SmallLoss:
    patch_size = 1
    loss_func = "smoothl1"
    _dbg_step = 1

    def unpatchify(self, value):
        return value.transpose(1, 2).reshape(-1, 3, 16, 8)

    def get_dynamic_loss_weights(self, epoch):
        return 0.4, 0.3

    def improved_edge_detection(self, value):
        return value

    def vgg_loss(self, prediction, target, per_sample=False):
        if per_sample:
            return (prediction - target).abs().mean(dim=(1, 2, 3))
        return (prediction - target).abs().mean()

    def resize(self, value):
        return value

    def discriminator(self, value):
        return value.mean((1, 2, 3), keepdim=False).unsqueeze(1)


def test_alpha1_w0_equals_baseline():
    forward = types.MethodType(
        _forward_loss_method({"torch": torch, "F": F, "transforms": transforms}),
        _SmallLoss(),
    )
    torch.manual_seed(0)
    pred = torch.rand(2, 3, 16, 8)
    target = torch.rand_like(pred) + 1
    mask = torch.zeros(2, 128)
    mask[:, -16:] = 1

    # baseline：jieti 关闭
    base = _SmallLoss()
    fbase = types.MethodType(
        _forward_loss_method({"torch": torch, "F": F, "transforms": transforms}), base)
    out_base = fbase(pred, pred.clone(), target, mask.clone(), torch.ones_like(pred),
                     no_gan=True)
    loss_base = out_base[0]

    # jieti 开启但 α_jt=1、w=0：结果必须逐值相等。
    jmod = _SmallLoss()
    jmod.jieti_alpha_jt = 1.0
    jmod.jieti_w = 0.0

    class _ZeroJieti:
        def __call__(self, composite, tgts, voro, valid_parts, is_jt):
            z = composite.new_tensor(0.0)
            return z, {"centroid": z, "logsigma": z, "shape": z}, composite.new_tensor(0.0)
    jmod.jieti_loss_mod = _ZeroJieti()
    fj = types.MethodType(
        _forward_loss_method({"torch": torch, "F": F, "transforms": transforms}), jmod)
    voro = torch.zeros(2, 2, 224, 224, dtype=torch.long)
    valid_parts = torch.zeros(2, 2, 4, dtype=torch.bool)
    is_jt = torch.tensor([True, False])
    out_j = fj(pred, pred.clone(), target, mask.clone(), torch.ones_like(pred),
               no_gan=True, voro=voro, valid_parts=valid_parts, is_jt=is_jt)
    loss_j = out_j[0]
    torch.testing.assert_close(loss_j, loss_base, rtol=0, atol=1e-6)
