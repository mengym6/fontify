# Fontify JT 边缘偏移 Loss 进度对齐

更新时间：2026-06-09

## 当前目标

给 JT semantic mask 样本增加一个可反向传播的边缘偏移约束，用来改善结体区域的边缘错位问题。

核心要求：

- 只作用于 `mask_mode == "jt_semantic"` 的样本。
- 保留现有 hard `edge_offset` 指标作为日志观察项。
- 新增 loss 必须可导，不能直接复用 `masked_edge_offset_metric()`。
- 尽量不破坏原有 `recon + style + edge + adv` loss 结构。

## 已有提交背景

最近相关提交：

```text
4f66538 添加JT语义遮盖边缘偏移指标
ad8e513 修复随机搜索验证图导出卡顿
bf5e108 finetune.sh参数设计，针对上一次commit的分阶段不同遮盖比例
5748ccd Add configurable mask mix sampling
```

`4f66538` 已完成：

- `PairDataset.__getitem__()` 返回第五项 `contour_valid`。
- 只有 `mask_mode == "jt_semantic"` 时，`contour_valid = 1.0`。
- `engine_train.py` 已能透传并记录 `edge_offset` 指标。
- `models_train.py` 中已有 `masked_edge_offset_metric()`，用于 hard edge offset 日志。

## 为什么不能直接把 `masked_edge_offset_metric()` 加进 loss

当前 `masked_edge_offset_metric()` 是 metric-only：

- 使用 `@torch.no_grad()`。
- 内部对 `edge_pred`、`edge_target`、`mask` 使用 `detach()`。
- 使用 hard threshold、`nonzero()`、`topk()`、最近邻点匹配。

因此它能观察：

```text
edge_offset
gt_to_pred
pred_to_gt
dx
dy
count
```

但不能给 `pred` 提供稳定梯度，不能直接参与反向传播。

## 当前已实现方案：方案 B

已按“方案 B”实现：

```text
JT-masked soft edge L1 + soft centroid offset vector loss
```

涉及文件：

```text
models_train.py
main_train.py
engine_train.py
finetune_font.sh
```

### 1. 新增可导 loss

新增函数：

```python
Fontify.jt_edge_offset_loss(edge_pred, edge_target, mask, contour_valid)
```

位置：

```text
models_train.py
```

它返回：

```text
loss_jt_edge
loss_jt_vec
jt_edge_valid_count
```

### 2. `loss_jt_edge`

作用：

只在 JT semantic mask 区域内，对 `edge_pred` 和 `edge_target` 做边缘强度 L1 对齐。

有效区域：

```python
jt_mask = mask[:, :1] * contour_valid.reshape(-1, 1, 1, 1)
```

只要 `contour_valid == 0`，该样本不贡献这个 loss。

### 3. `loss_jt_vec`

作用：

用 soft edge 权重计算预测边缘和目标边缘的软质心，然后约束两者偏移向量接近 0。

处理流程：

```python
soft_pred = sigmoid((edge_pred - jt_edge_tau) / jt_edge_temp)
soft_target = sigmoid((edge_target - jt_edge_tau) / jt_edge_temp)
```

再在 `[-1, 1]` 归一化坐标中计算：

```text
pred centroid
target centroid
offset_vec = pred centroid - target centroid
loss_jt_vec = SmoothL1(offset_vec, 0)
```

这里使用归一化坐标，不直接用像素坐标，避免 loss 数值随图像尺寸放大。

## 当前总 loss 形式

当前实现：

```python
loss_jt_total = loss_jt_edge + jt_edge_vec_weight * loss_jt_vec
jt_edge_weight = edge_weight * jt_edge_loss_weight
loss = loss_l1l2 + loss_vgg + edge_weight * loss_edge + jt_edge_weight * loss_jt_total
loss = loss + adv_weight * adv_loss
```

也就是：

```text
loss =
  recon
  + style
  + edge_weight * edge
  + edge_weight * jt_edge_loss_weight * (jt_edge + jt_edge_vec_weight * jt_vec)
  + adv_weight * adv
```

重要结论：

- 当前 JT loss 与原 `edge loss` 同步生效。
- `edge_weight == 0` 时，JT loss 只计算 raw 值和日志，不参与反传。
- `edge_weight > 0` 后，JT loss 才开始贡献梯度。
- random、BF、fallback random 样本不会贡献 JT loss。

## 新增参数

入口在 `main_train.py`：

```text
--jt_edge_loss_weight
--jt_edge_vec_weight
--jt_edge_tau
--jt_edge_temp
```

`finetune_font.sh` 当前显式设置：

```bash
--jt_edge_loss_weight 0.05 \
--jt_edge_vec_weight 0.1 \
--jt_edge_tau 0.45 \
--jt_edge_temp 0.05
```

### `jt_edge_loss_weight`

JT edge/vector loss 的外层总权重。

实际贡献还会乘以 `edge_weight`：

```text
effective_jt_edge_weight = edge_weight * jt_edge_loss_weight
```

当前设置为 `0.05`。

如果后期 `edge_weight = 0.4`，则：

```text
effective_jt_edge_weight = 0.4 * 0.05 = 0.02
```

### `jt_edge_vec_weight`

控制偏移向量项在 `loss_jt_total` 内部的比例。

当前：

```text
loss_jt_total = loss_jt_edge + 0.1 * loss_jt_vec
```

### `jt_edge_tau`

soft edge 阈值中心，当前 `0.45`。

越大越只关注强边缘；越小越容易让弱边缘参与。

### `jt_edge_temp`

soft threshold 温度，当前 `0.05`。

越小越接近 hard threshold；越大越平滑。

## 当前 TensorBoard 可观察项

训练阶段：

```text
train_loss
train_loss_detail/loss_l1l2
train_loss_detail/loss_vgg
train_loss_detail/loss_jt_edge
train_loss_detail/loss_jt_vec
train_loss_detail/loss_jt_total
train_loss_detail/loss_jt_contrib
train_contour_valid_ratio
train_jt_edge_valid_count
train_edge_offset_count
train_edge_offset_metric/edge_offset
train_edge_offset_metric/gt_to_pred
train_edge_offset_metric/pred_to_gt
train_edge_offset_metric/dx
train_edge_offset_metric/dy
```

重点看：

```text
loss_jt_contrib
train_jt_edge_valid_count
train_edge_offset_metric/edge_offset
train_edge_offset_metric/dx
train_edge_offset_metric/dy
```

解释：

- `loss_jt_edge`、`loss_jt_vec` 是 raw 值。
- `loss_jt_total` 是内部组合后的 raw 总值。
- `loss_jt_contrib` 是真正进入总 loss 的贡献值。
- 只要 `loss_jt_contrib == 0`，新 JT loss 就还没有影响参数更新。

验证阶段：

```text
test_loss/loss
test_jt_edge_loss/loss_jt_edge
test_jt_edge_loss/loss_jt_vec
test_jt_edge_loss/loss_jt_total
test_jt_edge_loss/loss_jt_contrib
test_jt_edge_valid_count
test_edge_offset_count
test_edge_offset_metric/*
```

注意：

当前验证集大概率仍是 half-mask 验证，常见 `test_jt_edge_valid_count == 0` 或 `test_edge_offset_count == 0` 是预期现象，不一定说明接线错误。

## 当前训练日志解读

用户贴过 epoch 0 日志：

```text
raw: recon=1.5829 style=0.3822 edge=0.0597 jt_edge=0.1021 jt_vec=0.0078 adv=0.7008
w: edge=0.000 jt_edge=0.000 jt_vec=0.100 adv=0.000
contrib: edge=0.0000 jt_edge=0.0000 adv=0.0000
share%: recon=80.6 style=19.4 edge=0.0 jt_edge=0.0 adv=0.0
```

解释：

- `jt_edge` 和 `jt_vec` raw 值已经算出来。
- `jt_edge_valid=2` 说明当前 batch 有 JT semantic 样本。
- 但 `edge_weight=0`，所以 `jt_edge_weight=0`。
- 因此 `jt_edge_loss_contrib=0`，当前还没有参与反传。

这符合当前同步设计。

## 当前设计争议：是否应与 `edge loss` 同步

当前实现：

```python
jt_edge_weight = edge_weight * jt_edge_loss_weight
```

优点：

- 保守。
- 不绕过原有 edge warmup 课程。
- 避免训练早期模型输出不稳定时，偏移向量 loss 过早干预。

缺点：

- epoch 0 到 edge warmup 开始前，JT loss 只记录不训练。
- 如果核心目标是尽早纠正 JT 结构偏移，起效可能太晚。

可以独立出来。

最简单改法：

```python
jt_edge_weight = jt_edge_loss_weight
```

但不推荐直接满权重从 epoch 0 打开。

更稳的后续方案：

```text
新增独立 warmup：
--jt_edge_start_epoch
--jt_edge_warmup_epochs
```

然后：

```python
if epoch < jt_edge_start_epoch:
    jt_edge_weight = 0.0
else:
    progress = min(1.0, (epoch - jt_edge_start_epoch) / jt_edge_warmup_epochs)
    jt_edge_weight = jt_edge_loss_weight * progress
```

这样 JT loss 不再依赖 `edge_weight`，但仍有独立 warmup。

建议初值：

```text
jt_edge_start_epoch = 1
jt_edge_warmup_epochs = 5
jt_edge_loss_weight = 0.05
jt_edge_vec_weight = 0.1
```

## 已验证命令

修改后已通过：

```bash
python -m py_compile models_train.py engine_train.py main_train.py data/pairdataset.py
bash -n finetune_font.sh
git diff --check
```

## 当前未提交改动

当前已修改但未提交：

```text
engine_train.py
finetune_font.sh
main_train.py
models_train.py
```

另外工作区里存在一个未由本轮操作产生的删除状态：

```text
generate_new_json.py
```

后续提交前需要确认是否要包含这个删除。

## 建议下一步

两个可选方向：

### 方向 A：保持当前同步设计

继续训练，等 `edge_weight` 开始大于 0 后观察：

```text
loss_jt_contrib
share%: jt_edge
edge_offset
dx
dy
```

适合先验证新 loss 是否稳定。

### 方向 B：改成独立 warmup

如果希望 JT 偏移 loss 更早参与训练，建议改为独立 warmup，而不是直接从 epoch 0 满权重生效。

需要改：

```text
models_train.py
main_train.py
finetune_font.sh
```

可保留 `engine_train.py` 当前日志接线。
