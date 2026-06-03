# Fontify Finetune Random Search 指南

本文档对应当前仓库中的 `random_search_runs.csv` 和 `run_random_search_finetune.sh`。

当前方案已经改为：

```text
JT 和 BF 从 epoch 0 开始同步混合训练
semantic_only_epochs 固定为 0
不再搜索旧的课程比例参数
```

这意味着：

```text
数据端不再做 JT-only 重定向
loss 端从 epoch 0 开始进入 jt_bf_sync
edge/adv warmup 从 epoch 0 开始按原逻辑计时
```

## 1. 当前实验结构

当前 random search 一共 24 组：

```text
baseline + rs01 ... rs23
```

启动脚本读取 `random_search_runs.csv`，根据 `STAGE` 选择 `short`、`mid` 或 `full` 阶段的 epoch 数。脚本不会自动筛选前几组，进入下一阶段的 `RUN_IDS` 需要人工根据 short/mid 结果指定。

输出目录规则：

```text
models/random_search/<stage>/<run_id>_eb<effective_batch>_bs<batch_size>_acc<accum_iter>_ep<epochs>_sem0/
```

每组目录中主要看：

```text
train.log
log.txt
run_params.txt
checkpoint-*.pth
logs/
val_images/
val_images/manifest.csv
```

其中 `val_images/` 是从 TensorBoard 里导出的 4 张验证图，每张图横向拼接：

```text
x | im_masked | y | tgt
```

含义分别是：

```text
输入图 | 被 mask 的目标图 | 模型输出 | 目标图
```

默认导出的是当前阶段最后一个 epoch：

```text
short: epoch 29
mid: epoch 69
full: epoch 99
```

random search 启动脚本会把 `VAL_IMAGE_LIMIT` 同步传给训练入口的 `--val_tb_image_limit`。默认 `VAL_IMAGE_LIMIT=4`，因此每个验证 epoch 最多向 TensorBoard 写 4 张 val 图，避免 event 文件过大导致导出阶段卡住。

## 2. 启动命令

2 卡 short：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
STAGE=short \
RUN_IDS=all \
bash ./run_random_search_finetune.sh
```

4 卡 short：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
STAGE=short \
RUN_IDS=all \
bash ./run_random_search_finetune.sh
```

进入 mid 时，不再用 `RUN_IDS=all`，而是填 short 后人工选出的组。例如：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
STAGE=mid \
RUN_IDS=rs02,rs07,rs11,baseline \
bash ./run_random_search_finetune.sh
```

进入 full 时同理，只填 mid 后留下的最终候选。例如：

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
STAGE=full \
RUN_IDS=rs07,baseline \
bash ./run_random_search_finetune.sh
```

注意：

```text
NPROC_PER_NODE 必须和 CUDA_VISIBLE_DEVICES 中的 GPU 数一致。
```

错误示例：

```bash
bash STAGE=short RUN_IDS=all ./run_random_search_finetune.sh
```

正确写法是把环境变量放在 `bash` 前面：

```bash
STAGE=short RUN_IDS=all bash ./run_random_search_finetune.sh
```

## 3. 搜索参数列表

当前真正参与 random search 的参数如下。

| 参数 | 当前范围 / 候选值 | 作用 | 解释重点 |
|---|---|---|---|
| `target_effective_batch` | `{48, 64, 96}` | 等效 batch size | 控制梯度噪声、每 epoch optimizer step 数、和 `lr` 的匹配关系 |
| `lr` | `1.5e-5` 到 `1e-4` | 主优化器学习率 | 最敏感参数之一，过大容易破坏结构，过小学不动风格 |
| `weight_decay` | `{0.03, 0.05, 0.08, 0.10, 0.15, 0.20}` | AdamW 正则强度 | 控制过拟合与细节保守程度 |
| `layer_decay` | `{0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95}` | 分层学习率衰减 | 越小越保护底层，越接近 `1.0` 各层学习率越接近 |
| `freeze_blocks` | `{6, 8, 9, 10, 12}` | 冻结 encoder 前多少个 block | 越大越保守，越小可训练 encoder 越多 |
| `drop_path` | `{0.00, 0.05, 0.10}` | stochastic depth 正则 | 小数据下可帮助正则，但过大可能欠拟合 |

### 3.1 等效 batch size 换算

CSV 中固定的是 `target_effective_batch`，不是固定 `accum_iter`。脚本会自动计算：

```text
accum_iter = target_effective_batch / (batch_size * NPROC_PER_NODE)
```

当前 `batch_size=2`，所以：

| `target_effective_batch` | 2 卡 `accum_iter` | 4 卡 `accum_iter` |
|---:|---:|---:|
| 48 | 12 | 6 |
| 64 | 16 | 8 |
| 96 | 24 | 12 |

这样做的目的：2 卡和 4 卡运行同一个 `run_id` 时，保持等效 batch size 不变，只改变梯度累积步数。

### 3.2 阶段参数

| 阶段 | `epochs` | `semantic_only_epochs` |
|---|---:|---:|
| `short` | 30 | 0 |
| `mid` | 70 | 0 |
| `full` | 100 | 0 |

也就是说，每个阶段都从一开始同步训练 JT 和 BF。

## 4. 固定参数

这些参数当前不参与搜索：

| 参数 | 固定值 |
|---|---|
| `batch_size` | `2` |
| `semantic_only_epochs` | `0` |
| `model` | `vit_base_patch16_input896x448_win_dec64_8glb_sl1` |
| `num_mask_patches` | `784` |
| `max_mask_patches_per_block` | `392` |
| `warmup_epochs` | `1` |
| `clip_grad` | `1.0` |
| `input_size` | `896 448` |
| `num_mask_annotations_bf` | `3` |
| `num_mask_annotations_jt` | `1` |
| `mask_coverage_threshold` | `0.1` |
| `PRETRAIN_CKPT` | 默认 `models/vit_base_font/checkpoint-14.pth` |
| `DATA_PATH` | 默认 `fontdata_example` |
| `SAVE_FREQ` | 默认 `10`，可用环境变量覆盖 |
| `VAL_IMAGE_LIMIT` | 默认 `4`，每个验证 epoch 写入/导出的 val 图数量 |
| `VAL_EXPORT_TIMEOUT` | 默认 `300` 秒，导图超时后给 warning 并继续后续 run |

启动脚本故意不传 `--auto_resume`，避免 random search 接上旧实验。

## 5. 当前课程与 loss 行为

当前所有 run 都传：

```text
--semantic_only_epochs 0
```

因此模型侧：

```text
get_loss_phase(epoch) -> jt_bf_sync
phase_epoch = epoch
```

数据侧：

```text
不会把 BF 样本重定向到 JT
JT 和 BF 从 epoch 0 起按原始采样分布同步出现
```

loss 端：

```text
smooth_l1 + VGG_style + edge_weight * edge + adv_weight * adv
```

其中 edge/adv 的 warmup 从 epoch 0 开始计时：

```text
adv_weight: 前 20 epoch 为 0，之后 20 epoch 线性升到最终值
edge_weight: 前 15 epoch 为 0，之后 20 epoch 线性升到最终值
```

注意：这和旧方案不同。旧方案前面有 JT-only warmup，edge/adv warmup 会从切换点重新开始；当前方案没有这个切换点。

## 6. Checkpoint 保存规则

训练代码按 epoch 编号保存，epoch 从 `0` 开始。

保存条件：

```text
epoch % save_freq == 0
或
epoch + 1 == epochs
```

默认 `SAVE_FREQ=10` 时：

| 阶段 | 保存 checkpoint |
|---|---|
| `short=30` | `checkpoint-0.pth`, `checkpoint-10.pth`, `checkpoint-20.pth`, `checkpoint-29.pth` |
| `mid=70` | `checkpoint-0.pth`, `checkpoint-10.pth`, ..., `checkpoint-60.pth`, `checkpoint-69.pth` |
| `full=100` | `checkpoint-0.pth`, `checkpoint-10.pth`, ..., `checkpoint-90.pth`, `checkpoint-99.pth` |

当前没有自动保存 `best.pth`。最终选择需要结合 `log.txt`、`val_images/` 和 checkpoint 对应输出判断。

## 7. 选组别规则

### 7.1 short 阶段

short 阶段运行全部 24 组：

```bash
STAGE=short RUN_IDS=all bash ./run_random_search_finetune.sh
```

short 的目的不是直接选最终最优，而是筛掉明显不稳定或明显质量差的组合。

short 后建议保留：

```text
3 到 5 个表现较好的组 + baseline
```

如果算力紧张：

```text
2 到 3 个表现较好的组 + baseline
```

### 7.2 short 筛选优先级

第一优先级：淘汰失败组。

淘汰条件包括：

```text
loss 为 NaN
loss 爆炸或持续异常震荡
生成图大片空白
字形结构明显散掉
笔画出现大量噪声、脏点或破碎
```

第二优先级：看 `val_images/`。

主要看：

```text
结构是否正确
重心是否稳定
部件位置是否对
笔画粗细是否合理
边缘是否干净
输出是否接近目标字体风格
```

第三优先级：看 `log.txt` 中最后几个 epoch 的 `test_loss`。

建议看最后 3 到 5 个 epoch 的趋势，而不是只看最后一行。

第四优先级：保留多样性。

进入 mid 的组合不要全是同一种设置。例如不要全是：

```text
target_effective_batch = 96
freeze_blocks = 12
layer_decay = 0.65
```

如果视觉质量接近，应保留不同 `target_effective_batch`、不同冻结深度或不同 `layer_decay` 的组合。

### 7.3 mid 阶段

mid 只跑 short 后筛出的组。例如：

```bash
STAGE=mid RUN_IDS=rs02,rs07,rs11,baseline bash ./run_random_search_finetune.sh
```

mid 的目标是验证 short 的结论是否稳定，尤其要看同步训练更久之后，结构和风格是否继续改善。

mid 后建议保留：

```text
1 到 2 个候选组 + baseline
```

如果 baseline 已经明显落后，可以不进 full；如果差距不大，应保留 baseline 做最终对照。

### 7.4 full 阶段

full 只跑最终候选。例如：

```bash
STAGE=full RUN_IDS=rs07,baseline bash ./run_random_search_finetune.sh
```

full 阶段主要用于最终比较，不建议再大范围改变参数。若 full 第一名和第二名差距很小，优先选择视觉更稳、结构更少出错的组合。

## 8. 调参解释规则

### 8.1 总原则

不要只看总 loss。当前总 loss 由多项组成：

```text
smooth_l1 + VGG_style + edge_weight * edge + adv_weight * adv
```

看 `[loss-dbg]` 时应按：

```text
raw 原始值 -> w 当前权重 -> contrib 加权贡献 -> share% 占比
```

当前没有 JT-only 阶段，所以所有 run 从 epoch 0 都是 `jt_bf_sync`。但 edge/adv 仍然有 warmup，因此 early epoch 的 `edge` 和 `adv` 贡献可能仍为 0 或很小。

### 8.2 `target_effective_batch`

`target_effective_batch` 越小：

```text
每个 epoch optimizer step 更多
梯度噪声更大
可能更容易适应小数据
训练曲线可能更抖
```

`target_effective_batch` 越大：

```text
梯度更平滑
每个 epoch optimizer step 更少
训练可能更稳但更保守
若 epoch 不变，实际更新次数更少
```

| 现象 | 可能解释 | 调整方向 |
|---|---|---|
| 训练很抖，val 图时好时坏 | batch 偏小或 `lr` 偏大 | 增大 `target_effective_batch`，或降低 `lr` |
| 训练很稳但风格学不进去 | batch 偏大、更新次数少或 `lr` 偏小 | 降低 `target_effective_batch`，或略增 `lr` |
| short 阶段几乎没有适应 | batch 偏大导致 step 少，或冻结太深 | 降低 batch，减少冻结，或增大 `lr` |
| 细节偶尔很好但不稳定 | 小 batch 噪声带来探索但不稳定 | 尝试 `64` 作为折中 |

### 8.3 `lr`

| 现象 | 可能解释 | 调整方向 |
|---|---|---|
| loss 几乎不动，输出接近初始状态 | `lr` 偏小 | 增大 `lr` |
| 字形骨架漂移、结构破坏 | `lr` 偏大 | 降低 `lr`，或增加冻结 |
| 笔画噪声明显、边缘脏 | `lr` 偏大或 GAN/edge 阶段不稳 | 降低 `lr`，观察同步阶段 |
| 风格学得慢但结构稳定 | `lr` 略小或冻结过深 | 略增 `lr`，或减少 `freeze_blocks` |

`lr` 和 `target_effective_batch` 要一起看。较大 batch 通常能承受略大的 `lr`，但这是小数据 fine-tune，不应简单线性放大。

### 8.4 `weight_decay`

| 现象 | 可能解释 | 调整方向 |
|---|---|---|
| train 下降明显，但 val 图变差 | 过拟合 | 增大 `weight_decay` |
| 风格细节学不进去，输出偏保守 | 正则过强 | 降低 `weight_decay` |
| 图像噪声和局部破碎较多 | 正则可能偏弱，也可能 `lr` 偏大 | 先看 `lr`，再考虑增大 `weight_decay` |
| 结构稳定但笔画缺乏目标风格 | 正则偏强或冻结偏深 | 降低 `weight_decay` 或减少冻结 |

### 8.5 `layer_decay`

```text
越小：越保护底层
越接近 1.0：各层学习率越接近
```

| 现象 | 可能解释 | 调整方向 |
|---|---|---|
| 字形结构容易坏 | 底层改动过大 | 降低 `layer_decay` |
| 风格学不进去，输出过于保守 | 底层和中层适应不足 | 增大 `layer_decay` |
| 解冻较多层后不稳定 | 分层保护不够 | 降低 `layer_decay` |
| 冻结很深但仍学得慢 | 可训练层学习率不足 | 增大 `layer_decay` 或减少冻结 |

### 8.6 `freeze_blocks`

当前模型是 `vit_base`，encoder 深度为 12。`freeze_blocks` 表示冻结前多少个 encoder block。

| `freeze_blocks` | 含义 |
|---:|---|
| `6` | 解冻较多，适应能力强，但风险高 |
| `8` | 中等偏激进 |
| `9` | 当前 baseline |
| `10` | 中等偏保守 |
| `12` | 冻结全部 encoder block，最保守 |

| 现象 | 可能解释 | 调整方向 |
|---|---|---|
| 结构漂移、部件位置错 | 解冻太多或学习率太大 | 增大 `freeze_blocks`，或降低 `lr` |
| 风格迁移很弱 | 冻结太深 | 减小 `freeze_blocks` |
| 训练稳定但上限低 | 可训练容量不足 | 减小 `freeze_blocks` 或增大 `layer_decay` |
| 小数据下过拟合明显 | 可训练参数太多 | 增大 `freeze_blocks` |

### 8.7 `drop_path`

| 现象 | 可能解释 | 调整方向 |
|---|---|---|
| 过拟合明显 | 正则不足 | 增大 `drop_path` |
| 欠拟合，风格和细节都弱 | 正则过强 | 降低 `drop_path` |
| 小数据但结构容易坏 | 可尝试中等 `drop_path` | 比较 `0.05` 和 `0.10` |

当前范围较小：

```text
0.00, 0.05, 0.10
```

不建议第一轮就超过 `0.10`。

## 9. 最终选择建议

最终模型不要只按一个指标选。建议按以下顺序：

```text
1. 排除失败组
2. 看 full 阶段 val_images 的结构稳定性
3. 看目标字体风格接近程度
4. 看最后 5 个 epoch 的 test_loss 趋势
5. 看 checkpoint 输出是否稳定
6. 若差距很小，选更保守、更少结构错误的组
```

如果两个组合非常接近：

```text
优先选择结构更稳的组合，而不是单张图风格最像的组合。
```

原因是字体生成里结构错误通常比风格略弱更难接受。

## 10. 常见误判

不要把 `checkpoint-29.pth` 理解成第 29 次训练。它表示 epoch 编号 29，是 30 epoch short 的最后一个 checkpoint。

不要把 4 卡运行下的 `accum_iter` 和 2 卡运行下的 `accum_iter` 直接比较。要比较的是 `effective_batch_size`。

不要用 `RUN_IDS=all` 跑 mid 或 full，除非你确实想所有组继续跑下去。正常流程是 short 后人工筛选 `RUN_IDS`。

不要沿用旧方案里的课程比例解释当前结果。当前所有 run 都是 JT/BF 从 epoch 0 同步混合训练。
