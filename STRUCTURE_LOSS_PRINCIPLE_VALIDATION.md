# Fontify 结体学习问题：结构损失原理验证方案

## 1. 实验目的

验证 CalliPhase 数据是否能改善 Fontify 的结体（Jieti）建模，以及改善是否来自显式结构约束，而不是仅来自数据分布或遮罩策略变化。

本方案针对已有现象设计：

- 电脑字体大规模预训练后，笔法可以通过约 1500 张书法数据微调学习；
- 结体特征几乎无法学习；
- 减少 encoder 冻结层数后效果更差；
- 单纯提高 JT 语义遮罩比例后效果也更差。

因此本轮不再把“解冻更多层”或“提高 JT mask 比例”作为主要假设，而验证以下假设：

> 当前 Jieti 标注仅作为遮罩位置使用，没有直接约束结构表示；加入稳定的显式结构损失，可能使模型在保持预训练结构先验的同时学习书法结体。

## 2. 当前基线与限制

基线 checkpoint：

```text
/Users/root1/Desktop/Fontify-main/models/vit_base_font/checkpoint-14.pth
```

当前模型流程为：

```text
参考图 imgs + 目标图 tgts
→ 两路 patch embedding
→ 目标 mask token 替换
→ ViT encoder
→ 第 3 个 block 后双流融合
→ 多层特征拼接
→ 卷积 decoder
→ pred
```

现有损失为：

$$L_{base}=L_{recon}+L_{style}+w_{edge}L_{edge}+w_{adv}L_{adv}.$$

其中重建损失只在 `mask * valid` 区域计算。CalliPhase 的 Jieti polygon 当前最终被转换为 patch 级二值 mask；18 类 Jieti 空间标签、笔画关系和结构位置没有直接进入 loss。

## 3. 实验总原则

所有实验固定：

1. 同一个 `checkpoint-14.pth`；
2. 同一训练集、验证集和 JSON 划分；
3. 同一参考字符、目标字符和推理脚本；
4. 同一训练步数、batch size、梯度累积和随机种子；
5. 同一冻结策略；
6. 同一数据增强；
7. 首轮统一关闭 GAN，避免判别器训练波动掩盖结构损失效果。

首轮建议保留当前冻结策略，不改变 encoder 可训练范围。若当前正式配置是 `--freeze_encoder --freeze_blocks 9`，三组实验全部使用该设置。

## 4. 实验分组（已取消数据消融）

### 当前执行选择

根据已有实验结果，本项目不再安排 A/B/C 数据消融。当前只验证两件事：阶段 2 是否能把 CalliPhase 的 Bifa/Jieti 先验注入 `checkpoint-14.pth`，以及阶段 3 的优化微调是否能把该先验转化为目标书法风格。历史最优的“冻结 9 层、随机/BF 主导”结果作为固定参照，不重复作为新实验变量。

## 5. 当前主行动路线：三阶段计划与登记

阶段 2 和阶段 3 都针对 CalliPhase 的 **Bifa（笔法）与 Jieti（结体）**，阶段 2 不是只解决结体，阶段 3 也不是只做风格外观。两阶段的区别在于：阶段 2 建立跨书法家、跨字符的书法先验，阶段 3 学习具体目标书法风格。

| 阶段 | 数据/初始化 | 目标 | 关键协议 | 完成登记 |
|---|---|---|---|---|
| 1. 通用字形预训练 | 数百万电脑字体；MAE 初始化 | 通用字形、笔画拓扑、结构重建和条件对应 | 原始 Fontify 训练协议 | **已完成**；checkpoint：`models/vit_base_font/checkpoint-14.pth` |
| 2. 书法 Bifa-Jieti 适配 | `checkpoint-14.pth`；电脑字体 + CalliPhase 混合 | 注入书法笔法和结体先验，保持通用结构能力 | 显式结构/区域损失；电脑字体 replay；优先冻结 encoder 或 adapter；随机 mask 主导 | 待执行 |
| 3. 目标书法风格微调 | 阶段 2 checkpoint；只用 CalliPhase/目标手写书法 | 学习目标书法家/风格的笔法和结体表现 | 冻结 9 层、随机/BF 主导；逐项测试蒸馏、布局损失、adapter/LoRA | 待执行 |

每完成一个阶段，登记：日期、训练命令、数据版本、初始化 checkpoint、最终 checkpoint、验证图、Bifa 指标、Jieti/结构指标及人工判断。只有登记完成后才能进入下一阶段；当前阶段 1 已确认完成，阶段 2、3 尚未完成。

### A：原始微调基线

只使用现有书法微调协议：

```text
输入：1500 张目标书法数据
损失：原有 recon + style + edge（GAN 首轮关闭）
结构损失：无
Jieti 显式标签：不使用
```

目的：重新得到一个可比较的基线，而不是直接拿历史结果比较。必须保存训练命令、checkpoint 和固定推理结果。

### B：加入 CalliPhase，但不加结构损失

在 A 的训练流程中加入 CalliPhase 样本，仍使用原有损失：

```text
损失：原有 recon + style + edge
CalliPhase：用于图像和语义 mask
显式结构损失：无
```

建议 CalliPhase 的采样比例从较低值开始，不让小规模数据完全改变原始训练分布：

```text
电脑字体/原书法数据 : CalliPhase = 4 : 1
```

如果暂时不能实现混合 DataLoader，可以先只用 CalliPhase 做短时中间适配，再进入 1500 张微调，但必须单独记录这是“阶段适配”而不是普通混合。

### C：CalliPhase + 显式结构损失

C 与 B 完全相同，只增加：

$$L=L_{base}+w_{structure}L_{structure}.$$

首轮测试：

```text
w_structure = 0.05
w_structure = 0.10
```

不要一开始使用大权重。结构损失应只对存在结构标注的 CalliPhase 样本启用；无标注样本的结构损失为 0。

## 5. 首版结构损失

### 5.1 输入前景

`pred` 和 `target` 当前处于 ImageNet 归一化空间。计算结构特征前必须先反归一化：

```python
img = img * imagenet_std + imagenet_mean
```

然后转灰度并构造 soft foreground。建议不要使用硬阈值，以保持梯度可传播。可使用 sigmoid：

```python
gray = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]
foreground = torch.sigmoid((threshold - gray) * temperature)
```

阈值和温度应固定，并在预测图与目标图上使用相同计算过程。

### 5.2 行列投影

对前景沿宽度和高度求和，再归一化：

$$P_{row}(y)=\frac{\sum_x F(y,x)}{\sum_{y,x}F(y,x)+\epsilon},$$

$$P_{col}(x)=\frac{\sum_y F(y,x)}{\sum_{y,x}F(y,x)+\epsilon}.$$

损失为预测与目标投影的 L1：

$$L_{row}=\left\|P_{row}(\hat y)-P_{row}(y)\right\|_1,$$

$$L_{col}=\left\|P_{col}(\hat y)-P_{col}(y)\right\|_1.$$

这两个项约束字符整体上下分布、左右分布和部件留白。

### 5.3 前景质心

定义归一化坐标 `x ∈ [-1,1]`、`y ∈ [-1,1]`，计算 soft foreground 的质心：

$$c_x=\frac{\sum_{x,y}xF(x,y)}{\sum_{x,y}F(x,y)+\epsilon},\quad
c_y=\frac{\sum_{x,y}yF(x,y)}{\sum_{x,y}F(x,y)+\epsilon}.$$

$$L_{centroid}=|c_x(\hat y)-c_x(y)|+|c_y(\hat y)-c_y(y)|.$$

### 5.4 前景面积

$$a=\frac{1}{HW}\sum_{x,y}F(x,y),$$

$$L_{area}=|a(\hat y)-a(y)|.$$

该项防止预测字形整体过细、过粗或前景覆盖范围明显错误。

### 5.5 外接框宽高比

首轮不建议直接对硬外接框求梯度。可以先将宽高比作为验证指标；若需要放入 loss，使用 soft quantile 或投影累计质量近似边界。

因此首版建议：

```text
训练 loss：L_row + L_col + L_centroid + L_area
监控指标：bbox aspect ratio
```

最终首版结构损失为：

$$L_{structure}=L_{row}+L_{col}+L_{centroid}+L_{area}.$$

各项最好先单独归一化，再求和，避免投影项数值远大于质心项。

## 6. 代码修改清单

### `data/pairdataset.py`

建议扩展返回值，使样本能够携带结构信息：

```python
return image, target, mask, valid, structure_target
```

其中 `structure_target` 可以是：

- 结构区域 mask；
- polygon 转换后的区域集合；
- Jieti 类别和位置标签；
- 无标注样本使用 `None`。

为降低首轮实现复杂度，也可以暂时不增加返回值，直接从 target 图像计算全局结构特征；这种做法只能验证“全局结构损失”是否有效，不能验证 Jieti 标签本身是否有效。

### `data/pair_transforms.py`

如果传递 polygon 或区域 mask，必须保证以下几何变换同步作用于图像和标注：

- resize；
- crop；
- pad；
- horizontal flip。

颜色抖动、归一化不应作用于结构标签。

### `models_train.py`

新增函数，例如：

```python
def compute_structure_loss(self, pred, target, structure_target=None):
    ...
```

在 `forward_loss` 中增加：

```python
loss_structure = pred.new_tensor(0.0)
if structure_loss_weight > 0:
    loss_structure = self.compute_structure_loss(pred, tgts, structure_target)
loss = loss + structure_loss_weight * loss_structure
```

`forward`、`engine_train.py` 和 `main_train.py` 都要同步传递和记录该值。

建议返回：

```python
loss, loss_l1l2, loss_vgg, loss_structure, y, mask, pred
```

如果担心破坏现有调用，可保留原返回顺序，只在日志中额外返回或挂载 scalar。

### `main_train.py`

增加参数：

```python
parser.add_argument('--structure_loss_weight', default=0.0, type=float)
parser.add_argument('--structure_loss_start_epoch', default=0, type=int)
```

首轮建议：

```text
A/B: --structure_loss_weight 0.0
C1:  --structure_loss_weight 0.05
C2:  --structure_loss_weight 0.10
```

### `engine_train.py`

训练循环中记录：

```text
train_loss
loss_l1l2
loss_vgg
loss_structure
grad_norm
lr
```

验证阶段也应记录结构指标，否则不能判断结构损失是否只改善训练集。

## 7. 建议训练配置

第一轮关闭 GAN：

```bash
--no_gan
```

保持 encoder 冻结范围不变，使用较小学习率。示例：

```bash
--batch_size 2 \
--accum_iter 32 \
--lr 5e-5 \
--freeze_encoder \
--freeze_blocks 9 \
--augmentation_policy finetune \
--epochs 10
```

若使用 CalliPhase mask，首轮采用低比例：

```bash
--mask_mix_probs 0.8 0.1 0.1
```

如果当前数据没有可靠的 JT semantic 文件，不得强行设置 JT 比例大于 0；应先验证文件存在、尺寸正确、坐标经过变换后仍覆盖目标区域。

## 8. 评价指标

### 图像重建指标

- masked 区域 Smooth-L1；
- 全图 L1/L2；
- VGG style loss；
- edge loss。

### 结构指标

- 前景质心误差；
- 行投影误差；
- 列投影误差；
- 前景面积误差；
- 外接框宽高比误差；
- 前景连通区域数量；
- 主要部件之间的中心距离。

### 人工视觉检查

固定参考字和目标字符，比较 A/B/C：

1. 整体重心是否接近目标书法；
2. 左右结构、上下结构比例是否正确；
3. 部件间距和留白是否接近目标；
4. 是否出现局部笔画正确但整体布局错误；
5. 加结构损失后笔法是否退化。

## 9. 结果判据

### 支持结构损失

满足以下多数条件才认为有效：

- C 的结构指标稳定优于 B；
- C 的视觉结体明显优于 B；
- C 没有明显损害笔法和重建质量；
- 改变 `w_structure` 后趋势具有一致性。

### 仅数据有效

如果 B 优于 A，但 C 与 B 接近，说明 CalliPhase 数据分布有帮助，但当前结构损失没有提供额外收益。此时应进一步研究标注利用方式，而不是继续增大权重。

### 结构损失无效或有害

如果 B、C 都不如 A，或 C 的笔法明显退化，应优先检查：

- polygon 是否与 crop/resize 同步；
- mask 是否覆盖了错误区域；
- target 是否被错误归一化；
- 结构损失是否在空白图或极少前景图上产生异常梯度；
- 结构损失数值是否远大于重建和 style loss。

## 10. 后续路线

只有 C 稳定有效后，才进入正式的三阶段训练：

```text
阶段 1：电脑字体大规模预训练
阶段 2：CalliPhase 结构适配，使用显式结构损失
阶段 3：1500 张目标书法风格微调
```

阶段 2 不等于原来的普通微调，因为它的目标是建立书法结构先验，并且使用 Jieti/Bifa 标注或结构损失；阶段 3 才负责适应具体书法家的风格。

如果结构损失有效但 encoder 解冻仍然有害，阶段 2 应继续冻结 encoder，或仅使用 adapter/LoRA；不要因为结构损失有效就自动解冻全部 ViT 层。

## 11. 最小可行实现顺序

推荐按照以下顺序修改，避免一次引入过多变量：

1. 先实现不依赖 CalliPhase polygon 的全局结构损失；
2. 用同一批 1500 张数据跑 A/C 对照；
3. 确认 loss 数值、梯度和 TensorBoard 正常；
4. 再加入 CalliPhase 数据，形成 B/C；
5. 最后才接入 polygon 和 18 类 Jieti 标签；
6. 每一步都保留 checkpoint、命令行和固定推理图。

这样可以区分三个问题：结构损失是否有效、CalliPhase 图像分布是否有效、CalliPhase 细粒度标签是否被正确利用。

## 12. 实际技术路线（当前执行版）

### 阶段 2：书法先验注入

初始化为 `models/vit_base_font/checkpoint-14.pth`，训练数据由电脑字体 replay 与 CalliPhase 组成。阶段 2 同时注入 Bifa 与 Jieti，不把 CalliPhase 当作只解决结体的附加数据。电脑字体 replay 用于保持原有字形拓扑和参考/目标对应能力，CalliPhase 用于提供书法笔画阶段、局部形态、空间区域和跨字符结构先验。

代码实施顺序：

1. 在 `data/pairdataset.py` 中保留原 `(image, target, mask, valid)` 兼容接口，增加可选的 CalliPhase 区域/类别标注字段；先确保 polygon 经 crop、resize、pad、flip 后与图像同步。
2. 在 `data/pair_transforms.py` 中实现标注同步变换，颜色抖动和归一化不作用于标签。
3. 在 `models_train.py` 中增加 `compute_structure_loss()`，首版采用 soft foreground 的行投影、列投影、质心和面积约束；Jieti 18 类分类和笔画关系图放到首版验证之后。
4. 在 `Fontify.forward()`、`forward_loss()`、`engine_train.py`、`main_train.py` 中贯通 `structure_target` 和 `loss_structure`，无标注电脑字体样本的结构损失为 0。
5. 增加 `--structure_loss_weight`、`--calliphase_json_path` 或等价的数据混合配置，并把命令行和数据比例写入输出目录。
6. 阶段 2 优先冻结 encoder；若需要容量，使用 adapter/LoRA，而不是直接解冻大量 ViT block。随机 mask 保持主导，语义 mask 作为辅助，避免再次出现提高 JT mask 后性能下降的问题。

阶段 2 的验收重点不是单纯总 loss，而是 CalliPhase 验证图中的 Bifa 与 Jieti：笔画起收、粗细变化、局部形态、整体重心、部件比例、间距和留白是否同时改善；并确认电脑字体上的基础重建能力没有明显退化。阶段 2 完成后登记最终 checkpoint，作为阶段 3 唯一初始化来源。

#### 阶段 2 首版固定协议

```text
初始化：checkpoint-14.pth
电脑字体：随机 replay 子集
CalliPhase：全部可用训练样本
增强：pretrain
encoder：冻结原参数 + adapter/LoRA
decoder：训练
结构损失：开启
GAN：关闭
```

以上是当前首版待执行配置。电脑字体 replay 的具体比例在实现时作为一个明确超参数记录，但不得使用“全部电脑字体全量重复训练”替代随机 replay；CalliPhase 样本应全部纳入阶段 2 的可用训练池。阶段 2 完成前不得将该协议标记为实验结论。

### 阶段 3：目标书法风格优化微调

阶段 3 只使用 CalliPhase/目标手写书法数据，目标是把阶段 2 的书法 Bifa-Jieti 先验适配到具体书法家或目标风格。默认沿用已经验证较优的冻结 9 层、随机 mask 主导和 BF 参与设置；不把提高 JT mask 或解冻更多层作为默认方案。

阶段 3 的优化按以下顺序逐项实施，每次只引入一个变量：

1. **结构布局损失**：在现有 recon/style/edge/adv 外加入 `w_structure * L_structure`，先试 `0.05` 和 `0.10`，只对有可靠标注的样本启用。
2. **特征蒸馏**：冻结阶段 2 模型作为 teacher，在选定 encoder 层约束当前模型特征，加入 `w_distill * ||f_student-f_teacher||2`，防止小数据微调破坏 Bifa-Jieti 先验。
3. **参数高效微调**：在冻结 ViT block 中加入 adapter、LoRA 或可训练 bias，仅让少量参数表达目标风格；比较 decoder-only 与 adapter/LoRA 的容量和稳定性。
4. **组合方案**：只有单项有效后，才组合结构损失与 adapter/LoRA 或蒸馏；不直接把所有技术同时加入。

阶段 3 每次实验都固定参考字、目标字符、推理预处理和初始化 checkpoint，记录笔法与结体的分项结果。必须同时观察笔法是否退化，避免结构指标提高但书法笔触变平。

### 阶段 3 代码接口

- `models_train.py`：增加可选 `teacher_model`、中间层特征缓存和 `compute_distill_loss()`；adapter/LoRA 模块应明确列入 optimizer 参数组。
- `main_train.py`：增加 `--distill_weight`、`--adapter_type`、`--adapter_dim` 等参数；冻结逻辑必须保证 teacher 不参与反向传播。
- `engine_train.py`：记录 `loss_structure`、`loss_distill`、可训练参数量和梯度范数；生成器与判别器交替更新逻辑保持不变。
- `tools/eval.py` 或新增评估脚本：输出前景质心、行列投影、面积、宽高比和部件间距，并保存固定样例图。

## 13. 阶段完成登记模板

每完成阶段 2 或阶段 3，复制以下模板填写后再进入下一阶段：

```text
阶段：
完成日期：
初始化 checkpoint：
最终 checkpoint：
训练命令：
训练数据版本与比例：
冻结/可训练模块：
结构损失、蒸馏或 adapter 配置：
Bifa 指标与视觉结论：
Jieti 指标与视觉结论：
是否出现基础能力退化：
下一阶段是否允许开始：是 / 否
```
