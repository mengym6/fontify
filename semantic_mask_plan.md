---
name: semantic-mask-plan
description: 用CVAT标注的语义mask（笔画BF/结体JT）替换Fontify现有随机遮盖策略的完整上下文
metadata: 
  node_type: memory
  type: project
  originSessionId: d2328276-dce9-48e8-92c6-3e904c297a71
---

## 目标

用 CVAT 标注的语义 mask 替换 Fontify 现有的随机矩形遮盖策略。

**Why:** 现有随机遮盖与字的笔画/结构无关，语义遮盖能让模型学会"根据剩余笔画补全缺失部件"或"根据半边结构补全另一半"。

**How to apply:** 修改数据加载和 mask 生成逻辑，从标注 JSON 中读取区域信息生成 patch 级 mask。

---

## 标注文件结构

路径规则：`font/train/new/{字体名}{BF或JT}/annotations/instances_default.json`

格式：COCO 格式，从 CVAT (cvat.ai) 导出。

顶层键：`licenses`, `info`, `categories`, `images`, `annotations`

### BF（笔法）文件

- 51 个 categories：各笔画×起笔/中笔/收笔 + 一个 `text`
- 笔画类型：横、竖、撇、捺、提、竖提、点、横钩、竖钩、弯钩、卧钩、横折、竖折、撇折、斜钩、横折钩、竖弯钩
- 每张图约 23 个标注（密集）
- 示例：DongqcBF 有 99 张图、2308 条标注

### JT（结体）文件

- 17 个 categories：左、右、上、中、下、左下、右上、左上、右下、中上、中下、下左、下中、下右、上左、上右 + `text`
- 每张图约 3.4 个标注（稀疏）
- 示例：DongqcJT 有 84 张图、283 条标注

### annotation 字段

- `id` — 标注唯一 ID
- `image_id` — 对应 images 列表中的图片
- `category_id` — 对应 categories 列表中的类别
- `bbox` — `[x, y, width, height]`
- `segmentation` — 两种格式（见下）
- `area` — 区域面积
- `iscrowd` — 0
- `attributes` — `{'occluded': False}`

### 两种 segmentation 格式

**1. 多边形（笔画部件用）：**
```json
"segmentation": [[x1, y1, x2, y2, x3, y3, ...]]
```
几个到十几个顶点坐标，用 `cv2.fillPoly` 可画成 mask。

**2. RLE（text 整字轮廓用）：**
```json
"segmentation": {"counts": [25924, 2, 875, 3, ...], "size": [H, W]}
```
列优先展平，交替记录"连续背景像素数"和"连续前景像素数"。用 `pycocotools.mask.decode` 解码。
JSON 中极大段的数字就是 RLE 编码（整字轮廓的压缩像素 mask），短列表数字是多边形顶点坐标（笔画部件轮廓）。

---

## 完整数据流（从原始图片到模型输入）

### 阶段 1：离线预处理（一次性）

原始碑帖图片（images/，非正方形，如 1152×877）
 → 白底黑字提取（convert_to_white_bg）
 → 膨胀 mask 去噪（mask_denoise）
 → pad 成正方形 + resize 到 448×448（pad_and_resize.py L14-23）
 → 存入 images_white_bg_mask_denoised/

标注坐标基于 images/ 中的原始尺寸图片，与 448×448 训练图存在尺寸偏差。

### 阶段 2：训练数据加载（pairdataset.py）

1. 从 pair JSON 加载 (source, target) 路径对
2. 加载 source 图（标准字体 448×448）和 target 图（碑帖 448×448）
3. 过 transform 链（见下）
4. 两对 pair 上下拼接：source1+source2 → imgs，target1+target2 → tgts
5. 可能过 seccrop 二次裁剪
6. 生成 mask（patch 网格级别的 0/1 矩阵）
7. 返回 (imgs, tgts, mask, valid)

### 阶段 3：模型前向（models_train.py）

```
imgs → patch_embed → x (source embedding，完整保留)
tgts → patch_embed → y → 被 mask 的 patch 替换为 mask_token
x, y 拼接后送入 ViT encoder → decoder → 预测被遮盖区域
```

---

## 训练时 Transform 链详解

### 裁剪 vs 遮盖（重要区分）

- **裁剪（crop）**：数据增强，改变模型"看到的图片范围"，裁掉的部分直接丢弃，模型不知道其存在。对 source 和 target 都操作，用相同随机参数。
- **遮盖（mask）**：图片完整输入模型，但标记某些 patch 为"不可见"，模型知道那里有内容需要预测。只对 target 操作。

执行顺序：先 crop 确定"看哪块" → 再 mask 确定"看到的这块里哪些 patch 被遮住"。

### Transform 配置（main_train.py）

**transform_train（L225-233）— 通用训练，增强最强：**

| 步骤 | 操作 | 作用 |
|------|------|------|
| 1 | PadToSquare(fill=255) | 非正方形图 pad 成正方形，fill=255 即白色填充（白底黑字不引入干扰） |
| 2 | RandomResizedCrop(input_size, scale=(min,1.0)) | 随机取图中一块区域，resize 到 input_size |
| 3 | ColorJitter(0.4, 0.4, 0.2, 0.1) | 随机扰动亮度/对比度/饱和度/色调 |
| 4 | RandomHorizontalFlip() | 50% 概率水平翻转 |
| 5 | ToTensor() | PIL Image → Tensor，[0,255] → [0,1] |
| 6 | Normalize(mean, std) | ImageNet 均值/方差标准化 |

**transform_train2（L234-238）— 弱增强：**

`PadToSquare(fill=255) → RandomResizedCrop(input_size, scale=0.9999~1.0) → ToTensor → Normalize`

跟 transform_train3 一样，几乎不裁剪只 resize，无颜色/翻转增强。用于非 font 类型但不需要强增强的场景。

**transform_train3（L239-243）— font 专用，实际走这条：**

`PadToSquare(fill=255) → RandomResizedCrop(input_size, scale=0.9999~1.0) → ToTensor → Normalize`

font 类型走这条路径（pairdataset.py L109 判断 `"font" in pair_type`）。
scale=0.9999~1.0 意味着几乎不做随机裁剪，只 resize 到 input_size。
无 ColorJitter、无 HorizontalFlip。

**transform_train_seccrop（L244-246）— 二次裁剪：**

`RandomResizedCrop(input_size, scale=(min_random_scale, 1.0), ratio=(0.3, 0.7))`

在双图拼接后再做一次非正方形裁剪（当 use_half_mask=False 时触发，pairdataset.py L130-133）。

**transform_val（L248-252）— 验证用：**

`PadToSquare(fill=255) → RandomResizedCrop(input_size, scale=0.9999~1.0) → ToTensor → Normalize`

跟 transform_train3 一样，只 resize 不裁剪。验证时不做任何数据增强。

### RandomResizedCrop 详解（pair_transforms.py L158-176）

这是 transform 中唯一改变图片尺寸/内容区域的操作。做两件事：

**第一步：get_params 决定裁哪块**
1. 目标面积 = 原图面积 × random(scale[0], scale[1])
2. 目标宽高比 = random(ratio[0], ratio[1])
3. 由面积和宽高比算出裁剪框 h, w
4. 随机选左上角 (i, j)

**第二步：F.resized_crop(img, i, j, h, w, self.size)**
- 从图片中取 img[i:i+h, j:j+w] 区域
- resize 到 self.size（即 input_size）
- interpolation=3 是 bicubic 插值

**对 source 和 target 用相同的 (i,j,h,w) 参数**，保证两张图裁剪区域一致。

对 transform3（font 专用）：scale=0.9999~1.0，裁出来的框几乎覆盖整张图，随机性极小。
- input_size=448 时：448→448，等于没变
- input_size=224 时：448→224，等比缩小一半，无随机 crop

### PadToSquare 详解（pair_transforms.py L40-51）

将非正方形图 pad 成正方形。fill=255 = 白色填充（RGB 最大值）。
因为训练图是白底黑字，补边用白色不会在边缘引入干扰。
对已经是 448×448 正方形的 font 图片，此步骤不触发。

### 路由逻辑（pairdataset.py）

- L109-113: `"font" in pair_type` → 使用 `transform3`（弱增强）
- L129-133: `use_half_mask` 为 False 且 `transforms_seccrop` 存在时，对拼接后的图再做 seccrop
- L139-144: mask 生成在所有 transform 之后

### 关键结论

font 类型走 `transform3`，坐标变换是**确定性的**（只有 resize，没有随机 crop）。
但 `seccrop` 会在双图拼接后引入随机裁剪，需要注意。

### 尺寸变换代码定位

| 操作 | 文件 | 行号 |
|------|------|------|
| 离线 pad+resize 到 448 | font-preload/pad_and_resize.py | L14-23 |
| PadToSquare | data/pair_transforms.py | L40-51 |
| RandomResizedCrop | data/pair_transforms.py | L158-176 |
| RandomHorizontalFlip | data/pair_transforms.py | L190-200 |
| font 走 transform3 | data/pairdataset.py | L109-113 |
| seccrop 二次裁剪 | data/pairdataset.py | L130-133 |
| mask 生成 | data/pairdataset.py | L139-144 |
| MaskingGenerator 随机矩形 | util/masking_generator.py | L65-93 |
| input_size 参数 | main_train.py | L50 |
| patch 网格计算 | main_train.py | L221 |

---

## 现有遮盖逻辑

文件：`util/masking_generator.py` + `data/pairdataset.py` + `models_train.py`

### mask 作用对象（重要）

**mask 只对目标字（target/tgt）操作，参考字（source/imgs）始终完整可见。**

模型代码（models_train.py L534-543）：
```python
x = self.patch_embed(imgs)   # source 参考字 embedding → 不受 mask 影响
y = self.patch_embed(tgts)   # target 目标字 embedding → 被 mask 替换
w = bool_masked_pos...
y = y * (1 - w) + mask_token * w   # 只替换 y 中被 mask 的 patch
```

### Patch 网格概念

**patch 是 ViT 把图片切成小块的基本单位，mask 在 patch 级别操作，不是像素级别。**

`patch_size = 16`（默认，models_train.py L296）。PatchEmbed 用 16×16 卷积核（stride=16）扫过图片：

```
输入图片 448×448 像素
 → 切成 (448/16) × (448/16) = 28×28 = 784 个 patch
 → 每个 patch 是 16×16×3 = 768 维像素块
 → 经过线性映射变成一个 embedding 向量
```

patch 网格大小 = input_size // patch_size（main_train.py L221）。

因为训练时 target 是两张图上下拼接，实际网格高度翻倍：
- 单张图：28×28 patch 网格
- 拼接后：56×28 patch 网格（mask 覆盖这个范围）

### MaskingGenerator 工作方式（util/masking_generator.py L65-93）

在 patch 网格上（不是像素上）随机放矩形块：

```
MaskingGenerator(input_size=28, num_masking_patches=118):
  → 在 28×28 网格上反复尝试放随机面积、随机宽高比的矩形
  → 凑够 118 个格子被标记为 1
  → 返回 28×28 的 0/1 矩阵
```

每个被标记为 1 的格子 = 图片中对应的 16×16 像素区域被遮盖。

### 训练时的图片布局

拼接后的图片结构（左右各一张大图）：
- 左侧大图（source/imgs）：参考字1（上）+ 参考字2（下），完整可见
- 右侧大图（target/tgts）：目标字1（上）+ 目标字2（下），被 mask 部分遮盖

mask 网格覆盖右侧整张拼接图（目标字1 + 目标字2 视为一整张图）。

### half_mask 机制

`half_mask`：把 patch 网格的下半部分全设为 1 → 第二对 pair 的目标字完全不可见。
模型必须纯靠参考字来生成第二对的目标字（"纯生成"模式）。

比例配置（main_train.py L74, L263, L266）：
- **训练集**：`half_mask_ratio = 0.1`（默认）→ 100 张中约 10 张是 half_mask
- **验证集**：`half_mask_ratio = 1.0`（写死）→ 100% 都是 half_mask（测试纯生成能力）

---

## 数据路径关联

训练 pair JSON（如 `train_json_new/font_train_DongqcBF.json`）格式：
```json
{
  "image_path": "ttf/SourceHanSansSC-Bold/月.png",
  "target_path": "font/train/new/DongqcBF/images_white_bg_mask_denoised/月.png",
  "type": "font_DongqcBF"
}
```

关联方式：`target_path` 中的文件名（如 `月.png`）== annotations JSON 中 `images[].file_name`

从 `type` 字段可推断标注 JSON 路径：
- `font_DongqcBF` → `font/train/new/DongqcBF/annotations/instances_default.json`

---

## 标注坐标与训练图的尺寸偏差

标注基于 images/ 中的原始图片（如 1152×877），训练图是 448×448。
预处理变换（pad_and_resize.py）：原图 → pad 成正方形（居中，白色填充）→ resize 到 448。

对于原图 1152×877：
- max_side = 1152, offset_x = 0, offset_y = (1152-877)//2 = 137
- 坐标变换：x_new = (x + offset_x) * (448/max_side), y_new = (y + offset_y) * (448/max_side)

训练时 transform3 的 RandomResizedCrop(scale=0.9999~1.0) 几乎不裁剪，只做 resize。
所以 font 路径下坐标变换是确定性的（只有 resize 比例）。

---

## 语义 mask 对齐方案

### 方案 A：mask 当图片，跟训练图一起过 transform（推荐）

预渲染 mask 为 448×448 二值图，作为第三张图跟 image/target 走相同空间变换。
- 优点：坐标自动对齐，不管 transform 怎么变都正确
- 缺点：需改 transform 链支持三张图；需预渲染 mask 图片

### 方案 B：mask 拼为第四通道，跟图片一起变换

mask 作为额外通道拼到 image 上（4通道），一起过 transform，最后拆出。
- 优点：不需要额外存储；坐标自动对齐
- 缺点：ColorJitter 等颜色变换不能作用于 mask 通道

### 方案 C：记录 transform 随机参数，事后对 mask 施加相同变换

transform 执行时记录 crop 参数 (i,j,h,w) 和 flip 状态，再对 mask 做相同操作。
- 优点：transform 链改动最小
- 缺点：新增空间变换时容易漏掉

---

## 待决策问题

1. **BF/JT 混用策略**：训练时 BF pair 用笔画 mask、JT pair 用结构 mask（天然分开），还是需要同一张图随机选用 BF 或 JT？
2. **遮盖比例控制**：
   - 方案 A：直接用标注区域实际面积（语义完整，面积波动大）
   - 方案 B：随机组合多个标注区域凑目标比例（BF 选几个笔画拼）
   - 方案 C：先取语义区域再对 patch 级 mask 做补充/裁剪至目标比例
3. **input_size 实际值**：决定坐标变换的缩放比（待用户确认）
