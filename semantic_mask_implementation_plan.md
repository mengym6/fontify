# 语义遮盖实现方案：代码修改计划

## 核心思路

将 CVAT 标注的语义区域预渲染为 448×448 二值 mask 图，作为第三张图跟 image/target 一起过所有空间变换（nearest 插值），最后下采样到 patch 网格级别。无论 transform 链中有什么随机裁剪/缩放，mask 都自动对齐。

---

## 当前进度

### 已完成

1. **`font-preload/render_semantic_mask.py` 脚本已写好并验证通过**
   - 测试模式（代码内 `TEST_MODE=True`）：随机抽样 N 张图，输出 mask + overlay 对比图
   - 批量模式（`TEST_MODE=False`）：遍历 `NEW_DIR` 下所有字体文件夹自动处理
   - 支持 `CLEAR_OUTPUT=True` 清空输出目录后重新生成
   - 核心逻辑：text RLE 解码 → 笔画 polygon 渲染 → 取交集 → pad+resize(NEAREST) → 保存

2. **验证结果**
   - DongqcJT "樂"：覆盖率 23%（JT 结体标注，几个结构区域）
   - DongqcBF "月"：覆盖率 16.8%（BF 笔画标注，所有笔画并集）
   - 坐标对齐正确，mask 精确贴合字形，无背景溢出

### 待解决的关键问题

**问题：当前脚本把所有标注 OR 成一张图，训练时无法选子集控制遮盖比例。**

BF 标注每张图约 20+ 个笔画部件，全部合并后 ≈ 整个字形。训练时需要只选其中一部分遮盖，剩余部分模型可见（作为"提示"）。

**拟定方案：改为逐标注存储，训练时随机选子集**

输出格式改为每张图一个 `.npy` 文件：

```
semantic_masks/
  月.npy    → shape (N, 448, 448), dtype=uint8, 每层是一个标注的独立 mask
```

训练时 `pairdataset.py` 的加载逻辑：
```python
layers = np.load("月.npy")              # (N, 448, 448)
k = random.randint(1, N)                # 或按目标比例选
indices = random.sample(range(N), k)
combined = np.any(layers[indices], axis=0).astype(np.uint8)  # OR 合并为单张 mask
# 然后 combined 作为 PIL Image 跟 image/target 一起过 transform
```

**待决策：**
- 选多少个标注？固定数量 vs 固定覆盖率 vs 随机范围
- `.npy` 文件可能较大（20层×448×448 ≈ 4MB/图），是否用 `.npz` 压缩？
- 是否同时保留一份合并版 PNG 用于快速可视化调试？

---

## 二、修改 `data/pair_transforms.py`

### 目标

让所有空间变换类支持可选的第三个参数 `mask`，对 mask 统一用 nearest 插值。

### 改动清单

#### 2.1 `Compose.__call__`

```python
# 现在：(img, tgt, interpolation1, interpolation2)
# 改为：(img, tgt, interpolation1, interpolation2, mask=None)
def __call__(self, img, tgt, interpolation1=None, interpolation2=None, mask=None):
    for t in self.transforms:
        img, tgt, mask = t(img, tgt, interpolation1, interpolation2, mask)
    return img, tgt, mask
```

#### 2.2 `PadToSquare.__call__`

```python
def __call__(self, img, tgt, interpolation1=None, interpolation2=None, mask=None):
    def _pad(image, fill):
        w, h = image.size
        size = max(w, h)
        new_img = Image.new(image.mode, (size, size), fill)
        new_img.paste(image, ((size - w) // 2, (size - h) // 2))
        return new_img
    img_out = _pad(img, self.fill)
    tgt_out = _pad(tgt, self.fill)
    mask_out = _pad(mask, 0) if mask is not None else None  # fill=0 即不遮盖
    return img_out, tgt_out, mask_out
```

#### 2.3 `RandomResizedCrop.forward`

```python
def forward(self, img, tgt, interpolation1=None, interpolation2=None, mask=None):
    i, j, h, w = self.get_params(img, self.scale, self.ratio)
    interp1 = InterpolationMode.NEAREST if interpolation1 == 'nearest' else InterpolationMode.BICUBIC
    interp2 = InterpolationMode.NEAREST if interpolation2 == 'nearest' else InterpolationMode.BICUBIC
    img_out = F.resized_crop(img, i, j, h, w, self.size, interp1)
    tgt_out = F.resized_crop(tgt, i, j, h, w, self.size, interp2)
    mask_out = F.resized_crop(mask, i, j, h, w, self.size, InterpolationMode.NEAREST) if mask is not None else None
    return img_out, tgt_out, mask_out
```

#### 2.4 `RandomHorizontalFlip.forward`

```python
def forward(self, img, tgt, interpolation1=None, interpolation2=None, mask=None):
    if torch.rand(1) < self.p:
        img, tgt = F.hflip(img), F.hflip(tgt)
        mask = F.hflip(mask) if mask is not None else None
    return img, tgt, mask
```

#### 2.5 非空间变换（`ColorJitter`、`ToTensor`、`Normalize`、`GaussianBlur`、`RandomErasing`）

只需透传 mask，不做任何处理：

```python
def forward(self, img, tgt, interpolation1=None, interpolation2=None, mask=None):
    # ... 原有逻辑不变 ...
    return img_out, tgt_out, mask  # mask 原样返回
```

`ToTensor` 特殊处理：把 mask 从 PIL 转为 tensor（单通道）

```python
def __call__(self, pic1, pic2, interpolation1=None, interpolation2=None, mask=None):
    mask_tensor = F.to_tensor(mask) if mask is not None else None
    return F.to_tensor(pic1), F.to_tensor(pic2), mask_tensor
```

`Normalize` 不对 mask 做归一化，直接透传。

### 兼容性

所有改动对 `mask=None`（无语义 mask 的场景）完全向后兼容，不影响现有训练流程。

---

## 三、修改 `data/pairdataset.py`

### 目标

加载预渲染的语义 mask 图，跟 image/target 一起过 transform，拼接后过 seccrop，最终转为 patch 级 mask。

### 改动清单

#### 3.1 `__init__` 新增参数

```python
def __init__(self, ..., semantic_mask_dir: Optional[str] = None, mask_coverage_threshold: float = 0.5):
    ...
    self.semantic_mask_dir = semantic_mask_dir  # 预渲染 mask 根目录，None 则走原有随机 mask
    self.mask_coverage_threshold = mask_coverage_threshold  # patch 内 mask 覆盖率阈值
```

#### 3.2 新增 mask 加载方法

```python
def _load_semantic_mask(self, target_path: str) -> Optional[Image.Image]:
    """根据 target_path 推断并加载对应的语义 mask 图"""
    if self.semantic_mask_dir is None:
        return None
    # target_path 示例: "font/train/new/DongqcBF/images_white_bg_mask_denoised/月.png"
    # mask 路径: "font/train/new/DongqcBF/semantic_masks/月.png"
    parts = target_path.split('/')
    # 替换 images_white_bg_mask_denoised → semantic_masks
    for i, p in enumerate(parts):
        if 'images' in p:
            parts[i] = 'semantic_masks'
            break
    mask_path = os.path.join(self.root, '/'.join(parts))
    if os.path.exists(mask_path):
        return Image.open(mask_path).convert('L')  # 单通道灰度
    return None  # 无标注时回退到随机 mask
```

#### 3.3 修改 `__getitem__` 核心逻辑

```python
def __getitem__(self, index):
    pair = self.pairs[index]
    image = self._load_image(pair['image_path'])
    target = self._load_image(pair['target_path'])
    sem_mask = self._load_semantic_mask(pair['target_path'])  # 新增

    # transform 现在返回三元组
    image, target, sem_mask = cur_transforms(image, target, interp1, interp2, mask=sem_mask)

    if self.use_two_pairs:
        # 第二对
        image2, target2, sem_mask2 = cur_transforms(image2, target2, interp1, interp2, mask=sem_mask2)

        image = self._combine_images(image, image2)
        target = self._combine_images(target, target2)
        # mask 也上下拼接
        if sem_mask is not None and sem_mask2 is not None:
            sem_mask = torch.cat([sem_mask, sem_mask2], dim=1)  # (1, 2H, W)
        elif sem_mask is not None:
            sem_mask = torch.cat([sem_mask, torch.zeros_like(sem_mask)], dim=1)
        elif sem_mask2 is not None:
            sem_mask = torch.cat([torch.zeros_like(sem_mask2), sem_mask2], dim=1)
        else:
            sem_mask = None

    # seccrop 也带上 mask
    if (self.transforms_seccrop is not None) and not use_half_mask:
        image, target, sem_mask = self.transforms_seccrop(image, target, interp1, interp2, mask=sem_mask)

    # 生成 patch 级 mask
    if use_half_mask:
        mask = ...  # 原有 half_mask 逻辑不变
    elif sem_mask is not None:
        mask = self._pixel_mask_to_patch_mask(sem_mask)
    else:
        mask = self.masked_position_generator()  # 回退到随机 mask

    return image, target, mask, valid
```

#### 3.4 新增像素 mask → patch mask 转换方法

```python
def _pixel_mask_to_patch_mask(self, sem_mask: torch.Tensor) -> np.ndarray:
    """将像素级 mask (1, H, W) 转为 patch 网格级 mask (Hp, Wp)"""
    # sem_mask: (1, 896, 448) tensor, 值为 0 或 ~1（经过 ToTensor 后 255→1.0）
    patch_size = 16
    h, w = sem_mask.shape[1], sem_mask.shape[2]
    Hp, Wp = h // patch_size, w // patch_size  # 56, 28

    # reshape 为 patch 网格，计算每个 patch 内的覆盖率
    mask_2d = sem_mask[0]  # (H, W)
    patches = mask_2d.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    # patches: (Hp, Wp, patch_size, patch_size)
    coverage = patches.mean(dim=(-1, -2))  # (Hp, Wp)，每个 patch 的平均覆盖率

    # 超过阈值的 patch 标记为遮盖
    patch_mask = (coverage > self.mask_coverage_threshold).numpy().astype(np.int32)
    return patch_mask
```

---

## 四、修改 `main_train.py`

### 改动清单

#### 4.1 新增命令行参数

```python
parser.add_argument('--semantic_mask_dir', default=None, type=str,
                    help='预渲染语义 mask 根目录，None 则使用随机 mask')
parser.add_argument('--mask_coverage_threshold', default=0.5, type=float,
                    help='patch 内 mask 像素覆盖率超过此阈值则标记为遮盖')
```

#### 4.2 传参给 PairDataset

```python
dataset_train = PairDataset(
    ...,
    semantic_mask_dir=args.semantic_mask_dir,
    mask_coverage_threshold=args.mask_coverage_threshold,
)
```

#### 4.3 修改 transform 构建

所有 `PairStandardTransform.__call__` 需要适配三元组返回值（见第二节）。

---

## 五、修改 `data/pair_transforms.py` 中的 `PairStandardTransform`

```python
class PairStandardTransform(StandardTransform):
    def __call__(self, input, target, interpolation1, interpolation2, mask=None):
        if self.transform is not None:
            input, target, mask = self.transform(input, target, interpolation1, interpolation2, mask)
        return input, target, mask
```

---

## 六、不需要修改的文件

| 文件 | 原因 |
|------|------|
| `models_train.py` | 模型只接收 patch 级 mask，接口不变 |
| `engine_train.py` | 训练循环只用 (image, target, mask, valid)，接口不变 |
| `util/masking_generator.py` | 作为 fallback 保留，不修改 |

---

## 七、数据流总结

```
离线阶段：
  COCO JSON + text RLE → render_semantic_mask.py → 448×448 二值 PNG

训练阶段：
  PairDataset.__getitem__:
    加载 image (PIL) + target (PIL) + sem_mask (PIL, 单通道)
    ↓
    transform3: PadToSquare → RandomResizedCrop → ToTensor → Normalize
                （三张图用相同空间参数，mask 用 nearest 插值）
    ↓
    两对拼接：image1+image2, target1+target2, mask1+mask2
    ↓
    seccrop: RandomResizedCrop（三张图同参数）
    ↓
    _pixel_mask_to_patch_mask: (1, 896, 448) → (56, 28) patch 级 0/1
    ↓
    返回 (image, target, patch_mask, valid) → 模型接口不变
```

---

## 八、回退机制

- `--semantic_mask_dir` 不传或为 None → 完全走原有随机 mask 逻辑
- 某张图没有对应的 semantic_mask PNG → 该样本回退到 `MaskingGenerator()` 随机 mask
- `half_mask` 模式优先级最高，不受语义 mask 影响

---

## 九、待确认事项

1. **遮盖比例控制**：语义 mask 的面积不固定（有的笔画多有的少），是否需要补充随机 patch 凑到固定比例（如 50%）？还是允许自然波动？
2. **BF/JT 分开渲染**：每个字体的 BF 和 JT 分别渲染独立的 mask 图，训练时根据 pair JSON 的 type 字段自动选对应目录。
3. **第二对 pair 无标注时的处理**：当前方案是该位置填 0（不遮盖），是否改为随机 mask 填充？
