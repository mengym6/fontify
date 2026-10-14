# FontDiffuser 复现手册

本文档整理 FontDiffuser 的论文、官方代码、数据格式、权重使用、训练流程、评估方案和常见问题。目标是区分三件事：

1. 论文与官方资源在哪里。
2. 如何用官方 checkpoint 做推理级复现。
3. 如何在自建字体数据集上做训练或 finetune 级复现。

> 结论先行：FontDiffuser 的推理级复现条件较好，因为官方 README 提供了 `FontDiffuser` 和 `SCR` checkpoint 下载入口；但严格从零复现论文训练结果并不完整，因为论文使用的 424-font 数据集没有直接公开，且 `SCR` 从零预训练流程在 README 中仍标为 coming soon。

## 1. 论文信息

**论文题目**

FontDiffuser: One-Shot Font Generation via Denoising Diffusion with Multi-Scale Content Aggregation and Style Contrastive Learning

**发表信息**

- AAAI 2024
- arXiv: `2312.12142`
- 论文链接：
  - arXiv 摘要页: <https://arxiv.org/abs/2312.12142>
  - arXiv HTML: <https://arxiv.org/html/2312.12142>

**官方代码**

- GitHub: <https://github.com/yeungchenwa/FontDiffuser>

**论文核心任务**

给定：

- 一个目标字符的内容图像，通常由标准字体渲染；
- 一个目标风格参考 glyph，来自目标字体；

生成：

- 目标字符在参考风格下的 glyph 图像。

这属于 one-shot font generation。模型本质上是基于扩散模型的图像生成方法，同时加入：

- 多尺度内容聚合；
- style contrastive learning；
- 内容编码器；
- 风格编码器；
- U-Net denoising backbone。

## 2. 官方资源可用性

| 资源 | 官方是否提供 | 说明 |
|---|---:|---|
| 论文 | 是 | AAAI 2024, arXiv 可访问 |
| 代码 | 是 | GitHub 仓库公开 |
| 推理 checkpoint | 是 | README Model Zoo 标注 `FontDiffuser` released |
| SCR checkpoint | 是 | README Model Zoo 标注 `SCR` released |
| 完整训练数据集 | 否 | 只给 `data_examples` 和数据格式 |
| 从零预训练 SCR 脚本 | 不完整 | README 中 `Pre-training of SCR` 标为 coming soon |
| 两阶段训练脚本 | 是 | `scripts/train_phase_1.sh`, `scripts/train_phase_2.sh` |

## 3. 论文训练规模

论文实验中，作者收集了 424 个中文字体：

| Split | 字体数 | 字符数 | 规模 |
|---|---:|---:|---:|
| Train | 400 seen fonts | 800 seen chars | 320,000 glyphs |
| Seen Font, Unseen Content | 100 seen fonts | 272 unseen chars | 27,200 glyphs |
| Unseen Font, Unseen Content | 24 unseen fonts | 300 unseen chars | 7,200 glyphs |
| Unseen Font, Seen Content | 24 unseen fonts | 800 seen chars | 19,200 glyphs |

训练配置来自论文和官方脚本：

| 项目 | 配置 |
|---|---|
| 图像分辨率 | `96 x 96` |
| Phase 1 steps | `440000` |
| Phase 2 steps | `30000` |
| Phase 1 batch size | `16` |
| Phase 2 batch size | `16` |
| Phase 1 lr | `1e-4` |
| Phase 2 lr | `1e-5` |
| 采样器 | DPM-Solver++ |
| Sampling steps | `20` |
| Guidance scale | `7.5` |
| 论文硬件 | 单张 RTX 3090 |

注意：论文训练用的完整 424-font 数据没有在仓库中直接发布。因此严格复刻论文数据分布不可直接完成，只能在自建数据集上按相同协议复现。

## 4. 环境准备

官方 README 的示例环境如下：

```bash
git clone https://github.com/yeungchenwa/FontDiffuser.git
cd FontDiffuser

conda create -n fontdiffuser python=3.8
conda activate fontdiffuser

pip install -r requirements.txt
```

官方 `requirements.txt` 中包含的关键依赖包括：

- `accelerate`
- `diffusers`
- `fonttools`
- `lpips`
- `pytorch-fid`
- `torch-fidelity`
- `pytorch-lightning`
- `transformers`
- `opencv-python`
- `scikit-image`

训练脚本使用 `accelerate launch`，因此建议先配置 accelerate：

```bash
accelerate config
```

如果只做单卡训练，按交互提示选择 single GPU 即可。

## 5. 数据格式

官方 README 中的数据目录示例为：

```text
data_examples/
└── train/
    ├── ContentImage
    └── TargetImage
        ├── font1
        ├── font2
        ├── ...
        └── fontn
```

结合 `dataset/font_dataset.py`，推荐整理为下面这种格式：

```text
data_examples/
└── train/
    ├── ContentImage/
    │   ├── 你.jpg
    │   ├── 我.jpg
    │   └── ...
    └── TargetImage/
        ├── styleA/
        │   ├── styleA+你.jpg
        │   ├── styleA+我.jpg
        │   └── ...
        ├── styleB/
        │   ├── styleB+你.jpg
        │   ├── styleB+我.jpg
        │   └── ...
        └── ...
```

### 5.1 重要实现细节

虽然 README 中示意用了 `.png`，但官方 `font_dataset.py` 里存在 `.jpg` 相关硬编码：

- `ContentImage/{content}.jpg`
- Phase 2 negative sample 使用 `{choose_style}+{content}.jpg`

因此为了少改代码，建议训练数据统一保存为 `.jpg`：

```text
ContentImage/字.jpg
TargetImage/style/style+字.jpg
```

如果想使用 `.png`，需要检查并修改 `dataset/font_dataset.py` 中的硬编码路径。

### 5.2 字体和字符覆盖要求

Phase 2 会为 style contrastive learning 采样负样本。代码逻辑要求：

- 不同 style 目录下最好具有相同字符集合；
- 对于当前目标字符 `content`，其他 style 目录中也应存在 `style+content.jpg`；
- style 数量要大于 `num_neg`。

官方训练脚本中：

```bash
--num_neg=16
```

因此如果你只用一个很小的数据集调试，应该降低 `num_neg`，例如：

```bash
--num_neg=2
```

否则负样本采样可能失败。

## 6. 官方 checkpoint 推理复现

这是最容易完成的复现层级。

### 6.1 下载 checkpoint

官方 README 的 Model Zoo 标注：

- `FontDiffuser`: released
- `SCR`: released

下载后建议整理为：

```text
ckpt/
├── unet.pth
├── content_encoder.pth
├── style_encoder.pth
└── scr_210000.pth
```

其中推理至少需要：

```text
unet.pth
content_encoder.pth
style_encoder.pth
```

### 6.2 单样本采样

官方 `scripts/sample.sh` 示例：

```bash
python sample.py \
  --ckpt_dir="ckpt" \
  --demo \
  --content_image_path="assets/枇.png" \
  --style_image_path="assets/拇.png" \
  --save_image \
  --save_image_dir="outputs" \
  --device="cuda:0" \
  --algorithm_type="dpmsolver++" \
  --guidance_type="classifier-free" \
  --guidance_scale=7.5 \
  --num_inference_steps=20 \
  --method="multistep" \
  --order=2
```

建议先使用官方 `assets/` 中示例图片跑通。

### 6.3 用 TTF 渲染内容字符

`sample.py` 支持通过 `--ttf_path` 和 `--content_character` 生成内容图像。官方示例：

```bash
python sample.py \
  --ckpt_dir="ckpt" \
  --demo \
  --style_image_path="assets/拇.png" \
  --save_image \
  --save_image_dir="outputs" \
  --device="cuda:0" \
  --algorithm_type="dpmsolver++" \
  --guidance_type="classifier-free" \
  --guidance_scale=7.5 \
  --num_inference_steps=20 \
  --method="multistep" \
  --order=2 \
  --content_character="隆" \
  --ttf_path="ttf/KaiXinSongA.ttf"
```

需要确认 `ttf_path` 对应字体能渲染目标字符，否则内容图像可能为空或异常。

## 7. 从自建数据训练

官方训练分为两个阶段。

### 7.1 Phase 1

官方脚本 `scripts/train_phase_1.sh` 使用：

```bash
accelerate launch train_phase_1.py \
  --report_to="wandb" \
  --experiment_name="FontDiffuser" \
  --train_batch_size=16 \
  --eval_batch_size=1 \
  --learning_rate=1e-4 \
  --mixed_precision="no" \
  --num_train_epochs=440000 \
  --save_ckpt_steps=20000 \
  --save_image_epochs=1 \
  --only_save_embeds=True \
  --data_root="data_examples/train" \
  --content_font="font1" \
  --style_font="font2" \
  --resolution=96 \
  --content_start_channel=3 \
  --style_start_channel=3 \
  --reduce_loss="mean" \
  --num_neg=16
```

这里 `num_train_epochs=440000` 实际更像按 step 规模使用。复现时建议在日志中记录：

- optimizer update steps；
- total seen samples；
- batch size；
- checkpoint step。

如果只做 smoke test，可以先改小：

```bash
--num_train_epochs=1000
--save_ckpt_steps=500
```

### 7.2 准备 Phase 1 checkpoint 目录

Phase 2 脚本需要：

```bash
--phase_1_ckpt_dir="phase_1_ckpt"
```

该目录中应包含 Phase 1 训练得到的：

```text
phase_1_ckpt/
├── unet.pth
├── content_encoder.pth
└── style_encoder.pth
```

如果文件名或保存目录不同，需要按 Phase 2 加载逻辑调整。

### 7.3 Phase 2

官方脚本 `scripts/train_phase_2.sh` 使用：

```bash
accelerate launch train_phase_2.py \
  --report_to="wandb" \
  --experiment_name="FontDiffuser" \
  --train_batch_size=16 \
  --eval_batch_size=1 \
  --learning_rate=1e-5 \
  --mixed_precision="no" \
  --num_train_epochs=30000 \
  --save_ckpt_steps=20000 \
  --save_image_epochs=1 \
  --only_save_embeds=True \
  --data_root="data_examples/train" \
  --content_font="font1" \
  --style_font="font2" \
  --resolution=96 \
  --content_start_channel=3 \
  --style_start_channel=3 \
  --reduce_loss="mean" \
  --num_neg=16 \
  --phase_1_ckpt_dir="phase_1_ckpt" \
  --scr_ckpt_path="ckpt/scr_210000.pth" \
  --sc_coe=0.01
```

Phase 2 需要 `SCR` checkpoint。官方 README 提供 SCR checkpoint 下载入口，但从零预训练 SCR 的脚本在 README 中仍标注为 coming soon。

因此推荐两种路线：

| 路线 | 可行性 | 说明 |
|---|---:|---|
| Phase 1 自训 + 官方 SCR + Phase 2 自训 | 高 | 推荐 |
| 从零训练 Phase 1 + 从零预训练 SCR + Phase 2 | 低 | 官方 SCR 预训练流程不完整 |
| 官方 FontDiffuser checkpoint 直接推理 | 最高 | 用于推理级复现 |

## 8. 在新数据集上 finetune

如果你的目标是在已有 FontDiffuser base 上适配新数据集，建议：

1. 使用官方 `FontDiffuser` checkpoint 初始化：
   - `unet.pth`
   - `content_encoder.pth`
   - `style_encoder.pth`
2. 将新数据集整理为官方 `data_root` 格式。
3. 使用 Phase 2 或自定义 finetune 脚本进行小学习率训练。

推荐从下面配置开始：

| 参数 | 建议 |
|---|---|
| resolution | `96`，先对齐论文 |
| lr | `1e-5` 或更小 |
| batch size | 根据显存设置，记录 total seen samples |
| steps | 例如 `10k`, `20k`, `30k` |
| num_neg | 数据集 style 数量足够时用 `16`；小数据先调低 |
| checkpoint | 每固定 step 保存 |

为了公平比较，应报告：

- base checkpoint 来源；
- finetune data；
- finetune steps；
- total seen glyph samples；
- batch size；
- image resolution；
- negative sample 数量；
- 是否使用 SCR；
- 是否冻结部分模块。

## 9. 评估建议

官方仓库主要提供训练和采样流程。论文报告了图像质量和感知指标，但实际复现时建议自行整理评估脚本。

推荐指标：

| 指标 | 方向 | 用途 |
|---|---:|---|
| L1 / MAE | 越低越好 | 像素误差 |
| SSIM | 越高越好 | 结构相似度 |
| LPIPS | 越低越好 | 感知相似度 |
| FID / KID | 越低越好 | 分布级视觉质量 |
| OCR Acc | 越高越好 | 字符内容是否正确 |
| Style Retrieval Acc | 越高越好 | 风格是否接近 reference |
| Human Preference | 越高越好 | 字体视觉主观质量 |

建议测试协议：

```text
Seen Style, Unseen Content
Unseen Style, Seen Content
Unseen Style, Unseen Content
```

如果用于你的新数据集论文，建议额外报告：

- no-finetune vs NewSet finetune；
- old-data finetune vs NewSet finetune；
- public-control finetune vs NewSet finetune；
- cross-dataset generalization。

## 10. 常见问题和排查

### 10.1 图片扩展名错误

症状：

```text
FileNotFoundError: ContentImage/某字.jpg
FileNotFoundError: TargetImage/style/style+某字.jpg
```

原因：

代码中部分位置硬编码 `.jpg`。

处理：

- 优先将训练数据保存为 `.jpg`；
- 或修改 `dataset/font_dataset.py`，统一支持 `.png`。

### 10.2 style 数量太少

症状：

```text
negative sample 采样失败
pop from empty list
```

原因：

`num_neg=16` 要求有足够多的其他 style。

处理：

小数据调试时使用：

```bash
--num_neg=2
```

### 10.3 字符覆盖不一致

症状：

```text
某些 style 下找不到 style+char.jpg
```

原因：

Phase 2 需要在其他 style 中找到同一 content 的 negative target。

处理：

- 保证每个 style 目录中字符集合一致；
- 或在 data loader 中加入缺失样本跳过逻辑。

### 10.4 训练数据不是论文原数据

问题：

官方没有发布完整 424-font 训练集。

处理：

论文复现时必须明确写：

```text
We reproduce the training protocol on our collected dataset, because the original training font set is not publicly released.
```

不要声称严格复现原论文数据结果。

### 10.5 SCR 从零预训练不可直接完成

问题：

README 中 `Pre-training of SCR` 标为 coming soon。

处理：

- 使用官方 released `scr_210000.pth`；
- 或自行实现 SCR 预训练流程；
- 在论文或报告中说明 SCR checkpoint 来源。

## 11. 推荐复现层级

| 层级 | 目标 | 是否推荐 |
|---|---|---:|
| Level 1 | 用官方 checkpoint 跑通 sample | 强烈推荐 |
| Level 2 | 用官方 checkpoint 在自建测试集上评估 | 强烈推荐 |
| Level 3 | Phase 1 自训 + 官方 SCR + Phase 2 自训 | 推荐 |
| Level 4 | 完全从零训练，包括 SCR | 不推荐，官方流程不完整 |
| Level 5 | 严格复现论文 424-font 结果 | 不可严格完成，原数据未公开 |

## 12. 最小可执行复现清单

### 推理级复现

- [ ] clone 官方仓库。
- [ ] 安装依赖。
- [ ] 下载官方 FontDiffuser checkpoint。
- [ ] 跑通 `scripts/sample.sh`。
- [ ] 保存生成图像。
- [ ] 在固定测试集上批量采样。
- [ ] 计算 LPIPS / SSIM / OCR Acc 等指标。

### 训练级复现

- [ ] 准备 `ContentImage/*.jpg`。
- [ ] 准备 `TargetImage/style/style+char.jpg`。
- [ ] 确保所有 style 具有一致字符集合。
- [ ] 根据 style 数量设置 `num_neg`。
- [ ] 跑通小步数 Phase 1 smoke test。
- [ ] 跑完整 Phase 1。
- [ ] 整理 `phase_1_ckpt/`。
- [ ] 下载或准备 `scr_210000.pth`。
- [ ] 跑 Phase 2。
- [ ] 用固定 test split 批量生成。
- [ ] 计算指标并保存可视化。

## 13. 来源链接

- Paper arXiv: <https://arxiv.org/abs/2312.12142>
- Paper HTML: <https://arxiv.org/html/2312.12142>
- Official GitHub: <https://github.com/yeungchenwa/FontDiffuser>
- Phase 1 script: <https://raw.githubusercontent.com/yeungchenwa/FontDiffuser/main/scripts/train_phase_1.sh>
- Phase 2 script: <https://raw.githubusercontent.com/yeungchenwa/FontDiffuser/main/scripts/train_phase_2.sh>
- Sample script: <https://raw.githubusercontent.com/yeungchenwa/FontDiffuser/main/scripts/sample.sh>
- Dataset loader: <https://raw.githubusercontent.com/yeungchenwa/FontDiffuser/main/dataset/font_dataset.py>
- Requirements: <https://raw.githubusercontent.com/yeungchenwa/FontDiffuser/main/requirements.txt>
