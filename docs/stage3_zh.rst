阶段 3：参考图条件化与损失标定
==============================

状态（2026-09-20）
------------------

实现代码与独立的 CPU 测试已就绪。完整模型的 CUDA smoke 测试、标定
（calibration）、checkpoint 选择以及阶段 3 训练均尚未运行。
用户需自行提供选定的阶段 2 checkpoint 与 CalliPhase 数据。
所有命令都应在仓库根目录下、在现有的 CUDA/detectron2 训练环境中运行。
VGG19 权重必须位于仓库中现有的路径。临时的本地测试环境不能替代该训练环境。

阶段 3 将 detail 损失固定为实验 2 的定义：Gaussian highpass + 0.1 * Sobel，
kernel 5，sigma 1，batch-region 归一化。此入口不提供逐样本（per-sample）切换；
若保存的配置中请求了该选项，将被拒绝。

数据约定
--------

需提供一个 manifest，其中的 JSON 路径相对于 manifest 文件本身::

    {
      "train": ["train.json"],
      "val_seen": ["val_seen.json"]
    }

只有这两个划分，其他键会被拒绝。每位书家都同时进入两者，因此不产出未见书家的
指标。manifest 可以直接从 CalliPhase 目录结构（``<书家>BF``/``<书家>JT`` 目录下的
``images_text_denoised`` 和 ``semantic_masks``）生成::

    python tools/build_stage3_manifest.py --data-root fontdata_example \
      --output fontdata_example/stage3_json

style_id 是书家名，同一书家的 BF/JT 目录共用；character 取文件名首字；
glyph_id 为 ``<书家>-<文件名主干>``。划分单元是 (书家, 字)：同一个字的全部
BF/JT 记录以及 ``永1`` 这类重复书写落在同一划分，val_seen 的字在所有书家范围内
整体留出。每位书家约 --val-ratio（默认 0.15，即原 train_json_new/val_json_new
的规则）的字进入 val_seen，且不少于 --min-val-characters。缺少 source 字形或
语义 mask 的目标会被丢弃并记录在 split_summary.json 中。

每个被引用的 JSON 是一个列表，包含原有的 image_path、target_path、type
字段，外加显式的 style_id、character、glyph_id（均为非空字符串）。
示例::

    {
      "image_path": "source/永.png",
      "target_path": "writer_a/images/永.png",
      "type": "BF",
      "style_id": "writer_a",
      "character": "永",
      "glyph_id": "writer_a-original-page17-char8",
      "source_dataset": "calliphase"
    }

图片路径相对于 --data-root。glyph_id 标识裁剪前的原始字形；该字形的所有
备选裁剪版本都保留同一个 glyph_id。错误的 glyph_id 无法从像素可靠地检测出来；
审计（audit）会结合提供的标识、目标文件路径以及精确的文件哈希来判断。
同一 split 内允许 JT/BF 重复；跨 split 的目标图重复会被拒绝。共享的源字体
图片是允许的。这并不保证能检测出任意未标注的近似重复样本。

val_seen 中的字符不得出现在 train 中，且 val_seen 的每种风格都必须出现在 train 中。
每个 split 都需要每种风格下有不同字符的参考图；train 还要求每种 type 下也满足
该条件，并且 BF 语义采样需要可用的语义源。训练时的参考图配对同时强制
风格一致且字符不同。调参命令不会读取任何测试集路径。请单独保留一份最终测试集。

评估时从同一评估 split 中确定性地选择参考图，并排除查询字符本身。因此它衡量的
是"带参考图条件的未见字符生成"，而不是没有风格样例可用的场景。若要获得完整的
敏感性覆盖，每种风格至少提供三个字符、且至少两种风格；缺失的备选参考图
（alternate-reference）用例会在 metrics.json 中被显式省略。图片包含
参考图 / 预测 / 目标，不带分数。

预检（Preflight）
-----------------

请使用新的输出目录；已存在的目录会被拒绝，训练也不会自动恢复（resume）。
请将以下示例替换为真实路径::

    python tools/stage3.py audit --manifest /data/calli/manifest.json \
      --data-root /data/calli --output /runs/stage3-audit

    python tools/stage3.py smoke --manifest /data/calli/manifest.json \
      --data-root /data/calli --checkpoint /models/selected-stage2.pth \
      --style-mode reference --output /runs/stage3-smoke

    python tools/stage3.py evaluate --manifest /data/calli/manifest.json \
      --data-root /data/calli --checkpoint /models/selected-stage2.pth \
      --output /runs/stage2-reference-check

对阶段 1 的 checkpoint-14 以及其他阶段 2 候选重复运行 evaluate。依据固定的验证
图片与指标来选择初始化，而不是依据最小训练损失。没有任何命令会自动宣布胜者。
请记录人工决策。

smoke 会真实执行前向/反向传播、一次带裁剪的优化器步骤、checkpoint 保存以及
严格模式重新加载并比较预测结果。CPU 单元测试不能替代它。

在扩展容量之前，先检查 32 样本拟合::

    torchrun --standalone --nproc_per_node=2 tools/stage3.py train \
      --manifest /data/calli/manifest.json --data-root /data/calli \
      --checkpoint /models/selected-stage2.pth --fit32 --updates 300 \
      --output /runs/stage3-fit32

fit32 使用 32 个固定且互不相同的目标、确定性的预处理和完整的查询遮盖。
它不使用语义增强。参数报告每 5 次更新记录一次参数数量、实际学习率、裁剪后的
梯度范数和更新范数。train.jsonl 中的损失分量标注为 last_microbatch_losses，
而非 epoch 平均值；loss_rank0 是 rank-0 上累积的均值，而非全局 DDP 均值。

结构损失标定与受控扫描
----------------------

在选定的初始化上运行标定，不带风格模块::

    python tools/stage3.py calibrate --manifest /data/calli/manifest.json \
      --data-root /data/calli --checkpoint /models/selected-stage2.pth \
      --output /runs/stage3-calibration

使用 32 个风格均衡、每批 2 个样本的 batch，固定查询遮盖，全精度。报告原始值、
下半部分查询区域的像素梯度、可训练参数梯度、两两余弦相似度以及 Gram 矩阵。
内部系数用于平衡像素梯度的中位数；另有一个独立的公共缩放因子用于保持组合
参数梯度范数的中位数不变。系数超出 [0.1, 100]、梯度为零或组合梯度退化都会
阻止标定通过。不要为了强行通过而放宽限制。梯度范数比值并不等于各损失对更新
的独立贡献份额。

原有的结构损失定义保持不变：它们计算拼接图像的全局软前景统计量，而 detail
使用 mask*valid。下半部分查询区域的梯度测量并不重新定义结构损失。

一次只生成一个阶段的命令::

    python tools/stage3_experiments.py internal \
      --manifest /data/calli/manifest.json --data-root /data/calli \
      --checkpoint /models/selected-stage2.pth \
      --semantic-mask-dir /data/calli/font/train/new \
      --coefficients /runs/stage3-calibration/calibration.json \
      --output-root /runs/stage3 --output-json /runs/internal-commands.json

每个条目包含一个可直接用于 subprocess.run(argv, check=True) 的 argv 列表。
它不会被自动执行。在视觉/指标评审之后，用选定的权重和可选的 --coefficients
生成下一阶段：

* internal：等系数 vs 标定系数，两个权重均为 0.05。
* structure：总权重 0/0.025/0.05/0.1，detail 固定为 0.05。
* detail：总权重 0/0.025/0.05/0.1，已选定的 structure 保持不变。
* style：off/reference/constant，两个已选定的损失保持不变。
* local：reference 模型，每次只将一个已选定的损失调为 0.5x 或 2x。已选定为零的
  损失保持为零。比较最终结果时请包含原始的 A/C。

使用 --structure-weight 和 --detail-weight 传递显式决策。若等系数胜出，则省略
--coefficients。使用 --seeds 1 2 对决赛候选及其基线做复现。只有当 checkpoint、
数据指纹、全部设置、随机种子和更新预算都一致时才可复用基线；命令的输出目录
永远不会覆盖先前的运行。

run.json 分别保存 JSON-manifest 指纹与图片/身份指纹。语义遮盖内容不包含在内
（semantic_masks_hashed=false）；请单独保存其数据集版本，遮盖变更后不要复用
基线。

训练默认值与兼容性
------------------

400 次优化器更新；有效 batch 为 128（2 GPU * batch 2 * 累积 32）。单 GPU 需要
--accum-iter 64。冻结前 9 个 block 与 embedding；不使用 GAN。学习率 1e-4，
layer decay 0.8，新条件化模块（conditioner）学习率 3e-4，梯度裁剪 3.0。
学习率先在 40 次更新内爬升，随后余弦衰减。Structure/detail 与 edge 在 40 次更新内
爬升；edge 目标为 0.3。每 50 次更新以及最后一次更新时评估并保存。
入口传入的遮盖模式为 random/JT/BF = 0.8/0/0.2、random 模式内半遮盖概率 0.5，
但 PairDataset 会对每条带语义 mask 的 CalliPhase 记录强制改写模式：JT 记录固定抽
1 层语义 mask，BF 记录固定抽 11 层，因此阶段 3 训练实际上从不使用随机块遮盖或
半遮盖（2026-09-21 决定：保持该行为）。0.8/0/0.2 只对没有语义源的记录生效，
而 manifest 生成器已把这类记录丢弃。训练保留现有的弱 finetune 变换。对于阶段 3 的 checkpoint，评估和两个现有推理入口
均使用方形白色填充与 bicubic 缩放，不再使用旧的 64 像素参考图下采样。

新的阶段 3 训练默认 --vgg-input-mode rgb：在 VGG 模块自身归一化之前，先撤销数据集
归一化。--vgg-input-mode legacy 仅用于受控的兼容性对比；在同一轮扫描内保持一致。
现有训练入口保留 legacy 默认值。

风格条件化使用可见的上半部分参考图 RGB 加可见性共四个通道；被遮盖的参考图像素为
白色，下半部分 GT 永远不会被读取。三个 Conv/GroupNorm/GELU 阶段（32/64/128）、
全局池化和三个零初始化的 head 对最后三个 ViT block 进行调制。constant 分支接收
相同的全 1 输入，使用相同的可训练架构。不使用 style-ID embedding。旧 checkpoint
迁移时只允许缺少新的 conditioner 键；其他缺失/多余的键和尺寸不匹配均视为错误。

报告与验收
----------

每次评估保存 metrics.json 以及 参考图/预测/GT 拼接条。指标只使用下半部分的查询
像素：固定 Gaussian highpass、Sobel 梯度误差、edge F1（阈值 0.1，精确像素匹配）、
前景几何（阈值 0.9）、质心、行/列投影、面积和 bbox 长宽比。这些是评估指标，
不是训练损失的缩放副本。错误/空白参考图的指标仅作诊断用；汇总时只使用正确
参考图的用例::

    python tools/stage3.py summarize --metrics /runs/A/eval-400/metrics.json \
      /runs/B/eval-400/metrics.json --output /runs/comparison.json

请结合固定的盲审图片与验证指标共同决策。比较相同种子的结果以及训练/验证差距。
小于种子间波动的微小变化视为未确认。容量、标定系数、detail 权重或新模块都不预设
为有效。若没有候选优于基线，则停止扩展实验。真实运行之后，将命令、数据哈希、
初始化、最终选定的 checkpoint、图表、指标和人工结论登记到 AGENTS.md 中。

本地验证
--------

    PYTHONPATH=. python -m pytest tests/test_stage3.py -q

测试使用小型 fixture 执行条件化模块以及真实的 encoder/损失方法体；损失测试用替身
替换了 VGG 特征提取器。它们不会实例化完整的预训练 ViT，也不能证明 CUDA/DDP
训练的正确性。
