# Fontify 项目说明与开发约束

## 项目目标

Fontify 是一个基于上下文学习（in-context learning）的单/少样本字体生成模型。输入为参考字形图像与需要生成的目标字形图像（训练时目标图像用于监督），输出为目标字体的重建图像。主模型是带 MAE 式遮盖输入的双流 Vision Transformer；仓库同时包含字体数据预处理、训练、批量推理和少样本推理代码。

## 目录与职责

- `main_train.py`：命令行参数、增强、数据集/采样器、模型初始化、预训练加载、冻结策略、AdamW、DDP/DeepSpeed、断点续训和 epoch 主循环。
- `models_train.py`：`Fontify`、ViT block、解码器、判别器以及重建/Gram 风格/VGG/边缘/对抗损失。
- `engine_train.py`：混合精度训练、梯度累积、生成器与判别器交替更新、日志和验证。
- `data/pairdataset.py`、`data/pair_transforms.py`、`data/sampler.py`：JSON 成对样本读取、同步图像增强、随机/半图/语义 patch mask、加权采样和分布式采样。
- `font-preload/`：字体 PNG、白底、裁剪缩放、去噪、反色、语义标注 mask 等预处理工具。
- `fontdata_example/`：字体转 PNG 和 JSON 生成示例；`train_json_new/*.json`、`val_json_new/*.json` 由训练脚本读取。
- `eval/`、`docs/eval/`：批量及少量参考字推理；`tools/`：评估和 TensorBoard 导出。
- `train_vit_base_font.sh`：4 GPU 预训练示例；`finetune_font.sh`：2 GPU、小学习率、部分冻结和语义遮罩微调示例。

## 数据契约

每个 JSON 是字典列表，至少包含 `image_path`（参考/已知字） 、`target_path`（目标字体字形）和 `type`（样本类别，如 `font_<字体名>`、JT、BF）。路径相对 `--data_path`；图像被转为 RGB。默认 `use_two_pairs=True`：同一 `type` 再随机采样一对，沿高度拼接为 `(3, 2H, W)`，因此默认模型输入尺寸为 `896x448`，patch 为 `16`，patch 网格为 `56x28`、共 `1568` 个 token。返回 `(image, target, mask, valid)`，图像经 ImageNet mean/std 归一化，`mask` 是 `(56,28)` 的 0/1 patch mask，1 表示遮盖，`valid` 与 target 同形状。

训练遮罩可为：`MaskingGenerator` 随机块、下半图强制遮盖（`half_mask_ratio`）、由 `semantic_masks/*.npy` 或 COCO annotation 生成的 JT/BF 语义遮罩。`mask_mix_probs` 按 `random, JT semantic, BF semantic` 归一化抽样；`semantic_only_epochs` 可使早期只采样 JT 随机遮盖。验证集通过 `half_mask_ratio=1.0` 强制遮盖目标区域。训练采样器是带 JSON 文件均衡权重的 `WeightedRandomSampler`，再包装为分布式采样器。

阶段 2 的 `train_json_mix`/`val_json_mix` 是固定 1:1 chinese/CalliPhase 混合清单。chinese 样本的 `image_path` 使用 `ttf/source` 下与 target 同名的字形图；CalliPhase 样本优先使用自身的 `semantic_masks/*.npy`。BF 的 `.npy` 当前按“每个非 text 标注一层”生成，因此 `num_mask_annotations_bf` 表示从全部起笔/中笔/收笔标签中随机抽取的单标签数量，而不是抽取笔画种类。生成固定混合 JSON 使用 `tools/build_stage2_mix_json.py`。

## 前向与梯度流

`Fontify.forward` 将参考图 `imgs` 与目标图 `tgts` 分别 patch embed；目标流中被 mask 的 token 替换为可学习 `mask_token`，两流分别加 `segment_token_x/y` 和绝对/相对位置编码。两流沿 batch 维拼接，通过 24 层 ViT-L（或 12 层 ViT-B）；第 3 个 block 后两流逐 token 平均融合，并从多个深度抽取特征。四个特征拼接，经线性层恢复 patch 像素，再由卷积解码器输出 `pred`，形状为 `(B,3,896,448)`。

生成器总损失为

$$L=L_{recon}+L_{style}+w_{edge}L_{edge}+w_{structure}L_{structure}+w_{detail}L_{detail}+w_{adv}L_{adv}.$$

`L_recon` 默认是仅在 `mask*valid` 区域计算的 Smooth-L1（`loss_func=smoothl1`）；`L_style` 是冻结 VGG19 特征的 Gram style L1（VGG content 不计入总损失）；`L_edge` 是温和 Gaussian+Sobel 边缘图的 L1；`L_structure` 是反归一化灰度前景的行/列投影、质心和面积损失；`L_detail` 是固定 Gaussian 高通与 Sobel 梯度损失；`L_adv` 是判别器对生成图判为真的 BCE-with-logits。JT-only 阶段 edge/adv 权重为 0；同步阶段按起始 epoch 和持续时间线性 warmup，默认最终权重 `adv=0.4`、`edge=0.3`。即使 `w_adv=0`，判别器分支仍保留在生成器计算图中以满足 DDP static graph。

当前阶段 2 的 `finetune_font.sh` 使用 `--no_gan`，实际训练总损失不包含有效 `L_adv`。detail loss 的代码默认权重为 `0.03`，使用 `kernel_size=5`、`sigma=1.0` 的高通和 `gradient_ratio=0.5` 的 Sobel 项；当前 `test5` 的显式覆盖值见下节，它与 structure loss 都只在 `mask*valid` 对应的像素区域内计算。若开启 GAN，注意当前非 DeepSpeed 路径中 D 每个 micro-batch 更新一次，而 G 每 `accum_iter=32` 更新一次；这会显著改变 GAN 动态，不能把阶段 2 的 `--no_gan` 配置直接去掉后视为同等实验。

生成器反向传播来自总损失：`NativeScaler` 在 bfloat16 autocast 下缩放、可选范数裁剪（默认 3.0），按 `accum_iter` 累积后更新。每个 batch 随后单独训练判别器：冻结非 discriminator 参数，真实 target 标签为 1，`pred.detach()` 标签为 0，使用独立 AdamW；判别器学习率为生成器的 `0.1`，betas 为 `(0.5,0.999)`。生成器 AdamW 默认 betas `(0.9,0.999)`，基础学习率按有效 batch `batch_size*accum_iter*world_size/256` 缩放（显式 `--lr` 时覆盖），配合 warmup、层衰减和 bias/norm 零 weight decay。

开启 GAN 时，discriminator 参数不得进入生成器 `param_groups_lrd`；当前实现先临时关闭 discriminator 的 `requires_grad`，构造 G optimizer 后再单独构造 D optimizer。D 的 `optimizer_d.step()` 后应立即 `discriminator.zero_grad(set_to_none=True)`，避免残留 D 梯度混入生成器梯度监控或通过 G optimizer 被二次更新。

`--grad_log_interval N` 开启时，rank 0 会把分组梯度写入 `output_dir/gradient_log.csv`。CSV 记录 `reported_grad_norm`、`pre_clip_grad_norm`、`post_clip_grad_norm`、`grad_clip_ratio` 以及各参数组范数；`torch.nn.utils.clip_grad_norm_()` 返回的是裁剪前范数，因此不要把 `reported_grad_norm` 直接当作裁剪后范数。

## 训练、加载与冻结

默认从 MAE ViT-Base checkpoint 加载；形状不匹配的 decoder/mask token 会删除，位置编码可插值。`--freeze_encoder` 冻结 patch embedding、位置/segment/mask token 和指定数量的 ViT blocks；`--freeze_blocks=-1` 时连最终 norm 一起冻结，只训练 decoder（以及启用时的 discriminator）。`--no_gan` 会冻结判别器并完全移除生成器对抗项。训练支持 DDP、DeepSpeed ZeRO、自动恢复和每 epoch checkpoint；输入默认应保持纵向二倍尺寸，否则 `patchify/unpatchify` 的断言会失败。

## 推理与评估

批量推理配置在 `infer_font.sh`（checkpoint、参考字目录、source/GT 目录）；只有少数参考字时运行 `eval/infer_few_font.py`。评估代码依赖与训练相同的 896x448 拼接和归一化约定。修改模型、mask 或损失时应同时检查 TensorBoard 图像（输入、遮盖目标、预测、GT）以及 `tools/eval.py` 指标。

## 修改注意事项

保持 JSON 字段和相对路径兼容；改变拼接方向、patch 尺寸或图像比例时必须同步修改 `patchify/unpatchify`、`args.window_size`、mask 生成器和推理脚本。改变损失权重/冻结范围/学习率会直接影响收敛和字体细笔画，需记录实际配置并运行至少一次可复现的前向或训练 smoke check。依赖环境以 Python 3.9、PyTorch 2.3、CUDA 12.1 为基准，另需 detectron2 和 VGG19 权重。

## 结体问题的原理验证方案（当前共识）

已有事实是：电脑字体大规模预训练后，约 1500 张手写书法数据微调时笔法可以学习，但结体几乎不能学习；减少冻结层数或提高 JT 遮罩比例的实验反而变差。因此不能继续假设“解冻更多层”或“增加 JT mask”必然有效。更可能的原因是：Jieti 标注在当前代码中只转换成二值遮罩，没有使用 18 类空间标签及笔画关系；Jieti 遮罩会提高重建难度但不直接形成结构表示；小数据解冻会造成预训练结构先验的灾难性遗忘。

首轮只做严格消融，固定同一个 `checkpoint-14.pth`、数据划分、参考字、推理字符、训练步数、随机种子和 GAN 设置，运行三组短实验：A 为现有基线，B 加入 CalliPhase 但仍只用现有损失，C 在 B 基础上增加结构损失。建议首轮关闭 GAN（`--no_gan`），CalliPhase 的 random/JT/BF 比例从 `0.8/0.1/0.1` 起步，避免 JT 遮罩占多数。

代码映射：`data/pairdataset.py` 保留原四元组接口并可选返回结构标注；`data/pair_transforms.py` 对 polygon/区域标签同步执行几何变换；`models_train.py::Fontify.forward_loss` 增加 `structure_loss_weight`，首版用预测与目标的前景行投影、列投影、质心、面积和宽高比差异；`engine_train.py` 记录 `loss_structure`；`main_train.py` 增加 `--structure_loss_weight`（先试 `0.05`、`0.1`）和 CalliPhase 数据路径/混合参数，并保存完整配置。无标注样本只计算原损失。

结构损失建议为：

$$L_{structure}=L_{row}+L_{col}+L_{centroid}+L_{area}+L_{aspect}.$$

先比较 A/B/C 的固定参考推理图和结构指标：前景质心、行列投影、外接框宽高比、部件间距，以及整体重心和空间留白。若 B/C 均不优于 A，优先检查标注坐标经过裁剪缩放后的映射及推理预处理一致性，不再继续调大 JT 比例。若 C 稳定改善，再建立三阶段路线：电脑字体预训练 → 使用显式结构监督的 CalliPhase 结构适配（冻结 encoder 或 adapter，并保留少量电脑字体 replay）→ 1500 张目标书法风格微调；这才区别于原先直接从电脑字体 checkpoint 进入目标书法微调的路线。

## 与用户讨论形成的完整问题判断与建议

用户已在电脑字体大规模预训练模型 `models/vit_base_font/checkpoint-14.pth` 基础上，用约 1500 张手写书法字进行续训。历史结果显示：笔法可以学习，但结体特征几乎没有学会；减少 encoder 冻结层数、提高 JT 遮罩比例的实验效果更差。因此后续实验不得默认“解冻更多层”或“增大 JT mask”一定有效。

当前失败的可能机制包括：Jieti mask 可能遮盖大块结构区域，破坏参考图与目标图的条件关系；Jieti 标注在当前代码中只转换为 patch 二值 mask，没有使用 18 类空间标签、笔画顺序及笔画关系；解冻更多层可能造成电脑字体预训练结构先验的灾难性遗忘；1500 张图对跨笔画、跨字符的结体规律覆盖不足，而笔法属于局部属性更容易学习。

因此微调目标应是保留预训练结构能力，只学习书法风格残差。优先保留当前冻结策略，比较 decoder/风格 token、adapter、LoRA 或可训练 bias；参数高效微调比直接解冻大量 block 更可能兼顾稳定性和风格容量。

Jieti mask 不应直接作为主要重建任务。可先以随机 mask 为主、BF semantic 为辅（如 `random=0.8, BF=0.2, JT=0`），同时加入显式结构约束。结构损失可比较预测图和目标图的前景质心、水平/垂直投影、外接框宽高比、skeleton/距离变换及语义区域面积和中心位置：

$$L_{layout}=\sum_k\left\|\phi_k(\hat y)-\phi_k(y)\right\|_1.$$

若 encoder 允许少量更新，可用冻结的预训练模型作为 teacher，约束微调模型中间特征：

$$L_{distill}=\left\|f_{finetune}(x)-f_{pretrain}(x)\right\|_2^2.$$

结构损失、结构蒸馏和 adapter/LoRA 必须先通过控制变量实验验证，不应一次全部加入。还必须检查推理时参考字拼接、目标 mask 覆盖、`source_dir`/`ref_dir`、字符集合、`896x448` 尺寸、归一化、patch 排列及微调 checkpoint 是否正确加载。

实验优先级为：复现当前冻结基线；加入 CalliPhase 但不加结构损失；加入全局结构损失；测试冻结 encoder + adapter/LoRA；最后再接入 polygon、18 类 Jieti 标签和笔画关系损失。只有显式结构损失稳定优于对照，才建立“电脑字体预训练 → CalliPhase 结构适配 → 1500 张目标书法风格微调”的三阶段路线；中间阶段必须使用显式结构监督，不能只是普通数据微调。

### 当前最终选择

两组列表中，第一组是数据/损失消融，第二组是后续技术路线。当前首选是第一组中的 **A/C 对照**：A 为原始冻结 encoder 微调基线，C 为相同配置下加入 CalliPhase 与显式结构损失。若历史基线不可复现，先单独重跑 A；若已有可靠历史基线，可直接实施 C，但必须保留 A 的同配置对照。由于已有结果表明单纯提高 JT mask 和解冻更多层会变差，首轮可暂时跳过 B（CalliPhase 但无结构损失）。若 C 有效，再测试第二组 D（冻结 encoder + adapter/LoRA）；若 C 无效，再补做 B 以区分 CalliPhase 数据分布和结构损失的作用；暂不直接做 E 组合实验。

在 detail loss 刚落地时，阶段 2 曾计划做代码级 A/B 对照：固定 `structure_loss_weight=0.05` 和 `--no_gan`，A 使用 `detail_loss_weight=0`，B 使用 `detail_loss_weight=0.03`；只有 B 的验证高频误差稳定下降且不损害结构指标时，才把 detail loss 纳入阶段 2 正式协议。该 A/B 对照没有完成，后续训练脚本已推进到 `test5` 并改成更高的 structure/detail 权重。因此当前不能再把上述权重计划当成正在执行的对照，也不能仅凭参数提交宣称 detail loss 或 structure loss 有效。

## 当前项目主要行动路线与阶段登记

用户已确认：已有最优续训配置是从百万电脑字体预训练 checkpoint `checkpoint-14.pth` 出发，只使用约 1500 张手写书法数据，随机 mask 主导、BF 参与、冻结 9 层；该路线能够学习笔法但结体不足。当前主路线改为三阶段，且阶段 2、阶段 3 同时关注 CalliPhase 的 Bifa（笔法）和 Jieti（结体），不能把 CalliPhase 仅理解为结体数据。当前不再安排数据消融实验，只实施阶段 2 先验注入和阶段 3 优化微调。

| 阶段 | 数据与初始化 | 主要目标 | 主要训练约束 | 状态 |
|---|---|---|---|---|
| 阶段 1：通用字形预训练 | 数百万张电脑字体；从 MAE 初始化 | 学习通用字形拓扑、参考/目标对应、基础笔画与结构重建能力 | 原始 Fontify 协议 | **已完成**：使用 `models/vit_base_font/checkpoint-14.pth` |
| 阶段 2：书法 Bifa-Jieti 结构适配 | 从 `checkpoint-14.pth` 初始化；电脑字体数据与 CalliPhase 混合 | 同时注入书法笔法和结体先验，减少小数据直接微调造成的遗忘 | 加入显式结构/区域监督；保留电脑字体 replay；优先冻结 encoder 或使用 adapter；随机 mask 为主，语义 mask 为辅 | 代码与混合数据已实现；训练待执行（当前 `test5` 配置已提交，尚无阶段 2 最终 checkpoint） |
| 阶段 3：目标书法风格微调 | 只使用 CalliPhase/目标手写书法数据；从阶段 2 checkpoint 初始化 | 学习具体书法家或目标风格的笔法和结体表现 | 沿用已验证较优的冻结 9 层、随机/BF 主导配置；第二组优化（蒸馏、布局损失、adapter/LoRA）在此阶段逐项消融 | 待执行 |

阶段登记规则：每完成一个阶段，必须记录训练命令、数据版本、初始化 checkpoint、最终 checkpoint、验证图、结构/笔法指标和人工结论，并将本表对应状态改为“已完成（日期、checkpoint 路径）”。当前仅阶段 1 可标记为已完成；阶段 2 和阶段 3 不得提前宣称完成。

截至本记录时，`main`/`origin/main` 的 `HEAD` 为 `32fb453`，当前阶段 2 训练入口为 `finetune_font.sh` 的 `test5` 配置：初始化 `checkpoint-14.pth`，混合 JSON 使用固定 1:1 chinese/CalliPhase 清单，`augmentation_policy=finetune`，`--no_gan`，`--freeze_encoder --freeze_blocks 9`，`--structure_loss_weight 2.0`、structure warmup 从 epoch 6 开始且提交版持续 8 个 epoch，`--detail_loss_weight 0.5`、detail warmup 从 epoch 10 开始且持续 6 个 epoch，edge 最终权重 `0.3`，BF 单标签数 `11`，JT 标签数 `1`，验证 TensorBoard 上限 `76` 张，梯度日志每 5 个 optimizer update 写一次。当前未提交工作区把 structure warmup duration 从 `8` 改为 `6`；该差异尚未提交。`test5` 只是已提交的执行参数，不代表结构损失或 detail loss 已训练验证有效；工作区内没有 `models/finetune_stele_test4`、`models/finetune_stele_test5` 目录、最终 checkpoint、`gradient_log.csv` 或验证结论。

## 最近 Git 与实现进度

截至 2026-09-18，本地 `main` 与 `origin/main` 均指向 `32fb453`。与本项目主路线直接相关的近期提交如下：

- `e99b127`（2026-09-14）：整合 finetune/pretrain 入口，引入 `augmentation_policy`，把损失权重控制移到 shell 脚本，更新原数据 source；同时提交本说明和结构损失原理验证文档。
- `fcf8320`（2026-09-15）：生成阶段 2 固定比例混合 JSON，并加入 `tools/build_stage2_mix_json.py`。
- `ba41105`（2026-09-16）：补齐 CalliPhase JT/BF 训练与验证 JSON、结构先验和路径更新逻辑。
- `51d0cc7`（2026-09-17）：chinese 样本改用原 source；BF mask 改为从 `.npy` 的非 text 单标签中抽取，而不是抽取笔画种类；更新 test2 的 structure/edge 权重。
- `20c44b4`（2026-09-17）：加入分组梯度 CSV 监控 `gradient_log.csv`，提交 test3 配置。
- `75e4fff`（2026-09-18）：加入固定 Gaussian 高通 + Sobel detail loss、`--detail_*` 参数、训练端 detail/high-pass/gradient 分量日志，并修复 discriminator 进入生成器 optimizer、D 残留梯度污染和 CSV tensor 字符串问题。该提交最初还加入了验证集 high-pass/gradient error 图像。
- `c82ad6a`（2026-09-18）：提交 `test4` 配置，输出目录为 `models/finetune_stele_test4`。
- `cc61ad2`（2026-09-18）：回退 `75e4fff` 加入的验证集组件指标和 high-pass/gradient error 写图路径，恢复原 TensorBoard 图像拼接方式。当前验证阶段不再写这两张误差图；训练阶段仍记录 detail、highpass、gradient 和 detail weight。
- `039044d`（2026-09-18）：把配置名改为 `test5`，将 structure/detail 权重从 `0.05/0.03` 提高到 `0.5/0.3`，并把 structure/detail warmup 起点及持续时间改为 6/8 和 10/6。
- `e6d1000`（2026-09-18）：修复梯度裁剪。`model.parameters()` 是生成器，当前代码先物化为列表，确保 unscale、裁剪前范数、裁剪和裁剪后范数使用同一组参数；否则前后两次遍历可能得到空参数集，导致裁剪和日志失真。
- `32fb453`（2026-09-18）：提交当前 `test5` 参数，进一步把 `structure_loss_weight` 调到 `2.0`、`detail_loss_weight` 调到 `0.5`。

当前工作区未提交状态：

- `.gitignore` 额外加入 `*.md`；`AGENTS.md` 已跟踪，因此该规则不会自动停止 `AGENTS.md` 的跟踪，但会影响其他新 Markdown 文档。
- `finetune_font.sh` 相对 `HEAD` 将 `structure_warmup_duration` 从 `8` 改为 `6`。
- `AGENTS.md` 正在补充本次长期记忆，修改尚未提交。

已确认但仍需训练验证的结论：

- 旧 `test3` 的 `gradient_log.csv` 显示实际 discriminator 有梯度，而 git 中 test3 配置带 `--no_gan`；这说明那次实际运行与 git 配置不一致，不能直接作为 test3 的官方对照。
- GAN 开启时旧代码存在 D 参数同时进入 G/D optimizer 的风险；该问题已修复，但尚未用开启 GAN 的完整训练确认稳定性。
- 梯度裁剪的生成器物化修复已提交，但尚未在完整 `test5` 运行中核对 `pre_clip_grad_norm`、`post_clip_grad_norm` 与 `grad_clip_ratio` 是否连续、合理。
- `test5` 已把 structure/detail 权重提高较多，但没有训练输出、验证指标或人工结体比较，不能宣称其优于 `test4`、A/B 对照或阶段 1。
- 当前阶段 2 首轮仍不应把 GAN、LoRA/adapter、蒸馏或额外 JT 语义标签同时加入。
- 工作区内 `models/vit_base_font/checkpoint-14.pth` 仍是唯一已完成阶段 checkpoint；未发现阶段 2 最终 checkpoint。
