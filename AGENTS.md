# Fontify 项目说明与开发约束

## 代码规范

- 后续新增和修改的代码必须遵循 PEP 8 规范。
- Python 代码应保持统一的 4 空格缩进、清晰的命名、合理的行长度、规范的导入顺序，并在适当位置补充必要的文档字符串和注释。
- 修改代码后，应根据改动范围执行适当的格式检查、语法检查或项目测试；不得仅以代码能够打开或静态配置存在作为验证结论。

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

当前阶段 2 的 `finetune_loss_diag.sh` 使用 `--no_gan`，实际训练总损失不包含有效 `L_adv`。detail loss 的代码默认权重为 `0.03`，使用 `kernel_size=5`、`sigma=1.0` 的高通和默认 `gradient_ratio=0.5` 的 Sobel 项；诊断脚本的显式覆盖值见下节，它与 structure loss 都只在 `mask*valid` 对应的像素区域内计算。若开启 GAN，注意当前非 DeepSpeed 路径中 D 每个 micro-batch 更新一次，而 G 每 `accum_iter=32` 更新一次；这会显著改变 GAN 动态，不能把阶段 2 的 `--no_gan` 配置直接去掉后视为同等实验。

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
| 阶段 2：书法 Bifa-Jieti 结构适配 | 从 `checkpoint-14.pth` 初始化；电脑字体数据与 CalliPhase 混合 | 同时注入书法笔法和结体先验，减少小数据直接微调造成的遗忘 | 加入显式结构/区域监督；保留电脑字体 replay；随机 mask 为主 | 损失诊断试验 1/2/3 已完成；用户反馈笔锋弱、结构混乱仍存在且有过拟合；进入阶段 3 前仍需选定初始化 checkpoint |
| 阶段 3：目标书法风格微调 | 仅 CalliPhase，多风格参考驱动；从选定阶段 2 checkpoint 初始化 | 验证参考风格注入，并解决 structure 子项/总权重与 detail 总权重 | 冻结前 9 层；实验 2 detail 定义；先损失控制变量对照，再轻量参考条件调制 | 2026-09-20：代码及独立 CPU 合约测试已落地；完整模型 GPU smoke、校准和训练未执行，未验证效果 |

阶段登记规则：每完成一个阶段，必须记录训练命令、数据版本、初始化 checkpoint、最终 checkpoint、验证图、结构/笔法指标和人工结论，并将本表对应状态改为“已完成（日期、checkpoint 路径）”。当前仅阶段 1 可标记为已完成；阶段 2 和阶段 3 不得提前宣称完成。

截至本记录时，`main`/`origin/main` 的 `HEAD` 为 `1e49c80`。阶段 2 诊断入口为 `finetune_loss_diag.sh`，试验 1/2/3 均已完成 35 个 epoch。三组公共配置为：初始化 `checkpoint-14.pth`，固定 1:1 chinese/CalliPhase 混合清单，`augmentation_policy=finetune`，`--no_gan`，`--freeze_encoder --freeze_blocks 9`，`--structure_loss_weight 0.05`，`--detail_loss_weight 0.05`，`--detail_kernel_size 5`，`--detail_sigma 1.0`。试验 1 的 `detail_gradient_ratio=0.0`；试验 2/3 为 `0.1`；仅试验 3 使用每样本归一化。阶段 3 明确不采用该方案。

## 最近 Git 与实现进度

截至 2026-09-19，本地 `main` 与 `origin/main` 均指向 `1e49c80`。与本项目主路线直接相关的近期提交如下：

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
- `ef32818`（2026-09-19）：实现阶段 2 损失诊断试验 1/2/3，加入 detail 每样本归一化开关、region/valid/target 日志和 35-epoch 运行脚本。
- `1e49c80`（2026-09-19）：提交试验 3 前置工具，加入 weighted gradient balance 检查和脚本导入路径修复。

当前工作区未提交状态：

- `AGENTS.md` 及阶段 3 实现尚未提交；进入本次实施前已有 `.gitignore`、`AGENTS.md`、`finetune_font.sh` 未提交修改，保留其原有改动。
- 三组日志已由用户提供至 Downloads，验证图主要在服务器；本次实现未执行服务器训练。

## 试验 1/2/3 实际执行与结论

试验 1/2/3 已分别完成 35 个 epoch，提供的 detail 日志各含 385 条更新记录。试验 1/2 的日志显示：

- 试验 1 全程 `gradient_contribution=0`，`loss_detail=loss_highpass`。
- 试验 2 全程满足 `loss_detail=loss_highpass+0.1*loss_gradient`，最大公式误差约 `9e-7`。
- 两组最后 5 个 epoch 的 `loss_highpass` 均值和变异系数几乎相同；弱 Sobel 没有明显破坏高通损失。
- 试验 2 的训练梯度 pre-clip 变异系数高于试验 1，说明 Sobel 仍带来额外的优化波动。
- 两组 `grad_ok` 均全程为 `111111`，没有发现结构或高频损失断链。
- 默认归一化与每样本归一化的 `tools/check_loss_gradients.py` 都曾显示 `gradient check passed`。但该脚本单样本时不能区分两种归一化。

使用 `tools/check_loss_gradient_balance.py`、两个样本、`checkpoint-14.pth`、冻结 9 层、`structure_weight=0.05`、`detail_weight=0.05`、`gradient_ratio=0.1` 得到的加权梯度占比为：

| 加权损失 | 梯度范数 | 占比 |
|---|---:|---:|
| `recon_weighted` | `1.36076` | `54.18%` |
| `style_weighted` | `0.98977` | `39.41%` |
| `edge_weighted` | `0.08345` | `3.32%` |
| `structure_weighted` | `0.06885` | `2.74%` |
| `detail_weighted` | `0.00862` | `0.34%` |

structure 子项 raw 梯度范数中，row/col 约为 `4.8e-4`，centroid 约 `0.210`，area 约 `1.563`。因此 structure 的主要梯度来自 area/centroid，row/col 几乎无效；area 的梯度还可能与 centroid 发生抵消。

当前结论：structure 存在梯度，但行列子项量级偏小的诊断仍需在更多 batch 上复核。用户确认各配置没有明显解决笔锋弱、结构混乱，且出现一定过拟合；tools 中梯度检查输出符合预期。试验 3 改变了归一化，不能仅凭其 loss 数值较高认定效果更差。当前转入阶段 3，采用实验 2 的 detail 定义，分别验证 structure 子项比例、structure/detail 总权重与风格条件注入。0.05 只是起始对照，不是已验证的最优权重；过拟合也不能证明参数不足。

已确认但仍需训练验证的结论：

- 旧 `test3` 的 `gradient_log.csv` 显示实际 discriminator 有梯度，而 git 中 test3 配置带 `--no_gan`；这说明那次实际运行与 git 配置不一致，不能直接作为 test3 的官方对照。
- GAN 开启时旧代码存在 D 参数同时进入 G/D optimizer 的风险；该问题已修复，但尚未用开启 GAN 的完整训练确认稳定性。
- 梯度裁剪的生成器物化修复已提交，但尚未在完整 `test5` 运行中核对 `pre_clip_grad_norm`、`post_clip_grad_norm` 与 `grad_clip_ratio` 是否连续、合理。
- `test5` 已把 structure/detail 权重提高较多，但没有训练输出、验证指标或人工结体比较，不能宣称其优于 `test4`、A/B 对照或阶段 1。
- 当前阶段 2 首轮仍不应把 GAN、LoRA/adapter、蒸馏或额外 JT 语义标签同时加入。
- 工作区内 `models/vit_base_font/checkpoint-14.pth` 仍是唯一已完成阶段 checkpoint；未发现阶段 2 最终 checkpoint。
- 用户已给出总体视觉反馈：笔锋与结体难题未明显改善；尚未给出阶段 2 最优 checkpoint 的具体路径。
- `tools/check_loss_gradient_balance.py` 的加权梯度结果只来自两个样本和一个初始化 checkpoint，属于问题定位证据，不能单独证明最终结构损失有效性。

## 阶段 3 实现与执行登记（2026-09-20，优先于前文历史方案）

- 当前入口：`tools/stage3.py`，支持 audit、smoke、calibrate、train、evaluate、summarize。完整命令与数据契约在 `docs/stage3.rst`；`tools/stage3_experiments.py` 按 internal/structure/detail/style/local 分阶段生成 argv，不自动挑选赢家或执行整组训练。
- 数据由用户更换为 CalliPhase。额外要求显式 `style_id`、`character`、原始字形 `glyph_id`；manifest 划分 train/val_seen（2026-09-21 起移除 val_unseen，manifest 出现其他键会报错）。检查跨划分路径/hash/原始字形重复；严格同风格、异字符配对。无法由程序证明用户填写的风格身份或 glyph_id 正确，不声称穷尽任意近重复裁剪。
- 2026-09-21 用户决定：9 位书家全部同时进入 train 和 val_seen，不设 val_unseen；val 随机选、不区分 BF/JT；比例沿用 `generate_new_json.py` 的 `VAL_RATIO=0.15`；target 统一用 `images_text_denoised`；暂不设最终测试集。`tools/build_stage3_manifest.py` 以 (书家, 字) 为划分单元、val 字全局留出，生成 `fontdata_example/stage3_json/{manifest,train,val_seen,split_summary}.json`：train 1594 条、val_seen 286 条（15.2%），丢弃 10 个无 source 字形的生僻字；`tools/stage3.py audit` 通过，`image_and_metadata_sha256=5af150dd…1105bc`。`type` 写为 `BF`/`JT`，target 缩放插值因此从阶段 2 `font_*` 的 nearest 变为 bicubic；实测该差别可忽略：CalliPhase 全部 1890 张 target 与 source 都是 448×448，finetune 的 `RandomResizedCrop(448, scale=(0.9999,1))` 在 2000 次采样中 99.6% 为逐像素恒等，其余 0.4% 为 447→448 的一像素拉伸，两种插值互差均值 0.49/255；阶段 3 评估的 `image_tensor` 对 448×448 输入同样恒等（max|Δ|≈6e-8）。
- `data/pairdataset.py` `__getitem__` 对 `source_dataset=='calliphase'` 且带 `.npy` 的样本会无条件把 mask 模式改成 jt/bf semantic，`tools/stage3.py` 传入的 `mask_mix_probs=[0.8,0,0.2]` 与 `half_mask_ratio=0.5` 被绕过，阶段 3 训练实际 100% 使用语义遮盖（JT 抽 1 层、BF 抽 11 层，无随机块、无半遮盖）。用户 2026-09-21 决定保持该行为不改代码；登记实验时按此描述遮盖设置，不得写成 0.8/0/0.2。
- 固定实验 2 detail：`highpass + 0.1 * gradient`，kernel=5、sigma=1，batch-region 归一化；阶段 3 入口不提供每样本归一化选项，配置中若请求该模式则报错。旧阶段 2 的试验 3 接口仅为历史兼容保留，不在阶段 3 使用。
- structure 增加 row/col/centroid/area 四系数（默认全 1）与独立公共倍率；保留原子项定义。代码实查：当前 structure 是拼接图的全局软前景统计，并未像 detail 一样直接乘 `mask*valid`；前文将二者都描述为 masked loss 不准确。本次不悄然修改其定义。校准测量下半 query 像素梯度及可训练参数梯度。
- 32 个风格均衡 batch 校准：预测像素梯度中位数逆比例、系数范围 [0.1,100]，以参数合成梯度中位数匹配公共倍率；零/非有限梯度、超限或退化则停止。范数比例不是独立的优化贡献占比，需同时看梯度夹角和合成梯度。
- 风格模块：上半可见参考 RGB + visibility，三层 Conv/GroupNorm/GELU（32/64/128）和池化，零初始化头调制最后三个 ViT block；off/reference/constant 三组对照。不读取下半 GT，不绕过参考 mask，不使用书家 ID embedding。
- VGG 输入重复归一化已由调用路径确认。旧入口仍默认 legacy；阶段 3 默认 rgb，在 VGG 内部归一化前还原输入。所有新对照必须采用相同模式，不把修正收益混算成风格模块收益。
- 默认训练：400 optimizer updates、40 updates LR/loss warmup、每 50 updates 评价保存；有效 batch=128，旧参数 LR=1e-4、新模块=3e-4、layer_decay=0.8、clip=3、no_gan、冻结前 9 层；入口参数 mask=0.8/0/0.2、random 内 half_mask_ratio=0.5，但实际生效的是上一条所述的全语义遮盖。fit32 使用固定 32 样本与全 query mask，最多 300 updates。所有组独立输出，不 auto-resume。train 加 `--tensorboard` 会在 `<output>/tensorboard` 镜像 train.jsonl 标量、参数组范数、eval 指标均值和条带图（`stage3_experiments.py --tensorboard` 透传）；jsonl/eval 目录仍是正式记录。
- 新 checkpoint 保存 stage3_config；训练与既有推理入口重建风格模块，加载缺失检查只允许旧 checkpoint 迁移时新增模块参数缺失。阶段 3 推理使用统一补白/缩放，移除旧参考字 64px 中间降采样的影响。
- 本地验证：独立 CPU 测试覆盖条件模块、真实 encoder/loss 方法的小张量合约、结构系数、实验 2 公式、梯度、序列化、校准停止条件与数据泄漏检查。loss 测试替代 VGG 特征提取器，不等于完整预训练模型运行。另执行新增代码 lint/语法与 CLI 检查。
- 未完成：用户选定阶段 2 checkpoint；完整 ViT GPU smoke/反向/DDP、校准数值、400-update 对照、三随机种子复核、视觉验收。数据身份与划分及真实数据 audit 已于 2026-09-21 完成（见上）。当前没有阶段 3 最终 checkpoint，不能标记训练或效果验证已完成。
- 实验登记模板：命令/seed、数据 hash、初始化 checkpoint、候选/最终 checkpoint、固定验证图、结构/细节指标、训练验证差距、人工结论。仅记录真实运行；不根据参数提交推断效果。
