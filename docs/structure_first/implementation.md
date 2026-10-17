# runc：实现与执行契约

2026-10-17。架构schema `structure-first-bifa-v1`，runner schema `structure-first-bifa-runtime-v1`；用户最新指令授权自主执行，不再逐项等待。技术质量门槛与基线保护仍执行，委托检查与用户人工检查分别记录。

## 实际主链

`Fontify.forward/predict_full` 仅在显式 `enable_structure_first` 后进入新路径；不启用不创建新模块、不改旧RunB计算。`util/checkpoint.py` 检查新metadata与state键一致并严格重建，拒绝漏键/多键、Tip或GScale迁移。

阶段一：先清除target下半真实GT，包括NaN/Inf，再用下半全遮的原MAE取得F和四格浅层特征。原JT Head输出Q、4槽occupancy/existence/classes。旧fusion残差保留，新条件交互增量为：

- 下半F `[B,28,28,3072]` 投影为Query `[B,784,256]`。
- Q `[B,784,256]` 投影；预测S的occupancy×existence对编码后的slot/class信息做空间加权，得到S条件场。Q与S构成KV空间条件，附一枚上半可见参考特征池化token。
- 单层、4头、256维Cross-Attention与FFN，经零初始化3072维投影输出新残差，叠加旧fusion后进入原Decoder。
- BF-only对每个head来源按样本detach；F与新adapter/decoder图像梯度保留。H/J仅有效JT，不能把BF缺标当全零GT。

阶段二：完整冻结阶段一，在eval/no_grad下实际计算P0/S；将原target上半参考与被既有BF patch mask遮盖的P0下半拼接，原电脑字上下拼图保持部署布局。此输入再经过冻结原MAE得到F_bf。没有小ViT或新CNN Encoder。

BF Refiner的可训练残差为：冻结Decoder在 `F_bf+新adapter` 与原F_bf上的输出差，加轻量RGB残差头。adapter显式用F_bf、预测Q/S与可见参考；RGB头用masked P0、预测gate、参考与mask。两个末层零初始化，初始P1逐值等于P0。最终P1=P0+Delta，允许整个下半字修正。

MAE与base Decoder可no_grad；adapter之后的Decoder虽然权重冻结，仍需保留对输入的梯度，否则会切断新adapter。所有原模型参数始终不进入阶段二optimizer。每次train()后重新固定原模块eval，防drop_path漂移。

## 参数与冻结

生产尺寸新结构adapter 2,509,824参数；BF Refiner 2,513,027，总新增5,022,851。
阶段一默认冻结ViT，训练JT Head、旧fusion、结构adapter、decoder_embed/decoder_pred，共59,232,704参数；可配置decoder_pred-only共8,884,672。该选项不是已证明更好的架构。
阶段二仅BF Refiner的2,513,027参数可训练。推理1轮需要两次ViT编码与三次Decoder调用（coarse一次、refine base/差分各一次）；实际速度/显存须由runtime报告记录。

## 数据与损失

`data/structure_first_dataset.py` 包装原DualDataset，不改缓存、1092条manifest或同书家同池参考选择。coarse三池均可访问，refine仅原BF-bearing 818条索引。一次原采样的下半mask保存为bf_patch_mask，repeat到像素后与valid取交；不重复抽样、不合并polygon。

BF统计记录requested/selected/source_layers/nonempty_source/nonempty_after_crop/target_patch_coverage，不增加RNG。JT-only在refine入口报错；空BF局部域贡献0并保持完整batch分母。

`util/structure_first_loss.py`：coarse为下半SmoothL1(P0,Gaussian(GT))+J(P0,原GT)+H；Gaussian7/1.5/reflect是用户确定的初始值。refine为下半完整SmoothL1/Gram/原edge，加实际mask区域SmoothL1/edge，加双标样本J(P1)。H仅监控。没有GAN/Tip/新detail。

Smooth/edge逐字按有效通道像素归一再对完整batch均值；J/H保持原逐字内部归一后fullbatch均值。评测J逐JT累加除JT count。VGG只读下半完整图，224px、ImageNet归一化一次；局部edge先在完整下半计算再mask，避免人为遮盖边界。

新训练outer weights与J内部三系数显式配置并另做多batch梯度标定；旧RunB lambda_full/wJ/wH不用于新目标。评测J系数与S分母保持封存固定口径，独立于训练系数。

## 推理与限制

自动mask得分=预测P0墨迹patch均值×预测slot gates最大值；按得分降序、raster index稳定打破并列，候选数量32、阈值0.1、1轮是可复现初始配置。空候选保留排序第一patch。多轮仅读上一轮模型输出。

独立infer命令仅接收目标电脑字、参考书法字、参考电脑字路径，不读取目标GT、annotation或JT/BF类别。训练BF11层与推理自动patch的分布差异仍未解决，报告mask/中间图不等于质量一致性证明。

## 工程入口与检查

`tools/structure_first.py` 提供audit/smoke/probe/train/evaluate/infer；`finetune_structure_first.sh` 调用独立入口。每次使用新输出目录，checkpoint只创建新文件。配置示例 `configs/runc/coarse_correctness.json` 的单位权重仅用于正确性，不是正式标定权重。

训练accum窗口全局JT count=0时将head grad设None，避免AdamW的weight decay/旧momentum改动；mixed窗口保留有效JT梯度，DDP同步count。阶段切换重建optimizer/DDP。CPU/Gloo的小模型静态图检查与真实CUDA/DDP检查分开记录。

图像预览由 `tools/runc_lowpass_preview.py` 在服务器真实固定val图生成，包含GT与sigma1/1.5/2的448整字和同位置细画放大；预览卷积与训练loss Gaussian在CPU逐值一致。阈值连通数/墨迹变化只是辅助代理量，不能单独认定语义拓扑正确。

完整模型CUDA、bf16、多GPU、训练与最终质量的证据见后续smoke与执行记录；本实现说明本身不宣称已训练有效。
