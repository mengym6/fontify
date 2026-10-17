# runc：T0 源码与基线审计

日期：2026-10-17。实施起点 `a7d7007`；分支 `codex/structure-first-bifa`。
本报告中的源码事实来自实施前版本；新接口和验证结果另见 implementation/smoke。

## 依据与授权

- 执行规格：`plan_20261017.md`，原文 SHA256 `039bbc4e3f215e4bb034afcc0f95611e4754c7ffda2c1d4fdff4de24f39fd952`。
- 已读 10-16 handoff、当前 PROGRESS/AGENTS、Q/F 与 G 尺度结论。
- Fontify 原论文与作者公开源码：[论文](https://www.sciencedirect.com/science/article/abs/pii/S0950705126000973)、[作者代码](https://github.com/YingXu124/Fontify/)。论文提供设计背景，当前执行路径以本仓源码为准。
- 用户提供的 CalliPhase 论文已本地阅读，其 phase polygon 与空间角色概念用于核对标注语义；论文文件和正文不随本次代码发布。
- 用户本轮明确授权新分支、实现、push、服务器 pull、训练测试和 sub 分工；阶段一结构验收后才能进入阶段二长训的条件继续有效。

## 原 Run B 的实际计算依赖

`models_train.py:558`：双流分别 patch embed，目标流在 token 层 mask；第3个 block 后两流平均，ViT-B 在 block 2/5/8/11 抽头。
`models_train.py:887`：拼接抽头为 `[B,56,28,3072]`，下半 `[B,28,28,3072]`。

`util/dual_view.py:96`：头读取下半电脑字/目标流浅层特征与四抽头，参考 context 来自上半两流。
`util/dual_view.py:109`：`Q=[B,784,256]`；4槽，存在性 `[B,4]`、占用 `[B,4,28,28]`、方位角色 `[B,4,20]`。`G` 是预测 gates 加权的槽/类别场，共276维；与Q拼为532维。

`util/dual_view.py:140` 与 `models_train.py:617`：联合 LayerNorm → Linear(532,3072) → 下半四抽头残差 → decoder_embed → 像素卷积 decoder。生产 decoder 宽度64（`models_train.py:1063`）。因此原模型已经使用F/Q/G，不将“第一次接入布局”当作本轮增量。

旧训练有两次独立前向：语义/随机遮盖 GT 的 `P_pen/L_pen`，全遮的 `P_full/L_full/J/H`（`models_train.py:920`）。新 runc 的第二次前向必须显式读实际P0，不能只更换旧前向顺序。

## 数据与 BF 11 层

- `main_train.py:531` 使用 `data/dual_dataset.py`，不走旧 PairDataset。
- `font-preload/render_semantic_mask.py:132`：每个非text非空annotation独立一层。
- `data/dual_dataset.py:235`：`random.sample(range(K),min(11,K))` 无放回，union>0；K=0报错，K<11取全部，没有补齐、重采、类别聚合或空间合并。
- `data/dual_dataset.py:247`、`:302`：四张图与语义层共用一次crop，mask NEAREST，16×16像素均值严格>0.1映射patch；两行分别抽样。空/全mask原实现无fallback。
- 新路径复用同一次 `get_pair()` 的最终下半patch mask；上半保留完整参考。局部loss域是该mask映射的像素域，与valid取交。
- train 下半query，上半同书家同池均匀参考，可抽自身；val保持封存固定配对行顺序。阶段二通过索引过滤BF-bearing 818条，不改原manifest或参考协议。

本地 seed0、identity geometry、每个BF-bearing目标一次的实测：1092目标=466双标+352BF-only+274JT-only。818个BF文件层数min/median/max=3/21/49，85个少于11层，30个恰11层，无空NPY层。patch覆盖率min/median/p95/max=4.464%/20.791%/33.693%/46.556%，空/全覆盖均0。这些统计不代表随机crop和整个训练频率分布。

## Loss 与梯度约束

`util/dual_loss.py:115`：J只看目标下半、真实未低通GT，逐字按有效部件归一，最终对完整batch平均；BF零项仍占分母。相对质心至少2部件，预测质量shielding可能停用部件，须记录count。
`util/dual_loss.py:39`：H离散实例匹配在no_grad中；占用/角色按匹配槽归一，存在性按4槽均值，JT门控后对完整batch平均。

`util/dual_view.py:147` 只在旧fusion入口detach layout。新adapter直接读取Q/slots/logits时必须逐来源按has_jt隔离。BF-only的共享F图像路径继续学习。

原图像loss（`models_train.py:645`）recon按mask×valid归一，但VGG/edge看整张双拼图，并会原地修改valid。新loss显式只看下半完整字，不能复用原函数后称作新的full/local监督。

AdamW即使grad=0也可能因weight decay/已有momentum更新参数。一个完整accum窗口全局JT count为0时应在step前将head grad设为None；混合窗口保留JT梯度，多卡需全局归约。

## 新接口的最小实施选择

1. 用户后续明确选择轻量Cross-Attention：hidden_dim=256、num_heads=4、num_layers=1；Query来自F下半，Key/Value来自Q与预测S编码，并保留可见参考条件。末层零初始化。初期slot池化候选被本决定替代，不作为最终实施。
2. 阶段一先冻结ViT，训练head/fusion/新adapter/decoder；decoder可配置full或pred-only。只是正确性候选，不据此断言容量足够或效果最佳。
3. 阶段二target是 `[参考书法字;P0]`，原电脑字双拼作为imgs，mask是 `[0;既有BF下半mask]`。冻结encoder再编码，新refiner可训练；decoder需传递输入梯度，不能被no_grad切断。
4. `P1=P0+delta`，末层零初始化；允许整图修正。初始adapter梯度可能为0，先更新末层后再检查内部梯度。
5. 每次train()后阶段一强制eval，防drop_path造成P0漂移；阶段切换重建optimizer/DDP。

## 服务器实物与保护边界

2026-10-17现场：`/root/autodl-tmp/fontify_dual_runb`，起始 `codex/q-spatial-control @ a7d7007`，2×RTX4090空闲，约187GB空闲。Python3.9.0/PyTorch2.3.0+cu121/timm0.3.2。保留既有stage3_json四文件删除及未跟踪数据/软链接，不reset/clean。

Fix Edge best epoch50：`models/finetune_dual_view_runb_fix_20261015/checkpoint-best.pth`，SHA256 `57ac8276e3468ad92ce2aff033736128b33862a4f922618c51910717166ae852`。无Tip/GAN/GScale初始化。

train manifest SHA256 `cf7fea0e08e831ebd5ff6bbee0a463af2018c016c090610ca1fdf7d6b68db545`；fixed240 SHA256 `3b9373097017165b5634122acf62db67b304c4166d34ed97ca2305643f331a30`，101JT/139BF。cache869个NPZ逐SHA通过，递归5959显式数据路径均存在。

评测coef与S分母在 `models/dual_view_pad_20261014/`。评测coef={centroid:1.2499891188181484,logsigma:0.22683043282211374,shape:1}；S分母={L1_JT:0.764010760453668,L1_BF:0.6017321960317145,J:0.5714969584847441}。它们是封存评测口径，不冒充新训练loss权重。

新功能关闭的真实checkpoint逐值回归、CUDA bf16、DDP、模型训练与推理验证仍需在新代码集成后单独运行；T0审计与旧代码62项CPU回归不替代这些证据。

用户后续明确Gaussian初始kernel_size=7、sigma=1.5、padding=reflect；正式训练前必须审核固定样本sigma=1.0/1.5/2.0对图。GT低通仅用于重建loss，J仍用原始GT。未经该图像确认，不启动正式训练；代码正确性测试可继续。

后续授权与执行更新：用户随后明确“无需等待授权，自行按计划推进”，图像审查与技术验收由Codex委托执行并记录，不伪称人类确认。完整CUDA/NCCL检查及阶段一340更新现已完成；结构验收未通过，阶段II未正式训练。当前状态以results.md为准，以上T0“待执行”条目保留为审计时点记录。
