双视图 run B：服务器启动记录（2026-10-12）
=========================================

用户已授权代码复核、Git 上传、服务器测试并启动训练，要求有效 batch 64。
每次服务器 pull 前均先运行 ``source /etc/network_turbo``。
本文记录已发生的检查与启动状态；51 epoch 完成及研究效果仍待后续验证。
代码版本为 ``codex/dual-view-runb @ 407fddd``，基于 ``d93ba30``，main 保留原状。

当前启动
--------

正式训练于北京时间 2026-10-12 14:21:44 进入运行阶段。
14:26:50 的只读审计确认 epoch 0/1/2 已完成，后续 epoch 正在运行，
共记录 63 次优化器更新，四卡实际参数为 ``4 x 2 x 8 = 64``。
实际 hparams 与合同的 51 epoch、seed 0、lr 0.00195699、warmup 10、
style 14.73、layer decay 0.8、drop path 0.1、clip 3、全解冻/no GAN 一致。
每 epoch 只消费 17 个完整累积窗口，不执行不足有效 batch 64 的尾批更新。
所有已记录训练损失、验证 S 和梯度均有限，固定 val 每轮为 101 JT+139 BF。
``checkpoint-probe-2.pth`` 已保存；``checkpoint-probe-25.pth`` 待对应 epoch。
早期分数不用于判断最终收益。

- 服务器工作树：``/root/autodl-tmp/fontify_dual_runb``。
- 训练输出：``models/finetune_dual_view_runb``。
- 持久化 screen：``45175.dual_runb_20261012``，流水线脚本 PID 45177。
- 启动脚本：``models/dual_view/run_pipeline_20261012.sh``，服务器本地封存。
- 流水线状态/事件：``models/dual_view/pipeline_20261012/{status.json,events.jsonl}``。
- 训练日志：``models/dual_view/pipeline_20261012/training.log``。
- 实际参数：``models/finetune_dual_view_runb/hparams.json``。
- 每轮指标：``models/finetune_dual_view_runb/log.txt``。
- 微批匹配/槽位/裁剪：``models/finetune_dual_view_runb/dual_diagnostics.jsonl``。
- 按组梯度：``models/finetune_dual_view_runb/gradient_log.csv``。

版本化只读启动凭据为 ``docs/dual_view_launch_20261012.json``，包含现场参数、
已完成 epoch、有效更新数、checkpoint 文件列表、GPU 状态和科学文件 SHA256。
本地完整科学结果副本在本工作树 ``outputs/dual_runb_launch_20261012``。
训练仍在继续，因此凭据是带时间戳的快照，不能视为之后的实时状态。

数据与几何
----------

服务器原图缺失已补齐，原图归档含 1828 个路径、1186 个唯一 MD5，
SHA256 为 ``6ef72be3e1cbb5b7b6c00ba0b7a4b6840547869df8ffffd9fc3764ab9cb04d83``。
服务器重建仍为 1092 张：466 双标注、352 BF-only、274 JT-only，val 不合并。
已封存全部记录所用原图、GT、source、BF NPY、JT 标注与固定参考输入。

服务器既有 GT 使用直接缩放，本地既有 GT 使用方形补白。
原补白缓存 ``fontdata_example/dual_targets_v1`` 有 540 条坐标失败，保留但不用。
新缓存显式使用 ``native_geometry=direct_resize``，路径为
``fontdata_example/dual_targets_direct_v1``；869 个目标通过原坐标门槛，
最低墨迹覆盖率 0.901106、最低 IoU 0.882620、最大质心偏差 0.015357。
原图、GT、BF NPY 与 0.8 墨迹覆盖门槛均保持原值。
这项修复适配已有服务器预处理，不改变既定实例监督定义。
历史划分存在 154 个原图 train/val 重叠；沿用既定划分，不能宣称无泄漏泛化。

已完成前置实验
--------------

本地当前全套 251 项测试通过，新增精度代码 lint/编译通过。
服务器上一提交的全套 239 项测试通过；当前提交已有实际 GPU 标定复测通过。
两者不混写成当前版本的服务器全套结果。

真实 checkpoint-14、CUDA/bf16 下，关闭 dual 时旧/新输出与损失逐值相等，
键集合相等；零融合两视图与旧输出相等，下半 GT 扰动不影响全遮预测。
初次和标定后四卡 NCCL/DDP static_graph 测试均通过，后者从预热权重出发，
BF-only/JT-only/双标注/混合各 2 次有效 batch 64 更新。
标定后峰值分配 18.1087 GiB/卡，每更新 3.747--3.967 秒。
融合打开后 L_full/J 到 JT 头的梯度为正；BF 头梯度为零、融合梯度为正。
融合打开的前置动作和路由探针都是独立诊断，不写入正式训练 checkpoint。

头部预热仅训练头，1 epoch/17 updates/effective batch 64，
占用损失 0.792883 -> 0.674508，全空率 1 -> 0.041667，全满率 0。
预热末离线探针已完成：80 个物理 batch，固定 96 JT+32 BF+32 混合，
train/bf16 模式，无优化器更新。
预热末融合投影仍为零，L_full/J 到头的零梯度符合零初始化定义，
不能把这个时点与融合打开后的路由检查混为一谈。

训练权重（固定）：

- lambda_full = 0.5352981960748198。
- w_J = 0.45699876676596235；w_H = 0.5265069904544747。
- J centroid/logsigma/shape = 1.4153295255171303 / 0.6572871637779003 / 1。
- 复测 decoder full/pen = 0.4999999714，ViT J/base = 0.4999702719，H/J = 0.4999858352。

最初标定遇到固定 seed 的 CUDA/bf16 反向舍入波动。
独立 4 batch、3 次重复的前向标量和占用图逐值一致，固定权重的比值为
0.5 / 0.499687529 / 0.499762087，最大相对差 0.000624943。
据此将复测相对容差显式设为 0.001、绝对容差 1e-6；平衡目标仍为 0.5，
没有反复调权重追逐噪声。零/非有限/源哈希检查保持。
首次失败日志、独立重复记录及 initial/lambda/inner_j/final 标定阶段均保留。

统一评测
--------

ctrl_g12 best ep46 的评测 J 系数固定为 centroid/logsigma/shape =
1.2576067579516932 / 0.25876234886192295 / 1，与训练系数分文件保存。
固定 clean val 的 240 条分母使用 baseline vggfix ep50：
L1_JT=0.7980608488663589，L1_BF=0.6197072351150376，J=0.5594191557109947，S=3。
ctrl_g12 同口径 S=2.8249748800148895；这些文件均在正式训练选 best 前封存。

历史旧 val L1 未记录 seed，随机一像素裁剪使它与 seed 0 略有偏差。
独立旧 d93ba30 路径 seed 0 得到 L1_JT=0.8137554537682307、
L1_BF=0.6168440857550481，新代码 legacy 路径逐值复现。
``legacy_seed0_reference.json`` 及来源证明另存，历史参考和严格对账容差保留。
同一固定预测图的 101 个 JT 新旧 J 比较值/梯度均有限、上半梯度为零；
实例、定义、计算域、聚合和系数同时变化，不作单因素因果归因。

后续自动任务与验收
------------------

流水线在正式训练结束后依次执行 ep2/25/50 离线探针、best/ep50 新 S、
漂移/裁剪报告、配对 bootstrap（10000 次、seed 0）与固定盲评图导出。
任何一步失败记录 failed 并停止，不静默跳过、不覆盖旧实验输出。
盲评在训练前已固定 40 字形、80 次比较，人工评分尚待用户验收。
训练结束、最终指标、置信区间和人工结论均未完成，不能据启动成功宣称效果。
