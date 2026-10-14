双视图 run B：独立运行协议
==========================

设计合同为 ``PLAN_dual_view_runB.md`` 和 ``PROGRESS.md`` 的 T2 登记。
本文记录可执行步骤，不代表服务器实验已经完成，也不预设研究收益。
所有新输出均选新目录或新文件；协议工具拒绝覆盖已有 JSON。

数据及固定口径
--------------

先刷新服务器 GPU、原图版本、checkpoint、可用磁盘。
本地的 1092 条合并统计不能代替服务器统计。新缓存包含 train 与固定 val
所需的 JT 原始层及源文件哈希；训练目标图仍取 BF 版去噪图。

.. code-block:: bash

   python tools/build_dual_manifest.py \
     --data-path fontdata_example \
     --input-dir fontdata_example/train_json_clean_v2 \
     --output-dir fontdata_example/train_json_dual_v1

   python tools/build_dual_targets.py \
     --data-path fontdata_example \
     --json-paths fontdata_example/train_json_dual_v1/train_dual.json \
       fontdata_example/val_json_clean_v2/*.json \
     --fixed-pair-path fontdata_example/val_pairs_fixed_clean.json \
     --output-dir fontdata_example/dual_targets_v1

``instance-lower-full-v1`` 是新的 J 定义标识。训练／评测系数分文件保存。
文件集合哈希按 basename 排序，再散列文件名和内容摘要，部署绝对路径不进入哈希。
缓存哈希绑定 ``index.json`` 与 ``metadata.json``，缓存加载另核对其内部源哈希。

评测系数与 S 分母
-----------------

在 ctrl_g12 best（ep46）上以新实例划分、整字全遮口径标定评测 J；保持模型
eval 模式，对可训练模型参数的三项梯度取均值范数，以 shape 为基准。
训练系数另外在预热后的 run B、train/bf16 下标定，不能互相替代。

.. code-block:: bash

   python tools/calibrate_dual_eval.py \
     --checkpoint models/finetune_jieti_ctrl_g12/checkpoint-best.pth \
     --data_path fontdata_example \
     --val_json_path fontdata_example/val_json_clean_v2/*.json \
     --fixed_pair_path fontdata_example/val_pairs_fixed_clean.json \
     --dual_target_cache fontdata_example/dual_targets_v1 \
     --output_json models/dual_view/eval_coef.json

``BASELINE_EP50`` 必须指向已核对的 baseline vggfix ep50；
``OLD_L1_REFERENCE`` 指向保存过的旧 val 同 checkpoint L1 两项参考结果。
同一命令先在旧 val 对账 L1，再计算 clean val 新 S 分母。
旧 J 不参与对账，旧 J/S 数值不与新 J/S 混用。

.. code-block:: bash

   python tools/eval_s_baseline.py \
     --checkpoint "$BASELINE_EP50" --data_path fontdata_example \
     --val_json_path fontdata_example/val_json_clean_v2/*.json \
     --fixed_pair_path fontdata_example/val_pairs_fixed_clean.json \
     --calibration_json models/dual_view/eval_coef.json \
     --dual_target_cache fontdata_example/dual_targets_v1 \
     --j_definition instance-lower-full-v1 --as_dual_denominator \
     --legacy_val_json_path fontdata_example/val_json_new/*.json \
     --legacy_fixed_pair_path fontdata_example/val_pairs_fixed.json \
     --legacy_l1_reference_json "$OLD_L1_REFERENCE" \
     --output_json models/dual_view/s_baseline_dual.json

用同一新口径评测 ctrl_g12 best，并保存全量 ``per_item``。在训练前从固定
val 生成盲评清单。左右模型身份保存在单独清单，展示和打分时隐藏身份。
清单哈希写入正式训练 args；只允许 left/right/tie 的结体、笔法两项偏好。

.. code-block:: bash

   python tools/dual_report.py blind-manifest \
     --evaluation models/dual_view/s_baseline_dual.json \
     --models run_b baseline_vggfix ctrl_g12 --seed 0 --limit 40 \
     --output models/dual_view/blind_manifest.json

固定探针目标与头部预热
----------------------

固定清单保存 64–128 个不同 JT 目标及上排参考、书家、池、几何增强种子、
forward 随机种子，抽样使用方案 D 权重，目标不放回，参考同书家同池均匀抽取。
保存期望池份额；有限不放回抽样的实际份额会有波动。
清单须覆盖双标注和仅 JT 池，附 BF-only 与三池混合诊断样本。
后续预检、标定、离线探针全部复用同一清单与 batch_size。

.. code-block:: bash

   python tools/dual_view_protocol.py samples \
     --data-path fontdata_example \
     --json-path fontdata_example/train_json_dual_v1/train_dual.json \
     --target-cache fontdata_example/dual_targets_v1 \
     --targets 96 --diagnostic-targets 32 --batch-size 2 --seed 0 \
     --output models/dual_view/probe_samples.json

   python tools/dual_view_protocol.py warmup \
     --data-path fontdata_example \
     --json-path fontdata_example/train_json_dual_v1/train_dual.json \
     --target-cache fontdata_example/dual_targets_v1 \
     --samples models/dual_view/probe_samples.json \
     --checkpoint models/vit_base_font/checkpoint-14.pth \
     --batch-size 2 --device cuda --seed 0 \
     --output models/dual_view/head-warmup.pth

预热是单独进程、单独 AdamW 生命周期：只有 ``dual_head`` 可训练，lr=1e-3、
betas=(0.9,0.999)、weight decay=0.1（一维参数除外）、clip=3、有效 batch=64。
直接用未乘外层 w_H 的 L_H。先 1 epoch；固定预检占用损失未下降或预测全空／
全满则再跑 1 epoch，仍不合格写诊断记录并报错。最多 2 epoch，不自动延长。
保存整模型、父 checkpoint 哈希、实际 updates、预检记录和 checkpoint JSON。
正式阶段加载整份预热模型，重新建 optimizer/DDP；不沿用预热 optimizer。

训练标定
--------

.. code-block:: bash

   python tools/dual_view_protocol.py calibrate \
     --data-path fontdata_example \
     --json-path fontdata_example/train_json_dual_v1/train_dual.json \
     --target-cache fontdata_example/dual_targets_v1 \
     --samples models/dual_view/probe_samples.json \
     --checkpoint models/dual_view/head-warmup.pth \
     --parent-checkpoint models/vit_base_font/checkpoint-14.pth \
     --val-json-path fontdata_example/val_json_clean_v2/*.json \
     --fixed-pair-path fontdata_example/val_pairs_fixed_clean.json \
     --eval-coef models/dual_view/eval_coef.json \
     --s-baseline models/dual_view/s_baseline_dual.json \
     --device cuda --seed 0 --output models/dual_view/train_coef.json

模型为 train 模式，bf16 autocast，每个 batch 的 loss 来源是同一次 forward。
``autograd.grad`` 分别取 L_pen、L_full、J 三分项、L_H 及实际合成 J；保存
ViT、decoder、fusion、head 的原始 6×6 与四目标 4×4 内积矩阵。
按每 batch 先计算合成范数，再跨 batch 平均，包含夹角和相互抵消。
探针不执行 optimizer.step，不写 parameter.grad。

定值顺序为：lambda 从 0.5 起，decoder weighted-full/pen 比值低于 0.5
时提高到 0.5；以 ViT shape 均值范数重标三项训练系数；令 ViT weighted-J/base
比值为 0.5，weighted-H/weighted-J 为 0.5。base 是 L_pen+lambda*L_full 的
合成梯度。edge 取 0。lambda 若提高先单独复测；内部系数设好后重新测量实际
合成 J 再定两项外层权重，最后全套权重复测，而非只打印代数估计。
另报告 BF-only、三池混合的实际比值、夹角和跨 batch 波动。
任何需作分母的梯度为 0、非有限或复测不符，均停止诊断。

CUDA/bf16 固定种子重复测量中，四项前向损失逐值相同，但反向比率仍出现约
0.063% 相对漂移。因此复测使用 0.1% 相对容差及 1e-6 绝对容差；目标仍为
0.5，不根据复测噪声反复调权重。零梯度、非有限值及真实比例偏差仍停止。
initial/lambda/inner_j/final 四阶段矩阵分别写入输出名后附阶段名的 JSON；
只有全部检查通过才写正式标定文件，失败测量也保留，旧文件不覆盖。

训练标定文件绑定 train、缓存、固定 probe、warm、parent、fixed val、val清单、
eval 系数、S 分母九项哈希。改了任一输入就重标，不能复用旧系数。

冒烟与正式训练
--------------

先在服务器运行独立 CUDA/DDP smoke，覆盖 BF-only/JT-only/dual/mixed、
fusion 打开后的逐项梯度路由、static_graph、显存和耗时。旧／新 J 的固定
预测图比较必须同时记录数值和像素梯度；``compare_partitions`` 保留两种归一化
定义，不能把一次静态比较当作研究收益。

.. code-block:: bash

   python -m torch.distributed.run --nproc_per_node=4 \
     tools/smoke_dual_view.py \
     --data-path fontdata_example \
     --json-path fontdata_example/train_json_dual_v1/train_dual.json \
     --target-cache fontdata_example/dual_targets_v1 \
     --checkpoint models/dual_view/head-warmup.pth \
     --train-calibration models/dual_view/train_coef.json \
     --batch-size 2 --iterations 2 \
     --output-json models/dual_view/smoke.json

   NPROC=4 BATCH_SIZE=2 bash finetune_dual_view_runb.sh

显存不足时 BATCH_SIZE=1，累积自动翻倍；有效 batch 始终为 64。
smoke 的每次 DDP update 同样覆盖 64 个目标字：4 卡 B=2 时每卡 8 个
microbatch，B=1 时每卡 16 个；mixed 在每卡 16 个目标中轮换三池。
DDP 建立前开融合层的步骤是本地诊断，每卡累积 64/world 个目标，但无
all-reduce；报告单独记录此步骤，不称其为 DDP 更新。逐项梯度路由用小批次
独立诊断，不更新参数。所有累积按 microbatch 目标数占本卡总数加权。
脚本禁止旧输出目录，严格校验新系数／分母／固定清单，现场检查 GPU bf16 与磁盘。
空间预算按 warm 模型大小估算 7 个含 optimizer checkpoint 加 2 个模型快照，留20%
余量；它不是服务器最终实际磁盘使用的测量值。

正式训练 seed0、epoch0–50、lr=0.00195699、warmup10、style14.73、clip3、
layer_decay0.8、drop_path0.1、edge warmup10/duration8/final0.2、noGAN、全解冻。
每10epoch、best、ep50，以及 ep2/25 仅模型探针快照独立保存。
``--dual_view`` 默认关闭。开关关闭不实例化新模块，并需通过旧路径逐值回归。

离线探针与最终报告
------------------

对 warm_end、ep2、ep25、ep50 逐一运行；POINT 与 CKPT 必须对应。
ep2 fusion末层仍全零时保存 graph_blocked 记录并报错，不当作有效测量。

.. code-block:: bash

   python tools/dual_gradient_probe.py \
     --point "$POINT" --checkpoint "$CKPT" \
     --calibration models/dual_view/train_coef.json \
     --samples models/dual_view/probe_samples.json \
     --data-path fontdata_example \
     --json-path fontdata_example/train_json_dual_v1/train_dual.json \
     --target-cache fontdata_example/dual_targets_v1 \
     --output "models/dual_view/probe_${POINT}.json"

   python tools/dual_report.py probes \
     --probe models/dual_view/probe_warm_end.json models/dual_view/probe_ep2.json \
       models/dual_view/probe_ep25.json models/dual_view/probe_ep50.json \
     --output models/dual_view/probe_drift.json

   python tools/dual_report.py clipping \
     --log models/finetune_dual_view_runb/dual_diagnostics.jsonl \
     --output models/dual_view/clipping.json

全量 clean val 统一新 J/eval 系数评测 run B best、ep50、baseline ep50、ctrl best。
保存 per_item 后做逐字差和 95% paired bootstrap；S 每次在 JT/BF 组内分别成对
重抽，保留 S 三项各自的 cohort 分母，不能用 BF 的零 J 稀释 JT 均值。
独立评测 ``tools/eval_s_baseline.py`` 使用新 J 时加
``--s_baseline_path models/dual_view/s_baseline_dual.json``，写绝对 ``S``、
``S_components`` 和分母文件 SHA256；固定配对、val 清单、target cache、eval
系数四项哈希须一致。首次生成新分母时自身 ``S=3``；未指定分母的评测可用于
标定前检查。paired 报告的 ``S`` 内同时写 ``run_S``、``reference_S`` 和双方
三个分项比值，三项始终按各自 JT/BF cohort 均值计算。

.. code-block:: bash

   python tools/dual_report.py paired \
     --run models/dual_view/run_b_best_eval.json \
     --reference models/dual_view/baseline_eval.json models/dual_view/ctrl_eval.json \
     --denominator models/dual_view/s_baseline_dual.json \
     --iterations 10000 --seed 0 --output models/dual_view/paired_best.json

   python tools/dual_report.py blind-ratings \
     --manifest models/dual_view/blind_manifest.json \
     --ratings models/dual_view/blind_ratings.csv \
     --output models/dual_view/blind_results.json

生成盲评图时只传入与预固定清单同名的模型键；Run B checkpoint 必须包含训练前
封存的 ``dual_blind_manifest_sha256``。导出只显示 Ref/GT/Left/Right，文件名仅为
blind_id；独立 ``answer_key.json`` 保存左右身份，``ratings.csv`` 留给人工填写。
盲评时只交付 ``images/`` 和空评分表，答案文件暂不展示。

.. code-block:: bash

   python tools/export_dual_blind.py \
     --manifest models/dual_view/blind_manifest.json \
     --models run_b=models/finetune_dual_view_runb/checkpoint-best.pth \
       baseline_vggfix="$BASELINE_EP50" \
       ctrl_g12=models/finetune_jieti_ctrl_g12/checkpoint-best.pth \
     --data-path fontdata_example \
     --val-json-path fontdata_example/val_json_clean_v2/*.json \
     --fixed-pair-path fontdata_example/val_pairs_fixed_clean.json \
     --target-cache fontdata_example/dual_targets_v1 \
     --eval-coef models/dual_view/eval_coef.json \
     --output-dir models/dual_view/blind_best

裁剪报告包括实际有日志的 optimizer update 数、触发率、范数、比例中位数/p05/p10。
梯度漂移和跨 checkpoint 匹配切换只是诊断证据；频繁切换需要训练日志补充。
盲评 CSV 用 blind_id,structure_preference,pen_preference 三列。
过程验收与研究效果分别写结论：GPU/DDP/51epoch 未执行则实施验收未完成；
指标与人工偏好冲突原样报告。单 seed 不表述为稳定收益，不作单因素归因。
