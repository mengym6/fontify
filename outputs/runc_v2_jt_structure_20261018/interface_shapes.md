# Run C v2 模型接口审计

实查时间：2026-10-18。当前结论区分源码、CPU 合约、完整服务器运行证据。

## 真实 Fontify 计算图

`models_train.py::forward_encoder` 的 ViT-B 深度为 12，实际抽头是
0-based `2、5、8、11`（即第 `3、6、9、12` 层）。第 3 个 block 后先保存
两流的四个上下象限，再对两流逐 token 平均融合。每个抽头经同一个 `norm`。

| 张量 | 生产形状 | 来源 |
| --- | --- | --- |
| source / target 输入 | `B×3×896×448` | 上下拼接，patch 16 |
| 四个 merge 前象限 | 各 `B×28×28×768` | `x_top/y_top/x_bottom/y_bottom` |
| 每个 ViT tap | `B×56×28×768` | idx `2/5/8/11` 后的共享 LayerNorm |
| `F_full` | `B×56×28×3072` | 四抽头沿 channel 拼接 |
| lower / reference F | 各 `B×28×28×3072` | `F_full` 的下半 / 上半 |
| `Q` | `B×784×256` | 新初始化 JT Head 空间 query |
| `S.occupancy_logits` | `B×4×28×28` | 四个预测实例槽 |
| `S.existence_logits` | `B×4` | 预测槽存在性 |
| `S.class_logits` | `B×4×20` | 预测方位角色 |
| `D0` / `O0_logits` | 各 `B×1×112×112` | 新 Coarse Head |
| `p0_lower` | `B×3×448×448` | `1-Upsample(D0)`，RGB 后一次 normalize |
| `p0` | `B×3×896×448` | 上半可见参考 + 下半生成 `p0_lower` |

v1 `util/structure_first.py::StructureConditioner` 使用 F query、Q/S KV，输出残差
回旧 decoder；v2 不调用该 conditioner、旧 fusion 或旧 decoder。v2 只复用其
通用单层 cross-attention block 实现。

## v2 条件作用与参数

`util/runc_v2.py::StructureQueryEncoder` 对四个槽使用同一运算，聚合
`sigmoid(occupancy)×sigmoid(existence)` 加权的 class probability、空间质量和二阶
gate 量。求和后除固定槽数；不使用 slot ID、顺序或独立槽 embedding。由此得到
结构 field，加 row/column position 后作为唯一 query。

K/V 分别来自真实 lower F、真实 Q、真实 reference F，三者使用独立 projection 和
source embedding。生产配置 hidden `256`、heads `4`、1 block，query token `784`，
K/V token `2352`（`784+784+784`）。输出经 `28→56→112` 轻量空间网络，分头预测
sigmoid density 与 occupancy logits。没有 F/Q 到输出的绕 S RGB/residual 捷径。

| 新模块 | 生产参数量 |
| --- | ---: |
| `dual_head` | 4,699,157 |
| `runc_v2_coarse` | 2,897,986 |
| 显式可选 `bf_refiner` | 107,683 |

上述参数量由本地 PyTorch 2.11 实际实例化计数。生产大小的独立 Head/Coarse CPU
forward 已运行，输出尺寸均为 `1×1×112×112`。CPU profiler 支持的算子记录
8,029,306,432 FLOPs，折算 4,014,653,216 MACs；fused SDPA 可能未计入，不能把
此值当作完整模型 MACs。完整 ViT、CUDA/BF16、峰值显存与双卡证据由 T3 单独报告。

## 冻结与部署接口

| 子阶段 | ViT | JT Head | Coarse | BF Refiner |
| --- | --- | --- | --- | --- |
| IA0 | 冻结 | 训练 | 冻结且不调用 | 不附加 |
| IA1 | `blocks.2/5/8/11` + 共享 norm | 训练 | 冻结且不调用 | 不附加 |
| IB | 冻结 | 冻结 | 训练 | 不附加 |
| IC | 同 IA1 小集合 | 训练 | 训练 | 不附加 |
| infer | 冻结 | 冻结 | 冻结 | 不附加 |
| refine（仅接口） | 冻结 | 冻结 | 冻结 | 仅此模块训练 |

按指定本地 ckpt14 的真实 tensor shape 计数，每个选中 block 为 7,098,496 参数；
四 block 加共享 norm 合计 28,395,520。因此 IA0 可训练 4,699,157，IA1 可训练
33,094,677，IB 可训练 2,897,986，IC 可训练 35,992,663。此数量描述配置，不代替
运行前后 SHA 冻结核验。

patch embedding、absolute position、segment/mask tokens、旧 decoder、VGG 与
discriminator 在所有本轮阶段均冻结。每次 forward 在 runner 的 `model.train()` 后
恢复冻结模块的 eval 状态。IA1/IC 的冻结中间 block 仍保留向已选 block 传递梯度的
autograd 路径；IA0/IB 的整个 encoder 使用 no-grad。

- `enable_runc_v2(legacy_model, config)`：必须在 legacy ckpt14 加载后执行；拒绝已有
  dual/Fix/v1/FiLM 模块。完整实际维度、tap 与 partial-unfreeze 清单写入 config。
- `set_runc_v2_stage(model, stage)`：应用冻结矩阵，返回精确可训练参数名。
- `runc_v2_parameter_report(model)`：按 vit/head/coarse/refiner/unused 给出名称与数量。
- `runc_v2_forward(..., render=None, return_features=False)`：IA 默认只计算 Head；
  IB/IC/infer 默认同时生成 coarse。仅接受 deployment source/reference，不接受 GT JT。
- `render_runc_v2(model, lower, reference, head, structure=None)`：固定真实 F/Q；可将
  S 的 occupancy/existence/class 三项整体替换，支持 no-S、平移与缩放条件探针。
- `runc_v2_checkpoint_metadata(model)`：深复制架构配置及可选 Refiner 配置；保存后
  改变 model stage 不会篡改已有 metadata。`restore_runc_v2(legacy_model, saved)`
  检查完整字段、BF config/keys/stage 一致性与 dtype，重建后严格加载全部 state。
  Refine checkpoint 必须显式保存 `runc_v2_refiner_config`，不能靠默认值猜配置。

下半 target GT 在 patch embedding 前被零张量替换，包括 NaN/Inf；无算术乘零的
NaN 泄漏。`p0` 的上半来自允许的参考字，下半只由预测 density 生成。

## ckpt14 初始化核查

本地指定文件 `models/vit_base_font/checkpoint-14.pth`：epoch 14，224 个 model keys；
其文件 SHA256 为
`339ae9781f4f3831b9c97a19f3c5160c72fa790dbc82157d1d7d6ddf7be17dac`。
sorted model tensor digest 为
`b7c83f9cfe10aaadc76c1a3997c0c01d297f36808500d0807646f2728d340592`。
逐 key bytes/dtype/shape 清单在 `checkpoint_local_keys.json`。

已核实关键 shape：patch `[768,3,16,16]`、mask `[1,1,1,768]`、absolute position
`[1,197,768]`、decoder projection `[16384,3072]`，确有完整 decoder。原服务器同名
checkpoint 文件与 192 个 model tensor bytes 不同，因此必须使用主控复制到独立 v2
初始化目录的本地指定文件，不能把同文件名当同权重。

模块提供 `load_runc_v2_pretrained` 用于严格 legacy key/shape 加载审计；runner 如需
容许 inactive VGG/discriminator/decoder 迁移，必须另行记录完整兼容/缺失清单，且
绝不容许有效 encoder 参数静默随机初始化。新 JT Head、Coarse 和未来 Refiner
均在预训练加载后由新模块 constructor 初始化，seed 由运行配置记录。

## 阶段 II 仅接口

`enable_runc_v2_refiner(model, config)` 显式新建 residual module，设置 refine-only
冻结。`runc_v2_refine_forward(..., bf_mask, inference=False)` 执行模型 P0 → 原 BF
patch mask → 第二次冻结 MAE ViT → 新 residual → P1。禁止读取下半 GT；不调用旧
fusion/decoder；零初始化最终卷积使未训练 P1 与 P0 逐值相等。

训练接口保存原 annotation 层数 11、`aggregate_bf_types=False`。dataloader 负责从
11 个原标注层生成既有二值 patch mask；此函数原样接收和验证该 mask，不在接口内
重新采样或按 BF 类型聚合。推理通过原自动结构分块实现，训练 / 推理 mask 分布
差异仍保留。此实现只提供接口 smoke 与配置草案，不表示 BF 改善或授权正式长训。

## 本地验证范围

`tests/test_runc_v2_model.py` 19 项 CPU 合约全部通过，覆盖 GT 下半 NaN/Inf 替换逐值
不变、槽置换不变、固定 F/Q 的 S 梯度与数值响应、四阶段冻结参数逐值检查、加载及
序列化、原路径关闭回归、11 层 mask 输入契约、第二次编码真实模型 P0、Refiner-only
梯度、stage-I 冻结、非零 Refiner 下多轮推理无 GT 泄漏、阶段二完整 checkpoint
逐值恢复，以及缺 metadata、缺 key、错误 dtype 的严格拒绝。

测试从实际 Fontify class AST 提取真实 encoder/decoder 方法，使用小线性 block 和
VGG 替代 fixture；不能冒充完整 ViT 或 CUDA 运行。旧模型/Run C v1 合约组合运行
48 项通过；另两项真实 attention 测试因本地缺 `timm` 未运行。后续服务器在
`134699051a9ba3f1f5875c06cccffb3b1ac0cc5c` 以 torch 2.3.0+cu121、禁用 GPU 运行
baseline/dual/v1/v2-model/v2-checkpoint 五组 CPU 回归，74 项通过，5 条既有
GradScaler/Pillow 警告，7.16 秒、退出码 0；包含这两项真实 attention 小网格测试。
原始输出见 `server_cpu_tests.log`；它仍不等于完整生产尺寸 ViT 的 GPU 验证。
随后服务器更新为 `990c3ea77d53993de5c33acc76b64866a6e19afe`，同样禁用 GPU 对
全部 `tests/test_runc_v2_*.py`（包含 health）运行 CPU 回归：78 项通过，4.59 秒，
退出码 0，原始输出单独保存于 `server_cpu_v2_health_tests_990c3ea.log`。
新增 Python 文件已通过 pycodestyle、py_compile，项目 diff 通过 `git diff --check`。
