# Run C v2 独立运行接口

`tools/runc_v2.py` 是独立入口，配置 schema 为 `runc-v2-runtime-v1`、experiment 为
`runc_v2`，阶段仅接受 `ia0/ia1/ib/ic`。主训练与推理经一个 DDP 可追踪的
`nn.Module.forward` 包装调用 `util.runc_v2.runc_v2_forward`；监督标签仅交给 loss。
运行入口再次把目标下半图清零，不能仅依赖数据集已清零。

命令包括 `audit/probe/smoke/train/evaluate/stage2-smoke`。首次初始化必须显式给出实查的
checkpoint14 SHA256，且只加载真实 key/shape 兼容项；主干缺项、意外参数或
Run B/v1 参数立即报错。v2 快照通过保存的模型元数据重新构建并严格加载。

`probe` 循环九个来源，报告每个独立原始损失在全可训练参数及 Head/ViT/Coarse
分组上的 Gram、范数和余弦，并记录当前权重下裁剪前后范数。真实 CUDA/BF16
结果必须在服务器生成；纯 Python 测试不代表数值或训练通过。

`train` 完整执行每个梯度累积窗口，AdamW 分离 ViT、Head、Coarse 学习率；
快照保存优化器、每 rank RNG、step、有效 batch、源码/输入权重/数据 SHA。
采样轨迹由全局 seed 重建，各 rank 取无重叠的 draws 位置。冻结参数前后逐字节
SHA 必须一致。只有同阶段、同 world size 的快照允许恢复优化器。

`evaluate` 使用原生 FP32 和固定配对，分别报告配对、唯一目标、书家-字符分组
以及逐实例的 P50/P90/P95。完整历史集要求 `101 JT 配对 / 216 实例 / 77 唯一目标`；
任何分母变化会阻断。固定集已经重复用于研究选择，不能称为独立泛化。
输出包含每样本 CSV、JSON 和至少 12 张带标签对照图。

正式 `train` 必须引用实测标定记录：`status=selected_from_measured_gradients`、
`stage`、`initial_checkpoint_sha256`、`selected_weights`、`rationale`、
`source_probe` 与 `source_probe_sha256`。源 probe 的九来源、实际起点 checkpoint、
AMP、模型配置和必要几何梯度均会检查，不能只放一个非空文件名。

IB/IC 还须 `stage_gate_record`，包含 `status=technical_gate_passed`、
`source_stage=ia1/ib`、目标 `stage=ib/ic`、起点 `checkpoint_sha256`、
`supporting_evaluations=[{path,sha256}]` 及主控的实测判断 `rationale`。
这是阶段切换技术凭据，不代替最终用户结构验收。

训练若设置 `evaluate_every_save=true`，从 step0 开始同时评价固定九来源 train18
与固定历史 val，分别保存，不把训练集改善当作验证改善。

`stage2-smoke` 只在既有 466 双标 + 352 BF-only 池取真实样本做一次 BF Refiner
更新，检查冻结、11 层原遮盖、J 的 JT 门控、GT 替换、自动 mask 和小快照恢复。
它没有正式 BF 长训入口；阶段一完成后仍须人工验收。
