# runc 验证状态

本地最终综合291项CPU测试通过，包括新模型/采样/loss/runner/runtime-check、阶段II标定及预取合约，以及RunB、checkpoint、Q空间、Tip回归。另实际双进程Gloo在coarse/refine均通过，覆盖混池、跨rank和累积窗口。CPU合约使用小网格和部分依赖替身，与完整生产模型CUDA证据分开记录。

PEP8相关Ruff E/F/I、Python3.9编译、shell语法与diff检查通过。新增参数5,022,851；阶段一默认可训练59,232,704；阶段二2,513,027。

真实完整ViT的9组CUDA检查已在d1ad239全部通过：legacy-off/零条件残差逐值回归；coarse/refine的FP32和BF16逐项梯度；两个精度的无GT泄漏、stage-I state SHA不变、strict checkpoint roundtrip；两个精度的预测S条件作用。峰值allocated最多4.34GiB。3步条件响应只证明条件路径可导且输出可响应，不作训练效果结论。

43a6c34的coarse/refine双卡NCCL BF16冒烟通过，含纯池、mixed batch、mixed accumulation、cross-rank mixed，两个rank梯度范数一致。BF-only head保持参数不变，stageII只有Refiner有梯度。实际DDP为find_unused_parameters=True、static_graph=False，不将CPU静态图合约误称CUDA静态图验证。

新训练损失已在九来源27纯池batch2加1个mixed上标定并复测，实际加权J/low与H/J中位数为0.500079/0.499963；J的独立autograd与Gram合成最大相对差0.3004%。跨batch波动和梯度抵消仍明显，不将标定称为最优权重或效果证据。阶段一340更新及全部固定240评测已完成，结构验收未通过，阶段II未正式训练；详见results.md。

低通固定图来自服务器真实clean240，抽选12个样本，sigma1/1.5/2、kernel7/reflect，整字与细画同坐标放大。实际Gaussian卷积与loss实现CPU逐值相等。用户最新委托自主推进；低通初始参数由Codex依据图像检查接受作首轮实验，不伪称用户已亲自确认。细尖变圆、窄间隙改变与小噪点消失仍作为风险记录。

首次提交34f9da1的完整CUDA：legacy-off的BF0/JT14与pull前tensor逐值相等；coarse四池/refine三池FP32逐项梯度通过。初始零残差完整240评测复现封存S=2.346697451864291（101JT/139BF，216实例）。BF16占用图写PNG时报dtype错误，safety验证脚本因Fix args-only元数据中断，失败记录保留；对应最小修复、FP32固定LPF与BF16 probe精度补丁进入第二提交后需重新验证，不将首次中断说成全通过。
