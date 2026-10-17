# runc 验证状态

本地最终综合273项CPU测试通过，包括新模型/采样/loss/runner/runtime-check及RunB、checkpoint、Q空间、Tip回归。另实际双进程Gloo在coarse/refine均通过，覆盖混池、跨rank和累积窗口。真实attention/ViT块只在小网格检查，VGG等可用替身，因此不是生产模型CUDA证据。

PEP8相关Ruff E/F/I、Python3.9编译、shell语法与diff检查通过。新增参数5,022,851；阶段一默认可训练59,232,704；阶段二2,513,027。

完整ViT的FP32/bf16、NCCL两卡、Fix真实checkpoint关闭开关逐值回归、多batch梯度标定与训练结果尚待服务器执行，随后追加实测证据。纯数值probe不作训练效果结论。

低通固定图来自服务器真实clean240，抽选12个样本，sigma1/1.5/2、kernel7/reflect，整字与细画同坐标放大。实际Gaussian卷积与loss实现CPU逐值相等。用户最新委托自主推进；低通初始参数由Codex依据图像检查接受作首轮实验，不伪称用户已亲自确认。细尖变圆、窄间隙改变与小噪点消失仍作为风险记录。

首次提交34f9da1的完整CUDA：legacy-off的BF0/JT14与pull前tensor逐值相等；coarse四池/refine三池FP32逐项梯度通过。初始零残差完整240评测复现封存S=2.346697451864291（101JT/139BF，216实例）。BF16占用图写PNG时报dtype错误，safety验证脚本因Fix args-only元数据中断，失败记录保留；对应最小修复、FP32固定LPF与BF16 probe精度补丁进入第二提交后需重新验证，不将首次中断说成全通过。
