# 干净模型运行检查汇总

阶段A持久控制器11个子进程均exit0：固定两目标IA0健康、IA1单卡FP32及双卡NCCL/BF16、IB单卡FP32及双卡NCCL/BF16、九来源IA0 probe/标定与4步训练、真实IA0端点再probe/标定与IA1正式400步。实际配置、PID、开始/结束时间与退出码见本目录pipeline.json。

阶段B四个子进程均exit0：九来源IB probe、独立标定、12步固定输入可学习性、正式320步。阶段B状态文件在../clean_run2_ib/pipeline.json。固定输入三个Renderer损失均下降且Head冻结；不把该检查作为质量验收。

所有阶段执行安全参考身份断言、下半GT替换、参数组/梯度、冻结SHA与checkpoint谱系检查；旧无policy快照/标定硬拒绝。干净源码发布前本地138项与15subtest、服务器132项核心CPU回归通过；随后新增条件审计/IB控制20项双端通过；阶段二增强8项双端通过。最后阶段二/运行时/条件三文件本地54项与15subtest通过。静态PEP8/Ruff与Python3.9语法检查单独记录，不将其等同GPU实验。

最终后置控制器两项GPU检查均exit0：12字固定真实F/Q条件响应、真实BF16一次Refiner接口更新。真实配置与报告在../post_ib_checks。各报告明确记录精度、TF32=false、checkpoint SHA与源码SHA；不把旧run1真实data probe列为有效证据。

训练/评测运行正确不代表结构质量合格；最终stage1_acceptance.md为未通过。阶段二未正式长训，I-C未执行。
