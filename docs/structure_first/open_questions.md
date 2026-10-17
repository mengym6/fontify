# runc：未决项与阶段门槛

已确定：P0 → 在P0上BF语义遮盖 → 第二次MAE/BF Refiner → P1；先coarse后refine，第二阶段完整冻结阶段一；BF训练采样仍11个annotation layers；refine只用466双标+352BF-only；GAN/Tip/新detail关闭。

用户已定初始技术配置：轻量Cross-Attention hidden_dim=256、num_heads=4、num_layers=1，F为Query，Q与预测S编码为Key/Value，零初始化残差；Gaussian kernel7/sigma1.5/reflect。它们是初始选择，非已验证最优参数。固定样本sigma1/1.5/2图已生成，用户随后明确委托自主推进；由Codex记录图像检查，不伪称人类已亲自确认。

其余最小实现候选：阶段一ViT冻结而decoder训练；推理按预测S/P0稳定排序选择patch。实际配置以代码/输出config为准。

正式实验前与阶段切换前需要具体证据：

- 多batch实际梯度范数与夹角，以及lambda_low/J0/H/img1/BF/J1、Gram、full/local edge的候选比例。旧RunB系数不可搬用。若新项退化/非有限，先修复并复测。
- 低通参数下P0是否保留短细画拓扑：给GT/LPF/P0图与实际尺度/位置/shape/J，发生明显拓扑损失记录阻断，不自动加高频loss。
- 阶段一预算、学习率与结构门槛；依据实际P0与head两层结果检查，不只看头部IoU。用户已委托自主推进，评估由Codex执行并记录依据，技术门槛不会被授权本身替代。
- 阶段二训练入口引用绑定具体coarse checkpoint SHA的验收记录；可以是真实人类确认或有用户委托依据的Codex检查，当前实现/梯度smoke本身不构成质量验收。

明确保留的研究限制：训练mask来自BF GT annotation的11层，推理来自预测S/P0自动patch候选，覆盖率与语义粒度不等价。本轮只验证无GT泄漏、确定性、可运行和可视化，不私改训练协议来消除差异。

固定clean val240已反复参与选择，并非独立泛化集；只有单seed或小样本结果时不宣称稳定收益或泛化。旧严格FP32与封存cuDNN后端分开记录，不能用后端差异当提升。

用户最新授权『不用等待我授权，自行按照计划推进』，允许按计划自主推送、服务器训练测试及有证据的委托验收；不授权删除旧产物、覆盖旧checkpoint、修改main或伪称用户作过人工验收。
