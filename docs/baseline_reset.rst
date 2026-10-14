checkpoint-14 基线（2026-09-28）
================================

当前入口为 `finetune_font.sh`，初始化 `models/vit_base_font/checkpoint-14.pth`。
删除旧 structure/detail、FiLM、校准和阶段 3 实验框架，不实现新部件/风格模型。

保留
----

- 原 recon、VGG Gram style（legacy）、edge 和可选 GAN；脚本仍使用 no_gan。
- G/D 优化器隔离、D step 后清梯度、参数列表化裁剪、裁剪前后范数及 CSV。
- 保留 CalliPhase 数据与审计工具；当前脚本按 type 随机配对，不启用严格同风格异字符配对。
- BF 单标注 mask、图像尺寸统一、原有遮盖选择逻辑和验证全目标遮盖。
- util/stage3_data.py 提供独立数据/固定参考/补白预处理；util/font_metrics.py 保留评价指标。
- 推理统一补白/bicubic，移除参考中间 64px 降采样；严格加载权重，拒绝含 FiLM 的旧模型。

训练
----

.. code-block:: bash

   bash finetune_font.sh

使用 train_json_new/*.json 和 val_json_new/*.json，不使用阶段 2 混合清单。
这些旧清单不要求 style_id/character 字段；参考字按相同 type 随机选择。
保留脚本原有 51 epochs、lr=1e-3、冻结 9 层等非目标超参；这些不是新路线最优配置。
保留原有 --auto_resume，自动恢复输出目录中的最新 checkpoint。
没有恢复文件时从 checkpoint-14 初始化。开始独立新实验时修改脚本中的 name。
旧文档和实验结果仅为历史资料。旧 checkpoint 中非参数损失设置不再接入训练。

验证边界
--------

测试中的小张量 loss 使用 VGG 替身验证实际 loss 方法与反向传播，不等于完整 ViT/VGG/CUDA 训练。
完整 GPU smoke 和训练须在具有 detectron2、VGG 权重及 CUDA 的环境执行。
