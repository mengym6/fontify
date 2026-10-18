# 阶段二真实接口检查

后置控制器两子进程均exit0。源3eb25f5，PyTorch2.3.0+cu121/CUDA12.1、RTX4090，TF32与cudnn benchmark均关闭；一次AdamW接口更新使用BF16，后续不变性前向使用FP32。阶段一初始模型为干净IB step320，SHA f3e39d8f1747ae5449c74e2a83f5d189d00b49edbc3b4da54f25d87b991970d4。

安全BF视图818=466双标+352BF-only，distinct-original-reference-v1身份guard实际通过。本次只取曹全碑双标“述”和BF-only“載”各一条，不冒称九书家或全池质量验证。真实annotation层数21/37，请求11/11、实际选11/11且裁剪前后均非空；原语义层均匀抽样后union，不聚合类别。输入输出语义patch mask逐值相同，覆盖116/72个下半patch（14.796%/9.184%）。

局部BF损失对Refiner的独立梯度范数为0.395412/0.907693，dual J(P1)为7.009272，BF-only J(P1)及梯度严格0。总损失2.450916只作运行凭据，不作训练改善结论。

真实encoder捕获检查覆盖语义训练、语义eval/替换GT、自动mask eval/替换GT五次前向：每次初始下半GT全隐藏，第二encoder输入逐值等于按实际mask遮盖的模型P0，未用oracle或GT替代。P0/P1/masked P0均[2,3,896,448]，自动mask[2,28,28]；语义及自动分支的P0/P1/mask/masked P0在替换隐藏GT后全部逐值相等。

161,979,611个stage-I冻结参数SHA前后为01775fc268b3e4dc14e8b0bb17588d704c4a55dae15f770395806aacaed27de0，完全不变。仅Refiner参与更新；小快照SHA55c543a6d4f2c3a5584224fd88f9db5f6cd10ac6cb3a9121b59ea028d2c44470，实际严格tensor/dtype/byte恢复到同一Refiner对象。此检查没有重建整个模型，不作该项承诺。

完整凭据在../post_ib_checks/stage2/stage2_interface_smoke.json与pipeline.json。stage_one_accepted=false、formal_bf_training_started=false；训练原annotation mask与推理自动mask仍有分布差别。本轮没有BF正式长训，配置草案configs/runc_v2/stage2_draft.json不可执行，LR/权重/预算未标定。
