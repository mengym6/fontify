# Run C v2：本轮最终结果（2026-10-19）

已完成独立分支实现、push/服务器pull、干净IA0四步、IA1四百步、IB三百二十步及最终测试。**阶段一结构质量未通过；不执行I-C或BF正式长训。** 早期run1的自身参考GT泄漏已修复，污染权重与标定全部作废，clean run2从唯一正确ckpt14重新开始。

同轨固定216实例：IA1 Head IoU 0.252255→0.449461，横纵logσ误差和0.900745→0.426860；IB原448坐标P0误差和0.984005→0.272103、shape0.491553→0.379154。IB的ViT/Head冻结SHA前后完全一致。所有训练与固定评测使用distinct-original-reference-v1，阶段一740个JT-bearing目标，验证101配对/77unique并另报74风险排除集。

生成图仍明显丢点、填充内部空白并合并部件：無的底部点画未恢复，惟/楨/容/雲等不可靠。固定13字、九书家、IA1端点及IB80/320并列图全部保留。oracle保留的关键结构在P0中仍丢失，不能把结果归咎于112表示地板，也不能用均值改善宣称结构合格。

固定真实F/Q的12字条件审计中，八种缩放/平移共96/96个方向与实际S一致，median gain约0.82—1.19；noS输出不相等、三类梯度有限非零、模型与缓存SHA不变。这支持S具有真实几何控制，不证明完整结构正确或泛化。

阶段二真实BF16单步接口通过：原11 annotation层抽样、BF局部与dual J独立梯度、BF-only J为0、模型P0第二编码、语义/自动mask隐藏GT替换与冻结SHA均实查。只有两条曹全碑样本；没有正式BF训练或效果验收。

训练源c54cf50（IA）/6411b26（IB），后置检查源3eb25f5；seed0、两张4090、effective batch32、TF32关闭。IA1快照SHA c2da2b87323cedfc3853acb414eb1dc3553a56bf67f8fab5e58d425fd2b7e757，IB快照SHA f3e39d8f1747ae5449c74e2a83f5d189d00b49edbc3b4da54f25d87b991970d4。大型权重留在服务器，不提交Git。

完整交付入口为仓库`outputs/runc_v2_jt_structure_20261018/index.html`，阶段判定为`clean_run2/stage1_acceptance.md`。包含分组/尾部/双轨曲线、checkpoint与配置manifest、oracle及失败图、条件干预与阶段二接口报告。用户最终人工审阅仍pending；下一轮必须先针对失败原因制定新实验，不能默认加步/解冻或转BF。
