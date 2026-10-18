# Clean run2 实测梯度标定

本报告仅使用修复自身参考泄漏后的数据策略 `distinct-original-reference-v1`。旧run1的probe、标定与训练权重全部invalid；未复用。每阶段从实际起点独立采集九书家、18个batch、每batch两目标的BF16前向及FP32梯度Gram；系数取跨batch中位数政策，不是搜索最优参数。

| 参数 | IA0 | IA1 | IB |
|---|---:|---:|---:|
| Head原损失 | 1 | 1 | 0 |
| Head几何总系数 | 2.48195141 | 1.14486818 | 0 |
| Head logσ内系数 | 1.01226649 | 1.75664170 | 不参与 |
| Head相对质心内系数 | 24.67161631 | 20.98723863 | 不参与 |
| Head shape内系数 | 1 | 1 | 不参与 |
| density | 0 | 0 | 1 |
| occupancy BCE | 0 | 0 | 0.30873417 |
| occupancy Dice | 0 | 0 | 0 |
| J_coarse总系数 | 0 | 0 | 1.72549437 |
| J logσ内系数 | 不参与 | 不参与 | 0.89258558 |
| J相对质心内系数 | 不参与 | 不参与 | 0.73696348 |
| J shape内系数 | 不参与 | 不参与 | 1 |

政策：Head几何内logσ/质心/shape加权范数中位数约2:1:1，再以原Head损失梯度为锚选择公共倍率。IB以density系数1为锚，occupancy范数约等于density、J内部仍2:1:1、J合成范数约为density的一半。组合使用实测Gram保留交叉项和抵消；单项范数不能解读为独立贡献百分比。零、非有限、过大系数与起点/策略不符均硬拒绝。BF16微小数值误差不作精确比例承诺。

唯一初始ckpt14 SHA为`339ae9781f4f3831b9c97a19f3c5160c72fa790dbc82157d1d7d6ddf7be17dac`；IA1起点clean IA0 step4为`5b380960bdc6c1e8046178bab807998b2252d8566e7e1bd9e20886e307368dfb`，IB起点clean IA1 step400为`c2da2b87323cedfc3853acb414eb1dc3553a56bf67f8fab5e58d425fd2b7e757`。

完整18batch明细、Gram、余弦、分组范数、预计裁剪和达到比例保存在各阶段`selection_ia0.json`、`selection_ia1.json`及`../clean_run2_ib/selection_ib.json`。对应probe SHA依次为`ff4f56915ccbc5e20e6500da5e47c17d3e297c077f600f4da8e2598d4945a6ae`、`58e5c1cea23d0045ed28ef0457952697bb944823780a3020ec2a4b9dfcff7cae`、`c3a7aaced3d210a82967f43ca448f83bfb2330e31fadfb8aa4b7da0457027f5b`。

IB固定两目标12步健康检查exit0：total3.74626→1.64960、density0.134252→0.062039、occupancy BCE0.711691→0.307805；冻结参数不变，Head指标逐值不变。这仅证明小样本可学习性，不代表val改善、完整结构或泛化通过。
