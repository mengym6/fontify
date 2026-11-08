Run B Fix Pixel：暂定投稿最终版
================================

版本定位
--------

用户于2026-11-08指定 Run B Fix Pixel 为当前暂定投稿最终版。
当前所指权重是像素遮盖续训版本的 best epoch66，不是原patch Fix、
Pixel首阶段epoch50或续训末轮epoch70。

源码分支：codex/runb。

权重身份
--------

* 检查点：models/runb_fix_pixel_continue20/checkpoint-best.pth
* 检查点内epoch：66（从0计数）
* SHA256：6973168c7c396a9d153d2dc7368ead203952f38839ec43e320166efa65b55340
* 已有服务器位置：
  /root/autodl-tmp/fontify_runb_pixel_20261107/models/runb_fix_pixel_continue20/checkpoint-best.pth

权重二进制不随Git提交。加载时应核验SHA，并通过checkpoint元数据严格重建，
不能因文件名相同就替换为另一个模型。

实际训练与推理
--------------

* 双视图共享Fontify主干、实例结体头和布局注入保持Run B Fix主线。
* BF局部训练按实际11个annotation层采样，笔法多边形与灰度>=230背景取并集；
  直接遮盖原生像素，不把部分命中的patch全部扩张遮盖。
* 首阶段51轮（epoch0--50）；从epoch50恢复模型、AdamW和AMP后追加20轮
  （epoch51--70），新增周期的峰值LR为1e-4，选择best epoch66。
* 这不是从起点采用71轮单一余弦周期训练的模型；源checkpoint缺少RNG状态，
  不宣称位级连续随机流。
* 推理使用整字全遮路径，不需要BF/JT真值标注。

源码入口
--------

* finetune_dual_view_runb.sh：显式设置DUAL_PIXEL_MASK=1启用像素遮盖。
* tools/run_runb_pixel_experiment.py：首阶段训练、统一评测和导图。
* tools/continue_runb_pixel.py：实际追加20轮协议及训练前向图导出。
* tools/build_callify_final.py：冻结输入、baseline及定版模型统一评测。
* tools/package_callify_final.py、tools/fetch_callify_final.py：打包、回收和校验。
* tools/export_runb_component_maps.py：导出实际结体头占用、存在性和方位预测。

比较口径
--------

后续比较使用冻结validation_268（129JT+139BF），保留historical_240用于旧表对账。
两者分别聚合、分别使用各自S分母，不混合数量或表述为新的独立测试。
现有交付位于桌面callify_final；该目录中的数值、图像、冻结输入和模型权重
均不因本次源码分支提交而重新生成或替换。
