Fontify 分支与模型定位（2026-11-08）
====================================

main
    全解冻 baseline。默认入口 finetune_font.sh 与原 baseline-vggfix
    的 1c8206e 配置一致：no_gan、no_jt、style_weight=14.73、lr=0.001、
    两卡 batch_size=2 / accum_iter=32、51 epoch、warmup=5。
    no_jt 表示 JT 使用随机遮盖，不是移除 JT 样本。
    合并 baseline-vggfix 历史，保留 main 中较新的共享源码和已经生效的
    VGG 二次归一化修复、BF 坐标检查及 TensorBoard 写图修复。
    Run B/结体监督均为显式可选模式，不由 baseline 默认入口启用。

codex/runb
    Run B Fix Pixel 暂定投稿最终版：像素遮盖训练后追加20轮，选择 best ep66。
    保留完整双视图、L_full、结体头、实例几何及原 edge；不是 tip、loss search
    或 strict 训练版本。精确权重身份见 RUNB_FIX_PIXEL_FINAL.rst。
    入口 finetune_dual_view_runb.sh 仍兼容原 patch 方式；复现 Pixel 训练必须设置
    DUAL_PIXEL_MASK=1。tools/continue_runb_pixel.py 保存实际51+20轮续训协议。
    模型、数据与生成图不纳入 Git；提交源码、测试、评测及交付工具。

codex/runc
    保留原 Run C v2 的历史、GT 泄漏修复与结构验收记录，并归档冻结 ep66
    部件条件直接生成实验、部件/轮廓输入预览及测试。
    该直接生成实验已完成51轮，但未达到 ep66 的质量，不能视为投稿最终版。
    最新研究方向是以部件概率图P为起点的条件扩散桥；设计待定、尚未实现。
    现有 experiments/runc_components/ 是历史直接生成模型，不是扩散桥。
    具体代码边界与历史复现配置见 experiments/runc_components/README.rst。

本地实验产物（2026-11-18）
--------------------------

桌面工作区的 ``outputs/`` 仅存放以下三组评测和生成结果，不含原始训练权重：

* ``RunB_Fix_Pixel_ep66/``：像素遮盖续训 best ep66。
* ``RunB_ep49/``：原 Run B best ep49，不是同轮数的消融模型。
* ``全解冻_baseline_ep50/``：修正 VGG 归一化的全解冻 baseline。

Run C 仍保留在 ``codex/runc`` 分支，不是第四个本地输出目录。
这些产物不随 Git 分发；从仓库检出代码不会还原本机图像或评测数据。

2026-10-28 清理记录（历史）
---------------------------

当时清理分支前曾生成未纳入 Git 的历史 bundle 和引用清单。2026-11-18
清理本地产物后，这些文件不再位于 ``outputs/git_cleanup_20261028/``。
当时的分支列表不代表当前分支；以 ``git branch -a`` 为准。

当时的验证：baseline 脚本与原 1c8206e 逐字节一致，实际启动参数通过当前解析器；
baseline、BF 坐标、双视图训练及 TensorBoard 共40项CPU回归通过。
本次没有重新进行GPU训练。
