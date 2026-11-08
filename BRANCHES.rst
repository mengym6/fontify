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

2026-10-28 清理与恢复记录
------------

本地与 origin 仅保留以上三条分支；远程删除范围已由用户明确确认。
其余实验和修复分支删除前，已保存：

* outputs/git_cleanup_20261028/repository-before.bundle：完整 Git 历史备份；
* outputs/git_cleanup_20261028/refs-before.txt：原分支名及完整提交号；
* outputs/git_cleanup_20261028/PROGRESS-before.md：整理前的进度记录。

例如需单独恢复补实验源码，可运行：

::

    git clone -b codex/calliphase-phase-ablation outputs/git_cleanup_20261028/repository-before.bundle /tmp/fontify-phase-ablation-recovered

bundle 位于忽略目录，不随 GitHub 推送。旧本地 worktree 保持原文件和提交，
改为 detached HEAD；没有删除数据、权重、实验输出或服务器目录。
本次仅整理本地及 GitHub 分支，不切换或更新服务器正在使用的代码。

验证：baseline 脚本与原 1c8206e 逐字节一致，实际启动参数通过当前解析器；
baseline、BF 坐标、双视图训练及 TensorBoard 共40项CPU回归通过。
本次没有重新进行GPU训练。
