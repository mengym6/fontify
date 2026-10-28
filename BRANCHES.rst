Fontify 保留分支（2026-10-28）
================================

main
    全解冻 baseline。默认入口 finetune_font.sh 与原 baseline-vggfix
    的 1c8206e 配置一致：no_gan、no_jt、style_weight=14.73、lr=0.001、
    两卡 batch_size=2 / accum_iter=32、51 epoch、warmup=5。
    no_jt 表示 JT 使用随机遮盖，不是移除 JT 样本。
    合并 baseline-vggfix 历史，保留 main 中较新的共享源码和已经生效的
    VGG 二次归一化修复、BF 坐标检查及 TensorBoard 写图修复。
    Run B/结体监督均为显式可选模式，不由 baseline 默认入口启用。

codex/runb
    原 edge、非 strict 的完整双视图 Run B Fix，保留 L_full。
    从清理前 main 的 4023916 建立，包含 BF 坐标与 TensorBoard 修复。
    入口 finetune_dual_view_runb.sh；不是 tip、loss search 或 strict 版本。

codex/runc
    Run C v2，对应原 codex/runc-v2-jt-structure 的 47bf5a6。
    包含最初 Run C 的提交历史和实现，保留 v2 的 GT 泄漏修复及实验记录。
    保留该实验分支不代表其结构验收成功。

清理与恢复
------------

本地仅保留以上三条分支；origin 尚未同步，等待确认远程删除范围。
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
