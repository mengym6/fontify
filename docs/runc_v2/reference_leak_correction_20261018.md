# Run C v2：自身参考泄漏纠正

旧训练采样允许query自身成为上半参考。上下使用同一裁剪，所以待预测GT经可见上半图进入encoder；清零下半GT与只替换下半张量的测试不能排除此通路。

精确CPU采样重放和真实图像张量检查确认：IA0_run1的128次暴露中3次自身参考；IA1_run1最终126次更新、4032次暴露中140次自身参考，前102次更新的116次可交叉对账。旧固定两字health中一个目标自身参考；其损失下降也不能作为安全路线的健康证据。IA1已终止，旧权重、标定和收益全部禁用，不从旧step4继续。

固定验证的101 JT配对没有同原图/像素SHA碰撞；BF139中有11个自身配对。原240文件保持不变。阶段一视图只暴露其中原JT101，不重配，仍为216实例/77唯一目标。排除3个训练同书家同字风险后的74目标继续单列，仍不是独立泛化集。

修复策略为 `distinct-original-reference-v1`：

- 预计算同书家/同pool的不同原图、不同target SHA参考候选；显式self reference与空候选硬拒绝，不回退自身，也不擅自增加异字符限制。
- JT740共有54,494条安全候选边，最少5个候选；全边与每目标64 seed审计均无自身参考。BF818安全视图保持原11个annotation层采样、不聚合类别，mask与显式安全reference下的旧实现逐值对账。
- dataset、collate、调用边界记录并核验query/reference身份及原图/target SHA，缺失、相同或不一致即停止。
- probe、标定选择与checkpoint metadata绑定该策略；旧run1缺少策略谱系，不能作为训练初始化或标定来源。
- 新增dataset层全局query-GT loader替换测试，覆盖此前遗漏的可见上半参考。

原oracle只由监督GT构造，不依赖训练参考选择，无需为此重新定义112表示。纯模型代数/形状测试保留；真实数据健康、梯度标定和训练将在新 `clean_run2` 目录从已锁定SHA的字体预训练checkpoint14重新执行，不复用污染权重。

安全修复及自动运行工具的本地整合验证为138项通过、15项subtest通过；对应Python3.9语法与Ruff E/F/I检查通过。后续实际服务器验证和clean训练以新目录的进程退出记录、快照、冻结SHA与固定评测为准，不将本地检查写作GPU完成。

逐次碰撞、BF自引用名单和候选审计保存在本轮 `self_reference_replay_audit.json`、`data_reference_policy_audit.json`、`data_old_health_probe_reference_audit.json`。旧原始文件保留供定位，不能因数值曾改善而恢复其有效性。
