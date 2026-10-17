# Run C v2 执行约定与 T0 初始化核查

2026-10-18。主方案为同目录 `plan_20261018.md`，与用户Downloads原文逐字节一致。

用户本对话已明确授权新分支开发、提交、push、服务器pull、阶段一训练和测试，必要时使用sub；本条授权覆盖方案中对同类操作的另行批准措辞。主控负责进度与汇总，sub不直接更新PROGRESS.md或询问用户。技术停止条件、最终人工结构验收以及阶段二暂不正式训练的范围不变。

## 起点与保护项

- 分支 `codex/runc-v2-jt-structure`，起点 `1106d03b18203f633b3e737a96f09cd03540ab02`。
- 进入时无tracked修改；已有 `fontdata_example/dual_targets_v1/`、`train_json_dual_v1/` 保留，不加入本轮提交。
- 服务器checkout `/root/autodl-tmp/fontify_dual_runb` 同起点SHA，2张RTX4090空闲，187GiB可用；原stage3_json删除、数据软链接状态保留。
- 所有新结果写入 `outputs/runc_v2_jt_structure_20261018/`；新模型写入 `models/runc_v2_jt_structure_20261018/`。不覆盖Run B、Run C v1或任何旧权重。

## 同名checkpoint不是相同权重

本地方案明确指定的 `models/vit_base_font/checkpoint-14.pth`：

- 文件SHA256 `339ae9781f4f3831b9c97a19f3c5160c72fa790dbc82157d1d7d6ddf7be17dac`，1,752,692,378字节。
- checkpoint epoch=14；args.output_dir=`models/vit_base_font`，finetune=`/root/autodl-tmp/pretrained/mae_pretrain_vit_base.pth`，epochs=15。
- 224个model keys；真实ViT-B 12层、768通道；包含原decoder与冻结VGG/判别器。

服务器旧同名文件SHA256为 `90df3369c8187c3ab6bd9e8a6f7af812eafc87c3657e2f4eff7879f578d99258`，1,752,590,682字节。其args.output_dir=`models/original_adv2`，finetune=`models/original_adv/checkpoint-9.pth`。逐张量检查194项不同，仅30项相同，主干也不同，不能解释为纯序列化差别。

本轮使用方案指定的**本地vit_base_font文件**作为唯一主初始化，复制到服务器新目录 `models/runc_v2_jt_structure_20261018/init/checkpoint-14.pth` 并核验同SHA。不覆盖服务器旧文件，也不把两者视为等价。加载兼容表、各key形状与哈希详见本轮 `checkpoint_manifest.json`；正式运行必须记录实际加载文件SHA。

## T0源码接口

`models_train.py:558` 的encoder有四个0-based taps `[2,5,8,11]`，每tap `B×56×28×768`，cat下半为 `B×28×28×3072`。idx2合并前保留四个 `B×28×28×768` quadrants。`util/dual_view.py:44` 的Head输出Q=`B×784×256`，四槽occupancy/existence/20个方位角色。

旧 `util/structure_first.py` 的conditioner以F为Query、Q/S/reference为KV，回注旧Decoder。v2改为S结构Query和新112密度/占用Head，不能复用v1路径充当新架构。

拟按四tap分别选择blocks `[2,5,8,11]` 和共享norm做IA1低学习率适配，Head单独学习率；实际参数名单、数量、梯度与冻结状态由运行检查确认，尚不宣称训练有效。IA0只作极短健康检查，IB冻结ViT/Head，只训练结构条件与粗图Head。

## 验证口径

T0真实数据为740=466双标+274JT-only，BF-only=0；以原图MD5/像素SHA与target SHA独立去重均可对账。历史101JT对只有77唯一目标；与train原图/target SHA均无交集，但有3个书家—字符重合，需单列排除此三字后的74唯一目标。此val长期用于研究选择，不称独立泛化。

本地初始旧路径回归88项通过，2项因本地缺timm无法运行；这两项须在依赖完整的服务器运行，不标作通过。本地torch2.11，服务器torch2.3.0+cu121；CPU合约与真实CUDA/NCCL检查分开记录。
