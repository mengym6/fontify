# Calli 笔画阶段与结体区域：YOLO26s 实例分割

一个模型同时输出50类笔画阶段（BF）与20类结体空间角色（JT）。直接使用Calli原始annotation，不合并整笔画。图像使用Fontify已有去噪PNG，不接入Fontify生成网络。

## 当前 baseline（2026-11-27 用户确认）

采用 `calli_search_v7_02_lr4x_20261127` 的官方 `best.pt`（第44轮；完整训练60轮），替代v3作为后续比较基线。服务器权重路径：

```text
/root/autodl-tmp/fontify_yolo/yolo/runs/calli_search_v7_02_lr4x_20261127/weights/best.pt
```

SHA256：`1e85c30244c45ea7213323a7517da91170416db013673ff06faf350fae80cfdf`。训练源码版本 `41dfe5d`，Ultralytics 8.4.218；本次仅登记baseline并更新启动默认值，不是新训练结果。

| 指标 | 当前baseline |
|---|---:|
| Box mAP50 / mAP50–95 | 58.097% / 32.188% |
| Mask mAP50 / mAP50–95 | 59.086% / 30.316% |
| BF mask mAP50–95 | 26.227% |
| JT mask mAP50–95 | 43.128% |
| 官方fitness | 0.62504 |

上述指标均对应第44轮同一个checkpoint，来自固定开发验证集，非独立test。数据1093 train / 122 val；split_id为`ccdbdad60ff49e8d4aa6d287ac384f1b7c648ef3fff9ee437d7b89e3f79e661c`。

训练参数：YOLO26s-seg从COCO初始化；60轮、batch24、imgsz640、seed42；AdamW `lr0=0.00054`、`momentum=0.9`（beta1，beta2=0.999）、`weight_decay=0.0005`（当前batch下普通权重实际0.0005625）；线性学习率衰减、`lrf=0.01`、预热3轮、`warmup_bias_lr=0`；全程BF/JT联合、`freeze=0`、`BF_ONLY_AFTER=0`、`STAGE_FREEZE=0`；`mask_ratio=4`、AMP开启、`patience=0`；box/cls/dfl增益7.5/0.5/1.5，辅助semantic倍率1。

增强：Mosaic1（最后10轮关闭）、translate0.1、scale0.5、HSV0.015/0.7/0.4；旋转、剪切、透视、翻转、MixUp、Copy-paste、CutMix均0。部分标注监督与`overlap_mask=False`保持。

`bash start_train.sh`是本baseline的默认复现入口，仍从COCO开始训练；第44轮是这次60轮训练的选模结果，不能将训练预算改成44轮视为等价复现。底层`train.py`的历史CLI默认值未变，直接调用它时应显式传入上述参数。本地与服务器启动脚本使用同一baseline默认参数。

## 数据与监督

默认数据为`dataset_full/`：1215张、22,695个实例，训练1093张、验证122张（约9:1），覆盖全部9个来源。BF-only 311张、双标671张、JT-only 233张；验证中对应31/67/24张。全部69个实际出现的类别在训练集有样本，JT“中左”在源数据中没有实例。

对18个原始COCO文件全量处理；原图身份共1217个，其中1个缺去噪图、1个仅有相互冲突的BF版本而排除。另有冲突或空标注的任务标为未知，仍保留该图可靠的另一任务。详情在`summary.json`和`manifest.json`，原始Calli不改动。

按跨书家的规范汉字、原图SHA和去噪像素SHA联合成组，再以整组分配，兼顾书家和标注池比例。同字不同书家、同图不同命名不会跨train/val。已核对四种身份交集均为0。验证集用于开发和选模，没有独立测试集；这衡量已覆盖书家中的未见字泛化，不能代表新书家泛化。

- BF-only仍输出JT，但JT类别负样本及辅助语义损失不计；JT-only同理。
- `task_valid[B,2]`从`annotation_availability.json`进入loss。框、实例mask和L1仅使用实际GT；类别BCE及辅助语义项屏蔽未知任务。
- 辅助语义目标按类别合并实例为multi-hot，允许同一像素同时属于BF/JT，避免原单类别目标压掉另一任务。
- 验证只统计有该任务标注的图；预览仍展示两种预测，缺失GT列标记`UNANNOTATED`。
- `overlap_mask=False`保留重叠实例。当前默认沿用v3增强：`mosaic=1`、`degrees=0`、`shear=0`、`perspective=0`、`translate=0.1`、`scale=0.5`；翻转保持关闭，HSV仍为`0.015/0.7/0.4`。这是下一次新训练的默认配置，旧run及resume仍沿用保存的原参数。

新划分从COCO `models/yolo26s-seg.pt`开始。旧Calli模型可能已见过新验证图，训练和后验评估入口会拒绝将缺少相同split记录的Calli权重用于新划分。旧`dataset/`与旧运行保留作历史复核，不能直接横向比较新旧划分得分。

## AutoDL启动

```bash
source /root/autodl-tmp/fontify_yolo_env/bin/activate
cd /root/autodl-tmp/fontify_yolo/yolo
bash start_train.sh --dry-run  # 只检查数据、参数
bash start_train.sh           # 后台训练并启动/复用TensorBoard
bash start_train.sh --tb-only # 只开TensorBoard
```

常用超参数在`start_train.sh`顶部，默认沿用服务器的60轮、batch24，也可临时覆盖：

```bash
EPOCHS=60 BATCH=24 RUN_NAME=calli_full_my_run bash start_train.sh
```

启动脚本默认全程联合训练BF/JT，不冻结浅层。历史v6分阶段方案仍可通过`BF_ONLY_AFTER=40 STAGE_FREEZE=4`显式启用；仅在启用时第41轮起只计算BF训练损失并冻结backbone模块0–3及其BatchNorm统计。验证始终包含两个任务，官方best选模规则不变。

阶段2会在匹配前移除JT GT，并关闭JT分类/辅助语义监督；JT-only样本不贡献训练损失，数据加载保持原池。关闭的是训练监督，深层共享参数仍可改变JT输出。阶段2的JT训练loss不再写入，CSV留空；JT验证loss和mAP继续正常显示，不能把关闭的训练loss当成0误差。

仅启用上述40轮切换方案时，第40轮额外保留`weights/joint_end.pt`用于对照；阶段切换不重建优化器、学习率或E2E调度。阶段配置保存在已有`partial_supervision.json`内，resume继承原计划；旧run没有阶段配置时仍全程联合训练。

`patience=0`关闭指标早停，`EPOCHS`仍是轮数上限。人工停止在当前epoch保存和预览完成后生效：

```bash
touch runs/<运行名>/STOP
```

默认保存`weights/best.pt`、`last.pt`，关闭周期checkpoint；只有启用分阶段训练才额外保存阶段边界`joint_end.pt`。正常结束会按Ultralytics逻辑剥离last/best优化器状态；中断续训使用尚未完成的完整checkpoint。继续训练前检查checkpoint状态，不能将已完成的权重冒充中断恢复点。

当前启动脚本显式使用`OPTIMIZER=AdamW`和`LR0=0.00054`。如改为`OPTIMIZER=auto`，原生逻辑会自行选择优化器与学习率、忽略手填`LR0`，不能将其视为当前baseline配置。

环境使用`/root/autodl-tmp/fontify_yolo_env`，复用系统CUDA PyTorch；Ultralytics固定8.4.218。安装其他机器时先准备匹配GPU的PyTorch，再运行：

```bash
python -m pip install -r requirements.txt -c autodl_constraints.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

代码由Git同步；`dataset_full/`、旧`dataset/`、models、runs均不进入Git，需单独传输。整个yolo目录可独立搬到训练机器。

## TensorBoard

默认读取`runs/tensorboard`，旧看板在`runs/dashboard`保留。服务器6006转发至本机即可访问；手动启动命令：

```bash
source /root/autodl-tmp/fontify_yolo_env/bin/activate
tensorboard --logdir /root/autodl-tmp/fontify_yolo/yolo/runs/tensorboard --host 127.0.0.1 --port 6006 --samples_per_plugin images=1000
```

每epoch对完整122张验证图评价。新增的官方mask mAP来自原生Validator，按细类别匹配、置信度排序和IoU阈值计算；BF/JT分别对实际有GT的类别求平均。原有三项诊断指标使用置信度0.25、mask IoU阈值0.5，组内先最大化一对一几何匹配数量，再最大化总IoU。所有曲线按任务分组：

| BF标签（JT同理） | 定义 |
|---|---|
| `BF/metrics/mask_mAP50_pct` | 官方mask AP在IoU=0.5下，对BF有效类别取平均 |
| `BF/metrics/mask_mAP50_95_pct` | 官方mask AP在IoU=0.5至0.95的10个阈值及BF有效类别上取平均 |
| `BF/metrics/recall_pct` | 几何匹配数 / GT区域数，是否找全 |
| `BF/metrics/precision_pct` | 几何匹配且细标签正确数 / 全部预测数，错标签和多框均扣分 |
| `BF/metrics/iou_pct` | 几何成功匹配的mask IoU均值，轮廓重合程度 |
| `BF/loss/train_combined` | BF训练主任务合并损失贡献 |
| `BF/loss/val_combined` | BF验证主任务合并损失贡献 |

漏检已经计入查全率及官方AP；precision的分母仅为预测数量，不单独包含FN。例如GT有10块，仅预测3块且都正确，precision=100%，recall=30%，并不代表整体识别正确。按当前诊断定义，查全先看区域找到没有，查准同时看细标签和多框，因此两者分子不同。IoU是成功匹配条件下的平均值，必须结合查全看；没有匹配时无定义，JSON记null、TB省略。不是像素准确率。

**选模型完全保留Ultralytics 8.4.218官方规则：`fitness = box mAP50–95 + mask mAP50–95`。** 不使用BF/JT均值选模，也不把mAP50加入选模权重。两组mAP仅用于分任务观察；整体官方mAP和fitness仍由原生验证与checkpoint流程处理。缺失标注任务仍排除在评价域之外。

分组mAP使用`seg.all_ap`及`seg.ap_class_index`，不会把验证集中不存在类别的填充值混入平均；有GT但没有正确预测的类别AP计0。固定Ultralytics版本，避免版本间AP插值口径变化影响比较。TB共14条标量：BF/JT各5条评价曲线及2条loss。新曲线从下一次使用此版本的训练开始，历史events保持原样。

四条loss读取实际one-to-one分支box+mask+cls+L1加权贡献，沿用共同分母，BF+JT等于该分支四项之和。验证不计算辅助语义项，所以两端曲线均排除该项；实际E2E反传仍包含两个分支与辅助项。曲线是可比较的主任务量，不能称为全部反传目标。官方细项和整体mAP保留在`results.csv`，14条TB曲线同时汇总到单个`metrics.csv`。

**每个epoch单独一个预览card**：`epoch:001/validation`、`epoch:002/validation`……。每个epoch是独立的顶层折叠分组，展开只有一个图框。与Fontify原始`add_image`写法一致，epoch放在tag，图片序号0–59放在step，因此拖动图框进度条切换该轮的60张固定图；没有额外的多样本选择器。固定图像身份和顺序跨epoch不变。每张仍为输入/BF GT/BF预测/JT GT/JT预测五列；图片只在内存编码后写入TB，不生成独立PNG、逐轮预览JSON或逐图指标文件。`previews/fixed_samples.json`绑定划分身份，训练中不重新抽样。

运行目录默认只保留`weights/`、`args.yaml`、`results.csv`、`metrics.csv`、恢复训练必需的loss/partial配置和固定预览清单；TB在`tensorboard/`。原生batch预览、混淆矩阵等绘图默认关闭。数据构建不再自动生成标签预览。

## 预测与后验评估

```bash
python predict.py --weights runs/<运行名>/weights/best.pt --source dataset_full/images/val --device 0
python evaluate_regions.py --weights runs/<运行名>/weights/best.pt --split val --save-previews --output runs/evaluation_full
python train.py --resume runs/<运行名>/weights/last.pt --device 0
```

预测导出类别、置信度、框、轮廓及保留孔洞/不连通部分的mask RLE。后验评估与训练使用同一三指标口径和标注可用性。对旧模型复核必须显式指定`--data dataset/data.yaml`，使用其原划分。

## 主要文件

- `build_full_dataset.py`：全量Calli转换、去噪图复制、分组9:1划分与清单。
- `staged_training.py`：阶段切换、浅层冻结、BF-only标注筛选和边界checkpoint。
- `partial_supervision.py`：部分标注Dataset/Trainer/Validator、损失屏蔽、实际分组loss及官方分组AP输出。
- `official_metrics.py`：从原生mask AP结果提取BF/JT mAP，保留官方fitness。
- `train.py`、`start_train.sh`：训练、参数入口与TB启动。
- `live_visualization.py`、`region_metrics.py`：固定预览与BF/JT三指标。
- `calli_loss.py`：辅助语义权重与恢复记录。
- `dataset_full/{manifest,summary,annotation_availability}.json`：来源、数量、划分审计和任务可用性。
- `tests/`：数据协议、未知任务梯度、数值一致性、统计与TB接口测试。

官方参考：[训练配置](https://docs.ultralytics.com/modes/train/)、[分割任务](https://docs.ultralytics.com/tasks/segment/)。

## 双图联合区域对应（pairing，代码阶段）

用户于2026-11-27确定新目标：同时输入Calli GT与生成字，直接输出两侧成对的BF/JT区域，不先做70类识别再配对。当前实现位于`yolo/pairing/`，原单图baseline与训练入口保持独立。

- `model.py`：共享YOLO26 backbone/neck/Proto26；将两侧空间特征和侧别编码一起输入Transformer decoder。每个共享query输出A/B两个mask及两侧存在分数。默认64个BF、16个JT query，只区分任务，不预测50/20细类别。
- `loss.py`：训练时把一个预测query同时分配给一个GT区域对，使用两侧联合代价做Hungarian分配；这是训练监督分配，不是推理后两图检测匹配。mask BCE、Dice、归一化质心、面积比、存在性分别加权。默认系数1/1/1/.25/1，尚未验证为有效训练配方。
- `data.py`：统一letterbox，保留图像几何；区别真实存在、明确缺失、未标注。未知侧不产生直接mask/存在性监督，未标注的整套任务不产生背景负例。
- `train.py`：单独的新任务训练入口，保存best/last与逐轮train/val损失；best暂按成对验证loss选取，不称为YOLO mAP。默认冻结继承的特征层及BN，先学习新头；`--unfreeze-encoder`可显式微调特征。不会继承旧训练optimizer或冒充resume。
- `predict.py`：对训练后的pair checkpoint同时推理两图，以相同query ID导出成对mask、存在性和置信度；mask RLE恢复到各自原图尺寸，不做细分类/NMS/后匹配。
- `build_pairs.py`：按ep49旧评测索引的精确目标路径绑定Calli身份，保留现有YOLO split/leak_group，不用同名字直接猜配。

初始化只复用当前best44的backbone、neck、prototype，70类分类头和辅助semantic头不作为对应输出使用。新增的双图decoder/query/成对mask系数及存在性头是随机初始化。完整pair checkpoint包含YOLO结构配置和全部模型state，可重载推理，不依赖原baseline权重文件再次初始化。

### 已配对的数据

用户指定生成字目录：`/Users/root1/Desktop/callify_final/02_推理图片/Run_B_best_ep49_单图`。

使用`outputs/RunB_ep49/visual_best_vs_vggfix_240/index.csv`的`file/query_path`字段进行绑定。240张生成字全部匹配，无排除；对应160张独立Calli原图、148个划分组。继承YOLO划分得到217组train、23组val，分别144/16张不同Calli原图。重复目标不跨split；这里的val不能据此声明为相对生成器训练历史独立的测试集。

实际清单位于`yolo/pair_data/runb_ep49/pairs.json`，汇总位于同目录`summary.json`，均忽略Git。图像采用原文件绝对路径引用，不复制或改写图片；迁移服务器时必须连同图像一起迁移并更新路径。文件SHA用于定位本次配对输入，不将目录名字当作已核验的生成器checkpoint SHA。

**本批只有图像级配对和Calli侧5055个区域标注记录，生成侧没有对应区域标注。** 生成侧`present=null`，没有复制GT多边形或制作伪标签。用户本轮明确要求暂不正式训练；当前训练入口也会拒绝以这批无生成侧监督的数据直接开跑。代码可执行不代表新模型已有对应识别能力。

### 配对格式

`pairs.json`顶层schema为`calli-paired-regions-v1`，`pairs`的每一项含`id`、`split`、`image_a`、`image_b`、`leak_group`、`complete`和`regions`。A为Calli，B为生成字。`complete`为2×2布尔矩阵，行A/B、列BF/JT，仅完整标注的任务允许监督未分配query为背景。

每个region包含唯一`pair_id`、`group`（BF/JT）和`a`/`b`。每侧：

- `present=true`：区域存在；`polygons`为该侧原图归一化坐标多边形列表。
- `present=false`：人工/明确监督确认缺失，惩罚存在分数，不虚构mask。
- `present=null`：不知道在哪里或是否缺失，既不是负例，也不能直接当作对应真值。
- `source`明确为human、synthetic、pseudo或unannotated。伪标签只有显式`--allow-pseudo`才接受，当前数据不含伪标签。

一对区域应至少有一侧确认存在。GT侧和生成侧的多边形可不同；需要人工标注或用户另行确认的监督构造方法才能进入正式对应训练。

### 命令（从仓库根目录运行）

重新准备当前图像配对：

```bash
python -m yolo.pairing.build_pairs \
  --generated '/Users/root1/Desktop/callify_final/02_推理图片/Run_B_best_ep49_单图' \
  --index-csv outputs/RunB_ep49/visual_best_vs_vggfix_240/index.csv \
  --dataset yolo/dataset_full --output yolo/pair_data/runb_ep49
```

仅检查监督状态，不初始化GPU或训练：

```bash
python -m yolo.pairing.train --check-only \
  --manifest yolo/pair_data/runb_ep49/pairs.json \
  --baseline /root/autodl-tmp/fontify_yolo/yolo/runs/calli_search_v7_02_lr4x_20261127/weights/best.pt \
  --output yolo/runs/pair_regions_v1
```

本批返回`training_ready=false`是预期结果。补齐经过确认的生成侧区域对应后，才可在新目录显式训练；训练入口没有自动启动、自动续训或继承夜间搜索预算的行为。未来推理入口：

```bash
python -m yolo.pairing.predict --checkpoint <pair-run>/best.pt \
  --image-a <Calli.png> --image-b <generated.png> --output <pairs.json>
```

当前训练入口只记录成对loss，尚未接入原单图mAP/TensorBoard图册，也未接入Fontify生成损失；这些不影响本轮代码与图像配对交付，不能宣称已完成正式训练或生成网络整合。

### 复制GT区域的弱监督版本（2026-11-27追加）

用户提出先尝试复制GT标签。`yolo/pairing/copy_gt.py`从未标注生成侧的配对清单建立独立副本：保留两侧相同pair_id，把Calli的归一化多边形、存在状态及已知任务范围复制到生成侧，`source=pseudo`、`method=copy_gt_normalized_coordinates`。70类细标签仍不进入模型输出或loss，只保留BF/JT分组及实例对身份。

```bash
python -m yolo.pairing.copy_gt \
  --manifest yolo/pair_data/runb_ep49/pairs.json \
  --output yolo/pair_data/runb_ep49_copygt
```

已实际生成`runb_ep49_copygt/pairs.json`与`summary.json`：240对均为448×448、5055个复制区域、217train/23val。原始清单和图片不变。训练入口仍需显式`--allow-pseudo`；当前未启动正式训练。

复制的是位置/大小/形状/存在性的弱假设，不是对真实对应的验证。生成字偏移、缺笔、粘连时复制标签可能错误；与复制val标签一致只能衡量拟合该假设，不能当作真实对应准确率。验证真实对应需要人工核查或具有可追踪真值的形变/错配测试。没有因忽略细类别tag而取消这些几何限制。

### 逐图审阅后的近似标签版本

`yolo/pair_data/runb_ep49_reviewed/pairs.json`保存240对经助手图像对级视觉检查后的区域修订，配准辅助迁移并对161个区域进行额外定位/范围修正、近似拆分或缺失处理。共有5055条区域记录、217train/23val；原GT和图片保持不变，旧copygt版本保留。`review_log.csv`逐图记录处理方式，`summary.json`汇总，`review_examples/`四个示例是训练标签叠加而非模型输出。

用户确认：易的中央短竖仍对应第三撇；粘连部分按GT相对位置近似拆分；無的中央粗点按左右两部分对应两个GT点。歌和思的两处可见笔迹缺失对应4个阶段区域，生成侧present=false。所有修订仍保留source=pseudo/approximate=true；没有将配准和图像对级目检宣称为5055区域逐边人工精标。字段copied_gt保留最初复制版本的历史来源，其旧的同位置/同存在性假设已被review_policy与各区域修订覆盖。

可将此文件作为未来`--manifest`并显式`--allow-pseudo`使用；当前没有启动正式训练。修订标签上的loss只能衡量拟合这些弱目标，不能作为独立的真实对应准确率。

### 生成侧多余区域（ERR）

用户确认“未”中主干与末端点画之间的额外连接应判错。对应标注为A.present=false、A.polygons=[]、B.present=true及错误区域；role/relation为extra_in_generated。当前reviewed数据新增6个这样的区域，合计5061条。预测入口用两侧presence输出relation与is_error，GT无且生成有时标extra_in_generated；训练时监督生成侧mask与两侧存在性，细类别tag不参与。浏览器用红色虚线ERR显示。此为代码/标注支持，尚未正式训练验证识别效果。

最新ERR复查（2026-11-27）：已覆盖240对全部图像，本轮确认多余区域共40处，分布于22个样本、11个汉字（未、曰、成、雲、注、卿、之、秀、崇、言、思）；当前数据共5095条区域记录。具体见reviewed/extra_errors.csv，浏览页可勾选“只列有ERR样本”。原仅3个未字/6处ERR是历史阶段记录。标签仍是助手审核的近似弱监督，未正式训练。

### 三态输出（当前规则）

正常对应：两侧都有区域；多余ERR：GT无而生成有；缺失MISS：GT有而生成无，但生成侧仍圈出期望位置。presence表示实际笔迹存在性，mask_supervised独立控制区域监督。缺失侧使用present=false与mask_role=expected_missing_region，非空多边形继续参与定位损失和推理输出。此规则覆盖前文缺失不输出mask的旧说明。当前数据5015正常、40 MISS、40 ERR；10项测试通过，尚未正式训练。

## v9：GT引导的生成字区域对应

v9代码位于`yolo/correspondence_v9/`，与v8的自由query三态输出分开。**输入两张图片以及GT侧已有BF/JT区域标注**；从每个GT区域池化图像特征、位置和任务信息，构成有明确GT ID的查询。联合读取两侧图像后，每个GT查询只输出生成侧mask和matchability，不重新检测GT，也不输出无GT来源的新区域。

输出两种状态：`matched`附带生成侧mask；`unmatched`标明哪个GT ID对应不上，`gen_mask=null`。推理不读取生成侧标注，生成侧真实/弱参考仅用于训练和评价。若未来输入只有未标注GT图，本接口还需要先提供GT区域，不能冒称当前已支持无标注GT端到端推理。

### 数据与监督

`prepare.py`将v8审阅数据转换成独立v9清单，保留图像身份、GT区域ID和split。原40个GT存在但生成侧缺失的实例变为`matchable=false`，丢弃其复制的生成侧GT占位框，防止把占位当成功匹配；原40个生成侧独有ERR不作为GT查询。当前240对、5055个GT锚点：train4574可对应/40不可对应，val441可对应/0不可对应。原v8清单和训练不改。

v9参考标签仍为`reviewed_weak`。验证集没有不可对应样本，拒绝召回率与错误接受率会返回null、support=0，不能据此验收拒绝能力；也不能把对这些参考标签的拟合当成独立人工精标准确率。

### 模型与损失

- `model.py`：GT-mask池化的条件query，双图交互；固定每个GT一个结果，查询次序改变只改变输出次序。继承YOLO best44的encoder/neck/prototype，或从v8 checkpoint迁移兼容的encoder、投影、decoder和生成侧输出层；GT查询构造是新增部分，需要学习。两类来源通过checkpoint schema区分，不把未训练v8随机头当预训练迁移。
- `loss.py`：按GT ID直接监督，不再Hungarian交换查询身份。可对应区域使用BCE+Dice+质心L1+0.25×面积对数比；所有已知锚点使用matchability BCE，正负类在同图中分别平均后等权组合；不可对应区域只监督拒绝，不要求输出复制框。
- 附加0.1×同任务预测过度重叠损失：仅对可对应GT锚点，预测重叠超过参考重叠加0.05时处罚；允许参考本来存在的重叠。初始权重是实现配置，未宣称最优。
- 未标注matchability为null（数据加载为-1），不当作不可对应负例。弱参考正式训练需显式`--allow-weak-labels`。

### 评价口径

`metrics.py`按各自GT ID核对，不通过重新排列输出获得好成绩。默认matchability阈值0.5、对应mask IoU阈值0.5，BF/JT分组和总体分别报告：

- 正确匹配、错配、误拒绝、正确拒绝、错误接受的计数。
- 匹配查准率=正确匹配/预测匹配；召回率=正确匹配/可对应GT锚点。
- 对应不上查准率=正确拒绝/预测拒绝；召回率=正确拒绝/不可对应GT锚点。
- false_reject_rate、false_accept_rate、接受的可对应样本平均IoU。
- 两个本来不重叠的同任务GT锚点，如果预测出近乎相同区域（参考IoU<0.5、预测IoU>0.9），记重复冲突，两者不算正确匹配。若预测明显更贴合另一个GT锚点的参考区域，也记错配。

定位评估使用输入letterbox分辨率上的参考mask，预测logit双线性上采样后阈值化；预测导出使用相同解码，避免先缩小GT引入额外别名误差。缺少分母的指标输出null，不补0或100%。这些是自定义对应指标，不是YOLO mAP。

### 入口

从仓库根目录执行。首次准备（当前本地清单已生成）：

```bash
python -m yolo.correspondence_v9.prepare \
  --source yolo/pair_data/runb_ep49_reviewed/pairs.json \
  --output yolo/pair_data/v9_gt2gen/pairs.json
```

仅核对，不训练：

```bash
python -m yolo.correspondence_v9.train --check-only \
  --manifest yolo/pair_data/v9_gt2gen/pairs.json \
  --init /root/autodl-tmp/fontify_yolo/yolo/runs/calli_search_v7_02_lr4x_20261127/weights/best.pt \
  --output yolo/runs/v9
```

正式训练需另行执行该入口并去掉`--check-only`，加`--allow-weak-labels`；默认20轮、batch2、320、lr1e-4、冻结encoder，支持`--unfreeze-encoder`。可用`--resume <run>/last.pt --epochs <总轮数>`恢复模型/optimizer/RNG，检查数据SHA、配置和历史轮次。`STOP`在epoch边界生效；不覆盖v8 run。

```bash
python -m yolo.correspondence_v9.predict --checkpoint <v9-run>/best.pt \
  --manifest <v9-pairs.json> --output <predictions.json>
python -m yolo.correspondence_v9.evaluate --checkpoint <v9-run>/best.pt \
  --manifest <v9-pairs.json> --output <evaluation.json>
```

训练保存best/last、metrics.jsonl与evaluation.json；best暂按v9验证loss选取。TensorBoard保留BF/JT两行，分别显示GT锚点、生成侧参考、生成侧匹配预测；拒绝时标出对应GT编号，不画假的生成框。当前仅实现、测试和同步，未运行正式v9训练。
