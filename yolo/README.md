# Calli 笔画阶段与结体区域：YOLO26s 实例分割

一个模型同时输出50类笔画阶段（BF）与20类结体空间角色（JT）。直接使用Calli原始annotation，不合并整笔画。图像使用Fontify已有去噪PNG，不接入Fontify生成网络。

## 数据与监督

默认数据为`dataset_full/`：1215张、22,695个实例，训练1093张、验证122张（约9:1），覆盖全部9个来源。BF-only 311张、双标671张、JT-only 233张；验证中对应31/67/24张。全部69个实际出现的类别在训练集有样本，JT“中左”在源数据中没有实例。

对18个原始COCO文件全量处理；原图身份共1217个，其中1个缺去噪图、1个仅有相互冲突的BF版本而排除。另有冲突或空标注的任务标为未知，仍保留该图可靠的另一任务。详情在`summary.json`和`manifest.json`，原始Calli不改动。

按跨书家的规范汉字、原图SHA和去噪像素SHA联合成组，再以整组分配，兼顾书家和标注池比例。同字不同书家、同图不同命名不会跨train/val。已核对四种身份交集均为0。验证集用于开发和选模，没有独立测试集；这衡量已覆盖书家中的未见字泛化，不能代表新书家泛化。

- BF-only仍输出JT，但JT类别负样本及辅助语义损失不计；JT-only同理。
- `task_valid[B,2]`从`annotation_availability.json`进入loss。框、实例mask和L1仅使用实际GT；类别BCE及辅助语义项屏蔽未知任务。
- 辅助语义目标按类别合并实例为multi-hot，允许同一像素同时属于BF/JT，避免原单类别目标压掉另一任务。
- 验证只统计有该任务标注的图；预览仍展示两种预测，缺失GT列标记`UNANNOTATED`。
- `overlap_mask=False`保留重叠实例。禁用翻转、Mosaic、MixUp、Copy-paste等会破坏标签语义或混合未知域的增强。

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

`patience=0`关闭指标早停，`EPOCHS`仍是轮数上限。人工停止在当前epoch保存和预览完成后生效：

```bash
touch runs/<运行名>/STOP
```

正常结束会按Ultralytics逻辑剥离last/best优化器状态；中断续训应使用尚未完成的完整checkpoint，或保留的周期checkpoint。继续训练前检查checkpoint状态，不能将已完成的权重冒充中断恢复点。

`OPTIMIZER=auto`会自行选择优化器与学习率，忽略手填`LR0`；需要控制学习率时选AdamW或SGD。原始训练超参数保留为可调选项，本版本没有自动采用上一轮分析产生的新优化方案。

环境使用`/root/autodl-tmp/fontify_yolo_env`，复用系统CUDA PyTorch；Ultralytics固定8.4.218。安装其他机器时先准备匹配GPU的PyTorch，再运行：

```bash
python -m pip install -r requirements.txt -c autodl_constraints.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

代码由Git同步；`dataset_full/`、旧`dataset/`、models、runs均不进入Git，需单独传输。整个yolo目录可独立搬到训练机器。

## TensorBoard

默认读取`runs/dashboard_full`，旧看板在`runs/dashboard`保留。服务器6006转发至本机即可访问；手动启动命令：

```bash
source /root/autodl-tmp/fontify_yolo_env/bin/activate
tensorboard --logdir /root/autodl-tmp/fontify_yolo/yolo/runs/dashboard_full --host 127.0.0.1 --port 6006 --samples_per_plugin images=1000
```

全部评价每epoch遍历完整122张验证图；固定置信度0.25、mask IoU匹配阈值0.5。BF/JT各自组内，先最大化一对一匹配数量，再最大化总IoU。所有曲线按任务分组：

| BF标签（JT同理） | 定义 |
|---|---|
| `BF/metrics/recall_pct` | 几何匹配数 / GT区域数，是否找全 |
| `BF/metrics/precision_pct` | 几何匹配且细标签正确数 / 全部预测数，错标签和多框均扣分 |
| `BF/metrics/iou_pct` | 几何成功匹配的mask IoU均值，轮廓重合程度 |
| `BF/loss/train_combined` | BF训练主任务合并损失贡献 |
| `BF/loss/val_combined` | BF验证主任务合并损失贡献 |

按用户定义，查全先看区域找到没有，查准同时看细标签和多框，因此两者分子不同。IoU是成功匹配条件下的平均值，必须结合查全看；没有匹配时无定义，JSON记null、TB省略。不是像素准确率。

四条loss读取实际one-to-one分支box+mask+cls+L1加权贡献，沿用共同分母，BF+JT等于该分支四项之和。验证不计算辅助语义项，所以两端曲线均排除该项；实际E2E反传仍包含两个分支与辅助项。曲线是可比较的主任务量，不能称为全部反传目标。细项和mAP仍保留在results.csv。

`previews/validation`是唯一图卡：每epoch写入同一批60张验证图，跨9个来源固定抽样。图卡的样本选择器切换0–59，epoch滑块切换轮次。每张仍为输入/BF GT/BF预测/JT GT/JT预测五列；60张PNG也完整保存于`runs/<运行名>/previews/epoch_XXX/`。`fixed_samples.json`绑定新划分身份，训练中不重新抽样。启动脚本默认保留1000个epoch的图片历史。

## 预测与后验评估

```bash
python predict.py --weights runs/<运行名>/weights/best.pt --source dataset_full/images/val --device 0
python evaluate_regions.py --weights runs/<运行名>/weights/best.pt --split val --save-previews --output runs/evaluation_full
python train.py --resume runs/<运行名>/weights/last.pt --device 0
```

预测导出类别、置信度、框、轮廓及保留孔洞/不连通部分的mask RLE。后验评估与训练使用同一三指标口径和标注可用性。对旧模型复核必须显式指定`--data dataset/data.yaml`，使用其原划分。

## 主要文件

- `build_full_dataset.py`：全量Calli转换、去噪图复制、分组9:1划分与清单。
- `partial_supervision.py`：部分标注Dataset/Trainer/Validator、损失屏蔽和实际分组loss。
- `train.py`、`start_train.sh`：训练、参数入口与TB启动。
- `live_visualization.py`、`region_metrics.py`：固定预览与BF/JT三指标。
- `calli_loss.py`：辅助语义权重与恢复记录。
- `dataset_full/{manifest,summary,annotation_availability}.json`：来源、数量、划分审计和任务可用性。
- `tests/`：数据协议、未知任务梯度、数值一致性、统计与TB接口测试。

官方参考：[训练配置](https://docs.ultralytics.com/modes/train/)、[分割任务](https://docs.ultralytics.com/tasks/segment/)。
