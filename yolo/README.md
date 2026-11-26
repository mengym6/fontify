# Calli 笔画阶段与结体区域：YOLO26s 实例分割

直接使用 Calli 原始标注训练一个模型：**50 类笔画阶段＋20 类结体区域，共 70 类**。每个非 `text` 的 annotation 是一个实例，不合并阶段、不恢复整笔，不接入 Fontify。

代码通过 Git 同步到 AutoDL；数据、模型和运行输出不进入 Git。下面使用服务器目录 `/root/autodl-tmp/fontify_yolo/yolo`，实际部署位置不同时替换该路径。

## AutoDL 环境与数据

在 VS Code 通过 Remote SSH 连接 AutoDL 后，打开服务器终端：

```bash
source /root/autodl-tmp/fontify_yolo_env/bin/activate
cd /root/autodl-tmp/fontify_yolo/yolo
python -m pip install -r requirements.txt -c autodl_constraints.txt
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'unavailable')"
```

本次环境 `/root/autodl-tmp/fontify_yolo_env` 通过 `venv --system-site-packages` 复用服务器现有 PyTorch 2.1.2+cu121／torchvision 0.16.2+cu121；约束 NumPy<2、OpenCV<4.12，避免旧版 PyTorch 与 NumPy 2 的 ABI 不兼容，不改系统环境。

使用已安装匹配 CUDA 版本 PyTorch 的 GPU 环境。`nvidia-smi` 能看到显卡，且 Python 输出 `CUDA: True` 后再训练。

`dataset/` 已被 Git 忽略，拉取代码不会自动得到数据。将本地 `yolo/dataset/` 完整传到服务器同级目录，最终至少应存在：

```text
yolo/
├── train.py
├── predict.py
├── requirements.txt
└── dataset/
    ├── data.yaml
    ├── classes.json
    ├── images/train/
    ├── images/val/
    ├── labels/train/
    └── labels/val/
```

检查数据，不下载模型、不启动训练：

```bash
python train.py --dry-run
```

## 统一启动脚本：参数、训练、TensorBoard

常用超参数集中在 **`start_train.sh` 顶部**：轮数、batch、输入分辨率、学习率、优化器、权重衰减、warmup、早停、AMP、存档间隔和预览间隔。默认保持现有训练配置。

```bash
cd /root/autodl-tmp/fontify_yolo/yolo
bash start_train.sh --dry-run  # 只检查数据与参数，不启动任何服务或训练
bash start_train.sh           # 自动启动/复用TensorBoard，后台启动训练
```

启动后终端会显示训练PID、日志、权重目录和TensorBoard地址。可以关闭终端；训练和TensorBoard独立后台运行。同一目录已有训练时只准备TensorBoard并显示已有PID，不会再启动一份训练。

也可以一次性覆盖参数，不修改文件：

```bash
EPOCHS=80 BATCH=8 OPTIMIZER=AdamW LR0=0.0003 bash start_train.sh
```

**`OPTIMIZER=auto` 时，Ultralytics会自行选择优化器和初始学习率，手填 `LR0` 不生效。** 要调整初始学习率，请选择 `AdamW` 或 `SGD`。`COS_LR=1` 启用余弦调度；`AMP=0` 关闭混合精度。`RUN_NAME` 默认带启动时间，也可手动指定；同名结果不会覆盖。

只启动或复用TensorBoard，不启动训练：

```bash
bash start_train.sh --tb-only
```

TensorBoard监听服务器 `127.0.0.1:6006`，日志根目录与脚本的 `PROJECT_DIR` 一致。端口被其他服务或其他日志目录占用时会明确提示；修改顶部 `TB_PORT` 可换端口。无需自动打开远程浏览器。

在本地VS Code的 **Ports（端口）→ Forward a Port（转发端口）** 输入 `6006`，打开转发后的本机地址。曲线每轮更新，Images中的固定4个验证字在首轮、每5轮、结束轮更新。

如果希望手动启动TensorBoard，可在服务器另一个终端执行：

```bash
source /root/autodl-tmp/fontify_yolo_env/bin/activate
tensorboard --logdir /root/autodl-tmp/fontify_yolo/yolo/runs --host 127.0.0.1 --port 6006
```

默认最多100轮、batch8、640输入、patience25。第一次训练会下载COCO初始化到 `models/`；现有数据、权重和运行结果均不由Git覆盖。当前在跑的训练不需要为新增启动脚本重启；修改脚本参数只影响下一次新训练。

## 验证、预测与续训

训练结束会完成验证。使用所选权重对已有验证图或新的单字图预测：

```bash
python predict.py --weights runs/calli_phase_jieti_yolo26s/weights/best.pt --source dataset/images/val --device 0
```

`predict.py` 输出叠加图、每实例类别／中文名／置信度／框／轮廓，以及保留孔洞和不连通区域的无损二值 mask RLE。建议输入单字、黑字白底、周围有留白的图像。

若训练中断，可用对应运行的 `last.pt` 继续：

```bash
python train.py --resume runs/calli_phase_jieti_yolo26s/weights/last.pt --device 0
```

续训默认写入该权重所属的运行目录，也可通过 `--resume-output` 指定新目录。已经完成且不含优化器状态的权重不能作为中断训练恢复点；`best.pt` 用于按验证集表现选模和推理。

## 文件

| 文件／目录 | 用途 |
|---|---|
| `start_train.sh` | 顶部调整超参数；自动TensorBoard与后台训练 |
| `train.py` | 训练入口，支持数据检查、实时可视化和续训 |
| `live_visualization.py` | 每轮标量与固定样本分组预览，独立载入当前轮权重 |
| `autodl_constraints.txt` | 已有CUDA PyTorch环境的NumPy/OpenCV兼容约束 |
| `predict.py` | 预测叠加图与实例结果导出 |
| `requirements.txt` | 固定训练与数据转换依赖 |
| `dataset/images/train`、`dataset/images/val` | 实际去噪图像，不依赖原项目路径 |
| `dataset/labels/train`、`dataset/labels/val` | YOLO 分割标签，每行一个原始标注实例 |
| `dataset/data.yaml`、`dataset/classes.json` | 70 类索引、英文训练名和中文显示名 |
| `dataset/manifest.json`、`dataset/summary.json` | 来源 annotation、去噪源图、原图身份、划分和类别数量 |
| `previews/` | 数据标签叠加示例，**不是模型预测效果** |
| `runs/<运行名>/previews/` | 训练中固定验证字的实际预测预览 |
| `build_dataset.py`、`tests/` | 原标注转换器与必要检查；已有数据训练不需重建 |

## 已生成的数据

- 训练：480张，12,808个实例（11,720阶段＋1,088结体）。
- 验证：171张，4,401个实例（4,012阶段＋389结体）。
- 合计：651张，17,209个原始标注实例。
- 70个类别索引中，训练集实际覆盖67类；结体“下中、中左、中右”没有训练实例，不能声称模型已学会这些类。
- 共有676个双标原图身份；首次导出排除25个（3个版本冲突、1个无text、1个无JT实例、20个缩放后轮廓退化），当前图像更新保持这651个样本。

## 标签、图像与划分

- 笔画阶段来自17种笔画的起／中／收组合，点只有起／收，共50类。
- 结体来自20个空间角色类别，例如左、右、上、下、内、外；它们不是“木、氵”等偏旁字典名称。
- 原 annotation 的身份和所属阶段保留。不要求一个字具有完整的起／中／收组合，不把同名阶段合成一笔。
- 阶段与结体区域可以重叠，因此训练设置 `overlap_mask=False`。
- 训练／验证图直接复制 Fontify 已有 `images_text_denoised` 去噪PNG：648张来自BF，3张BF去噪缺失时使用同源JT（欧阳询“洪”、苏轼“黄”、赵孟頫“歸”）。全部输出与所选源PNG字节一致，没有再次调亮度、反相或阈值化。
- 去噪图采用居中补方到448的几何，与BF/JT原始标注的pad变换一致；训练时再缩放到640。图像替换保持原651个样本ID、train/val划分及标签文件。
- 分割TXT中的多轮廓使用往返细桥连接；精确实例RLE保存在manifest。TXT采用像素中心坐标，17,209个导出实例回读重绘均与448栅格标注一致。

同一模型需要同时学习两种标注，因此使用BF/JT同源双标字形，按书家＋原图SHA256配对，不按文件名猜测。BF-only／JT-only图不直接加入普通完整监督，避免未提供的另一套标签被当成背景。冲突的重复标注版本单独排除，原始数据不修改。

原ep66验证记录涉及的原图始终留在val；其余接受的双标原图进train，同源副本不跨split。没有独立test集，val用于开发和选模。具体数量与类别频次见 `dataset/summary.json`。

## 可选：重新转换数据

整个 `yolo/` 目录可以独立搬到训练机器，训练和预测不需要 Fontify 代码。只有重新生成数据才需要原始 Calli `new/` 目录、已有去噪图和验证记录：

```bash
python build_dataset.py --help
```

按帮助指定源路径和新的输出目录。若要保持当前样本，使用已有 `dataset/manifest.json` 作为 `--selection-manifest`，并启用 `--allow-jt-fallback`。已经生成的数据可直接训练，无需重复转换。

官方参考：[分割任务](https://docs.ultralytics.com/tasks/segment/)、[训练配置](https://docs.ultralytics.com/modes/train/)、[标签格式](https://docs.ultralytics.com/datasets/segment/)。
