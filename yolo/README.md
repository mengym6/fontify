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

## 训练与实时查看

默认输入大小为640、batch为8、最多100个epoch、早停耐心值为25。可以直接执行：

```bash
python train.py --epochs 100 --batch 8 --imgsz 640 --device 0
```

显存不足时先将 `--batch` 改为4。第一次正式训练会把 COCO 初始化权重下载到 `models/`；这份初始化本身不能识别本项目的阶段和结体类别。

默认运行输出位于 `runs/calli_phase_jieti_yolo26s/`；同名目录已有运行时会顺延编号，**以程序打印的实际目录为准**。若希望终端断开后继续运行，可以使用：

```bash
nohup python train.py --epochs 100 --batch 8 --imgsz 640 --device 0 > train.log 2>&1 &
```

TensorBoard 默认启用：标量每个epoch更新；固定4个验证字的预测预览每5个epoch更新，首个和最后一个epoch也保存。预测图片同时写入该运行的 `previews/` 和 TensorBoard 的 **Images** 页面。这些图使用训练中的模型输出，和本目录 `previews/` 中的标注示例不同。

另开一个服务器终端启动 TensorBoard：

```bash
source /root/autodl-tmp/fontify_yolo_env/bin/activate
cd /root/autodl-tmp/fontify_yolo/yolo
tensorboard --logdir runs --host 127.0.0.1 --port 6006
```

在本地 VS Code 的 **Ports（端口）** 面板选择 **Forward a Port（转发端口）**，输入 `6006`，点击转发地址，即可在浏览器查看训练曲线和预测图。如果本地6006已占用，使用 VS Code 显示的实际转发地址。

更新代码时通过 Git 拉取；现有 `dataset/`、`runs/`、`models/` 不由 Git 覆盖。训练过程中的最新权重、标量和预览都保存在服务器运行目录中。

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
