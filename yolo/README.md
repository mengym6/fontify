# Calli 笔画阶段与结体区域：YOLO26s 实例分割

直接使用 Calli 原始标注训练一个模型：**50 类笔画阶段＋20 类结体区域，共 70 类**。每个非 `text` 的 annotation 是一个实例，不合并阶段、不恢复整笔，不接入 Fontify。

本目录已准备训练数据与代码，由你在 VS Code 连接的 Google Colab 中训练。本轮没有执行模型训练，没有训练后的 `best.pt`。

## 在 VS Code / Colab 中使用

1. 将 `yolo_colab.zip` 上传到 Google Drive 的 `MyDrive/CalliYOLO/`。压缩包内已有顶层 `yolo/`；也可以把整个 `yolo/` 文件夹上传到同一位置。
2. 打开 `train_colab.ipynb`，连接 Colab GPU 内核。VS Code 本地文件不会自动出现在远端内核中，需要先上传。
3. 依次执行准备文件、安装依赖、检查数据、训练、验证和预测。训练默认 `yolo26s-seg.pt`、640输入、batch8、100epoch上限、patience25；显存不足时改batch4。
4. 默认输出保存在 Drive 的 `MyDrive/CalliYOLO/runs/`，断线后用实际运行目录的 `weights/last.pt` 续训。

命令行方式（进入本目录后）：

```bash
python -m pip install -r requirements.txt
python train.py --dry-run
python train.py --epochs 100 --batch 8 --device 0
python predict.py --weights runs/calli_phase_jieti_yolo26s/weights/best.pt --source dataset/images/val --device 0
```

实际输出目录以训练程序打印为准；同名运行会顺延编号。`--dry-run` 只检查数据，不下载模型。真正训练时才把 COCO 初始化下载到 `models/`，初始化本身不能识别本项目阶段与结体类别。

## 文件

| 文件／目录 | 用途 |
|---|---|
| `train_colab.ipynb` | Colab 文件准备、训练、验证、推理及续训 |
| `train.py` | 训练入口；可迁移路径；`--dry-run` 数据检查 |
| `predict.py` | 输出叠加图、每实例类别／框／轮廓及无损二值 mask RLE |
| `requirements.txt` | 固定 `ultralytics==8.4.218` 及数据转换依赖 |
| `dataset/images/train`、`dataset/images/val` | 实际图像文件，不依赖原项目路径 |
| `dataset/labels/train`、`dataset/labels/val` | YOLO 分割标签，每行一个原始标注实例 |
| `dataset/data.yaml`、`dataset/classes.json` | 70 类索引、英文训练名和中文显示名 |
| `dataset/manifest.json`、`dataset/summary.json` | 来源 annotation、原图身份、划分和类别数量 |
| `previews/` | 标签叠加预览，**不是模型预测效果** |
| `build_dataset.py`、`tests/` | 原标注转换器与必要检查；已有数据训练不需重建 |

## 已生成的数据

- 训练：480张，12,808个实例（11,720阶段＋1,088结体）。
- 验证：171张，4,401个实例（4,012阶段＋389结体）。
- 合计：651张，17,209个原始标注实例。
- 70个类别索引中，训练集实际覆盖67类；结体“下中、中左、中右”没有训练实例，不能声称模型已学会这些类。
- 共有676个双标原图身份；排除25个（3个版本冲突、1个无text、1个无JT实例、20个缩放后轮廓退化），保留651个。

## 标签与图像

- 笔画阶段来自 17 种笔画的起／中／收组合，点只有起／收，共50类。
- 结体来自20个空间角色类别，例如左、右、上、下、内、外；它们不是“木、氵”等偏旁字典名称。
- 原 annotation 的身份和所属阶段保留。不要求一个字具有完整的起／中／收组合，不把同名阶段合成一笔。
- 笔法阶段与结体区域可以重叠，因此训练设置 `overlap_mask=False`。
- 输入从同一原图生成，利用原始整字text域统一为黑字白底并保留前景灰度；图片和全部标签一起等比补白缩放至448。训练再缩放到640。
- 分割TXT中的多轮廓使用往返细桥连接，导出时检查重绘IoU；精确原始实例RLE保存在manifest。预测JSON中的 `mask_rle` 也无损保存孔洞及不连通区域。TXT采用像素中心坐标，本次17,209个导出实例回读重绘均与448栅格标注一致。

## 数据划分

同一个模型需要同时学习两种标注，因此使用BF/JT同源双标字形，按书家＋原图SHA256配对，不按文件名猜测。BF-only／JT-only图不直接加入普通完整监督，避免未提供的另一套标签被当成背景。冲突的重复标注版本单独排除，原始数据不修改。

原ep66验证记录涉及的原图始终留在val；其余接受的双标原图进train，同源副本不跨split。没有独立test集，val用于开发和选模。具体数量与类别频次见 `dataset/summary.json`；某些类别很少或未出现，定义了类别索引不等于模型已经学会该类。

## 可选：重新转换

整个目录可独立搬到Colab，训练与预测不需要Fontify代码。只有重建数据才需原始Calli `new/` 目录及验证记录：

```bash
python build_dataset.py --help
```

按帮助指定源路径和新的输出目录。已经生成的图片／标签可直接使用，不必在Colab重复转换。

## 结果

训练后 `best.pt` 是按val表现选出的模型，`last.pt` 用于续训。续训默认在当前权重对应的运行目录保存，也可用 `--resume-output` 指定新目录；已完成且不含优化器状态的权重不能当作中断训练恢复。预测输出包含类别、中文名、置信度、实例框、便捷轮廓、无损mask RLE和叠加图。建议推理输入采用单字黑字白底、周围留白的图像。

当前只完成数据和代码准备，尚无识别精度结论。官方参考：[分割任务](https://docs.ultralytics.com/tasks/segment/)、[训练配置](https://docs.ultralytics.com/modes/train/)、[标签格式](https://docs.ultralytics.com/datasets/segment/)。
