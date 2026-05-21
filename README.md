# Fontify [![版本徽章](https://img.shields.io/badge/version-1.0.0-blue)]((https://shields.io/))

> Fontify: One-shot Font Generation via In-context Learning

##  Preparing Environments
### Requirements
Our code is tested on Python = 3.9, PyTorch = 2.3.0, cuda = 12.1, gcc = 9.1.0
```bash
pip install -r requirements.txt
```
Install [detectron2](https://github.com/facebookresearch/detectron2), following the instructions in [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html). 
Or simply use the following command.
```bash
git clone https://github.com/facebookresearch/detectron2
python -m pip install -e detectron2
```
Please manually modify the following file in your Conda environment:
/path/to/your/env/lib/python3.9/site-packages/timm/models/layers/helpers.py

Locate this line:
```bash
from torch._six import container_abcs
```
Replace it with:
```bash
import collections.abc as container_abcs
```

### Datasets
We provide an example of the dataset in fontdata_example.  
You can also create the datasets in the following way:
- Download the PNG files from [here](https://github.com/ligoudaner377/font_translator_gan#how-to-use)
- Download the TTF files to `fontdata_example/ttf` and convert them to PNG files in the following way
- ```bash
  python fontdata_example/font_to_png.py
  ```


## Training
Download pre-trained MAE ViT-Base model from [here](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_base.pth) and update `path/to/mae_pretrain_vit_base.pth` in `$Fontify_ROOT/train_vit_base_font.sh`. 

Download the VGG weights from [here](https://download.pytorch.org/models/vgg19-dcbb9e9d.pth) and place it in `$Fontify_ROOT/vgg19 directory`.

Run training script
```bash
bash train_vit_base_font.sh
```
## Inference
### If you want to generate font in batches
The configuration file: infer_font.sh

Please read and modify the configuration file:
```
ckpt_path: the path of the saved model
ref_dir: the path of reference characters, each font should have its own folder
source_dir: the path of source font
gen_dir:  the path of GT font, each font should have its own folder
```
Run training script
```bash
bash infer_font.sh
```
### If you only have a single or a few reference characters
Place the reference PNG characters in `font_datasets/reference`, run the following command, and enter the Chinese characters to be generated:
```bash
python eval/infer_few_font.py --ckpt_path path/to/ckpt.pth
```
The generated characters will be in `font_datasets/generated_fonts`.