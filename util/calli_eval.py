"""探针与生成对照共用的推理式输入构造和模型加载。

输入与推理一致：条件流 = [参考字 source; query source]，目标流 = [参考字书法; 白]，
下半 query 全遮盖。参考字固定为同 type、不同字符，按种子确定性抽取。
"""

import json
import random
from pathlib import Path

import torch

import models_train
from util.calli_labels import LabelRenderer, label_kind
from util.stage3_data import fixed_sample

ARCH = "vit_base_patch16_input896x448_win_dec64_8glb_sl1"


def read_records(json_paths):
    records = []
    for path in sorted(json_paths):
        with open(path, "r", encoding="utf-8") as f:
            records.extend(json.load(f))
    return records


def char_of(record):
    return Path(record["target_path"]).stem


def fixed_references(records, seed=0):
    """每条记录配一个同 type、不同字符的参考字，结果只由 seed 决定。"""
    rng = random.Random(seed)
    by_type = {}
    for i, record in enumerate(records):
        by_type.setdefault(record["type"], []).append(i)
    refs = []
    for i, record in enumerate(records):
        pool = [j for j in by_type[record["type"]] if char_of(records[j]) != char_of(record)]
        refs.append(rng.choice(pool))
    return refs


class QueryDataset(torch.utils.data.Dataset):
    """返回推理式输入、下半 query GT 及其 (70,28,28) 标签。"""

    def __init__(self, root, records, refs, with_labels=True):
        self.root = root
        self.records = records
        self.refs = refs
        self.renderer = LabelRenderer(root) if with_labels else None

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        query = self.records[index]
        ref = self.records[self.refs[index]]
        source, context, supervision, mask = fixed_sample(self.root, query, ref)
        item = {
            "index": index,
            "source": source,
            "context": context,
            "supervision": supervision,
            "mask": mask,
            "kind": label_kind(query["type"]),
        }
        if self.renderer is not None:
            label, _ = self.renderer.render(query)
            item["label"] = torch.from_numpy(label)
        return item


def build_model(checkpoint=None, device="cuda"):
    """checkpoint=None 为随机初始化；含 label_head 的 checkpoint 会自动建头。"""
    model = models_train.__dict__[ARCH]()
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")["model"]
        if "label_head.weight" in state:
            model.enable_label_head(0.0)
        missing, unexpected = model.load_state_dict(state, strict=False)
        critical = [
            k for k in missing
            if not k.startswith(("vgg_loss", "discriminator"))
        ]
        if critical:
            raise RuntimeError(f"{checkpoint} missing weights: {critical[:5]}")
        print(f"loaded {checkpoint} (unexpected={len(unexpected)})")
    return model.to(device).eval()


@torch.no_grad()
def encode(model, batch, device, visible=False):
    """返回 4 个抽头 (B,56,28,E) 和预测图 (B,3,896,448)。

    visible=True 时不遮盖 query 目标，只用作"可见墨迹"上界对照。
    """
    source = batch["source"].to(device)
    context = (batch["supervision"] if visible else batch["context"]).to(device)
    mask = batch["mask"].to(device).flatten(1)
    if visible:
        mask = torch.zeros_like(mask)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        latent = model.forward_encoder(source, context, mask)
        pred = model.forward_decoder(latent)
    return latent, pred
