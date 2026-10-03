import os.path
import json
from typing import Any, Callable, List, Optional, Tuple
import random

from PIL import Image, ImageDraw
import numpy as np

import torch
from torchvision.datasets.vision import VisionDataset, StandardTransform
import torch.nn.functional as F

from util.jieti_partition import build_partition, ink_label_map


class PairDataset(VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        root: str,
        json_path_list: list,
        transform: Optional[Callable] = None,
        transform2: Optional[Callable] = None,
        transform3: Optional[Callable] = None,
        transform_seccrop: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
        masked_position_generator: Optional[Callable] = None,
        use_two_pairs: bool = True,
        half_mask_ratio:float = 0.,
        semantic_mask_dir: Optional[str] = None,
        num_mask_annotations_bf: int = 3,
        num_mask_annotations_jt: int = 1,
        mask_coverage_threshold: float = 0.5,
        mask_mix_probs: Optional[List[float]] = None,
        annotation_subdir: str = "annotations",
        annotation_filename: str = "instances_default.json",
        use_annotation_masks: Optional[bool] = None,
        annotation_mask_size: int = 448,
        strict_style_pairing: bool = False,
        no_jt: bool = False,
        return_jieti: bool = False,
        jieti_k_max: int = 4,
        jieti_pool: int = 224,
        jieti_valid_ink: int = 200,
        jieti_struct_pair_prob: float = 1.0,
        fixed_pair_path: Optional[str] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self.pairs = []
        self.weights = []
        type_weight_list = [1.0] * len(json_path_list)
        #type_weight_list= [0.1, 0.2, 0.15, 0.25, 0.2, 0.15, 0.05, 0.05]
        for idx, json_path in enumerate(json_path_list):
            cur_pairs = json.load(open(json_path))
            self.pairs.extend(cur_pairs)
            cur_num = len(cur_pairs)
            self.weights.extend([type_weight_list[idx] * 1./cur_num]*cur_num)
            #print(json_path, type_weight_list[idx])
        # 固定 val：物理过滤到 fixed 列出的样本（已跳过无候选 JT），这样 val 的
        # DistributedSampler 也只遍历这批，与所有组、baseline 用同一份 val。
        if fixed_pair_path is not None and os.path.exists(fixed_pair_path):
            with open(fixed_pair_path, "r", encoding="utf-8") as _f:
                self._fixed_pairs = json.load(_f)  # target_path → pair2 完整记录
            # 固定 val 只遍历 fixed 列出的 query（已跳过无候选 JT）。pair2 参考字直接
            # 从记录里读，不需要留在 self.pairs 里，所以 val 的 DistributedSampler 也
            # 只遍历这批 query。
            keep = [i for i, p in enumerate(self.pairs)
                    if p["target_path"] in self._fixed_pairs]
            self.pairs = [self.pairs[i] for i in keep]
            self.weights = [self.weights[i] for i in keep]
        else:
            self._fixed_pairs = None
        self.use_two_pairs = use_two_pairs
        self.strict_style_pairing = strict_style_pairing
        if self.use_two_pairs:
            self.pair_type_dict = {}
            for idx, pair in enumerate(self.pairs):
                if "type" in pair:
                    if pair["type"] not in self.pair_type_dict:
                        self.pair_type_dict[pair["type"]] = [idx]
                    else:
                        self.pair_type_dict[pair["type"]].append(idx)
            for t in self.pair_type_dict:
                print(t, len(self.pair_type_dict[t]))
        self.transforms = PairStandardTransform(transform, target_transform) if transform is not None else None
        self.transforms2 = PairStandardTransform(transform2, target_transform) if transform2 is not None else None
        self.transforms3 = PairStandardTransform(transform3, target_transform) if transform3 is not None else None
        self.transforms_seccrop = PairStandardTransform(transform_seccrop, target_transform) if transform_seccrop is not None else None
        self.masked_position_generator = masked_position_generator
        self.half_mask_ratio = half_mask_ratio
        self.semantic_mask_dir = semantic_mask_dir
        self.num_mask_annotations_bf = num_mask_annotations_bf
        self.num_mask_annotations_jt = num_mask_annotations_jt
        self.mask_coverage_threshold = mask_coverage_threshold
        self.annotation_subdir = annotation_subdir
        self.annotation_filename = annotation_filename
        self.use_annotation_masks = semantic_mask_dir is not None if use_annotation_masks is None else use_annotation_masks
        self.annotation_mask_size = annotation_mask_size
        self._annotation_cache = {}
        self.mask_mix_probs = None
        self.no_jt = no_jt
        # 结体（Jieti）相关：返回部件划分、有效部件掩码、is_jt，并实现同结构异字配对。
        self.return_jieti = return_jieti
        self.jieti_k_max = jieti_k_max
        self.jieti_pool = jieti_pool
        self.jieti_valid_ink = jieti_valid_ink
        self.jieti_struct_pair_prob = jieti_struct_pair_prob
        self.fixed_pair_path = fixed_pair_path
        if mask_mix_probs is not None:
            if len(mask_mix_probs) != 3:
                raise ValueError("mask_mix_probs must contain 3 values: random, JT semantic, BF semantic")
            mask_mix_probs = [float(p) for p in mask_mix_probs]
            if any(p < 0 for p in mask_mix_probs):
                raise ValueError("mask_mix_probs values must be non-negative")
            prob_sum = sum(mask_mix_probs)
            if prob_sum <= 0:
                raise ValueError("mask_mix_probs sum must be positive")
            self.mask_mix_probs = [p / prob_sum for p in mask_mix_probs]
        self._semantic_indices_by_type = {}
        self._jt_semantic_indices = []
        self._bf_semantic_indices = []
        for i, pair in enumerate(self.pairs):
            if not self._has_semantic_source(pair) and not (
                self.no_jt and 'JT' in pair.get('type', '')
            ):
                continue
            pair_type = pair.get('type', '')
            self._semantic_indices_by_type.setdefault(pair_type, []).append(i)
            if 'JT' in pair_type:
                self._jt_semantic_indices.append(i)
            elif 'BF' in pair_type:
                self._bf_semantic_indices.append(i)
        self._jt_semantic_weights = [self.weights[i] for i in self._jt_semantic_indices]
        self._bf_semantic_weights = [self.weights[i] for i in self._bf_semantic_indices]
        if self.mask_mix_probs is not None:
            if self.mask_mix_probs[1] > 0 and not self._jt_semantic_indices:
                raise ValueError(
                    "mask_mix_probs requests JT samples, but none were found"
                )
            if self.mask_mix_probs[2] > 0 and not self._bf_semantic_indices:
                raise ValueError("mask_mix_probs requests BF semantic masks, but no BF semantic mask files were found")

        if self.return_jieti:
            self._build_jieti_pairing()

    def _jt_structure_signature(self, pair: dict) -> Optional[tuple]:
        """目标字的 JT 类别名集合（去 text、去笔画前缀数字），作为"同结构组合"键。

        无标注或无 JT 实例时返回 None。
        """
        if 'JT' not in pair.get('type', ''):
            return None
        ann_path = self._annotation_path(pair)
        if ann_path is None:
            return None
        index = self._load_annotation_index(ann_path)
        if index is None:
            return None
        target_name = os.path.basename(pair["target_path"])
        char_name = os.path.splitext(target_name)[0]
        image_info = index["images_by_name"].get(target_name) or index["images_by_stem"].get(char_name)
        if image_info is None:
            return None
        anns = index["anns_by_image_id"].get(image_info["id"], [])
        names = set()
        for ann in anns:
            cat_id = ann.get("category_id")
            if cat_id in index["text_ids"]:
                continue
            name = index["cat_id_to_stroke"].get(cat_id)
            if name:
                names.add(name)
        return tuple(sorted(names)) if names else None

    def _build_jieti_pairing(self):
        """为 JT 样本预建"同书家(type)、同结构组合、异字"候选池；无候选者剔除。

        character 用 target 文件名的字（stem）；"同书家"用 type 承载（type 含书家）。
        """
        self._jt_struct_sig = {}
        self._jt_char = {}
        for i, pair in enumerate(self.pairs):
            if 'JT' not in pair.get('type', ''):
                continue
            self._jt_struct_sig[i] = self._jt_structure_signature(pair)
            self._jt_char[i] = os.path.splitext(os.path.basename(pair['target_path']))[0]
        # (type, 结构签名) → index 列表
        bucket = {}
        for i, sig in self._jt_struct_sig.items():
            if sig is None:
                continue
            key = (self.pairs[i]['type'], sig)
            bucket.setdefault(key, []).append(i)
        self._jt_struct_pool = {}  # index → 同结构异字候选 index 列表
        self._jt_dropped = []
        for i, sig in self._jt_struct_sig.items():
            if sig is None:
                self._jt_dropped.append(i)
                continue
            key = (self.pairs[i]['type'], sig)
            cands = [j for j in bucket[key] if self._jt_char[j] != self._jt_char[i]]
            if cands:
                self._jt_struct_pool[i] = cands
            else:
                self._jt_dropped.append(i)
        # p=0 时不走同结构配对，训练集保持与 baseline 一致，不剔除（T1-C）。
        if self.jieti_struct_pair_prob <= 0:
            print(f"[jieti] struct_pair_prob={self.jieti_struct_pair_prob}: dropped 0 JT samples "
                  f"({len(self._jt_dropped)} without candidate kept)", flush=True)
            return
        # 从训练/评测中剔除无候选 JT 样本：置零采样权重并记录。
        for i in self._jt_dropped:
            self.weights[i] = 0.0
        if self._jt_dropped:
            print(f"[jieti] dropped {len(self._jt_dropped)} JT samples without "
                  f"same-structure different-character candidate", flush=True)

    def _load_image(self, path: str) -> Image.Image:
        while True:
            try:
                img = Image.open(os.path.join(self.root, path))
            except OSError as e:
                print(f"Catched exception: {str(e)}. Re-trying...")
                import time
                time.sleep(1)
            else:
                break

        img = img.convert("RGB")
        return img

    def _combine_images(self, image, image2, interpolation='bicubic'):
        # image under image2
        h, w = image.shape[1], image.shape[2]
        dst = torch.cat([image, image2], dim=1)
        return dst

    def _abs_data_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        return os.path.join(self.root, path)

    def _font_dir_from_target(self, target_path: str) -> Optional[str]:
        parts = target_path.split('/')
        for i, p in enumerate(parts):
            if 'images' in p:
                return '/'.join(parts[:i])
        return None

    def _semantic_mask_path(self, pair: dict) -> Optional[str]:
        explicit = pair.get("semantic_mask_path")
        if explicit:
            explicit_path = self._abs_data_path(explicit)
            if os.path.exists(explicit_path):
                return explicit_path

        target_path = pair['target_path']
        char_name = os.path.splitext(os.path.basename(target_path))[0]
        font_dir = self._font_dir_from_target(target_path)
        if font_dir is None:
            return None
        npy_path = os.path.join(self.root, font_dir, 'semantic_masks', f'{char_name}.npy')
        if not os.path.exists(npy_path):
            return None
        return npy_path

    def _annotation_path(self, pair: dict) -> Optional[str]:
        explicit = pair.get("annotation_path")
        if explicit:
            explicit_path = self._abs_data_path(explicit)
            if os.path.exists(explicit_path):
                return explicit_path

        font_dir = self._font_dir_from_target(pair['target_path'])
        if font_dir is None:
            return None
        ann_dir = os.path.join(self.root, font_dir, self.annotation_subdir)
        preferred = os.path.join(ann_dir, self.annotation_filename)
        if os.path.exists(preferred):
            return preferred
        if os.path.isdir(ann_dir):
            candidates = sorted(
                os.path.join(ann_dir, name)
                for name in os.listdir(ann_dir)
                if name.lower().endswith(".json")
            )
            if candidates:
                return candidates[0]
        return None

    def _extract_stroke_name(self, category_name: str) -> Optional[str]:
        if category_name == "text":
            return None
        base = category_name.rsplit("-", 1)[0]
        base = base.lstrip("0123456789-")
        return base if base else None

    def _load_annotation_index(self, ann_path: str) -> Optional[dict]:
        if ann_path in self._annotation_cache:
            return self._annotation_cache[ann_path]

        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            self._annotation_cache[ann_path] = None
            return None

        categories = data.get("categories", [])
        text_ids = {c["id"] for c in categories if c.get("name") == "text"}
        cat_id_to_stroke = {}
        for cat in categories:
            stroke = self._extract_stroke_name(cat.get("name", ""))
            if stroke:
                cat_id_to_stroke[cat["id"]] = stroke

        images = data.get("images", [])
        images_by_name = {img.get("file_name"): img for img in images}
        images_by_stem = {
            os.path.splitext(os.path.basename(img.get("file_name", "")))[0]: img
            for img in images
        }
        anns_by_image_id = {}
        for ann in data.get("annotations", []):
            anns_by_image_id.setdefault(ann.get("image_id"), []).append(ann)

        index = {
            "text_ids": text_ids,
            "cat_id_to_stroke": cat_id_to_stroke,
            "images_by_name": images_by_name,
            "images_by_stem": images_by_stem,
            "anns_by_image_id": anns_by_image_id,
        }
        self._annotation_cache[ann_path] = index
        return index

    def _annotation_has_target(self, pair: dict) -> bool:
        if not self.use_annotation_masks:
            return False
        ann_path = self._annotation_path(pair)
        if ann_path is None:
            return False
        index = self._load_annotation_index(ann_path)
        if index is None:
            return False
        char_name = os.path.splitext(os.path.basename(pair["target_path"]))[0]
        file_name = os.path.basename(pair["target_path"])
        return file_name in index["images_by_name"] or char_name in index["images_by_stem"]

    def _has_semantic_source(self, pair: dict) -> bool:
        return self._semantic_mask_path(pair) is not None or self._annotation_has_target(pair)

    def _decode_rle(self, segmentation, h: int, w: int) -> Optional[np.ndarray]:
        try:
            from pycocotools import mask as mask_util
        except ImportError:
            return None
        rle = segmentation
        if isinstance(rle.get("counts"), list):
            rle = mask_util.frPyObjects(rle, h, w)
        return mask_util.decode(rle).astype(np.uint8)

    def _render_polygon(self, segmentation, h: int, w: int) -> np.ndarray:
        mask_img = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask_img)
        if not segmentation:
            return np.zeros((h, w), dtype=np.uint8)
        polygons = segmentation
        if isinstance(segmentation[0], (int, float)):
            polygons = [segmentation]
        for poly in polygons:
            if len(poly) < 6:
                continue
            pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
            draw.polygon(pts, fill=1)
        return np.array(mask_img, dtype=np.uint8)

    def _render_segmentation(self, segmentation, h: int, w: int) -> Optional[np.ndarray]:
        if segmentation is None:
            return None
        if isinstance(segmentation, dict):
            return self._decode_rle(segmentation, h, w)
        return self._render_polygon(segmentation, h, w)

    def _pad_and_resize_annotation_layer(self, layer: np.ndarray) -> np.ndarray:
        h, w = layer.shape[:2]
        max_side = max(h, w)
        square = np.zeros((max_side, max_side), dtype=np.uint8)
        offset_x = (max_side - w) // 2
        offset_y = (max_side - h) // 2
        square[offset_y:offset_y + h, offset_x:offset_x + w] = layer
        mask_img = Image.fromarray((square > 0).astype(np.uint8) * 255, mode="L")
        mask_img = mask_img.resize((self.annotation_mask_size, self.annotation_mask_size), Image.NEAREST)
        return (np.array(mask_img) > 0).astype(np.uint8)

    def _render_annotation_layers(self, pair: dict) -> Optional[np.ndarray]:
        if not self.use_annotation_masks:
            return None
        ann_path = self._annotation_path(pair)
        if ann_path is None:
            return None
        index = self._load_annotation_index(ann_path)
        if index is None:
            return None

        target_name = os.path.basename(pair["target_path"])
        char_name = os.path.splitext(target_name)[0]
        image_info = index["images_by_name"].get(target_name) or index["images_by_stem"].get(char_name)
        if image_info is None:
            return None

        h = int(image_info["height"])
        w = int(image_info["width"])
        anns = index["anns_by_image_id"].get(image_info["id"], [])
        stroke_groups = {}
        for ann in anns:
            cat_id = ann.get("category_id")
            if cat_id in index["text_ids"]:
                continue
            stroke = index["cat_id_to_stroke"].get(cat_id)
            if not stroke:
                continue
            stroke_groups.setdefault(stroke, []).append(ann)

        layers = []
        for anns in stroke_groups.values():
            combined = np.zeros((h, w), dtype=np.uint8)
            for ann in anns:
                layer = self._render_segmentation(ann.get("segmentation"), h, w)
                if layer is not None:
                    combined |= layer.astype(np.uint8)
            if combined.any():
                layers.append(self._pad_and_resize_annotation_layer(combined))

        if not layers:
            return None
        return np.stack(layers, axis=0)

    def _sample_mask_mode(self) -> str:
        if self.mask_mix_probs is None:
            return "auto"
        modes = ["random", "jt_semantic", "bf_semantic"]
        return random.choices(modes, weights=self.mask_mix_probs, k=1)[0]

    def _sample_semantic_index(self, mask_mode: str) -> int:
        if mask_mode == "jt_semantic":
            return random.choices(self._jt_semantic_indices, weights=self._jt_semantic_weights, k=1)[0]
        if mask_mode == "bf_semantic":
            return random.choices(self._bf_semantic_indices, weights=self._bf_semantic_weights, k=1)[0]
        raise ValueError(f"Unsupported semantic mask mode: {mask_mode}")

    def _load_semantic_layers(self, pair: dict) -> Optional[np.ndarray]:
        npy_path = self._semantic_mask_path(pair)
        if npy_path is not None:
            return np.load(npy_path)  # (N, 448, 448)
        return self._render_annotation_layers(pair)

    def _sample_semantic_block_mask(self, layers: np.ndarray, num_blocks: int) -> Image.Image:
        """Randomly select labeled semantic mask layers generated from annotations.

        For BF, each layer is one individual start/middle/end annotation. For JT,
        each layer is one grouped spatial label.
        """
        n = layers.shape[0]
        k = min(max(0, int(num_blocks)), n)
        if k == 0:
            combined = np.zeros(layers.shape[1:], dtype=np.uint8)
        else:
            indices = random.sample(range(n), k)
            combined = np.any(layers[indices], axis=0).astype(np.uint8) * 255
        return Image.fromarray(combined, mode='L')

    def _load_semantic_mask(self, pair: dict, pair_type: str) -> Optional[Image.Image]:
        """加载 .npy 并生成 JT/BF 对应的语义遮盖。"""
        layers = self._load_semantic_layers(pair)
        if layers is None:
            return None
        N = layers.shape[0]
        if N <= 0:
            return None
        if 'JT' in pair_type:
            return self._sample_semantic_block_mask(layers, self.num_mask_annotations_jt)

        return self._sample_semantic_block_mask(layers, self.num_mask_annotations_bf)

    def _pixel_mask_to_patch_mask(self, sem_mask: torch.Tensor) -> np.ndarray:
        """将像素级 mask (1, H, W) 转为 patch 网格级 mask (Hp, Wp)"""
        patch_size = 16
        h, w = sem_mask.shape[1], sem_mask.shape[2]
        Hp, Wp = h // patch_size, w // patch_size
        mask_2d = sem_mask[0]  # (H, W)
        patches = mask_2d.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
        coverage = patches.mean(dim=(-1, -2))
        patch_mask = (coverage > self.mask_coverage_threshold).numpy().astype(np.int32)
        return patch_mask

    def _load_jieti_label_map(self, pair: dict) -> Optional[Image.Image]:
        """JT 部件墨迹层 → 单张整数标签图 PIL(L)，0=背景，1..k_max=部件。"""
        if 'JT' not in pair.get('type', ''):
            return None
        layers = self._load_semantic_layers(pair)
        if layers is None:
            return None
        label_map = ink_label_map(layers, self.jieti_k_max)
        return Image.fromarray(label_map, mode='L')

    def _build_jieti_tensors(self, lm_top, lm_bot, is_jt):
        """由上/下半增强后的部件标签图构造 (2,R,R) 划分与 (2,k_max) 有效掩码。

        lm_top/lm_bot: ToTensor 后的 (1,448,448) int64，或 None。
        """
        k = self.jieti_k_max
        R = self.jieti_pool
        voro = torch.zeros(2, R, R, dtype=torch.long)
        valid_parts = torch.zeros(2, k, dtype=torch.bool)
        if is_jt:
            for h, lm in enumerate((lm_top, lm_bot)):
                if lm is None:
                    continue
                arr = lm[0].numpy().astype(np.int64)  # (448,448)，增强后
                valid_k = np.array(
                    [int((arr == (c + 1)).sum()) >= self.jieti_valid_ink
                     for c in range(k)]
                )
                part = build_partition(arr, valid_k, k, R)
                voro[h] = torch.from_numpy(part)
                valid_parts[h] = torch.from_numpy(valid_k)
        return voro, valid_parts

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        mask_mode = self._sample_mask_mode()
        # 阶段 2 混合数据：原电脑字体使用随机 mask，CalliPhase 优先使用样本自带的
        # semantic_masks/*.npy。source_dataset 由混合 JSON 写入；旧 JSON 保持原行为。
        source_dataset = self.pairs[index].get('source_dataset', '')
        if source_dataset == 'calliphase':
            pair_type_hint = self.pairs[index].get('type', '')
            if self._has_semantic_source(self.pairs[index]):
                mask_mode = 'jt_semantic' if 'JT' in pair_type_hint else 'bf_semantic'
            else:
                mask_mode = 'random'
        if mask_mode in ("jt_semantic", "bf_semantic") and source_dataset != 'calliphase':
            index = self._sample_semantic_index(mask_mode)
        pair = self.pairs[index]
        if self.no_jt and 'JT' in pair.get('type', ''):
            mask_mode = 'random'
        image = self._load_image(pair['image_path'])
        target = self._load_image(pair['target_path'])

        # decide mode for interpolation
        pair_type = pair['type']
        if "font" in pair_type:
            interpolation1 = 'bicubic'
            interpolation2 = 'nearest'
        else:
            interpolation1 = 'bicubic'
            interpolation2 = 'bicubic'

        if mask_mode == "random":
            sem_mask = None
        else:
            sem_mask = self._load_semantic_mask(pair, pair_type)

        is_jt = self.return_jieti and 'JT' in pair_type
        jieti_lm = self._load_jieti_label_map(pair) if is_jt else None

        # no aug for instance segmentation
        if "font" in pair['type'] and self.transforms3 is not None:
            cur_transforms = self.transforms3
        else:
            cur_transforms = self.transforms

        image, target, sem_mask, jieti_lm = cur_transforms(
            image, target, interpolation1, interpolation2, mask=sem_mask, mask2=jieti_lm
        )

        if self.use_two_pairs:
            pair_type = pair['type']
            # sample the second pair belonging to the same type
            pair2_index = random.choice(self.pair_type_dict[pair_type])
            if source_dataset == 'calliphase':
                calli_pool = [i for i in self.pair_type_dict[pair_type]
                              if self.pairs[i].get('source_dataset') == 'calliphase']
                if calli_pool:
                    pair2_index = random.choice(calli_pool)
            if mask_mode in ("jt_semantic", "bf_semantic"):
                pair2_pool = self._semantic_indices_by_type.get(pair_type, [])
                if pair2_pool:
                    pair2_index = random.choice(pair2_pool)
            fixed_pair2 = None
            if self._fixed_pairs is not None and pair['target_path'] in self._fixed_pairs:
                # val 固定配对：直接用预存的 pair2 完整记录（见 U4）。
                fixed_pair2 = self._fixed_pairs[pair['target_path']]
            elif is_jt and index in getattr(self, '_jt_struct_pool', {}):
                # Q3：JT 以概率 p 走"同书家、同结构组合、异字"配对，否则随机同 type。
                if random.random() < self.jieti_struct_pair_prob:
                    pair2_index = random.choice(self._jt_struct_pool[index])
            if self.strict_style_pairing:
                pool = self.pair_type_dict[pair_type]
                if mask_mode in ("jt_semantic", "bf_semantic"):
                    pool = self._semantic_indices_by_type.get(pair_type, [])
                pool = [
                    i for i in pool
                    if self.pairs[i]["style_id"] == pair["style_id"]
                    and self.pairs[i]["character"] != pair["character"]
                ]
                if not pool:
                    raise ValueError(
                        "No distinct-character reference for style/type: "
                        f"{pair['style_id']}/{pair_type}"
                    )
                pair2_index = random.choice(pool)
            pair2 = fixed_pair2 if fixed_pair2 is not None else self.pairs[pair2_index]
            image2 = self._load_image(pair2['image_path'])
            target2 = self._load_image(pair2['target_path'])
            if mask_mode == "random":
                sem_mask2 = None
            else:
                sem_mask2 = self._load_semantic_mask(pair2, pair_type)
            jieti_lm2 = self._load_jieti_label_map(pair2) if is_jt else None
            assert pair2['type'] == pair_type
            image2, target2, sem_mask2, jieti_lm2 = cur_transforms(
                image2, target2, interpolation1, interpolation2, mask=sem_mask2, mask2=jieti_lm2
            )

            image = self._combine_images(image, image2, interpolation1)
            target = self._combine_images(target, target2, interpolation2)
            # 两个 target 都必须被遮盖（source 是参考字，始终完整可见）
            if sem_mask is not None and sem_mask2 is not None:
                sem_mask = torch.cat([sem_mask, sem_mask2], dim=1)
            else:
                sem_mask = None
        else:
            jieti_lm2 = None

        if self.mask_mix_probs is not None:
            # mask_mix_probs 控制 random/JT semantic/BF semantic 主比例；
            # half_mask_ratio 只在 random 类内部生效，避免破坏三类主比例。
            use_half_mask = mask_mode == "random" and torch.rand(1)[0] < self.half_mask_ratio
        elif self.half_mask_ratio >= 1.0:
            # val 验证集：强制完全遮盖整个 target
            use_half_mask = True
        else:
            # train：按 half_mask_ratio 概率全程走 half mask，JT/BF 都生效。
            use_half_mask = torch.rand(1)[0] < self.half_mask_ratio
        if self.no_jt and 'JT' in pair_type:
            use_half_mask = False
        if (self.transforms_seccrop is None) or use_half_mask:
            pass
        else:
            # seccrop 只在 pretrain 增强策略下非 None；结体用 finetune 策略，不走这里。
            image, target, sem_mask, _ = self.transforms_seccrop(image, target, interpolation1, interpolation2, mask=sem_mask)

        valid = torch.ones_like(target)

        if use_half_mask:
            num_patches = self.masked_position_generator.num_patches
            mask = np.zeros(self.masked_position_generator.get_shape(), dtype=np.int32)
            mask[mask.shape[0]//2:, :] = 1
        elif sem_mask is not None:
            mask = self._pixel_mask_to_patch_mask(sem_mask)
        elif mask_mode in ("jt_semantic", "bf_semantic"):
            raise RuntimeError(f"{mask_mode} was selected, but semantic mask loading failed")
        else:
            mask = self.masked_position_generator()

        if self.return_jieti:
            # 上半=pair（风格参考，始终可见），下半=pair2（query 目标）。
            voro, valid_parts = self._build_jieti_tensors(jieti_lm, jieti_lm2, is_jt)
            return image, target, mask, valid, voro, valid_parts, torch.tensor(bool(is_jt))
        return image, target, mask, valid

    def __len__(self) -> int:
        return len(self.pairs)


class PairStandardTransform(StandardTransform):
    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        super().__init__(transform=transform, target_transform=target_transform)

    def __call__(self, input: Any, target: Any, interpolation1: Any, interpolation2: Any, mask=None, mask2=None) -> Tuple[Any, Any, Any, Any]:
        if self.transform is not None:
            input, target, mask, mask2 = self.transform(
                input, target, interpolation1, interpolation2, mask=mask, mask2=mask2
            )
        return input, target, mask, mask2
