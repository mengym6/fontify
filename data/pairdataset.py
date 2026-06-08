import os.path
import json
from typing import Any, Callable, List, Optional, Tuple
import random

from PIL import Image
import numpy as np

import torch
from torchvision.datasets.vision import VisionDataset, StandardTransform
import torch.nn.functional as F


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
        semantic_only_epochs: int = 0,
        mask_mix_probs: Optional[List[float]] = None,
        semantic_mask_fallback_to_random: bool = False,
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
        self.use_two_pairs = use_two_pairs
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
        self.semantic_only_epochs = semantic_only_epochs
        self.semantic_mask_fallback_to_random = semantic_mask_fallback_to_random
        self.mask_mix_probs = None
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
        self.current_epoch = 0  # 由训练循环每个 epoch 更新
        # 课程学习用：前期只抽 JT；切换后恢复原始采样，JT/BF 同步训练。
        self._jt_indices = [i for i, p in enumerate(self.pairs) if 'JT' in p.get('type', '')]
        self._jt_weights = [self.weights[i] for i in self._jt_indices]
        self._bf_indices = [i for i, p in enumerate(self.pairs) if 'BF' in p.get('type', '')]
        self._bf_weights = [self.weights[i] for i in self._bf_indices]
        self._semantic_indices_by_type = {}
        self._jt_semantic_indices = []
        self._bf_semantic_indices = []
        for i, pair in enumerate(self.pairs):
            if self._semantic_mask_path(pair['target_path']) is None:
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
            missing_semantic_dir = (
                (self.mask_mix_probs[1] > 0 or self.mask_mix_probs[2] > 0)
                and self.semantic_mask_dir is None
            )
            if missing_semantic_dir:
                if not self.semantic_mask_fallback_to_random:
                    raise ValueError("semantic_mask_dir is required when JT/BF semantic mask probability is > 0")
                self.mask_mix_probs[0] += self.mask_mix_probs[1] + self.mask_mix_probs[2]
                self.mask_mix_probs[1] = 0.0
                self.mask_mix_probs[2] = 0.0
                print("[WARN] semantic_mask_dir is missing; semantic mask probabilities fallback to random masking")

            if self.mask_mix_probs[1] > 0 and not self._jt_semantic_indices:
                if not self.semantic_mask_fallback_to_random:
                    raise ValueError("mask_mix_probs requests JT semantic masks, but no JT semantic mask files were found")
                self.mask_mix_probs[0] += self.mask_mix_probs[1]
                self.mask_mix_probs[1] = 0.0
                print("[WARN] JT semantic mask files were not found; JT semantic probability fallback to random masking")
            if self.mask_mix_probs[2] > 0 and not self._bf_semantic_indices:
                if not self.semantic_mask_fallback_to_random:
                    raise ValueError("mask_mix_probs requests BF semantic masks, but no BF semantic mask files were found")
                self.mask_mix_probs[0] += self.mask_mix_probs[2]
                self.mask_mix_probs[2] = 0.0
                print("[WARN] BF semantic mask files were not found; BF semantic probability fallback to random masking")

    def set_epoch(self, epoch: int) -> None:
        """训练循环每个 epoch 调用，供课程学习判断当前阶段"""
        self.current_epoch = epoch

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

    def _semantic_mask_path(self, target_path: str) -> Optional[str]:
        if self.semantic_mask_dir is None:
            return None
        parts = target_path.split('/')
        char_name = os.path.splitext(os.path.basename(target_path))[0]
        font_dir = None
        for i, p in enumerate(parts):
            if 'images' in p:
                font_dir = '/'.join(parts[:i])
                break
        if font_dir is None:
            return None
        npy_path = os.path.join(self.root, font_dir, 'semantic_masks', f'{char_name}.npy')
        if not os.path.exists(npy_path):
            return None
        return npy_path

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

    def _load_semantic_mask(self, target_path: str, pair_type: str) -> Optional[Image.Image]:
        """加载 .npy 并随机选 K 个标注 OR 合并，返回 PIL Image (mode='L')"""
        npy_path = self._semantic_mask_path(target_path)
        if npy_path is None:
            return None
        layers = np.load(npy_path)  # (N, 448, 448)
        N = layers.shape[0]
        # 根据 pair_type 选择对应的标注数量
        if 'JT' in pair_type:
            num_ann = self.num_mask_annotations_jt
        else:
            num_ann = self.num_mask_annotations_bf
        k = min(num_ann, N)
        indices = random.sample(range(N), k)
        combined = np.any(layers[indices], axis=0).astype(np.uint8) * 255
        return Image.fromarray(combined, mode='L')

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

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        mask_mode = self._sample_mask_mode()
        # curriculum 仅训练集启用：前 N 个 epoch 只用 JT；
        # epoch>=N 后恢复原始采样分布，让 JT 随机遮盖和 BF 语义遮盖同步训练。
        # 验证集没有 semantic_mask_dir，不重定向。
        if self.mask_mix_probs is None and self.semantic_mask_dir is not None and self.semantic_only_epochs > 0:
            pair_type_cur = self.pairs[index].get('type', '')
            if (self.current_epoch < self.semantic_only_epochs
                    and self._jt_indices
                    and 'JT' not in pair_type_cur):
                index = random.choices(self._jt_indices, weights=self._jt_weights, k=1)[0]
        elif mask_mode in ("jt_semantic", "bf_semantic"):
            index = self._sample_semantic_index(mask_mode)
        pair = self.pairs[index]
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
            sem_mask = self._load_semantic_mask(pair['target_path'], pair_type)

        # no aug for instance segmentation
        if "font" in pair['type'] and self.transforms3 is not None:
            cur_transforms = self.transforms3
        else:
            cur_transforms = self.transforms

        image, target, sem_mask = cur_transforms(image, target, interpolation1, interpolation2, mask=sem_mask)

        if self.use_two_pairs:
            pair_type = pair['type']
            # sample the second pair belonging to the same type
            pair2_index = random.choice(self.pair_type_dict[pair_type])
            if mask_mode in ("jt_semantic", "bf_semantic"):
                pair2_pool = self._semantic_indices_by_type.get(pair_type, [])
                if pair2_pool:
                    pair2_index = random.choice(pair2_pool)
            pair2 = self.pairs[pair2_index]
            image2 = self._load_image(pair2['image_path'])
            target2 = self._load_image(pair2['target_path'])
            if mask_mode == "random":
                sem_mask2 = None
            else:
                sem_mask2 = self._load_semantic_mask(pair2['target_path'], pair_type)
            assert pair2['type'] == pair_type
            image2, target2, sem_mask2 = cur_transforms(image2, target2, interpolation1, interpolation2, mask=sem_mask2)

            image = self._combine_images(image, image2, interpolation1)
            target = self._combine_images(target, target2, interpolation2)
            # 两个 target 都必须被遮盖（source 是参考字，始终完整可见）
            if sem_mask is not None and sem_mask2 is not None:
                sem_mask = torch.cat([sem_mask, sem_mask2], dim=1)
            else:
                sem_mask = None

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
        if (self.transforms_seccrop is None) or use_half_mask:
            pass
        else:
            image, target, sem_mask = self.transforms_seccrop(image, target, interpolation1, interpolation2, mask=sem_mask)

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

        contour_valid = torch.tensor(1.0 if mask_mode == "jt_semantic" else 0.0, dtype=torch.float32)
        return image, target, mask, valid, contour_valid

    def __len__(self) -> int:
        return len(self.pairs)


class PairStandardTransform(StandardTransform):
    def __init__(self, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None) -> None:
        super().__init__(transform=transform, target_transform=target_transform)

    def __call__(self, input: Any, target: Any, interpolation1: Any, interpolation2: Any, mask=None) -> Tuple[Any, Any, Any]:
        if self.transform is not None:
            input, target, mask = self.transform(input, target, interpolation1, interpolation2, mask=mask)
        return input, target, mask
