"""Run B three-pool pairing and synchronized image/instance augmentation."""

import json
import random
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from tools.build_dual_manifest import normalize_record
from util.dual_targets import (
    CACHE_SCHEMA,
    DEFINITION,
    SOURCE_SEAL,
    build_instance_targets,
    empty_instance_targets,
    load_cached_targets,
    pool_weights,
    record_source_paths,
    sha256_file,
    transform_layers,
)

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


@contextmanager
def fixed_augmentation_seed(seed):
    """Make a probe reproducible without advancing training's global RNG."""
    if seed is None:
        yield
        return
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.random.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        # Set the CPU generator only; dataset probes must not mutate CUDA RNG.
        torch.random.default_generator.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)


class DualDataset(Dataset):
    """Return lower-query targets and two-row image pairs.

    Train uses merged manifests. The sampled record is the lower query and its
    upper reference is uniform in the same author/pool. Validation preserves
    old fixed-pair row order: JSON query on top, fixed partner on the bottom.
    One crop is sampled for the entire four-image grid; every BF/JT layer uses
    that exact crop with nearest interpolation before target derivation.
    """

    def __init__(
        self,
        root,
        json_path_list,
        target_cache_dir,
        masked_position_generator,
        is_train=True,
        fixed_pair_path=None,
        image_size=448,
        patch_size=16,
        num_mask_annotations_bf=11,
        mask_coverage_threshold=0.1,
        jieti_pool=224,
        jieti_valid_ink=200,
        crop_scale=(0.9999, 1.0),
        crop_ratio=(0.75, 4.0 / 3.0),
        verify_sources=True,
    ):
        self.root = Path(root)
        self.target_cache_dir = Path(target_cache_dir)
        self.masked_position_generator = masked_position_generator
        self.is_train = is_train
        self.image_size, self.patch_size = image_size, patch_size
        self.num_mask_annotations_bf = num_mask_annotations_bf
        self.mask_coverage_threshold = mask_coverage_threshold
        self.jieti_pool, self.jieti_valid_ink = jieti_pool, jieti_valid_ink
        self.crop_scale, self.crop_ratio = crop_scale, crop_ratio
        if image_size != 448 or patch_size != 16:
            raise ValueError("Run B requires 448-square images and 16-pixel patches")
        if num_mask_annotations_bf < 1:
            raise ValueError("BF semantic masking requires at least one layer")
        self.pairs = []
        for path in json_path_list:
            rows = json.loads(Path(path).read_text(encoding="utf-8"))
            if not isinstance(rows, list):
                raise ValueError(f"Expected dataset record list: {path}")
            self.pairs.extend(rows)
        self.pairs = [p if "has_jt" in p else normalize_record(p) for p in self.pairs]
        if is_train and any("record_id" not in p for p in self.pairs):
            raise ValueError("Training requires the merged dual manifest")
        self._fixed_pairs = None
        if not is_train:
            if not fixed_pair_path:
                raise ValueError("Dual validation requires fixed clean pairs")
            raw = json.loads(Path(fixed_pair_path).read_text(encoding="utf-8"))
            self._fixed_pairs = {
                key: p if "has_jt" in p else normalize_record(p)
                for key, p in raw.items()
            }
            self.pairs = [p for p in self.pairs if p["target_path"] in raw]
            if Counter(p["target_path"] for p in self.pairs) != Counter(raw.keys()):
                raise ValueError(
                    "Fixed validation queries missing or duplicated in JSON inputs"
                )
        self.weights = pool_weights(self.pairs)
        self.pair_type_dict = defaultdict(list)
        for index, pair in enumerate(self.pairs):
            key = (pair["calligrapher"], pair["pool_type"])
            self.pair_type_dict[key].append(index)
        cache_index_path = self.target_cache_dir / "index.json"
        self._cache_index = json.loads(cache_index_path.read_text(encoding="utf-8"))
        self._cache_metadata = json.loads(
            (self.target_cache_dir / "metadata.json").read_text(encoding="utf-8")
        )
        if (
            self._cache_metadata.get("definition") != DEFINITION
            or self._cache_metadata.get("status") != "complete"
        ):
            raise ValueError("Instance cache is incomplete or has the wrong definition")
        schema = self._cache_metadata.get("schema")
        if schema == CACHE_SCHEMA:
            if (
                self._cache_metadata.get("source_seal") != SOURCE_SEAL
                or not isinstance(self._cache_metadata.get("source_inputs"), list)
                or not isinstance(self._cache_metadata.get("source_sha256"), dict)
            ):
                raise ValueError("Instance cache full source seal is incomplete")
            self.source_seal = SOURCE_SEAL
        elif schema == "dual-target-cache-v1":
            # Legacy cache hashes cover JT COCO/raw/GT only, never all inputs.
            self.source_seal = "jt-cache-only-legacy"
        else:
            raise ValueError(f"Unknown instance cache schema: {schema}")
        self.source_verification_scope = (
            self.source_seal if verify_sources else "disabled"
        )
        if sha256_file(cache_index_path) != self._cache_metadata["index_sha256"]:
            raise ValueError("Instance cache index hash mismatch")
        input_hashes = set(self._cache_metadata["input_sha256"].values())
        for path in json_path_list:
            if sha256_file(path) not in input_hashes:
                raise ValueError(f"Dataset JSON hash differs from target cache: {path}")
        if not is_train:
            if sha256_file(fixed_pair_path) != self._cache_metadata.get(
                "fixed_pair_sha256"
            ):
                raise ValueError("Fixed validation pair hash differs from target cache")
        # Validate hashes once at construction rather than inside every epoch.
        relevant = self.pairs + list((self._fixed_pairs or {}).values())
        verified = set()
        for pair in relevant:
            entry = None
            if pair["has_jt"]:
                if pair["target_path"] not in self._cache_index:
                    raise ValueError(
                        f"JT instance cache missing: {pair['target_path']}"
                    )
                entry = self._cache_index[pair["target_path"]]
            if not verify_sources:
                continue
            if self.source_seal == SOURCE_SEAL:
                paths = record_source_paths(pair).values()
            elif entry is not None:
                paths = (
                    entry["annotation_path"],
                    pair["target_path"],
                    entry["coordinate_audit"]["original_path"],
                )
            else:
                continue
            for path in paths:
                if path not in verified:
                    expected = self._cache_metadata["source_sha256"].get(path)
                    if expected is None:
                        raise ValueError(f"Instance cache source hash missing: {path}")
                    if sha256_file(self.root / path) != expected:
                        raise ValueError(f"Instance cache source changed: {path}")
                    verified.add(path)

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, path, is_target=False):
        with Image.open(self.root / path) as image:
            image = image.convert("RGB")
        if is_target and image.size != (self.image_size, self.image_size):
            raise ValueError(f"GT image is not 448-square: {path}")
        width, height = image.size
        if width != height:
            side = max(width, height)
            square = Image.new("RGB", (side, side), "white")
            square.paste(image, ((side - width) // 2, (side - height) // 2))
            image = square
        if image.size != (self.image_size, self.image_size):
            image = image.resize(
                (self.image_size, self.image_size), Image.Resampling.BICUBIC
            )
        return image

    def _bf_mask(self, pair):
        path = self.root / pair["bf_semantic_mask_path"]
        layers = np.load(path, allow_pickle=False)
        if layers.ndim != 3 or layers.shape[1:] != (self.image_size, self.image_size):
            raise ValueError(f"BF mask geometry mismatch: {path}: {layers.shape}")
        if layers.shape[0] == 0:
            raise ValueError(f"BF semantic source has no layers: {path}")
        selected = random.sample(
            range(len(layers)), min(self.num_mask_annotations_bf, len(layers))
        )
        return np.any(layers[selected] > 0, axis=0).astype(np.uint8)

    def _transform_row(self, pair, crop, needs_targets):
        image = self._load_image(pair["image_path"])
        target = self._load_image(pair["target_path"], is_target=True)
        image = TF.resized_crop(
            image, *crop, [self.image_size] * 2, InterpolationMode.BICUBIC
        )
        target = TF.resized_crop(
            target, *crop, [self.image_size] * 2, InterpolationMode.NEAREST
        )
        semantic = None
        if self.is_train and pair["has_bf"]:
            semantic = transform_layers(self._bf_mask(pair), crop)[0]
        if needs_targets and pair["has_jt"]:
            raw = load_cached_targets(
                self.target_cache_dir,
                pair,
                index=self._cache_index,
                metadata=self._cache_metadata,
            )
            regions = transform_layers(raw["regions"], crop)
            text = transform_layers(raw["text"], crop)[0]
            targets = build_instance_targets(
                regions,
                text,
                raw["classes"],
                self.patch_size,
                self.jieti_pool,
                self.jieti_valid_ink,
            )
        else:
            targets = empty_instance_targets(
                self.image_size, self.patch_size, self.jieti_pool
            )
        image = TF.normalize(TF.to_tensor(image), MEAN, STD)
        target = TF.normalize(TF.to_tensor(target), MEAN, STD)
        return image, target, semantic, targets

    def get_pair(self, index, ref_index=None, augmentation_seed=None):
        """Fetch a deterministic probe; train index denotes the lower query."""
        with fixed_augmentation_seed(augmentation_seed):
            query = self.pairs[index]
            if self.is_train:
                key = (query["calligrapher"], query["pool_type"])
                if ref_index is None:
                    ref_index = random.choice(self.pair_type_dict[key])
                reference = self.pairs[ref_index]
                if (reference["calligrapher"], reference["pool_type"]) != key:
                    raise ValueError("Dual pairing may not cross author or pool")
                top, bottom = reference, query
            else:
                if ref_index is not None:
                    raise ValueError(
                        "Validation pairing is fixed by the clean pair file"
                    )
                top, bottom = query, self._fixed_pairs[query["target_path"]]
            if self.is_train:
                dummy = Image.new("RGB", (self.image_size, self.image_size))
                crop = RandomResizedCrop.get_params(
                    dummy, self.crop_scale, self.crop_ratio
                )
            else:
                # Deterministic identity geometry, with no RNG draw in validation.
                crop = (0, 0, self.image_size, self.image_size)
            upper = self._transform_row(top, crop, needs_targets=False)
            lower = self._transform_row(bottom, crop, needs_targets=True)
            image, target = (
                torch.cat((upper[0], lower[0]), 1),
                torch.cat((upper[1], lower[1]), 1),
            )
            if not self.is_train:
                grid = self.image_size // self.patch_size
                mask = np.zeros((2 * grid, grid), dtype=np.int64)
                mask[grid:] = 1
            elif bottom["has_bf"]:
                if upper[2] is None or lower[2] is None:
                    raise ValueError("Both BF rows must have semantic masks")
                pixels = np.concatenate((upper[2], lower[2]), axis=0)
                grid = self.image_size // self.patch_size
                coverage = pixels.reshape(
                    2 * grid, self.patch_size, grid, self.patch_size
                ).mean((1, 3))
                mask = (coverage > self.mask_coverage_threshold).astype(np.int64)
            else:
                mask = np.asarray(self.masked_position_generator(), dtype=np.int64)
            expected = (
                2 * self.image_size // self.patch_size,
                self.image_size // self.patch_size,
            )
            if mask.shape != expected:
                raise ValueError(f"Mask shape {mask.shape} differs from {expected}")
            sample = {
                "image": image,
                "target": target,
                "mask": torch.from_numpy(mask),
                "valid": torch.ones_like(target),
                "has_bf": torch.tensor(bool(bottom["has_bf"])),
                "has_jt": torch.tensor(bool(bottom["has_jt"])),
                "target_path": bottom["target_path"],
                "reference_path": top["target_path"],
                "sample_key": bottom.get("record_id", bottom["target_path"]),
                "query_target_path": bottom["target_path"],
                "reference_target_path": top["target_path"],
                "geometry_seed": -1 if augmentation_seed is None else augmentation_seed,
                "geometry_crop": torch.tensor(crop, dtype=torch.long),
            }
            sample.update(
                {name: torch.from_numpy(value) for name, value in lower[3].items()}
            )
            return sample

    def __getitem__(self, index):
        return self.get_pair(index)


DualPairDataset = DualDataset
