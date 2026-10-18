"""JT-only deployment pairs and low-resolution supervision for Run C v2."""

import random
from collections import Counter, defaultdict

import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode

from data.dual_dataset import MEAN, STD, DualDataset, fixed_augmentation_seed
from util.dual_targets import (
    build_instance_targets,
    empty_instance_targets,
    load_cached_targets,
    transform_layers,
)

REFERENCE_POLICY = "distinct-original-reference-v1"


def coarse_targets_from_normalized(target, ink_threshold=0.5, valid=None):
    """Build loss-only 112 targets from once-normalized 448 RGB images.

    ``target`` is CHW or BCHW, with ImageNet normalization applied once.
    Ink is ``1 - (.299 R + .587 G + .114 B)``. The default threshold matches
    the existing 8-bit grayscale ``gray < 128`` foreground convention at
    values 127/128. Density is never thresholded. Validity is averaged over
    channels and each 4-square cell, and remains a separate loss weight.
    """
    if not 0 < ink_threshold < 1:
        raise ValueError("Ink threshold must be strictly between zero and one")
    squeeze = target.ndim == 3
    image = target.unsqueeze(0) if squeeze else target
    if image.ndim != 4 or image.shape[1:] != (3, 448, 448):
        raise ValueError("Coarse supervision requires RGB 448-square targets")
    image = image.float()
    mean = image.new_tensor(MEAN).view(1, 3, 1, 1)
    std = image.new_tensor(STD).view(1, 3, 1, 1)
    rgb = image * std + mean
    gray = (rgb * image.new_tensor((0.299, 0.587, 0.114)).view(1, 3, 1, 1)).sum(
        1, keepdim=True
    )
    ink = (1 - gray).clamp(0, 1)
    density = F.avg_pool2d(ink, 4, 4)
    occupancy = F.max_pool2d((ink > ink_threshold).to(ink.dtype), 4, 4)
    if valid is None:
        valid_coarse = torch.ones_like(density)
    else:
        weights = valid.unsqueeze(0) if valid.ndim == 3 else valid
        if (
            weights.ndim != 4
            or weights.shape[0] != image.shape[0]
            or weights.shape[-2:] != (448, 448)
            or weights.shape[1] not in (1, 3)
        ):
            raise ValueError("Valid weights must be B1HW/B3HW on the target grid")
        weights = weights.to(image).mean(1, keepdim=True).clamp(0, 1)
        valid_coarse = F.avg_pool2d(weights, 4, 4)
    targets = {
        "density_target": density,
        "occupancy_target": occupancy,
        "valid_coarse": valid_coarse,
    }
    return {key: value[0] if squeeze else value for key, value in targets.items()}


class RuncV2Dataset(Dataset):
    """Select JT training queries without ever invoking BF mask sampling.

    The sealed DualDataset remains the source/cache verifier. This view builds
    deployment pairs directly instead of calling its BF-aware ``get_pair``.
    Train references are sampled uniformly from precomputed same-author/pool
    candidates with a distinct original identity and target SHA. A query must
    never become its own visible reference. Validation keeps the fixed pairs;
    actual lower-target metadata is returned separately from upper JSON rows.
    """

    def __init__(self, base_dataset, ink_threshold=0.5, query_kind="jt"):
        if query_kind not in ("jt", "bf"):
            raise ValueError("Run C v2 query kind must be jt or bf")
        self.base_dataset = base_dataset
        self.is_train = base_dataset.is_train
        self.ink_threshold = float(ink_threshold)
        self.image_size = base_dataset.image_size
        self.patch_size = base_dataset.patch_size
        self.root = base_dataset.root
        self.reference_policy = REFERENCE_POLICY
        self.query_kind = query_kind
        eligible_field = "has_jt" if query_kind == "jt" else "has_bf"
        self.indices = [
            index
            for index, pair in enumerate(base_dataset.pairs)
            if (
                pair
                if self.is_train
                else base_dataset._fixed_pairs[pair["target_path"]]
            )[eligible_field]
        ]
        self.pairs = [base_dataset.pairs[index] for index in self.indices]
        if not self.pairs:
            raise ValueError("Run C v2 has no eligible records")
        if self.is_train:
            identities = [
                (pair["calligrapher"], pair["original_md5"]) for pair in self.pairs
            ]
            if len(set(identities)) != len(identities):
                raise ValueError("Duplicate original-image identity in JT train")
        self.weights = [base_dataset.weights[index] for index in self.indices]
        self.pair_type_dict = defaultdict(list)
        for index, pair in enumerate(self.pairs):
            self.pair_type_dict[(pair["calligrapher"], pair["pool_type"])].append(index)
        relevant = self.pairs + list((base_dataset._fixed_pairs or {}).values())
        self._record_identities = {}
        source_hashes = base_dataset._cache_metadata.get("source_sha256", {})
        for pair in relevant:
            original_path = pair.get("original_path")
            if not original_path:
                raise ValueError("Original path is required for reference identity")
            original_sha = source_hashes.get(original_path)
            target_sha = source_hashes.get(pair["target_path"])
            if not original_sha or not target_sha:
                raise ValueError(
                    "Sealed original/target SHA is required for references"
                )
            self._record_identities[pair["target_path"]] = {
                "original_identity": f"{pair['calligrapher']}:{original_sha}",
                "original_sha256": original_sha,
                "target_sha256": target_sha,
            }
        self.reference_candidates = {}
        if self.is_train:
            for index, query in enumerate(self.pairs):
                key = (query["calligrapher"], query["pool_type"])
                candidates = tuple(
                    candidate
                    for candidate in self.pair_type_dict[key]
                    if self._references_distinct(query, self.pairs[candidate])
                )
                if not candidates:
                    raise ValueError(
                        f"No distinct-original reference for {query['target_path']}"
                    )
                self.reference_candidates[index] = candidates
            self.reference_candidate_summary = {
                "queries": len(self.pairs),
                "empty_queries": 0,
                "minimum_candidates": min(map(len, self.reference_candidates.values())),
                "maximum_candidates": max(map(len, self.reference_candidates.values())),
            }
        else:
            for reference in self.pairs:
                query = base_dataset._fixed_pairs[reference["target_path"]]
                self._assert_distinct_reference(query, reference)

    def _references_distinct(self, query, reference):
        query_id = self._record_identities[query["target_path"]]
        reference_id = self._record_identities[reference["target_path"]]
        return (
            query_id["original_sha256"] != reference_id["original_sha256"]
            and query_id["target_sha256"] != reference_id["target_sha256"]
        )

    def _assert_distinct_reference(self, query, reference):
        if not self._references_distinct(query, reference):
            raise ValueError(
                "Reference must have a distinct original identity and target SHA: "
                f"query={query['target_path']}, reference={reference['target_path']}"
            )

    def _sample_identity_fields(self, query, reference):
        query_identity = self._record_identities[query["target_path"]]
        reference_identity = self._record_identities[reference["target_path"]]
        return {
            "reference_policy": self.reference_policy,
            "original_identity": query_identity["original_identity"],
            **{f"query_{key}": value for key, value in query_identity.items()},
            **{f"reference_{key}": value for key, value in reference_identity.items()},
        }

    def __len__(self):
        return len(self.pairs)

    def _row(self, pair, crop, needs_targets):
        base = self.base_dataset
        source = base._load_image(pair["image_path"])
        target = base._load_image(pair["target_path"], is_target=True)
        source = TF.resized_crop(source, *crop, [448, 448], InterpolationMode.BICUBIC)
        target = TF.resized_crop(target, *crop, [448, 448], InterpolationMode.NEAREST)
        if needs_targets and pair["has_jt"]:
            raw = load_cached_targets(
                base.target_cache_dir,
                pair,
                index=base._cache_index,
                metadata=base._cache_metadata,
            )
            regions = transform_layers(raw["regions"], crop)
            text = transform_layers(raw["text"], crop)[0]
            arguments = (regions, text, raw["classes"], self.patch_size)
            targets = build_instance_targets(
                *arguments,
                pool=base.jieti_pool,
                valid_ink=base.jieti_valid_ink,
            )
            coarse = build_instance_targets(
                *arguments,
                pool=112,
                valid_ink=base.jieti_valid_ink,
            )
        else:
            targets = empty_instance_targets(pool=base.jieti_pool)
            coarse = empty_instance_targets(pool=112)
        targets.update(
            {
                "coarse_voro": coarse["voro"],
                "valid_parts_coarse": coarse["valid_parts"],
                "coarse_partition_lost": (
                    coarse["j_downsample_lost"] | coarse["j_partition_lost"]
                ),
            }
        )
        source = TF.normalize(TF.to_tensor(source), MEAN, STD)
        target = TF.normalize(TF.to_tensor(target), MEAN, STD)
        return source, target, targets

    def get_pair(self, index, ref_index=None, augmentation_seed=None):
        """Return full-mask deployment input plus strictly loss-only GT fields."""
        with fixed_augmentation_seed(augmentation_seed):
            query = self.pairs[index]
            if self.is_train:
                key = (query["calligrapher"], query["pool_type"])
                if ref_index is None:
                    ref_index = random.choice(self.reference_candidates[index])
                if ref_index not in self.reference_candidates[index]:
                    raise ValueError(
                        "Explicit reference is not a distinct-original candidate"
                    )
                top, bottom = self.pairs[ref_index], query
                if (top["calligrapher"], top["pool_type"]) != key:
                    raise ValueError("JT reference may not cross author/pool")
                dummy = Image.new("RGB", (448, 448))
                crop = RandomResizedCrop.get_params(
                    dummy,
                    self.base_dataset.crop_scale,
                    self.base_dataset.crop_ratio,
                )
            else:
                if ref_index is not None:
                    raise ValueError("Validation references are fixed")
                top = query
                bottom = self.base_dataset._fixed_pairs[query["target_path"]]
                crop = (0, 0, 448, 448)
            self._assert_distinct_reference(bottom, top)
            upper = self._row(top, crop, needs_targets=False)
            lower = self._row(bottom, crop, needs_targets=True)
            image = torch.cat((upper[0], lower[0]), 1)
            target = torch.cat((upper[1], lower[1]), 1)
            input_target = target.clone()
            input_target[:, 448:] = 0
            mask = torch.zeros((56, 28), dtype=torch.long)
            mask[28:] = 1
            author = bottom["calligrapher"]
            identity_fields = self._sample_identity_fields(bottom, top)
            identity = identity_fields["original_identity"]
            sample = {
                "image": image,
                "target": target,
                "input_target": input_target,
                "valid": torch.ones_like(target),
                "mask": mask,
                "has_bf": torch.tensor(bool(bottom["has_bf"])),
                "has_jt": torch.tensor(bool(bottom["has_jt"])),
                "target_path": bottom["target_path"],
                "reference_path": top["target_path"],
                "query_target_path": bottom["target_path"],
                "reference_target_path": top["target_path"],
                "sample_key": identity,
                **identity_fields,
                "author": author,
                "calligrapher": author,
                "character": bottom["character"],
                "geometry_seed": (
                    -1 if augmentation_seed is None else augmentation_seed
                ),
                "geometry_crop": torch.tensor(crop, dtype=torch.long),
                "bf_sampling_enabled": torch.tensor(False),
            }
            sample.update(
                {key: torch.from_numpy(value) for key, value in lower[2].items()}
            )
            sample.update(
                coarse_targets_from_normalized(
                    lower[1],
                    ink_threshold=self.ink_threshold,
                )
            )
            return sample

    def __getitem__(self, index):
        return self.get_pair(index)


class RuncV2RefineDataset(RuncV2Dataset):
    """Safe BF-bearing reference selection around the unchanged BF sampler.

    The wrapper supplies an explicit distinct-original reference to the old
    StructureFirstDataset. BF annotation selection, crop, patch coverage and
    actual mask-to-loss mapping remain in that implementation. Safer candidate
    selection changes the reference RNG trajectory; BF layer sampling retains
    its original distribution and requests exactly 11 annotation layers.
    """

    def __init__(self, base_dataset):
        from data.structure_first_dataset import StructureFirstDataset

        super().__init__(base_dataset, query_kind="bf")
        self.legacy_view = StructureFirstDataset(
            base_dataset=base_dataset, stage="refine"
        )
        self._legacy_indices = {
            pair["target_path"]: index
            for index, pair in enumerate(self.legacy_view.pairs)
        }

    def get_pair(self, index, ref_index=None, augmentation_seed=None):
        with fixed_augmentation_seed(augmentation_seed):
            query = self.pairs[index]
            if self.is_train:
                if ref_index is None:
                    ref_index = random.choice(self.reference_candidates[index])
                if ref_index not in self.reference_candidates[index]:
                    raise ValueError(
                        "Explicit reference is not a distinct-original candidate"
                    )
                reference = self.pairs[ref_index]
                legacy_reference = self._legacy_indices[reference["target_path"]]
            else:
                if ref_index is not None:
                    raise ValueError("Validation references are fixed")
                reference = query
                query = self.base_dataset._fixed_pairs[reference["target_path"]]
                legacy_reference = None
            self._assert_distinct_reference(query, reference)
            legacy_query = self._legacy_indices[self.pairs[index]["target_path"]]
            # The outer seed context owns reference selection and BF sampling.
            # Passing None prevents a second reset of the BF sampler's RNG.
            sample = self.legacy_view.get_pair(
                legacy_query, ref_index=legacy_reference, augmentation_seed=None
            )
            sample.update(self._sample_identity_fields(query, reference))
            sample.update(
                {
                    "sample_key": sample["original_identity"],
                    "author": query["calligrapher"],
                    "calligrapher": query["calligrapher"],
                    "character": query["character"],
                    "geometry_seed": (
                        -1 if augmentation_seed is None else augmentation_seed
                    ),
                }
            )
            return sample


def _build_sealed_base(config, validation):
    data = config["data"]
    return DualDataset(
        root=data["root"],
        json_path_list=data["val_json"] if validation else data["train_json"],
        target_cache_dir=data["cache"],
        masked_position_generator=None,
        is_train=not validation,
        fixed_pair_path=data.get("fixed_pairs") if validation else None,
        verify_sources=data.get("verify_sources", True),
    )


def build_runc_v2_dataset(config, validation=False):
    """Build JT740 train or the unchanged JT101 subset of fixed validation."""
    data = config["data"]
    base = _build_sealed_base(config, validation)
    dataset = RuncV2Dataset(base, ink_threshold=data.get("ink_threshold", 0.5))
    if not validation:
        pools = Counter(pair["pool_type"] for pair in dataset.pairs)
        if len(dataset) != 740 or pools != {"dual": 466, "jt_only": 274}:
            raise ValueError(f"JT manifest must contain 466+274 targets: {pools}")
    return dataset


def build_runc_v2_refine_dataset(config, validation=False):
    """Build safe BF818 queries for stage-II interface smoke, not long training."""
    dataset = RuncV2RefineDataset(_build_sealed_base(config, validation))
    if not validation:
        pools = Counter(pair["pool_type"] for pair in dataset.pairs)
        if len(dataset) != 818 or pools != {"dual": 466, "bf_only": 352}:
            raise ValueError(f"BF manifest must contain 466+352 targets: {pools}")
    return dataset
