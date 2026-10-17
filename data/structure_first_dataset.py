"""Run C stage views over the unchanged Run B pairing and BF sampler."""

import torch
from torch.utils.data import Dataset

from data.dual_dataset import DualDataset


class StructureFirstDataset(Dataset):
    """Preserve source seals and references while exposing deployment inputs.

    The base dataset retains its complete manifest and all author/pool lookup
    indices. Only the externally addressable query indices are filtered for
    refinement. Instrumentation observes the very same BF random sample; it
    never samples another set of annotation layers.
    """

    def __init__(self, *args, stage="coarse", base_dataset=None, **kwargs):
        if stage not in ("coarse", "refine"):
            raise ValueError("Run C dataset stage must be coarse or refine")
        if base_dataset is not None and (args or kwargs):
            raise ValueError(
                "Pass either base_dataset or DualDataset arguments"
            )
        if base_dataset is None:
            kwargs["collect_bf_mask_stats"] = True
            base_dataset = DualDataset(*args, **kwargs)
        else:
            base_dataset.collect_bf_mask_stats = True
        self.base_dataset = base_dataset
        self.stage = stage
        self.indices = [
            index for index, pair in enumerate(base_dataset.pairs)
            if stage == "coarse" or pair["has_bf"]
        ]
        if not self.indices:
            raise ValueError("Run C dataset stage has no eligible queries")
        self.pairs = [base_dataset.pairs[index] for index in self.indices]
        self.weights = [base_dataset.weights[index] for index in self.indices]
        self.image_size = base_dataset.image_size
        self.patch_size = base_dataset.patch_size
        self.is_train = base_dataset.is_train
        self.root = base_dataset.root

    def __len__(self):
        return len(self.indices)

    def get_pair(self, index, ref_index=None, augmentation_seed=None):
        """Return an original pair, actual query BF masks and safe input."""
        original_index = self.indices[index]
        original_reference = (
            None if ref_index is None else self.indices[ref_index]
        )
        sample = self.base_dataset.get_pair(
            original_index, ref_index=original_reference,
            augmentation_seed=augmentation_seed,
        )
        if self.stage == "refine" and not bool(sample["has_bf"]):
            raise ValueError("JT-only samples cannot enter BF refinement")
        sample = dict(sample)
        legacy_mask = sample["mask"]
        grid = self.image_size // self.patch_size
        bf_patch = legacy_mask[grid:].clone()
        # Validation's old lower mask is fully masked, not a BF annotation.
        if not self.is_train or not bool(sample["has_bf"]):
            bf_patch.zero_()
        bf_pixel = bf_patch.repeat_interleave(self.patch_size, 0)
        bf_pixel = bf_pixel.repeat_interleave(self.patch_size, 1)
        bf_pixel = bf_pixel.unsqueeze(0).to(sample["valid"].dtype)
        full_mask = torch.zeros_like(legacy_mask)
        full_mask[grid:] = 1
        input_target = sample["target"].clone()
        input_target[:, self.image_size:] = 0
        sample.update({
            "legacy_mask": legacy_mask,
            "mask": full_mask,
            "input_target": input_target,
            "bf_patch_mask": bf_patch,
            "bf_pixel_mask": bf_pixel,
            "bf_loss_mask": bf_pixel * sample["valid"][:, self.image_size:],
            "structure_stage": self.stage,
        })
        return sample

    def __getitem__(self, index):
        return self.get_pair(index)


def build_structure_dataset(base_dataset, stage="coarse"):
    """Build a stage view without rebuilding or rewriting the sealed cache."""
    return StructureFirstDataset(base_dataset=base_dataset, stage=stage)
