"""Run C v2 matching, geometry, phase and denominator contracts on CPU."""

from dataclasses import asdict

import pytest
import torch

from util.dual_loss import DualJietiLoss, InstanceHeadLoss
from util.jieti_loss import IMAGENET_MEAN, IMAGENET_STD
from util.runc_v2_loss import (
    CoarseJietiLoss,
    MatchedHeadGeometry,
    RunCV2LossWeights,
    head_metric_sums,
    runc_v2_losses,
)


def inputs(batch_size=2, pool=8, coarse=16):
    torch.manual_seed(93)
    occupancy = torch.zeros(batch_size, 4, pool, pool)
    occupancy[:, 0, 1:5, 1:3] = 1
    occupancy[:, 1, 2:7, 5:7] = 1
    valid = torch.tensor([[True, True, False, False]]).expand(batch_size, -1)
    head = {
        "occupancy_logits": torch.randn(
            batch_size, 4, pool, pool, requires_grad=True
        ),
        "existence_logits": torch.randn(batch_size, 4, requires_grad=True),
        "class_logits": torch.randn(batch_size, 4, 20, requires_grad=True),
    }
    voro = torch.ones(batch_size, coarse, coarse, dtype=torch.long)
    voro[:, :, coarse // 2:] = 2
    density = torch.zeros(batch_size, 1, coarse, coarse)
    density[:, :, 2:9, 2:6] = 0.6
    density[:, :, 3:13, 10:14] = 0.8
    batch = {
        "instance_occupancy": occupancy,
        "instance_valid": valid.clone(),
        "instance_classes": torch.tensor([[2, 3, -1, -1]]).expand(
            batch_size, -1
        ),
        "has_jt": torch.ones(batch_size, dtype=torch.bool),
        "density_target": density,
        "occupancy_target": (density > 0).float(),
        "valid_coarse": torch.ones_like(density),
        "coarse_voro": voro,
        "valid_parts": valid.clone(),
        "valid_parts_coarse": valid.clone(),
    }
    outputs = {
        "head": head,
        "density": torch.rand(
            batch_size, 1, coarse, coarse, requires_grad=True
        ),
        "occupancy_logits": torch.randn(
            batch_size, 1, coarse, coarse, requires_grad=True
        ),
    }
    return outputs, batch


def head_args(outputs, batch):
    return (
        outputs["head"], batch["instance_occupancy"], batch["instance_valid"],
        batch["instance_classes"], batch["has_jt"],
    )


def slice_batch(value, start, end):
    return {key: tensor[start:end] for key, tensor in value.items()}


def slice_outputs(value, start, end):
    return {
        key: slice_batch(tensor, start, end) if isinstance(tensor, dict)
        else tensor[start:end] for key, tensor in value.items()
    }


def test_one_matching_reused_with_original_loss():
    outputs, batch = inputs()

    class CountMatching(InstanceHeadLoss):
        calls = 0

        def forward(self, *args):
            self.calls += 1
            return super().forward(*args)

    matcher = CountMatching()
    loss = runc_v2_losses(outputs, batch, head_criterion=matcher)
    expected, details = InstanceHeadLoss()(*head_args(outputs, batch))
    assert matcher.calls == 1
    torch.testing.assert_close(loss["raw_head"], expected)
    assert torch.equal(loss["head_parts"]["assignment"], details["assignment"])
    assert torch.equal(
        loss["head_geometry"]["assignment"], details["assignment"]
    )


def test_permutation_invariant_loss_and_geometry():
    outputs, batch = inputs()
    first = runc_v2_losses(outputs, batch, stage="IC")
    order = torch.tensor([2, 0, 3, 1])
    permuted = {**outputs, "head": {
        key: value[:, order] for key, value in outputs["head"].items()
    }}
    second = runc_v2_losses(permuted, batch, stage="IC")
    for key in ("total", "raw_head", "raw_head_logsigma", "raw_head_centroid",
                "raw_head_shape"):
        torch.testing.assert_close(first[key], second[key])


@pytest.mark.parametrize("term", [
    "raw_head_logsigma_x", "raw_head_logsigma_y", "raw_head_centroid",
    "raw_head_shape",
])
def test_head_geometry_each_term_has_finite_matched_gradient(term):
    outputs, batch = inputs()
    result = runc_v2_losses(outputs, batch)
    gradient = torch.autograd.grad(
        result[term], outputs["head"]["occupancy_logits"]
    )[0]
    valid = result["head_geometry"]["valid"]
    assert torch.isfinite(gradient).all()
    assert gradient[valid].abs().sum() > 0
    assert torch.count_nonzero(gradient[~valid]) == 0


def test_near_empty_head_prediction_is_not_shielded():
    outputs, batch = inputs(batch_size=1)
    outputs["head"]["occupancy_logits"] = torch.full(
        (1, 4, 8, 8), -40.0, requires_grad=True
    )
    result = runc_v2_losses(outputs, batch)
    assert result["count_head_parts"] == 2
    assert result["raw_head_logsigma"] > 1
    result["total"].backward()
    gradient = outputs["head"]["occupancy_logits"].grad
    assert torch.isfinite(result["total"])
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_zero_instance_and_missing_jt_stay_finite_and_counted():
    outputs, batch = inputs()
    batch["has_jt"][:] = False
    batch["instance_valid"][:] = False
    batch["valid_parts"][:] = False
    batch["valid_parts_coarse"][:] = False
    result = runc_v2_losses(outputs, batch, stage="IA")
    assert result["count_head_parts"] == 0
    assert result["total"] == 0
    result["total"].backward()
    for tensor in outputs["head"].values():
        assert torch.isfinite(tensor.grad).all()
        assert torch.count_nonzero(tensor.grad) == 0


def test_relative_centroid_requires_two_parts_but_absolute_does_not():
    outputs, batch = inputs()
    batch["instance_valid"][:, 1] = False
    result = runc_v2_losses(outputs, batch)
    assert result["raw_head_centroid"] == 0
    reductions = result["head_geometry"]["reductions"]
    assert reductions["centroid_relative"]["count"] == 0
    assert result["raw_head_centroid_absolute"] > 0
    assert reductions["centroid_absolute"]["count"] == 2


def test_shape_descriptor_normalizes_nonempty_mass_at_head_grid():
    criterion = MatchedHeadGeometry(pool=28)
    mass = torch.zeros(1, 4, 28, 28)
    mass[:, 0, 4:10, 3:8] = 0.7
    mass[:, 1, 17:23, 19:24] = 0.2
    descriptor = criterion.geometry.descriptor(
        mass, criterion.geometry.moments(mass)
    )
    torch.testing.assert_close(descriptor[0, :2].sum((-1, -2)), torch.ones(2))
    assert torch.count_nonzero(descriptor[0, 2:]) == 0
    # The two translated rectangles have the same normalized shape/mass.
    torch.testing.assert_close(
        descriptor[0, 0], descriptor[0, 1], atol=2e-6, rtol=2e-6
    )


def test_identical_density_oracle_has_zero_geometry_and_no_resize():
    _, batch = inputs(batch_size=1)
    density = batch["density_target"].clone().requires_grad_()
    criterion = CoarseJietiLoss(pool=16)
    result = criterion(
        density, density.detach(), batch["coarse_voro"],
        batch["valid_parts"], batch["has_jt"],
    )
    for name in ("logsigma_x", "logsigma_y", "centroid_relative", "shape"):
        assert result["reductions"][name]["mean"].abs() < 1e-7
    assert result["valid"].sum() == 2


def test_coarse_geometry_matches_old_j_on_same_grid_except_zero_floor():
    outputs, batch = inputs(batch_size=1)
    prediction, target = outputs["density"], batch["density_target"]

    def canvas(ink):
        rgb = (1 - ink).repeat(1, 3, 1, 1)
        mean = rgb.new_tensor(IMAGENET_MEAN)[None, :, None, None]
        std = rgb.new_tensor(IMAGENET_STD)[None, :, None, None]
        normalized = (rgb - mean) / std
        return torch.cat((normalized, normalized), dim=2)

    _, old, _ = DualJietiLoss(pool=16)(
        canvas(prediction), canvas(target), batch["coarse_voro"],
        batch["valid_parts"], batch["has_jt"],
    )
    new = CoarseJietiLoss(pool=16)(
        prediction, target, batch["coarse_voro"], batch["valid_parts"],
        batch["has_jt"],
    )["reductions"]
    torch.testing.assert_close(
        new["logsigma_x"]["mean"] + new["logsigma_y"]["mean"],
        old["logsigma"], atol=2e-6, rtol=2e-6,
    )
    torch.testing.assert_close(
        new["shape"]["mean"], old["shape"], atol=2e-6, rtol=2e-6
    )
    # The original sqrt(d^2+eps) contributes a small positive identity floor;
    # v2 subtracts sqrt(eps) before dividing by the same GT character sigma.
    assert 0 < old["centroid"] - new["centroid_relative"]["mean"] < 0.001


def test_coarse_losses_use_oracle_on_same_grid_and_keep_empty_pixel_penalty():
    outputs, batch = inputs(batch_size=1)
    outputs["density"] = batch["density_target"].clone().requires_grad_()
    correct = runc_v2_losses(outputs, batch, stage="IB")
    assert correct["raw_density"] == 0
    assert correct["raw_j_coarse"].abs() < 1e-7
    outputs["density"] = torch.zeros_like(
        outputs["density"], requires_grad=True
    )
    empty = runc_v2_losses(outputs, batch, stage="IB")
    assert empty["count_j_coarse_shield"] == 2
    assert empty["raw_density"] > 0
    assert empty["raw_occupancy"] > 0
    empty["total"].backward()
    assert torch.isfinite(outputs["density"].grad).all()
    assert outputs["density"].grad.abs().sum() > 0


@pytest.mark.parametrize("stage", ["IA", "IA0", "IA1"])
def test_head_only_stage_requires_no_coarse_output(stage):
    outputs, batch = inputs()
    result = runc_v2_losses({"head": outputs["head"]}, batch, stage=stage)
    result["total"].backward()
    assert outputs["density"].grad is None
    assert outputs["occupancy_logits"].grad is None
    assert outputs["head"]["occupancy_logits"].grad.abs().sum() > 0


def test_phase_objectives_and_probe_leaves():
    outputs, batch = inputs()
    weights = RunCV2LossWeights(
        head=2, head_geo=3, density=4, occupancy=5, j_coarse=6,
        head_logsigma=0.7, head_centroid=0.8, head_shape=0.9,
    )
    ia = runc_v2_losses(outputs, batch, asdict(weights), "IA")
    ib = runc_v2_losses(outputs, batch, weights, "IB")
    ic = runc_v2_losses(outputs, batch, weights, "IC")
    torch.testing.assert_close(
        ia["total"], 2 * ia["raw_head"] + 3 * ia["raw_head_geo"]
    )
    torch.testing.assert_close(
        ib["total"], 4 * ib["raw_density"] + 5 * ib["raw_occupancy"]
        + 6 * ib["raw_j_coarse"]
    )
    torch.testing.assert_close(ic["total"], ia["total"] + ib["total"])
    assert "head" not in ib["probe_terms"]
    assert "head_logsigma_x" not in ia["probe_terms"]
    assert "j_coarse" not in ib["probe_terms"]
    ib["total"].backward()
    assert outputs["head"]["occupancy_logits"].grad is None
    assert outputs["density"].grad.abs().sum() > 0
    assert outputs["occupancy_logits"].grad.abs().sum() > 0


def test_equal_microbatch_and_unequal_rank_sample_weighting():
    outputs, batch = inputs(batch_size=3)
    # Deliberately vary the number of valid instances per character.
    batch["instance_valid"][1, 1] = False
    batch["instance_valid"][2] = False
    combined = runc_v2_losses(outputs, batch, stage="IC")["total"]
    separate = [
        runc_v2_losses(
            slice_outputs(outputs, i, i + 1), slice_batch(batch, i, i + 1),
            stage="IC",
        )["total"] for i in range(3)
    ]
    torch.testing.assert_close(combined, sum(separate) / 3)
    ranks = [runc_v2_losses(slice_outputs(outputs, 0, 1),
                            slice_batch(batch, 0, 1), stage="IC")["total"],
             runc_v2_losses(slice_outputs(outputs, 1, 3),
                            slice_batch(batch, 1, 3), stage="IC")["total"]]
    # DDP averages rank gradients. Multiply local means by world*n/N.
    simulated_ddp = (ranks[0] * 2 / 3 + ranks[1] * 4 / 3) / 2
    torch.testing.assert_close(combined, simulated_ddp)


def test_instance_metrics_use_instance_counts_and_separate_relative_domain():
    outputs, batch = inputs()
    batch["instance_valid"][0, 1] = False
    result = runc_v2_losses(outputs, batch)
    metrics = head_metric_sums(outputs["head"], batch, result)
    assert metrics["head_instance_count"] == 3
    assert metrics["head_centroid_relative_count"] == 2
    assert metrics["head_soft_iou_count"] == 3
    expected = metrics["per_instance"]["logsigma_x"][metrics["valid"]].sum()
    torch.testing.assert_close(metrics["head_logsigma_x_sum"], expected)


def test_zero_valid_area_has_zero_coarse_loss_and_finite_gradient():
    outputs, batch = inputs(batch_size=1)
    batch["valid_coarse"][:] = 0
    result = runc_v2_losses(outputs, batch, stage="IB")
    assert result["raw_density"] == 0
    assert result["raw_occupancy"] == 0
    assert result["raw_occupancy_dice"] == 0
    assert result["raw_j_coarse"] == 0
    result["total"].backward()
    assert torch.isfinite(outputs["density"].grad).all()
    assert torch.count_nonzero(outputs["density"].grad) == 0


def test_cpu_bfloat16_autocast_keeps_geometry_float_and_gradients_finite():
    outputs, batch = inputs(batch_size=1, pool=28, coarse=16)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = runc_v2_losses(outputs, batch, stage="IC")
    assert result["head_geometry"]["shape"].dtype == torch.float32
    assert result["coarse_geometry"]["shape"].dtype == torch.float32
    result["total"].backward()
    assert torch.isfinite(outputs["head"]["occupancy_logits"].grad).all()
    assert torch.isfinite(outputs["density"].grad).all()


@pytest.mark.parametrize("weights", [
    {"head": -1}, {"density": float("nan")}, {"j_shape": float("inf")},
    {"head": True},
])
def test_invalid_weights_rejected(weights):
    with pytest.raises(ValueError):
        RunCV2LossWeights(**weights)


def test_invalid_stage_and_coarse_geometry_rejected():
    outputs, batch = inputs()
    with pytest.raises(ValueError):
        runc_v2_losses(outputs, batch, stage="refine")
    with pytest.raises(ValueError):
        CoarseJietiLoss(pool=8)(
            outputs["density"], batch["density_target"], batch["coarse_voro"],
            batch["valid_parts"], batch["has_jt"],
        )
