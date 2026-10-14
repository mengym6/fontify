"""Run B matching, per-character reductions and gradient contracts on CPU."""

import itertools

import pytest
import torch
from torch.nn import functional as F

from util.dual_loss import DualJietiLoss, InstanceHeadLoss
from util.jieti_loss import JietiLoss


def outputs(batch=1):
    torch.manual_seed(42)
    return {
        "occupancy_logits": torch.randn(batch, 4, 4, 4, requires_grad=True),
        "existence_logits": torch.randn(batch, 4, requires_grad=True),
        "class_logits": torch.randn(batch, 4, 20, requires_grad=True),
    }


@pytest.mark.parametrize("count", range(5))
def test_matching_exhaustive_and_class_independent(count):
    prediction = outputs()
    targets = torch.eye(4).reshape(4, 4, 1).expand(4, 4, 4).unsqueeze(0)
    valid = torch.arange(4).unsqueeze(0) < count
    classes = torch.tensor([[1, 2, 3, 4]])
    criterion = InstanceHeadLoss()
    _, details = criterion(prediction, targets, valid, classes,
                           torch.ones(1, dtype=torch.bool))
    matches = details["matched_slots"][0]
    assert matches.sum().item() == count
    if count:
        logits = prediction["occupancy_logits"][0].detach()
        probability = logits.sigmoid()
        matrix = torch.empty(4, count)
        for slot in range(4):
            for target_index in range(count):
                target = targets[0, target_index]
                bce = F.binary_cross_entropy_with_logits(logits[slot], target)
                dice = 1 - ((2 * (probability[slot] * target).sum() + 1e-6)
                            / (probability[slot].sum() + target.sum() + 1e-6))
                matrix[slot, target_index] = (
                    bce + dice
                    + F.softplus(-prediction["existence_logits"][0, slot])
                )
        permutations = list(itertools.permutations(range(4), count))
        costs = [sum(matrix[slot, index] for index, slot in enumerate(order))
                 for order in permutations]
        best = permutations[torch.stack(costs).argmin()]
        assert matches.nonzero().flatten().tolist() == sorted(best)
        for index, slot in enumerate(best):
            assert details["assignment"][0, slot].item() == index
    changed = {**prediction, "class_logits": -100 * prediction["class_logits"]}
    _, new_details = criterion(changed, targets, valid, classes,
                               torch.ones(1, dtype=torch.bool))
    assert torch.equal(details["assignment"], new_details["assignment"])
    assert torch.equal(details["matched_slots"], new_details["matched_slots"])


@pytest.mark.parametrize("count", range(5))
def test_unmatched_slots_only_receive_existence_supervision(count):
    prediction = outputs()
    valid = torch.arange(4).unsqueeze(0) < count
    loss, details = InstanceHeadLoss()(
        prediction, torch.zeros(1, 4, 4, 4), valid,
        torch.ones(1, 4, dtype=torch.long), torch.ones(1, dtype=torch.bool),
    )
    loss.backward()
    unmatched = ~details["matched_slots"][0]
    assert torch.count_nonzero(
        prediction["occupancy_logits"].grad[0, unmatched]
    ) == 0
    assert torch.count_nonzero(
        prediction["class_logits"].grad[0, unmatched]
    ) == 0
    assert (prediction["existence_logits"].grad[0, unmatched] > 0).all()


def test_head_no_jt_has_zero_gradients_and_full_batch_mean():
    prediction = outputs(batch=2)
    occupancy = torch.rand(2, 4, 4, 4)
    valid = torch.ones(2, 4, dtype=torch.bool)
    classes = torch.zeros(2, 4, dtype=torch.long)
    loss, _ = InstanceHeadLoss()(
        prediction, occupancy, valid, classes, torch.tensor([True, False]),
    )
    single, _ = InstanceHeadLoss()(
        {key: value[:1] for key, value in prediction.items()},
        occupancy[:1], valid[:1], classes[:1], torch.tensor([True]),
    )
    torch.testing.assert_close(loss, single / 2)
    loss.backward()
    for tensor in prediction.values():
        assert tensor.grad is not None
        assert torch.count_nonzero(tensor.grad[1]) == 0


def j_inputs():
    torch.manual_seed(4)
    image = torch.rand(2, 3, 16, 8) * 0.7
    prediction = (image + torch.randn_like(image) * 0.12).requires_grad_()
    voro = torch.ones(2, 8, 8, dtype=torch.long)
    voro[:, :, 4:] = 2
    valid = torch.tensor([[True, True, False, False]]).expand(2, -1)
    return prediction, image, voro, valid


def test_j_only_lower_character_and_full_batch_mean():
    prediction, target, voro, valid = j_inputs()
    criterion = DualJietiLoss(pool=8, shape_bins=4)
    loss, _, _ = criterion(prediction, target, voro, valid,
                            torch.tensor([True, False]))
    single, _, _ = criterion(prediction[:1], target[:1], voro[:1], valid[:1],
                              torch.tensor([True]))
    torch.testing.assert_close(loss, single / 2)
    loss.backward()
    assert torch.count_nonzero(prediction.grad[:, :, :8]) == 0
    assert torch.count_nonzero(prediction.grad[1]) == 0
    assert torch.count_nonzero(prediction.grad[0, :, 8:]) > 0
    changed_top = prediction.detach().clone()
    changed_top[:, :, :8] += 100
    changed, _, _ = criterion(changed_top, target, voro, valid,
                               torch.tensor([True, False]))
    torch.testing.assert_close(loss, changed, rtol=0, atol=0)


def test_single_character_matches_old_j_lower_statistics():
    prediction, target, voro, valid = j_inputs()
    prediction, target, voro, valid = (
        value[:1] for value in (prediction, target, voro, valid)
    )
    new = DualJietiLoss(pool=8, shape_bins=4)
    old = JietiLoss(pool=8, shape_bins=4)
    actual, actual_parts, _ = new(prediction, target, voro, valid,
                                 torch.tensor([True]))
    old_valid = torch.stack((torch.zeros_like(valid), valid), 1)
    expected, expected_parts, _ = old(
        prediction, target, torch.stack((voro, voro), 1), old_valid,
        torch.tensor([True]),
    )
    torch.testing.assert_close(actual, expected)
    for name in ("centroid", "logsigma", "shape"):
        torch.testing.assert_close(actual_parts[name], expected_parts[name])


def test_j_microbatch_mean_matches_combined_batch():
    prediction, target, voro, valid = j_inputs()
    criterion = DualJietiLoss(pool=8, shape_bins=4)
    active = torch.ones(2, dtype=torch.bool)
    combined = criterion(prediction, target, voro, valid, active)[0]
    separate = sum(criterion(prediction[i:i + 1], target[i:i + 1],
                             voro[i:i + 1], valid[i:i + 1], active[i:i + 1])[0]
                   for i in range(2)) / 2
    torch.testing.assert_close(combined, separate)
