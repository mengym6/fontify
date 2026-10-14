"""Instance matching and lower-character J losses for dual-view run B."""

import itertools

import torch
from torch import nn
from torch.nn import functional as F

from util.jieti_loss import JietiLoss


class InstanceHeadLoss(nn.Module):
    """Four-slot matching; unmatched slots receive only negative existence."""

    def __init__(self, slots=4, class_weight=0.1, eps=1e-6):
        super().__init__()
        self.slots = slots
        self.class_weight = class_weight
        self.eps = eps

    def forward(self, outputs, occupancy, valid, classes, has_jt):
        logits = outputs["occupancy_logits"].float()
        exists = outputs["existence_logits"].float()
        class_logits = outputs["class_logits"].float()
        occupancy = occupancy.to(device=logits.device, dtype=logits.dtype)
        valid = valid.to(device=logits.device, dtype=torch.bool)
        classes = classes.to(device=logits.device, dtype=torch.long)
        active = has_jt.to(device=logits.device, dtype=torch.bool)
        batch, slots, height, width = logits.shape
        if slots != self.slots or occupancy.shape != logits.shape:
            raise ValueError("Expected four occupancy targets per character")
        if valid.shape != (batch, slots) or classes.shape != (batch, slots):
            raise ValueError("Instance validity/classes must have shape (B, 4)")
        if ((classes[valid] < 0) | (classes[valid] >= class_logits.shape[-1])).any():
            raise ValueError("Valid instance class outside classifier range")

        # The discrete assignment is label-dependent supervision, never a model
        # input. Class logits are deliberately absent from the matching cost.
        with torch.no_grad():
            pred_pair = logits[:, :, None].expand(
                batch, slots, slots, height, width
            )
            gt_pair = occupancy[:, None].expand_as(pred_pair)
            cost_bce = F.binary_cross_entropy_with_logits(
                pred_pair, gt_pair, reduction="none"
            ).mean((-1, -2))
            probabilities = pred_pair.sigmoid()
            intersection = (probabilities * gt_pair).sum((-1, -2))
            mass = probabilities.sum((-1, -2)) + gt_pair.sum((-1, -2))
            cost_dice = 1 - (2 * intersection + self.eps) / (mass + self.eps)
            cost = cost_bce + cost_dice + F.softplus(-exists)[:, :, None]
            assignment = torch.zeros(batch, slots, device=logits.device,
                                     dtype=torch.long)
            matched = torch.zeros_like(valid)
            for index in range(batch):
                gt_indices = valid[index].nonzero(as_tuple=False).flatten()
                count = gt_indices.numel()
                if count:
                    permutations = torch.tensor(
                        list(itertools.permutations(range(slots), count)),
                        device=logits.device, dtype=torch.long,
                    )
                    candidates = cost[index][permutations, gt_indices]
                    chosen = permutations[candidates.sum(-1).argmin()]
                    assignment[index, chosen] = gt_indices
                    matched[index, chosen] = True

        gather = assignment[:, :, None, None].expand_as(occupancy)
        assigned_occupancy = occupancy.gather(1, gather)
        assigned_classes = classes.gather(1, assignment).clamp_min(0)
        bce = F.binary_cross_entropy_with_logits(
            logits, assigned_occupancy, reduction="none"
        ).mean((-1, -2))
        probability = logits.sigmoid()
        intersection = (probability * assigned_occupancy).sum((-1, -2))
        mass = probability.sum((-1, -2)) + assigned_occupancy.sum((-1, -2))
        dice = 1 - (2 * intersection + self.eps) / (mass + self.eps)
        classification = F.cross_entropy(
            class_logits.transpose(1, 2), assigned_classes, reduction="none"
        )
        matched_float = matched.to(logits.dtype)
        count = matched_float.sum(-1).clamp_min(1)
        occupancy_per_word = ((0.5 * bce + 0.5 * dice)
                              * matched_float).sum(-1) / count
        class_per_word = (classification * matched_float).sum(-1) / count
        # Existence has all four slots as its supervision domain, including
        # negative unmatched slots and zero-instance characters.
        existence_per_word = F.binary_cross_entropy_with_logits(
            exists, matched_float, reduction="none"
        ).mean(-1)
        gate = active.to(logits.dtype)
        per_word = (occupancy_per_word + existence_per_word
                    + self.class_weight * class_per_word) * gate
        loss = per_word.mean()
        details = {
            "occupancy": (occupancy_per_word * gate).mean(),
            "bce": (((bce * matched_float).sum(-1) / count) * gate).mean(),
            "dice": (((dice * matched_float).sum(-1) / count) * gate).mean(),
            "existence": (existence_per_word * gate).mean(),
            "classification": (class_per_word * gate).mean(),
            "assignment": assignment,
            "matched_slots": matched,
            "per_sample": per_word,
        }
        return loss, details


class DualJietiLoss(JietiLoss):
    """New J: lower character only, per character then full-batch mean.

    Existing JietiLoss is unchanged. Statistics and mass shielding reuse its
    definitions; only the spatial domain and reduction contract differ.
    """

    def forward(self, prediction, target, voro, valid, has_jt):
        batch, _, height, _ = prediction.shape
        if height != 2 * prediction.shape[-1]:
            raise ValueError("J requires the vertical two-character canvas")
        if voro.shape != (batch, self.pool, self.pool):
            raise ValueError("Dual J partition must have shape (B, pool, pool)")
        if valid.shape != (batch, self.k_max):
            raise ValueError("Dual J valid parts must have shape (B, 4)")
        pred = prediction[:, :, height // 2:].float()
        tgt = target[:, :, height // 2:].float()
        a_pred = F.adaptive_avg_pool2d(
            self._soft_fg(pred).unsqueeze(1), (self.pool, self.pool)
        ).squeeze(1)
        a_gt = F.adaptive_avg_pool2d(
            self._soft_fg(tgt).unsqueeze(1), (self.pool, self.pool)
        ).squeeze(1)
        voro = voro.to(device=prediction.device, dtype=torch.long)
        valid = valid.to(device=prediction.device, dtype=torch.bool)
        active = has_jt.to(device=prediction.device, dtype=torch.bool)
        part = torch.stack([
            (voro == index + 1).float() for index in range(self.k_max)
        ], 1)
        mp, cxp, cyp, sxp, syp = self._part_stats(a_pred, part)
        mg, cxg, cyg, sxg, syg = self._part_stats(a_gt, part)
        base_valid = valid & (mg > self.eps) & active[:, None]
        pred_ok = mp >= self.pred_mass_ratio * mg
        shield_count = (base_valid & ~pred_ok).float().sum()
        pv = (base_valid & pred_ok).float()
        count = pv.sum(-1).clamp_min(1)
        logsigma = ((torch.log(sxp + self.eps)
                     - torch.log(sxg + self.eps)).abs()
                    + (torch.log(syp + self.eps)
                       - torch.log(syg + self.eps)).abs())
        per_logsigma = (logsigma * pv).sum(-1) / count

        mpv, mgv = mp * pv, mg * pv
        center_px = (cxp * mpv).sum(-1) / (mpv.sum(-1) + self.eps)
        center_py = (cyp * mpv).sum(-1) / (mpv.sum(-1) + self.eps)
        center_gx = (cxg * mgv).sum(-1) / (mgv.sum(-1) + self.eps)
        center_gy = (cyg * mgv).sum(-1) / (mgv.sum(-1) + self.eps)
        dx = ((cxp - center_px[:, None])
              - (cxg - center_gx[:, None]))
        dy = ((cyp - center_py[:, None])
              - (cyg - center_gy[:, None]))
        distance = torch.sqrt(dx.square() + dy.square() + self.eps)
        character = (voro > 0).float()
        xcoord = self.coord.view(1, 1, self.pool)
        ycoord = self.coord.view(1, self.pool, 1)
        char_mass = (a_gt * character).sum((-1, -2))
        gcx = (a_gt * character * xcoord).sum((-1, -2))
        gcx = gcx / (char_mass + self.eps)
        gcy = (a_gt * character * ycoord).sum((-1, -2))
        gcy = gcy / (char_mass + self.eps)
        variance = (a_gt * character
                    * ((xcoord - gcx[:, None, None]).square()
                       + (ycoord - gcy[:, None, None]).square()))
        variance = variance.sum((-1, -2)) / (char_mass + self.eps)
        sigma = torch.sqrt(variance + self.eps).detach()
        per_centroid = (distance / (sigma[:, None] + self.eps) * pv)
        per_centroid = per_centroid.sum(-1) / count
        per_centroid = per_centroid * (pv.sum(-1) >= 2).float()

        desc_pred = self._descriptor(
            a_pred, part, cxp.detach(), cyp.detach(),
            sxp.detach(), syp.detach(),
        )
        desc_gt = self._descriptor(
            a_gt, part, cxg.detach(), cyg.detach(),
            sxg.detach(), syg.detach(),
        )
        shape = (desc_pred - desc_gt).abs().sum((-1, -2))
        per_shape = (shape * pv).sum(-1) / count
        parts = {
            "centroid": per_centroid.mean(),
            "logsigma": per_logsigma.mean(),
            "shape": per_shape.mean(),
            "per_sample": (self.w_centroid * per_centroid
                           + self.w_logsigma * per_logsigma
                           + self.w_shape * per_shape),
        }
        loss = parts["per_sample"].mean()
        return loss, parts, shield_count


# Public name used by evaluation/protocol tooling.
LowerJietiLoss = DualJietiLoss
