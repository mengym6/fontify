"""Direct GT-ID supervision, matchability rejection and excess-overlap control."""

import torch
from torch import nn
from torch.nn import functional as F


class CorrespondenceLoss(nn.Module):
    def __init__(
        self, mask=1.0, dice=1.0, center=1.0, area=0.25, match=1.0, duplicate=0.1
    ):
        super().__init__()
        self.weights = dict(
            mask=mask,
            dice=dice,
            center=center,
            area=area,
            match=match,
            duplicate=duplicate,
        )

    def forward(self, outputs, targets):
        if len(outputs) != len(targets):
            raise ValueError("Output/target batch mismatch")
        zero = sum(
            o["mask_logits"].sum() * 0 + o["match_logits"].sum() * 0 for o in outputs
        )
        terms = {k: zero for k in self.weights}
        matched_sides = 0
        decisions = 0
        dup_batches = 0
        for o, t in zip(outputs, targets):
            p = o["mask_logits"]
            labels = t["matchable"].to(p.device)
            known = labels >= 0
            if p.shape[0] != len(labels):
                raise ValueError("One output required per GT anchor")
            if known.any():
                # Balance available positive/negative classes within the image.
                parts = []
                for value in (0, 1):
                    selected = labels == value
                    if selected.any():
                        parts.append(
                            F.binary_cross_entropy_with_logits(
                                o["match_logits"][selected], labels[selected].float()
                            )
                        )
                terms["match"] = terms["match"] + torch.stack(parts).mean()
                decisions += 1
            good = labels == 1
            if not good.any():
                continue
            pred = p[good].float()
            gt = F.interpolate(
                t["gen_masks"][good.cpu(), None].to(p.device),
                p.shape[-2:],
                mode="area",
            )[:, 0]
            if (gt.flatten(1).sum(1) == 0).any():
                raise ValueError("Matchable mask vanished at output resolution")
            prob = pred.sigmoid()
            n = len(pred)
            matched_sides += n
            terms["mask"] = (
                terms["mask"]
                + F.binary_cross_entropy_with_logits(pred, gt, reduction="none")
                .mean((1, 2))
                .sum()
            )
            pa = prob.sum((1, 2))
            ga = gt.sum((1, 2))
            terms["dice"] = (
                terms["dice"]
                + (1 - (2 * (prob * gt).sum((1, 2)) + 1) / (pa + ga + 1)).sum()
            )
            h, w = pred.shape[-2:]
            yy, xx = torch.meshgrid(
                torch.linspace(0, 1, h, device=p.device),
                torch.linspace(0, 1, w, device=p.device),
                indexing="ij",
            )
            pc = torch.stack(
                (
                    (prob * xx).sum((1, 2)) / pa.clamp_min(1e-6),
                    (prob * yy).sum((1, 2)) / pa.clamp_min(1e-6),
                ),
                1,
            )
            gc = torch.stack(
                ((gt * xx).sum((1, 2)) / ga, (gt * yy).sum((1, 2)) / ga), 1
            )
            terms["center"] = terms["center"] + (pc - gc).abs().mean(1).sum()
            terms["area"] = (
                terms["area"] + torch.log(pa.clamp_min(1e-6) / ga).abs().sum()
            )
            groups = t["groups"][good.cpu()].to(p.device)
            pairs = (groups[:, None] == groups[None]) & ~torch.eye(
                n, dtype=torch.bool, device=p.device
            )
            if pairs.any():
                a = prob.flatten(1)
                b = gt.flatten(1)
                pi = a @ a.T
                gi = b @ b.T
                piou = pi / (a.sum(1)[:, None] + a.sum(1)[None] - pi).clamp_min(1e-6)
                giou = gi / (b.sum(1)[:, None] + b.sum(1)[None] - gi).clamp_min(1e-6)
                terms["duplicate"] = (
                    terms["duplicate"] + F.relu(piou - giou - 0.05)[pairs].mean()
                )
                dup_batches += 1
        for k in ("mask", "dice", "center", "area"):
            terms[k] = terms[k] / max(matched_sides, 1)
        terms["match"] = terms["match"] / max(decisions, 1)
        terms["duplicate"] = terms["duplicate"] / max(dup_batches, 1)
        total = sum(v * self.weights[k] for k, v in terms.items())
        return total, {k: float(v.detach()) for k, v in terms.items()}
