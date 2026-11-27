"""Train paired queries with a single assignment to jointly annotated regions."""

import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment


class PairLoss(nn.Module):
    def __init__(
        self,
        mask_weight=1.0,
        dice_weight=1.0,
        center_weight=1.0,
        area_weight=0.25,
        presence_weight=1.0,
        empty_weight=0.1,
    ):
        super().__init__()
        self.weights = dict(
            mask=mask_weight,
            dice=dice_weight,
            center=center_weight,
            area=area_weight,
            presence=presence_weight,
        )
        self.empty_weight = empty_weight

    def forward(self, output, targets):
        logits = output["masks"]
        presence = output["presence"]
        device = logits.device
        terms = {k: logits.sum() * 0 + presence.sum() * 0 for k in self.weights}
        group_terms = [{k: logits.new_tensor(0.) for k in self.weights} for _ in range(2)]
        for batch, t in enumerate(targets):
            gt = t["masks"].to(device)
            visible = t["present"].to(device)
            mask_known = t.get("mask_supervised", t["present"] == 1).to(device)
            groups = t["groups"].to(device)
            if len(gt):
                gt = F.interpolate(gt, logits.shape[-2:], mode="nearest")
            complete = t["complete"].to(device)
            for group in (0, 1):
                before = {k: v.detach().clone() for k, v in terms.items()}
                qi = (output["groups"] == group).nonzero().flatten()
                ti = (groups == group).nonzero().flatten()
                if len(ti) > len(qi):
                    raise ValueError(
                        "More GT pairs than query slots; increase query count"
                    )
                assignment = []
                if len(ti):
                    with torch.no_grad():
                        cost = torch.zeros(len(qi), len(ti), device=device)
                        for side in (0, 1):
                            p = logits[batch, qi, side].float().flatten(1)
                            target = gt[ti, side].float().flatten(1)
                            probs = p.sigmoid()
                            bce = (
                                F.softplus(p).mean(-1)[:, None]
                                - p @ target.T / p.shape[-1]
                            )
                            dice = 1 - (2 * probs @ target.T + 1) / (
                                probs.sum(-1)[:, None] + target.sum(-1)[None] + 1
                            )
                            pos = mask_known[ti, side].float()
                            known = (visible[ti, side] >= 0).float()
                            s = presence[batch, qi, side].float()[:, None]
                            cost += (bce + dice) * pos[None] + (
                                F.softplus(s) - s * visible[ti, side].clamp_min(0)[None]
                            ) * known[None]
                        rr, cc = linear_sum_assignment(cost.cpu().numpy())
                        assignment = list(zip(rr.tolist(), cc.tolist()))
                target_presence = torch.zeros(len(qi), 2, device=device)
                weights = (
                    complete[:, group].float()[None].expand(len(qi), -1).clone()
                    * self.empty_weight
                )
                for row, col in assignment:
                    q, g = qi[row], ti[col]
                    for side in (0, 1):
                        v = visible[g, side]
                        weights[row, side] = float(v >= 0)
                        target_presence[row, side] = v.clamp_min(0)
                        if not mask_known[g, side]:
                            continue
                        pred = logits[batch, q, side].float()
                        target = gt[g, side].float()
                        prob = pred.sigmoid()
                        terms["mask"] = terms[
                            "mask"
                        ] + F.binary_cross_entropy_with_logits(pred, target)
                        terms["dice"] = (
                            terms["dice"]
                            + 1
                            - (2 * (prob * target).sum() + 1)
                            / (prob.sum() + target.sum() + 1)
                        )
                        h, w = pred.shape
                        yy, xx = torch.meshgrid(
                            torch.linspace(0, 1, h, device=device),
                            torch.linspace(0, 1, w, device=device),
                            indexing="ij",
                        )
                        pa, ga = (
                            prob.sum().clamp_min(1e-6),
                            target.sum().clamp_min(1e-6),
                        )
                        pc = torch.stack(
                            ((prob * xx).sum() / pa, (prob * yy).sum() / pa)
                        )
                        gc = torch.stack(
                            ((target * xx).sum() / ga, (target * yy).sum() / ga)
                        )
                        terms["center"] = terms["center"] + F.l1_loss(pc, gc)
                        terms["area"] = terms["area"] + torch.abs(torch.log(pa / ga))
                pl = F.binary_cross_entropy_with_logits(
                    presence[batch, qi].float(), target_presence, reduction="none"
                )
                terms["presence"] = terms["presence"] + (
                    pl * weights
                ).sum() / weights.sum().clamp_min(1)
                for k in terms:
                    group_terms[group][k] += terms[k].detach() - before[k]
        # Expected missing masks supervise location while presence remains zero.
        count = sum(
            int(t.get("mask_supervised", t["present"] == 1).sum()) for t in targets
        )
        for k in ("mask", "dice", "center", "area"):
            terms[k] = terms[k] / max(count, 1)
        terms["presence"] = terms["presence"] / max(len(targets) * 2, 1)
        total = sum(terms[k] * v for k, v in self.weights.items())
        logs = {k: float(v.detach()) for k, v in terms.items()}
        for group, name in enumerate(("BF", "JT")):
            logs[name + "/combined"] = sum(float(group_terms[group][k]) * w / (max(len(targets)*2,1) if k == "presence" else max(count,1)) for k,w in self.weights.items())
        return total, logs
