"""结体（Jieti）可微结构损失：相对质心、尺度 log σ、形状描述子。

只在 JT 样本上生效（is_jt），在拼接图的上下两半分别计算；门控、软前景、
计算分辨率与归一化见 PROGRESS.md 第 3 节 T1 的 Q5–Q9。

部件区域 P_k 由数据侧用 GT 各 JT 部件的墨迹做 Voronoi 划分得到（整数标签图，
detach，不回传梯度）。本模块只在 P_k 上统计可微软前景 a 的质量分布，梯度经
a 回传到整个生成器（结体项没有专属参数）。

三项（都对 pred 的软前景与 GT 的软前景分别求统计量再比较）：
- 相对质心：部件质心减整字质心，再除以 GT 整字 σ（detach）。关系项，要求该半图
  有效部件数 K ≥ 2。
- 尺度 log σ：部件沿 x/y 两轴的质量二阶矩的 log 差，捕捉大小/拉伸。
- 形状描述子：按部件自身质心和 σ（detach）归一化后的 bins×bins 软质量分布，
  对平移/缩放不变，捕捉旋转/剪切。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class JietiLoss(nn.Module):
    def __init__(self, k_max=4, pool=224, soft_fg="linear", sigmoid_scale=10.0,
                 shape_bins=8, shape_range=3.0, eps=1e-6, pred_mass_ratio=0.1,
                 w_centroid=1.0, w_logsigma=1.0, w_shape=1.0):
        super().__init__()
        self.k_max = k_max
        self.pool = pool
        self.soft_fg = soft_fg
        self.sigmoid_scale = sigmoid_scale
        self.shape_bins = shape_bins
        self.shape_range = shape_range
        self.eps = eps
        self.pred_mass_ratio = pred_mass_ratio
        # 三项相对系数：由 check_loss_gradient_balance.py 标定出的固定常数，所有
        # 搜索组共用（排序指标 S 的 J 也用这组常数），与各组自己的 w 无关。
        self.w_centroid = w_centroid
        self.w_logsigma = w_logsigma
        self.w_shape = w_shape
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        # 像素坐标（0..pool-1），注册为 buffer 以跟随 device/dtype。
        self.register_buffer("coord", torch.arange(pool).float())

    def _soft_fg(self, img_norm):
        """ImageNet 归一化图 → 去归一化灰度 g → 软前景 a∈[0,1]。

        img_norm: (M,3,H,W) → 返回 (M,H,W)。
        """
        img = img_norm * self.std + self.mean
        g = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]  # (M,H,W)
        if self.soft_fg == "sigmoid":
            # sigmoid 可选项：调试时与线性比较梯度（Q5）。
            return torch.sigmoid(self.sigmoid_scale * (0.5 - g))
        return (1.0 - g).clamp(0.0, 1.0)

    def _tri(self, bin_coord):
        """三角（线性）软分箱权重。

        bin_coord: (...,R) 每个像素在 [0,bins) 的分箱坐标。
        返回 (...,R,bins)：到各箱中心(c+0.5)距离 1 以内线性衰减。
        """
        centers = torch.arange(self.shape_bins, device=bin_coord.device,
                               dtype=bin_coord.dtype) + 0.5
        dist = (bin_coord.unsqueeze(-1) - centers).abs()
        return (1.0 - dist).clamp(0.0, 1.0)

    def _descriptor(self, a, part, cx, cy, sx, sy):
        """按部件自身质心/σ 归一化的 bins×bins 软质量分布。

        a:    (M,R,R) 软前景；part: (M,K,R,R) 部件掩码；
        cx,cy,sx,sy: (M,K) 该部件质心/标准差（调用方已 detach）。
        返回 (M,K,bins,bins)，每部件在 bins² 上和为 1。
        """
        R = self.pool
        bins = self.shape_bins
        rng = self.shape_range
        idx = self.coord.view(1, 1, R)  # (1,1,R)
        # 归一化坐标 u=(idx-c)/σ 映射到分箱单位 (u+rng)/(2rng)*bins。
        bx = ((idx - cx.unsqueeze(-1)) / (sx.unsqueeze(-1) + self.eps)
              + rng) / (2.0 * rng) * bins  # (M,K,R)  宽度方向
        by = ((idx - cy.unsqueeze(-1)) / (sy.unsqueeze(-1) + self.eps)
              + rng) / (2.0 * rng) * bins  # (M,K,R)  高度方向
        wx = self._tri(bx)  # (M,K,R,bins)
        wy = self._tri(by)  # (M,K,R,bins)
        mass = a.unsqueeze(1) * part  # (M,K,R,R)  [y,x]
        # hist[by,bx] = Σ_y Σ_x mass[y,x]·wy[y,by]·wx[x,bx]
        tmp = torch.einsum("mkyx,mkxb->mkyb", mass, wx)  # (M,K,R,bins_x)
        hist = torch.einsum("mkyb,mkyc->mkcb", tmp, wy)  # (M,K,bins_y,bins_x)
        total = hist.sum((-1, -2), keepdim=True)
        return hist / (total + self.eps)

    def _part_stats(self, a, part):
        """部件一阶/二阶矩。

        a: (M,R,R)；part: (M,K,R,R)。
        返回 m,(cx,cy),(sx,sy)，形状均 (M,K)。
        """
        R = self.pool
        X = self.coord.view(1, 1, 1, R)  # 宽度索引 (列)
        Y = self.coord.view(1, 1, R, 1)  # 高度索引 (行)
        am = a.unsqueeze(1) * part  # (M,K,R,R)
        m = am.sum((2, 3))  # (M,K)
        cx = (am * X).sum((2, 3)) / (m + self.eps)
        cy = (am * Y).sum((2, 3)) / (m + self.eps)
        vx = (am * (X - cx[..., None, None]) ** 2).sum((2, 3)) / (m + self.eps)
        vy = (am * (Y - cy[..., None, None]) ** 2).sum((2, 3)) / (m + self.eps)
        sx = torch.sqrt(vx + self.eps)
        sy = torch.sqrt(vy + self.eps)
        return m, cx, cy, sx, sy

    def forward(self, composite, target, voro, valid, is_jt):
        """
        composite: (N,3,896,448) 拼接图（遮盖区用预测、可见区用 GT），ImageNet 归一化。
        target:    (N,3,896,448) GT 拼接图。
        voro:      (N,2,R,R) long，0=背景，1..k_max=部件；dim1=[上(风格字),下(目标字)]。
                   已在数据侧按"增强→降到 224→精确最近墨迹划分"构造，是常量掩码。
        valid:     (N,2,k_max) bool，GT 部件增强后（448 分辨率）墨量达阈值。
        is_jt:     (N,) bool。

        返回 (J, 分项 dict, shield_count)。无 JT 样本时 J=0（仍保留在计算图里，
        满足 DDP static_graph）。
        """
        N = composite.shape[0]
        R = self.pool
        half = composite.shape[2] // 2
        W = composite.shape[3]
        # 拆上下两半并展平到 M=N*2 的批维，统一处理。
        comp = torch.stack([composite[:, :, :half], composite[:, :, half:]], dim=1)
        tgt = torch.stack([target[:, :, :half], target[:, :, half:]], dim=1)
        M = N * 2
        comp = comp.reshape(M, 3, half, W).float()
        tgt = tgt.reshape(M, 3, half, W).float()
        a_pred = self._soft_fg(comp).unsqueeze(1)  # (M,1,448,448)
        a_gt = self._soft_fg(tgt).unsqueeze(1)
        # 池化到 R×R（Q6：上下两半各平均池化到 224×224）。
        a_pred = F.adaptive_avg_pool2d(a_pred, (R, R)).squeeze(1)  # (M,R,R)
        a_gt = F.adaptive_avg_pool2d(a_gt, (R, R)).squeeze(1)
        # 划分已是 R×R 常量标签（数据侧构造）。
        voro = voro.reshape(M, R, R).long()
        valid = valid.reshape(M, self.k_max)
        half_active = is_jt.unsqueeze(1).expand(N, 2).reshape(M)  # (M,) bool

        part = torch.stack(
            [(voro == k + 1).float() for k in range(self.k_max)], dim=1
        )  # (M,K,R,R)
        mpred, cxp, cyp, sxp, syp = self._part_stats(a_pred, part)
        mgt, cxg, cyg, sxg, syg = self._part_stats(a_gt, part)

        # 部件有效性：GT 墨量达标（valid，由数据侧按 448 分辨率判定）且 GT 有质量；
        # 预测侧墨量低于 GT 的 pred_mass_ratio 时屏蔽该部件并计数（Q9）。
        gt_has_mass = mgt > self.eps
        pred_ok = mpred >= self.pred_mass_ratio * mgt
        base_valid = valid & gt_has_mass & half_active[:, None]
        shield_count = (base_valid & (~pred_ok)).float().sum()
        part_valid = base_valid & pred_ok  # (M,K)
        pv = part_valid.float()
        n_parts = pv.sum().clamp_min(1.0)

        # --- 尺度 log σ（逐部件，K≥1）---
        logsig = ((torch.log(sxp + self.eps) - torch.log(sxg + self.eps)).abs()
                  + (torch.log(syp + self.eps) - torch.log(syg + self.eps)).abs())
        loss_logsigma = (logsig * pv).sum() / n_parts

        # --- 相对质心（关系项，要求该半图有效部件数 ≥ 2）---
        mp_v = mpred * pv
        mg_v = mgt * pv
        cgx_p = (cxp * mp_v).sum(1) / (mp_v.sum(1) + self.eps)  # (M,) 整字质心(pred)
        cgy_p = (cyp * mp_v).sum(1) / (mp_v.sum(1) + self.eps)
        cgx_g = (cxg * mg_v).sum(1) / (mg_v.sum(1) + self.eps)  # (M,) 整字质心(gt)
        cgy_g = (cyg * mg_v).sum(1) / (mg_v.sum(1) + self.eps)
        rxp = cxp - cgx_p[:, None]
        ryp = cyp - cgy_p[:, None]
        rxg = cxg - cgx_g[:, None]
        ryg = cyg - cgy_g[:, None]
        d = torch.sqrt((rxp - rxg) ** 2 + (ryp - ryg) ** 2 + self.eps)  # (M,K)
        # GT 整字 σ：整半图 GT 软前景二阶矩，detach（Q7）。
        allp = (voro > 0).float()  # (M,R,R)
        X = self.coord.view(1, 1, R)
        Y = self.coord.view(1, R, 1)
        m_all = (a_gt * allp).sum((1, 2))  # (M,)
        gcx = (a_gt * allp * X).sum((1, 2)) / (m_all + self.eps)
        gcy = (a_gt * allp * Y).sum((1, 2)) / (m_all + self.eps)
        var_all = (a_gt * allp
                   * ((X - gcx[:, None, None]) ** 2
                      + (Y - gcy[:, None, None]) ** 2)).sum((1, 2)) / (m_all + self.eps)
        sigma_char = torch.sqrt(var_all + self.eps).detach()  # (M,)
        d_norm = d / (sigma_char[:, None] + self.eps)
        half_valid_count = pv.sum(1)  # (M,)
        rel_active = (half_valid_count >= 2).float()  # (M,)
        per_half = (d_norm * pv).sum(1) / half_valid_count.clamp_min(1.0)
        loss_centroid = (per_half * rel_active).sum() / rel_active.sum().clamp_min(1.0)

        # --- 形状描述子（逐部件，各用自身 detach 的质心/σ 作参考系）---
        desc_p = self._descriptor(a_pred, part, cxp.detach(), cyp.detach(),
                                  sxp.detach(), syp.detach())
        desc_g = self._descriptor(a_gt, part, cxg.detach(), cyg.detach(),
                                  sxg.detach(), syg.detach())
        shape_err = (desc_p - desc_g).abs().sum((-1, -2))  # (M,K)
        loss_shape = (shape_err * pv).sum() / n_parts

        J = (self.w_centroid * loss_centroid
             + self.w_logsigma * loss_logsigma
             + self.w_shape * loss_shape)
        parts = {
            "centroid": loss_centroid,
            "logsigma": loss_logsigma,
            "shape": loss_shape,
        }
        return J, parts, shield_count
