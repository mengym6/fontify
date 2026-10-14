"""结体部件的 Voronoi 划分（数据侧，精确最近墨迹，numpy only）。

P_k 只依赖 GT：每个 JT 部件实例是一张墨迹二值图，像素按"到哪个部件墨迹最近"
划分到该部件。划分是常量掩码，不参与梯度（梯度只经预测侧软前景 a）。

精确性：用 Felzenszwalb–Huttenlocher 的一维平方欧氏距离变换，沿两轴可分离，
结果是精确平方 EDT；对各有效部件求 EDT 后按像素取 argmin 得到划分。只用 numpy，
不引入 scipy。

计算顺序（与 Q6/Q9 对齐）：数据增强已完成 → 墨迹标签图按最近邻降到 224 →
在 224 网格上划分。有效部件由调用方在 448 分辨率按墨量阈值判定后传入。
"""

import numpy as np

INF = 1e20


def _dt_1d(f):
    """一维平方距离变换：D(p) = min_q f(q) + (p - q)^2。

    f: (n,) float。返回 (n,) float。标准下包络算法，精确。
    """
    n = f.shape[0]
    d = np.empty(n, dtype=np.float64)
    v = np.zeros(n, dtype=np.intp)  # 抛物线顶点横坐标
    z = np.empty(n + 1, dtype=np.float64)  # 相邻抛物线交点
    k = 0
    v[0] = 0
    z[0] = -INF
    z[1] = INF
    for q in range(1, n):
        s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
        while s <= z[k]:
            k -= 1
            s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = INF
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        d[q] = (q - v[k]) ** 2 + f[v[k]]
    return d


def _edt_sq(binary):
    """精确平方欧氏距离变换：每个像素到最近 True 像素的平方距离。

    binary: (H, W) bool。返回 (H, W) float。全 False 时返回全 INF。
    """
    f = np.where(binary, 0.0, INF)
    out = np.empty_like(f, dtype=np.float64)
    for i in range(f.shape[0]):  # 沿列（每行独立）
        out[i, :] = _dt_1d(f[i, :])
    for j in range(f.shape[1]):  # 沿行（每列独立）
        out[:, j] = _dt_1d(out[:, j])
    return out


def build_partition(label_map, valid, k_max, size):
    """由墨迹标签图构造 224 网格上的部件划分。

    label_map: (H, W) 整数，0=背景，1..k_max=部件墨迹（增强后，448 分辨率）。
    valid:     (k_max,) bool，部件是否有效（墨量达阈值，448 判定）。
    k_max:     最大部件数。
    size:      输出边长（如 224）。
    返回 (size, size) int32：0=无有效部件，1..k_max=最近的有效部件。
    """
    # 墨迹标签图按最近邻降采样到目标网格（增强后 → 降采样 → 划分）。
    H = label_map.shape[0]
    idx = (np.arange(size) * (H / size)).astype(np.intp)
    small = label_map[np.ix_(idx, idx)]
    partition = np.zeros((size, size), dtype=np.int32)
    best = np.full((size, size), INF, dtype=np.float64)
    for k in range(k_max):
        if not valid[k]:
            continue
        ink = small == (k + 1)
        if not ink.any():
            continue
        dist = _edt_sq(ink)
        closer = dist < best
        best = np.where(closer, dist, best)
        partition = np.where(closer, k + 1, partition)
    return partition


def ink_label_map(layers, k_max):
    """把 (N, H, W) 的部件墨迹层并成单张标签图 (H, W)，后出现的层覆盖先出现的。

    layers: (N, H, W)，>0 视为墨迹；只取前 k_max 层。
    """
    h, w = layers.shape[1], layers.shape[2]
    label_map = np.zeros((h, w), dtype=np.uint8)
    for k in range(min(layers.shape[0], k_max)):
        label_map[layers[k] > 0] = k + 1
    return label_map
