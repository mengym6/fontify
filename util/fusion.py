"""encoder 中电脑字路与风格字路的融合（纯函数，只用切片和算术，torch / numpy 都能用）。"""


def fuse_streams(x, lam=0.5):
    """x 沿第 0 维前半是电脑字路、后半是风格字路，返回 (1-λ)·前半 + λ·后半。"""
    n = x.shape[0] // 2
    if lam == 0.5:
        # 保留原式：(1-λ)·x+λ·y 与 (x+y)·0.5 在浮点上不保证逐位相同
        return (x[:n] + x[n:]) * 0.5
    return x[:n] * (1 - lam) + x[n:] * lam
