"""tools/export_fusion_sweep.py 与 util/fusion.py 的 numpy/PIL 单测（可直接 python 运行）。"""

import ast
import csv
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.export_fusion_sweep as fs  # noqa: E402
import tools.export_side_by_side as sbs  # noqa: E402
from tools.export_side_by_side import GAP  # noqa: E402
from util.fusion import fuse_streams  # noqa: E402

LAMBDAS = [0.3, 0.5, 0.7]
S_BASE = {"L1_JT": 0.8141332842054821, "L1_BF": 0.6168024233587972, "J": 0.3065800441162927}


def _font_path():
    """本地用 matplotlib 自带的 DejaVuSans（与服务器默认字体同一款），找不到就用服务器路径。"""
    try:
        import matplotlib
        p = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
        if p.is_file():
            return str(p)
    except ImportError:
        pass
    return fs.DEFAULT_FONT


FONT = _font_path()


def _solid(v, cell):
    return np.full((cell, cell, 3), v, dtype=np.uint8)


def _recs(n_jt, n_bf, seed=0):
    kinds = ["JT"] * n_jt + ["BF"] * n_bf
    np.random.default_rng(seed).shuffle(kinds)
    return [{"idx": i, "kind": k, "writer": "W", "char": chr(0x4E00 + i), "ref_path": f"r{i}",
             "query_path": f"q{i}"} for i, k in enumerate(kinds)]


# ---------------------------------------------------------------------------
# 融合公式
# ---------------------------------------------------------------------------

def _orig(x):
    """改动前 models_train.forward_encoder 里的原表达式。"""
    return (x[:x.shape[0] // 2] + x[x.shape[0] // 2:]) * 0.5


def test_fuse_default_bitwise_equal_original():
    rng = np.random.default_rng(0)
    for dtype in (np.float32, np.float64, np.float16):
        x = (rng.standard_normal((4, 7, 5, 6)) * 3).astype(dtype)
        for got in (fuse_streams(x), fuse_streams(x, 0.5)):
            assert got.dtype == x.dtype and got.shape == (2, 7, 5, 6)
            assert np.array_equal(got.view(np.uint8), _orig(x).view(np.uint8))
    # 边界值：上溢时原式得 inf，(1-λ)·x+λ·y 得有限值，所以 λ=0.5 必须走原式
    big = np.full((2, 3), np.finfo(np.float32).max, dtype=np.float32)
    with np.errstate(over="ignore"):
        assert np.isinf(_orig(big)).all() and np.isinf(fuse_streams(big)).all()
        assert np.isfinite(big[:1] * 0.5 + big[1:] * 0.5).all()


def test_fuse_other_lambdas():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((6, 4, 3)).astype(np.float32)
    a, b = x[:3].astype(np.float64), x[3:].astype(np.float64)
    for lam in (0.3, 0.7, 0.0, 1.0):
        got = fuse_streams(x, lam)
        assert got.shape == (3, 4, 3) and got.dtype == np.float32
        np.testing.assert_allclose(got, (1 - lam) * a + lam * b, rtol=1e-6, atol=1e-6)
    # 前半是电脑字路、后半是风格字路：λ=0 只剩前半，λ=1 只剩后半
    assert np.array_equal(fuse_streams(x, 0.0), x[:3])
    assert np.array_equal(fuse_streams(x, 1.0), x[3:])
    # 0.3 与 0.7 互为镜像
    np.testing.assert_allclose(fuse_streams(x, 0.3), fuse_streams(np.concatenate([x[3:], x[:3]]), 0.7),
                               rtol=1e-6, atol=1e-7)
    assert not np.allclose(fuse_streams(x, 0.3), fuse_streams(x, 0.5))


def test_models_train_uses_fuse_streams_at_merge():
    """forward_encoder 在 merge_idx（=2）处调用 fuse_streams，λ 取 fusion_lambda，默认 0.5。"""
    tree = ast.parse((ROOT / "models_train.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Fontify")
    fe = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward_encoder")
    src = ast.get_source_segment((ROOT / "models_train.py").read_text(encoding="utf-8"), fe)
    assert "merge_idx = 2" in src
    assert 'x = fuse_streams(x, getattr(self, "fusion_lambda", 0.5))' in src
    assert "* 0.5" not in src  # 原表达式只在 util/fusion.py 里保留一份
    # 训练相关的 __init__ 里没有注册 fusion_lambda（不进 state_dict）
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    assert "fusion_lambda" not in ast.dump(init)


def _numpy_encoder():
    """用 AST 取出真实的 forward_encoder，配 numpy 版的最小模块执行（本地无 torch）。"""
    import copy
    import types
    tree = ast.parse((ROOT / "models_train.py").read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Fontify")
    node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward_encoder")
    ns = {"torch": types.SimpleNamespace(cat=lambda t, dim=0: np.concatenate(t, axis=dim))}
    exec(compile(ast.Module(body=[copy.deepcopy(node)], type_ignores=[]), "fe", "exec"), ns)

    class Emb:  # 模拟 patch_embed 输出：size() 与加法、乘法
        def __init__(self, a):
            self.a = a

        def size(self):
            return self.a.shape

        def __add__(self, o):
            return self.a + o

        def __mul__(self, o):
            return self.a * o

    class Mask:  # 模拟 bool_masked_pos 的 unsqueeze/type_as/reshape
        def __init__(self, a):
            self.a = a

        def unsqueeze(self, d):
            return Mask(np.expand_dims(self.a, d))

        def type_as(self, t):
            return Mask(self.a.astype(t.dtype))

        def reshape(self, *s):
            return self.a.reshape(*s)

    rng = np.random.default_rng(0)
    ws = [rng.standard_normal((3, 3)).astype(np.float32) for _ in range(12)]
    m = types.SimpleNamespace(
        patch_embed=lambda im: Emb(im.transpose(0, 2, 3, 1)), depth=12,
        blocks=[(lambda w: (lambda x: np.tanh(x @ w)))(w) for w in ws], norm=lambda x: x,
        mask_token=types.SimpleNamespace(
            expand=lambda b, h, w, _: np.full((b, h, w, 3), 0.1, np.float32)),
        segment_token_x=rng.standard_normal((1, 1, 1, 3)).astype(np.float32),
        segment_token_y=rng.standard_normal((1, 1, 1, 3)).astype(np.float32), pos_embed=None)
    imgs = rng.standard_normal((2, 3, 8, 4)).astype(np.float32)
    tgts = rng.standard_normal((2, 3, 8, 4)).astype(np.float32)
    bm = np.zeros((2, 32), bool)
    bm[:, 16:] = True

    def run(lam=None):
        if lam is None:
            m.__dict__.pop("fusion_lambda", None)
        else:
            m.fusion_lambda = lam
        return ns["forward_encoder"](m, imgs, tgts, Mask(bm))

    def reference(lam):
        """按改动前的数据流手写：block idx 2 之后融合，λ=0.5 用原式。"""
        w = bm[..., None].astype(np.float32).reshape(-1, 8, 4, 1)
        x = imgs.transpose(0, 2, 3, 1) + m.segment_token_x
        y = tgts.transpose(0, 2, 3, 1) * (1 - w) + 0.1 * w + m.segment_token_y
        h = np.concatenate([x, y], 0)
        out = []
        for i, blk in enumerate(m.blocks):
            h = blk(h)
            if i == 2:
                h = (h[:2] + h[2:]) * 0.5 if lam == 0.5 else h[:2] * (1 - lam) + h[2:] * lam
            if i in (2, 5, 8, 11):
                out.append(h)
        return out

    return run, reference


def test_forward_encoder_default_bitwise_and_lambda_effect():
    run, reference = _numpy_encoder()
    same = lambda a, b: all(np.array_equal(p.view(np.uint8), q.view(np.uint8)) for p, q in zip(a, b))
    ref = reference(0.5)
    assert len(ref) == 4 and ref[0].shape == (2, 8, 4, 3)  # 融合后 batch 减半
    assert same(run(), ref) and same(run(0.5), ref)        # 无属性 / 0.5 都与原式逐位相同
    for lam in (0.3, 0.7):
        got = run(lam)
        for p, q in zip(got, reference(lam)):
            np.testing.assert_allclose(p, q, rtol=1e-6, atol=1e-6)
        # 抽头 idx 2 在融合之后，所以 4 个抽头都随 λ 改变
        assert all(not np.allclose(p, q) for p, q in zip(got, ref))


# ---------------------------------------------------------------------------
# 图：列布局、列头、分页、index
# ---------------------------------------------------------------------------

def test_column_heads_and_lambdas():
    assert fs.column_heads(LAMBDAS) == ["ref (upper GT)", "GT (lower)", "λ=0.3", "λ=0.5", "λ=0.7"]
    assert fs.lam_key(0.5) == "0.5" and fs.lam_key(0.30000000000000004) == "0.3"
    fs.check_lambdas(LAMBDAS)
    for bad in ([0.3, 0.7], [0.5, 0.5], [0.5, 1.2], [-0.1, 0.5]):
        try:
            fs.check_lambdas(bad)
            raise AssertionError(f"{bad} 应报错")
        except ValueError:
            pass


def test_font_has_lambda_glyph():
    """默认字体能画出 λ：与 DejaVu 里没有的字（汉字，画成缺字框）不同。"""
    from PIL import ImageDraw
    f = ImageFont.truetype(FONT, 24)

    def draw(s):
        im = Image.new("L", (40, 40), 255)
        ImageDraw.Draw(im).text((5, 5), s, font=f, fill=0)
        return np.asarray(im)

    lam = draw("λ")
    assert (lam < 128).any()
    assert not np.array_equal(lam, draw("字")) and not np.array_equal(lam, draw("l"))
    # 对照：两个不同的汉字在 DejaVu 里都是同一个缺字框
    assert np.array_equal(draw("字"), draw("佛"))


def test_compose_lambda_layout():
    cell = 40
    heads = fs.column_heads(LAMBDAS)
    rows = [[_solid(10 * (r * 5 + c + 1), cell) for c in range(5)] for r in range(2)]
    hh = sbs.header_height(cell)
    img = sbs.compose(rows[:1], heads, cell, font_path=FONT)
    assert img.shape == (hh + cell, 5 * cell + 4 * GAP, 3)
    for c in range(5):
        x = c * (cell + GAP)
        assert (img[hh:hh + cell, x:x + cell] == 10 * (c + 1)).all()
    # 列头：第 c 列的列头区域画的就是 heads[c]（换成别的 λ 后只有该列变化）
    alt = sbs.compose(rows[:1], heads[:2] + ["λ=0.3", "λ=0.9", "λ=0.7"], cell, font_path=FONT)
    for c in range(5):
        x = c * (cell + GAP)
        same = (img[:hh, x:x + cell] == alt[:hh, x:x + cell]).all()
        assert same == (c != 3)
    # 列头有字（不是空白）
    for c in range(5):
        x = c * (cell + GAP)
        assert (img[:hh, x:x + cell] < 128).any()
    # 拼图：带标签列，5 列
    labels = [(["#000", "W", "U+5B57"], _solid(0, 48)) for _ in rows]
    g = sbs.compose(rows, heads, cell, labels=labels, label_w=cell, font_path=FONT)
    x0 = cell + GAP
    assert g.shape == (hh + 2 * cell + GAP, x0 + 5 * cell + 4 * GAP, 3)
    for r in range(2):
        y = hh + r * (cell + GAP)
        for c in range(5):
            x = x0 + c * (cell + GAP)
            assert (g[y:y + cell, x:x + cell] == 10 * (r * 5 + c + 1)).all()


def _items(recs, cell_src=40):
    for r in recs:
        yield r, [_solid((r["idx"] + c) % 256, cell_src) for c in range(5)], _solid(0, cell_src)


def _per_sample_for(recs, l1_fn, j_fn):
    rows = []
    for r in recs:
        for lam in LAMBDAS:
            rows.append({"idx": r["idx"], "kind": r["kind"], "lambda": lam,
                         "l1": l1_fn(r, lam), "J": j_fn(r, lam) if r["kind"] == "JT" else None})
    return rows


def test_write_outputs_lambda_index_pages_metrics():
    cell_src, cell = 40, 16
    recs = _recs(n_jt=17, n_bf=31)
    heads = fs.column_heads(LAMBDAS)
    per_sample = _per_sample_for(recs, lambda r, lam: 0.5 + lam + r["idx"] * 1e-3,
                                 lambda r, lam: 0.2 + lam / 10)
    expect = [float(np.mean([0.5 + 0.5 + r["idx"] * 1e-3 for r in recs if r["kind"] == k]))
              for k in ("JT", "BF")] + [0.25]
    fin = fs.make_finalizer(per_sample, LAMBDAS, S_BASE, expect, 1e-4, {"model_name": "m"})
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "fs"
        got, pages = sbs.write_outputs(_items(recs, cell_src), out, heads, 15, cell,
                                       font_path=FONT, before_rename=fin)
        assert not sbs.tmp_dir(out).exists()
        assert sorted(p.name for p in out.iterdir()) == sorted(
            ["per_item", "index.csv", "metrics.json", "per_sample.csv"]
            + [sbs.grid_filename("JT", p) for p in (1, 2)]
            + [sbs.grid_filename("BF", p) for p in (1, 2, 3)])
        # index 与 per_item 一一对应
        files = sorted(p.name for p in (out / "per_item").iterdir())
        with open(out / "index.csv", encoding="utf-8") as f:
            idx_rows = list(csv.DictReader(f))
        assert len(idx_rows) == 48 and sorted(r["file"] for r in idx_rows) == files
        for r in idx_rows:
            assert r["file"] == f"{r['kind']}_{int(r['idx']):03d}_{chr(0x4E00 + int(r['idx']))}.png"
        # 单张图 5 列，原分辨率
        im = np.asarray(Image.open(out / "per_item" / idx_rows[0]["file"]))
        hh = sbs.header_height(cell_src)
        assert im.shape == (hh + cell_src, 5 * cell_src + 4 * GAP, 3)
        for c in range(5):
            x = c * (cell_src + GAP)
            assert (im[hh:, x:x + cell_src] == (int(idx_rows[0]["idx"]) + c) % 256).all()
        # 拼图：每页行数与 index 的 page 一致，5 列加标签列
        for kind in ("JT", "BF"):
            for p in range(1, len(pages[kind]) + 1):
                n_rows = sum(r["kind"] == kind and int(r["page"]) == p for r in idx_rows)
                g = np.asarray(Image.open(out / sbs.grid_filename(kind, p)))
                assert g.shape == (sbs.header_height(cell) + n_rows * (cell + GAP) - GAP,
                                   cell + GAP + 5 * cell + 4 * GAP, 3)
        # per_sample.csv：每条样本 × 每个 λ 一行，BF 的 J 为空
        with open(out / "per_sample.csv", encoding="utf-8") as f:
            ps = list(csv.DictReader(f))
        assert list(ps[0].keys()) == list(fs.PER_SAMPLE_FIELDS)
        assert len(ps) == 48 * 3
        assert [(int(r["idx"]), r["lambda"]) for r in ps] == [
            (i, k) for i in range(48) for k in ("0.3", "0.5", "0.7")]
        kinds = {int(r["idx"]): r["kind"] for r in idx_rows}
        for r in ps:
            assert r["kind"] == kinds[int(r["idx"])]
            assert (r["J"] == "") == (r["kind"] == "BF")
            assert float(r["l1"]) == 0.5 + float(r["lambda"]) + int(r["idx"]) * 1e-3
        # metrics.json
        m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
        assert m["model_name"] == "m" and m["lambdas"] == ["0.3", "0.5", "0.7"]
        assert m["s_baseline"] == S_BASE
        assert m["per_lambda"]["0.5"]["n_jt"] == 17 and m["per_lambda"]["0.5"]["n_bf"] == 31
        assert abs(m["per_lambda"]["0.7"]["J"] - 0.27) < 1e-12
        assert m["reconcile"]["rel_diff"]["L1_JT"] < 1e-12


def test_summarize_synthetic():
    per_sample = [
        {"idx": 0, "kind": "JT", "lambda": 0.3, "l1": 0.8, "J": 0.3},
        {"idx": 1, "kind": "JT", "lambda": 0.3, "l1": 0.6, "J": 0.1},
        {"idx": 2, "kind": "BF", "lambda": 0.3, "l1": 0.5, "J": None},
        {"idx": 0, "kind": "JT", "lambda": 0.5, "l1": 1.0, "J": 0.4},
        {"idx": 1, "kind": "JT", "lambda": 0.5, "l1": 0.6, "J": 0.2},
        {"idx": 2, "kind": "BF", "lambda": 0.5, "l1": 0.7, "J": None},
    ]
    s_base = {"L1_JT": 0.8, "L1_BF": 0.5, "J": 0.25}
    m = fs.summarize(per_sample, [0.3, 0.5], s_base)
    assert set(m) == {"0.3", "0.5"}
    a, b = m["0.3"], m["0.5"]
    assert np.isclose(a["L1_JT"], 0.7) and np.isclose(a["L1_BF"], 0.5) and np.isclose(a["J"], 0.2)
    assert np.isclose(a["L1_JT_ratio"], 0.875) and np.isclose(a["L1_BF_ratio"], 1.0)
    assert np.isclose(a["J_ratio"], 0.8) and np.isclose(a["S"], 2.675)
    assert np.isclose(b["L1_JT"], 0.8) and np.isclose(b["L1_BF"], 0.7) and np.isclose(b["J"], 0.3)
    assert np.isclose(b["S"], 1.0 + 1.4 + 1.2)
    assert a["n_jt"] == 2 and a["n_bf"] == 1
    # 用 baseline 自身数值算 S 应为 3
    same = [{"idx": 0, "kind": "JT", "lambda": 0.5, "l1": S_BASE["L1_JT"], "J": S_BASE["J"]},
            {"idx": 1, "kind": "BF", "lambda": 0.5, "l1": S_BASE["L1_BF"], "J": None}]
    assert fs.summarize(same, [0.5], S_BASE)["0.5"]["S"] == 3.0


def test_reconcile():
    expect = [0.795422, 0.551588, 0.283544]
    got = {"0.5": {"L1_JT": 0.795421700108619, "L1_BF": 0.5515879868314817,
                   "J": 0.28354447540782746}}
    rep = fs.reconcile(got, expect, 1e-4)
    assert max(rep["rel_diff"].values()) < 1e-5
    # 每一项单独超差都要报错；nan 也报错
    for k, v in (("L1_JT", 0.7956), ("L1_BF", 0.5517), ("J", 0.28360), ("J", float("nan"))):
        bad = {"0.5": dict(got["0.5"], **{k: v})}
        try:
            fs.reconcile(bad, expect, 1e-4)
            raise AssertionError(f"{k}={v} 应对账失败")
        except RuntimeError as e:
            assert "对账失败" in str(e)
    # 恰好在容差内通过
    edge = {"0.5": dict(got["0.5"], L1_JT=expect[0] * (1 + 0.9e-4))}
    fs.reconcile(edge, expect, 1e-4)


def test_reconcile_failure_leaves_no_output():
    recs = _recs(n_jt=3, n_bf=3)
    per_sample = _per_sample_for(recs, lambda r, lam: 1.0, lambda r, lam: 1.0)
    fin = fs.make_finalizer(per_sample, LAMBDAS, S_BASE, [0.795422, 0.551588, 0.283544], 1e-4, {})
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "fs"
        try:
            sbs.write_outputs(_items(recs), out, fs.column_heads(LAMBDAS), 15, 16,
                              font_path=FONT, before_rename=fin)
            raise AssertionError("对账失败应报错")
        except RuntimeError as e:
            assert "对账失败" in str(e)
        assert list(Path(d).iterdir()) == []
    # 逐样本条数不对（少了一档 λ）也报错
    fin = fs.make_finalizer(per_sample[:-1], LAMBDAS, S_BASE, [1.0, 1.0, 1.0], 1e-4, {})
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "fs"
        try:
            sbs.write_outputs(_items(recs), out, fs.column_heads(LAMBDAS), 15, 16,
                              font_path=FONT, before_rename=fin)
            raise AssertionError("条数不对应报错")
        except RuntimeError as e:
            assert "逐样本条数" in str(e)
        assert list(Path(d).iterdir()) == []


def test_main_rejects_existing_output_and_bad_lambdas():
    with tempfile.TemporaryDirectory() as d:
        base = ["x", "--model_ckpt", "a", "--model_name", "m", "--data_path", "d",
                "--val_json_path", "v", "--fixed_pair_path", "f", "--calibration_json", "c",
                "--s_baseline_json", "s", "--expect_lambda05", "1", "1", "1",
                "--font_path", FONT]
        cases = []
        out = Path(d) / "exists"
        out.mkdir()
        cases.append((base + ["--output_dir", str(out)], FileExistsError))
        out2 = Path(d) / "fresh"
        sbs.tmp_dir(out2).mkdir()
        cases.append((base + ["--output_dir", str(out2)], FileExistsError))
        cases.append((base + ["--output_dir", str(Path(d) / "x3"), "--lambdas", "0.3", "0.7"],
                      ValueError))
        cases.append((base + ["--output_dir", str(Path(d) / "x4")], FileNotFoundError))
        argv = sys.argv
        try:
            for args, exc in cases:
                sys.argv = args
                try:
                    fs.main()
                    raise AssertionError(f"{args} 应报 {exc.__name__}")
                except exc:
                    pass
        finally:
            sys.argv = argv
        assert sorted(p.name for p in Path(d).iterdir()) == ["exists", "fresh.tmp"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} passed")
