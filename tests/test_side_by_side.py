"""tools/export_side_by_side.py 的 numpy/PIL 部分单测（可直接 python 运行）。"""

import csv
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.export_side_by_side as sbs  # noqa: E402
from tools.export_side_by_side import GAP  # noqa: E402
from tools.export_jieti_ab import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402


def _recs(n_jt=105, n_bf=143):
    """模拟 fixed val 的混排顺序：JT/BF 交错出现，idx 连续。"""
    kinds = ["JT"] * n_jt + ["BF"] * n_bf
    rng = np.random.default_rng(0)
    rng.shuffle(kinds)
    return [{"idx": i, "kind": k, "writer": "W", "char": "字", "ref_path": f"r{i}",
             "query_path": f"q{i}"} for i, k in enumerate(kinds)]


def _solid(v, cell):
    return np.full((cell, cell, 3), v, dtype=np.uint8)


def test_paginate_counts_and_order():
    recs = _recs()
    pages = sbs.paginate(recs, 15)
    assert len(pages["JT"]) == 7 and len(pages["BF"]) == 10  # ceil(105/15), ceil(143/15)
    assert [len(p) for p in pages["BF"]] == [15] * 9 + [8]
    for kind, n in (("JT", 105), ("BF", 143)):
        flat = [r for p in pages[kind] for r in p]
        assert len(flat) == n and all(r["kind"] == kind for r in flat)
        assert [r["idx"] for r in flat] == sorted(r["idx"] for r in flat)  # 保持原顺序
        for p, page in enumerate(pages[kind], start=1):
            assert all(r["page"] == p for r in page)


def test_denorm_and_clamp():
    # 归一化空间里的白/黑/越界值：反归一化后截断到 [0,255]
    white = ((1.0 - IMAGENET_MEAN) / IMAGENET_STD)[:, None, None] * np.ones((3, 4, 4))
    black = ((0.0 - IMAGENET_MEAN) / IMAGENET_STD)[:, None, None] * np.ones((3, 4, 4))
    assert (sbs.to_rgb_u8(white) == 255).all() and (sbs.to_rgb_u8(black) == 0).all()
    assert (sbs.to_rgb_u8(white + 5.0) == 255).all()
    assert (sbs.to_rgb_u8(black - 5.0) == 0).all()
    mid = ((0.5 - IMAGENET_MEAN) / IMAGENET_STD)[:, None, None] * np.ones((3, 2, 3))
    out = sbs.to_rgb_u8(mid)
    assert out.shape == (2, 3, 3) and out.dtype == np.uint8 and (out == 128).all()


def test_compose_layout():
    cell = 32
    heads = sbs.column_heads("baseline", "ctrl_g12")
    rows = [[_solid(10 * (r * 4 + c + 1), cell) for c in range(4)] for r in range(3)]
    hh = sbs.header_height(cell)
    # 无标签列（单张图）
    img = sbs.compose(rows[:1], heads, cell)
    assert img.shape == (hh + cell, 4 * cell + 3 * GAP, 3)
    for c in range(4):
        x = c * (cell + GAP)
        assert (img[hh:hh + cell, x:x + cell] == 10 * (c + 1)).all()
    # 有标签列（拼图）
    lw = cell
    labels = [(["#000", "W", "U+5B57"], _solid(0, 48)) for _ in rows]
    img = sbs.compose(rows, heads, cell, labels=labels, label_w=lw)
    # 对照图：同样的字形但不写文字；以及既无文字也无字形
    img_notext = sbs.compose(rows, heads, cell, labels=[([], _solid(0, 48)) for _ in rows],
                             label_w=lw)
    img_blank = sbs.compose(rows, heads, cell, labels=[([], None) for _ in rows], label_w=lw)
    x0 = lw + GAP
    assert img.shape == (hh + 3 * cell + 2 * GAP, x0 + 4 * cell + 3 * GAP, 3)
    for r in range(3):
        y = hh + r * (cell + GAP)
        for c in range(4):
            x = x0 + c * (cell + GAP)
            assert (img[y:y + cell, x:x + cell] == 10 * (r * 4 + c + 1)).all()
        # 间隔是灰色
        assert (img[y:y + cell, x0 + cell:x0 + cell + GAP] == sbs.BG).all()
        # 标签列：写了文字的和不写文字的逐像素不同；贴了字形的和空白的不同，空白的全白
        assert (img[y:y + cell, :lw] != img_notext[y:y + cell, :lw]).any()
        assert (img_notext[y:y + cell, :lw] != img_blank[y:y + cell, :lw]).any()
        assert (img_blank[y:y + cell, :lw] == 255).all()


def test_char_code():
    assert sbs.char_code("字") == "U+5B57"
    assert sbs.char_code("佛1") == "U+4F5B 1"


def test_write_outputs_index_files_consistent():
    cell_src, cell = 40, 16
    recs = _recs(n_jt=17, n_bf=31)
    for r in recs:
        r["char"] = chr(0x4E00 + r["idx"])
    heads = sbs.column_heads("baseline", "ctrl_g12")

    def gen():
        for r in recs:
            panels = [_solid((r["idx"] + c) % 256, cell_src) for c in range(4)]
            yield r, panels, _solid(0, cell_src)

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "sbs"
        got, pages = sbs.write_outputs(gen(), out, heads, rows_per_page=15, grid_cell=cell)
        files = sorted(p.name for p in (out / "per_item").iterdir())
        with open(out / "index.csv", encoding="utf-8") as f:
            idx_rows = list(csv.DictReader(f))
        assert list(idx_rows[0].keys()) == list(sbs.INDEX_FIELDS)
        assert len(idx_rows) == 48 and [int(r["idx"]) for r in idx_rows] == list(range(48))
        assert sorted(r["file"] for r in idx_rows) == files
        for r in idx_rows:
            i = int(r["idx"])
            assert r["file"] == sbs.item_filename(r["kind"], i, chr(0x4E00 + i))
            assert r["file"] == f"{r['kind']}_{i:03d}_{chr(0x4E00 + i)}.png"
            assert r["ref_path"] == f"r{i}" and r["query_path"] == f"q{i}"
        # 单张图：原分辨率 4 列，面板内容就是传入的面板
        r0 = idx_rows[0]
        im = np.asarray(Image.open(out / "per_item" / r0["file"]))
        hh = sbs.header_height(cell_src)
        assert im.shape == (hh + cell_src, 4 * cell_src + 3 * GAP, 3)
        for c in range(4):
            x = c * (cell_src + GAP)
            assert (im[hh:, x:x + cell_src] == (int(r0["idx"]) + c) % 256).all()
        # 拼图：页数、文件名、每页行数与 index 的 page 一致
        grids = sorted(p.name for p in out.glob("grid_*.png"))
        assert grids == sorted([sbs.grid_filename("JT", p) for p in (1, 2)]
                               + [sbs.grid_filename("BF", p) for p in (1, 2, 3)])
        for kind in ("JT", "BF"):
            for p in range(1, len(pages[kind]) + 1):
                n_rows = sum(r["kind"] == kind and int(r["page"]) == p for r in idx_rows)
                g = np.asarray(Image.open(out / sbs.grid_filename(kind, p)))
                assert g.shape[0] == sbs.header_height(cell) + n_rows * (cell + GAP) - GAP
                assert g.shape[1] == cell + GAP + 4 * cell + 3 * GAP
        # 每页内 idx 升序：第一行左上面板的灰度 = 该页最小 idx
        bf2 = [r for r in idx_rows if r["kind"] == "BF" and r["page"] == "2"]
        g = np.asarray(Image.open(out / sbs.grid_filename("BF", 2)))
        y, x = sbs.header_height(cell), cell + GAP
        assert int(g[y + cell // 2, x + cell // 2, 0]) == int(bf2[0]["idx"]) % 256

        # 输出目录已存在时 main 在加载模型之前就报错，目录内容不变
        before = sorted(str(p) for p in out.rglob("*"))
        argv = sys.argv
        sys.argv = ["x", "--baseline_ckpt", "a", "--model_ckpt", "b", "--model_name", "m",
                    "--data_path", "d", "--val_json_path", "v", "--fixed_pair_path", "f",
                    "--output_dir", str(out)]
        try:
            sbs.main()
            raise AssertionError("已存在的输出目录应报错")
        except FileExistsError:
            pass
        finally:
            sys.argv = argv
        assert sorted(str(p) for p in out.rglob("*")) == before


def _gen_items(n, cell_src=40, fail_at=None):
    for r in _recs(n_jt=n // 2, n_bf=n - n // 2):
        if r["idx"] == fail_at:
            raise RuntimeError("boom")
        yield r, [_solid((r["idx"] + c) % 256, cell_src) for c in range(4)], _solid(0, cell_src)


def test_write_outputs_failure_cleans_tmp():
    heads = sbs.column_heads("baseline", "ctrl_g12")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "sbs"
        tmp = sbs.tmp_dir(out)
        assert tmp == Path(d) / "sbs.tmp"
        # 写了 3 条单张图之后生成器抛异常：异常原样抛出，out 与临时目录都不存在
        try:
            sbs.write_outputs(_gen_items(6, fail_at=3), out, heads, rows_per_page=15, grid_cell=16)
            raise AssertionError("应抛出 RuntimeError")
        except RuntimeError as e:
            assert str(e) == "boom"
        assert not out.exists() and not tmp.exists()
        assert list(Path(d).iterdir()) == []
        # 清理后可以直接重跑
        sbs.write_outputs(_gen_items(6), out, heads, rows_per_page=15, grid_cell=16)
        assert out.is_dir() and not tmp.exists()


def test_write_outputs_success_and_tmp_guard():
    heads = sbs.column_heads("baseline", "ctrl_g12")
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "sbs"
        recs, pages = sbs.write_outputs(_gen_items(6), out, heads, rows_per_page=15, grid_cell=16)
        assert not sbs.tmp_dir(out).exists()
        assert sorted(p.name for p in Path(d).iterdir()) == ["sbs"]
        assert len(list((out / "per_item").iterdir())) == 6
        assert (out / "index.csv").is_file()
        assert sorted(p.name for p in out.glob("grid_*.png")) == sorted(
            [sbs.grid_filename("JT", 1), sbs.grid_filename("BF", 1)])
        # 临时目录已存在（例如别的进程在写）：报错，且不删除它、不建 out
        out2 = Path(d) / "sbs2"
        tmp2 = sbs.tmp_dir(out2)
        tmp2.mkdir()
        (tmp2 / "keep.txt").write_text("x")
        for call in (lambda: sbs.write_outputs(_gen_items(2), out2, heads, 15, 16),
                     lambda: sbs.check_output_free(out2)):
            try:
                call()
                raise AssertionError("临时目录已存在应报错")
            except FileExistsError:
                pass
        assert (tmp2 / "keep.txt").read_text() == "x" and not out2.exists()
        # out 已存在：write_outputs 本身也报错，内容不变
        before = sorted(str(p) for p in out.rglob("*"))
        try:
            sbs.write_outputs(_gen_items(2), out, heads, 15, 16)
            raise AssertionError("out 已存在应报错")
        except FileExistsError:
            pass
        assert sorted(str(p) for p in out.rglob("*")) == before
        assert not sbs.tmp_dir(out).exists()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"{len(tests)} passed")
