"""固定 val 上 baseline 与指定模型的并排视觉图（不盲化，T1 最优组合看图用）。

每条样本一行 4 列：上半参考字（GT）| 下半 GT | baseline 预测 | 模型预测。
- 数据集构造与 tools/eval_s_baseline.py / export_jieti_ab.py 逐项相同，下半整字全遮；
  两个模型对同一份输入前向。
- 下半全遮时拼接图（pred·mask + GT·(1−mask)）的下半就是 pred，这里直接取纯 pred，
  反归一化后截断到 [0,1]（to_rgb_u8，与 export_jieti_ab 相同）。
- 输出：per_item/{kind}_{idx:03d}_{字}.png（原分辨率）、grid_{kind}_pageNN.png（按 idx 顺序
  分页，单元格缩到 --grid_cell）、index.csv。输出目录或其临时目录已存在就报错退出；
  先写到同级临时目录 {out}.tmp，全部写完再改名为 out，中途出错删掉临时目录。
- 服务器没有中文字体，图里的字用 U+ 码表示，并在行标签里贴印刷体字形（模型输入的下半）；
  文件名和 index.csv 里是字本身。

numpy/PIL 部分可以在没有 torch 的环境单测；torch 与模型代码在 run_models() 里延迟导入。
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.export_jieti_ab import to_rgb_u8  # noqa: E402

GAP = 4            # 单元格之间的间隔（像素）
BG = 200           # 间隔处的灰色，和白底字图区分开
INDEX_FIELDS = ("idx", "kind", "writer", "char", "ref_path", "query_path", "file", "page")


# ---------------------------------------------------------------------------
# numpy / PIL 部分（可在无 torch 环境测试）
# ---------------------------------------------------------------------------

def column_heads(baseline_name, model_name):
    return ["ref (upper GT)", "GT (lower)", baseline_name, model_name]


def item_filename(kind, idx, char):
    return f"{kind}_{idx:03d}_{char}.png"


def grid_filename(kind, page):
    return f"grid_{kind}_page{page:02d}.png"


def char_code(char):
    """非 ASCII 字符写成 U+XXXX（服务器无中文字体），ASCII 原样保留。"""
    return " ".join(c if c.isascii() else f"U+{ord(c):04X}" for c in char)


def paginate(recs, rows_per_page):
    """按 kind 分组、组内按 idx 升序，每 rows_per_page 条一页，页码从 1 开始。

    返回 {kind: [[rec, ...], ...]}，并就地给每条 rec 写入 "page"。
    """
    pages = {}
    for kind in ("JT", "BF"):
        group = sorted((r for r in recs if r["kind"] == kind), key=lambda r: r["idx"])
        pages[kind] = [group[i:i + rows_per_page] for i in range(0, len(group), rows_per_page)]
        for p, page in enumerate(pages[kind], start=1):
            for r in page:
                r["page"] = p
    return pages


def _font(size):
    from PIL import ImageFont
    return ImageFont.load_default(size=size)


def header_height(cell):
    return max(12, cell // 14) + 8


def _resize(arr, cell):
    from PIL import Image
    return np.asarray(Image.fromarray(arr).resize((cell, cell), Image.LANCZOS))


def compose(rows, heads, cell, labels=None, label_w=0):
    """拼一张图：顶部列头 + 每行 4 个 cell×cell 的面板，左侧可选标签列。

    rows: 每行 4 张 (cell, cell, 3) uint8；labels: 每行 (文字行列表, 字形 uint8 或 None)。
    第 r 行第 c 列的左上角：x = x0 + c·(cell+GAP)，y = header_h + r·(cell+GAP)，
    x0 = label_w + GAP（有标签列）或 0。
    """
    from PIL import Image, ImageDraw

    hh = header_height(cell)
    x0 = label_w + GAP if labels is not None else 0
    width = x0 + len(heads) * (cell + GAP) - GAP
    height = hh + len(rows) * (cell + GAP) - GAP
    canvas = Image.new("RGB", (width, height), (BG, BG, BG))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, width - 1, hh - 1], fill=(255, 255, 255))
    font = _font(max(12, cell // 14))
    for c, head in enumerate(heads):
        x = x0 + c * (cell + GAP)
        tw = draw.textlength(head, font=font)
        draw.text((x + (cell - tw) / 2, 4), head, fill=(0, 0, 0), font=font)
    for r, panels in enumerate(rows):
        y = hh + r * (cell + GAP)
        for c, p in enumerate(panels):
            canvas.paste(Image.fromarray(p), (x0 + c * (cell + GAP), y))
        if labels is not None:
            draw.rectangle([0, y, label_w - 1, y + cell - 1], fill=(255, 255, 255))
            lines, glyph = labels[r]
            lfont = _font(max(10, cell // 12))
            step = max(10, cell // 12) + 4
            for k, line in enumerate(lines):
                draw.text((4, y + 4 + k * step), line, fill=(0, 0, 0), font=lfont)
            if glyph is not None:
                g = cell // 2
                canvas.paste(Image.fromarray(_resize(glyph, g)),
                             ((label_w - g) // 2, y + cell - g - 4))
    return np.asarray(canvas)


def row_label(rec):
    return [f"#{rec['idx']:03d}", rec["writer"], char_code(rec["char"])]


def save_png(arr, path):
    from PIL import Image
    Image.fromarray(arr).save(path)


def write_index(path, recs):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(recs, key=lambda r: r["idx"]))


def tmp_dir(out):
    return out.parent / (out.name + ".tmp")


def check_output_free(out):
    for p in (out, tmp_dir(out)):
        if p.exists():
            raise FileExistsError(f"{p} 已存在，不覆盖")


def write_outputs(items, out, heads, rows_per_page, grid_cell):
    """items: [(rec, panels(4 张原分辨率 uint8), glyph uint8)]，逐条写单张，最后写拼图和 index。

    全部写进临时目录，成功后改名为 out；任何异常（含模型加载失败、Ctrl-C）都删掉
    本次创建的临时目录再原样抛出，不留半成品挡住重跑。
    """
    check_output_free(out)
    tmp = tmp_dir(out)
    tmp.mkdir(parents=True, exist_ok=False)  # 已存在则报错，且不进入下面的删除分支
    try:
        per_item = tmp / "per_item"
        per_item.mkdir()
        recs = []
        for rec, panels, glyph in items:
            rec = dict(rec)
            rec["file"] = item_filename(rec["kind"], rec["idx"], rec["char"])
            save_png(compose([panels], heads, panels[0].shape[0]), per_item / rec["file"])
            # 拼图只留缩小后的面板，省内存
            rec["_cells"] = [_resize(p, grid_cell) for p in panels]
            rec["_glyph"] = glyph
            recs.append(rec)
        pages = paginate(recs, rows_per_page)
        for kind, kpages in pages.items():
            for p, page in enumerate(kpages, start=1):
                img = compose([r["_cells"] for r in page], heads, grid_cell,
                              labels=[(row_label(r), r["_glyph"]) for r in page],
                              label_w=grid_cell)
                save_png(img, tmp / grid_filename(kind, p))
        write_index(tmp / "index.csv", recs)
        tmp.rename(out)
    except BaseException:
        shutil.rmtree(tmp)
        raise
    return recs, pages


# ---------------------------------------------------------------------------
# torch / 模型部分（仅在服务器运行）
# ---------------------------------------------------------------------------

def run_models(args):
    """逐条产出 (rec, [ref, gt, base_pred, model_pred], glyph)，图都是 (448,448,3) uint8。"""
    import torch

    from data.pairdataset import PairDataset
    from tools.eval_s_baseline import build_val_transform, load_model
    from util.masking_generator import MaskingGenerator

    torch.manual_seed(args.seed)
    # 构造参数与 eval_s_baseline / export_jieti_ab / eval_band_error 逐项相同
    dataset = PairDataset(
        args.data_path, args.val_json_path, transform=build_val_transform(),
        masked_position_generator=MaskingGenerator(
            (56, 28), num_masking_patches=784, max_num_patches=392),
        use_two_pairs=True, half_mask_ratio=1.0,
        semantic_mask_dir=args.semantic_mask_dir,
        num_mask_annotations_jt=1, num_mask_annotations_bf=11,
        mask_coverage_threshold=0.1, return_jieti=True,
        fixed_pair_path=args.fixed_pair_path,
    )
    models = [load_model(args.baseline_ckpt).to(args.device).eval(),
              load_model(args.model_ckpt).to(args.device).eval()]

    with torch.no_grad():
        for idx in range(len(dataset)):
            image, target, mask, _valid, _voro, _vp, is_jt = dataset[idx]
            kind = "JT" if bool(is_jt.item()) else "BF"
            if not mask[mask.shape[0] // 2:].all() or mask[:mask.shape[0] // 2].any():
                raise RuntimeError(f"sample {idx}: val mask 不是下半整字全遮")
            images = image.unsqueeze(0).to(args.device)
            targets = target.unsqueeze(0).to(args.device)
            bmp = torch.from_numpy(mask).bool().flatten()[None].to(args.device)
            hp = mask.shape[0] // 2 * 16  # 下半 query 起始行（像素）

            # target_path 是上半参考字（固定配对的 key），被评测的字是下半 query
            ref_path = dataset.pairs[idx]["target_path"]
            q = dataset._fixed_pairs[ref_path]
            rec = {"idx": idx, "kind": kind,
                   "writer": q["type"].replace("font_", "").replace(kind, ""),
                   "char": Path(q["target_path"]).stem,
                   "ref_path": ref_path, "query_path": q["target_path"]}
            tgt = target.double().numpy()
            panels = [to_rgb_u8(tgt[:, :hp, :]), to_rgb_u8(tgt[:, hp:, :])]
            for model in models:
                pred = model.forward_decoder(model.forward_encoder(images, targets, bmp))
                panels.append(to_rgb_u8(pred[0].double().cpu().numpy()[:, hp:, :]))
            glyph = to_rgb_u8(image.double().numpy()[:, hp:, :])
            yield rec, panels, glyph
            if (idx + 1) % 20 == 0:
                print(f"[sbs] {idx + 1}/{len(dataset)} samples done", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline_ckpt", required=True)
    parser.add_argument("--model_ckpt", required=True)
    parser.add_argument("--baseline_name", default="baseline")
    parser.add_argument("--model_name", required=True, help="列头里显示的模型名")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--val_json_path", nargs="+", required=True)
    parser.add_argument("--semantic_mask_dir", default=None)
    parser.add_argument("--fixed_pair_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--rows_per_page", type=int, default=15)
    parser.add_argument("--grid_cell", type=int, default=256, help="拼图里每个面板的边长（像素）")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out = Path(args.output_dir)
    check_output_free(out)  # 加载模型之前就检查 out 与临时目录
    heads = column_heads(args.baseline_name, args.model_name)
    recs, pages = write_outputs(run_models(args), out, heads, args.rows_per_page, args.grid_cell)
    n = {k: sum(r["kind"] == k for r in recs) for k in ("JT", "BF")}
    print(f"[sbs] done: {len(recs)} items (JT {n['JT']} / BF {n['BF']}), "
          f"pages JT {len(pages['JT'])} / BF {len(pages['BF'])} -> {out}", flush=True)


if __name__ == "__main__":
    main()
