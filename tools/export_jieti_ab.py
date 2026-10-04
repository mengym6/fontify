"""J 与人工结体判断是否一致：导出 baseline / 对照组的配对 A/B 盲评页（T1-V）。

用户 2026-10-04 定的设计：
- 只用固定 val 的 JT 样本（下半整字全遮），逐样本算 baseline 与对照组的 J，
  口径与 tools/eval_s_baseline.py 完全相同（同一 JietiLoss、同一三项系数、同一拼接图）。
- 选样：|ΔJ| 最大的 n_top 条 + 从其余样本里无放回随机抽 n_rand 条（默认 30 + 20）。
- 盲评：每题显示 GT、A、B 三张下半字，A/B 左右随机，题目顺序随机；页面里不出现
  模型名、样本 idx、分层名、J 值。评判标准只看结体（部件位置、大小比例、相对关系）。
- 映射表写在 _key/ 子目录，评完再看。

ΔJ = J_ctrl − J_base，ΔJ < 0 表示 J 认为对照组的结体更接近 GT。

numpy 部分（选样、盲化、叠加图、网页）可以在没有 torch 的环境单测；
torch 与模型代码在 run_models() 里延迟导入，只在服务器运行。
"""

import argparse
import base64
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float64)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float64)
MODEL_TAGS = ("base", "ctrl")
J_PARTS = ("centroid", "logsigma", "shape")


# ---------------------------------------------------------------------------
# numpy 部分（可在无 torch 环境测试）
# ---------------------------------------------------------------------------

def to_rgb_u8(x_norm):
    """(3, H, W) ImageNet 归一化 → (H, W, 3) uint8，截断到 [0,1]（与保存/观看一致）。"""
    img = x_norm * IMAGENET_STD[:, None, None] + IMAGENET_MEAN[:, None, None]
    return (np.clip(img, 0.0, 1.0).transpose(1, 2, 0) * 255.0 + 0.5).astype(np.uint8)


def select_pairs(rows, n_top=30, n_rand=20, seed=0):
    """按 |ΔJ| 取前 n_top 条（并列按 idx 升序），再从剩余里无放回随机抽 n_rand 条。

    rows: 含 "idx"、"dJ" 的字典列表。返回 [(idx, stratum)]，stratum ∈ {"top", "rand"}。
    """
    if n_top + n_rand > len(rows):
        raise ValueError(f"需要 {n_top + n_rand} 条，只有 {len(rows)} 条")
    order = sorted(rows, key=lambda r: (-abs(r["dJ"]), r["idx"]))
    top = [r["idx"] for r in order[:n_top]]
    rest = sorted(r["idx"] for r in order[n_top:])
    rng = np.random.default_rng(seed)
    rand = sorted(int(i) for i in rng.choice(rest, size=n_rand, replace=False)) if n_rand else []
    return [(i, "top") for i in top] + [(i, "rand") for i in rand]


def assign_blind(selected, seed=0):
    """打乱题目顺序并随机分配左右。返回题目列表，item 从 1 开始编号。

    left 是放在左侧（A）的模型；另一个模型放右侧（B）。
    """
    rng = np.random.default_rng(seed + 1)  # 与选样用不同的随机流
    perm = rng.permutation(len(selected))
    lefts = rng.integers(0, 2, size=len(selected))
    items = []
    for item_no, k in enumerate(perm, start=1):
        idx, stratum = selected[int(k)]
        left = MODEL_TAGS[int(lefts[item_no - 1])]
        right = MODEL_TAGS[1 - MODEL_TAGS.index(left)]
        items.append({"item": item_no, "idx": int(idx), "stratum": stratum,
                      "left": left, "right": right})
    return items


def ink_overlay(gt_u8, pred_u8):
    """GT 与预测的墨迹叠加：只在 GT 有墨 → 红，只在预测有墨 → 蓝，两者重合 → 黑。"""
    g_gt = gt_u8.astype(np.float64).mean(axis=2) / 255.0
    g_pr = pred_u8.astype(np.float64).mean(axis=2) / 255.0
    a_gt = 1.0 - g_gt
    a_pr = 1.0 - g_pr
    rgb = np.stack([1.0 - a_pr, 1.0 - np.maximum(a_gt, a_pr), 1.0 - a_gt], axis=2)
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def png_b64(arr_u8):
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr_u8).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>结体盲评</title>
<style>
:root { color-scheme: light; }
body { font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
       background: #f6f6f4; color: #222; margin: 0; padding: 16px 24px; }
h1 { font-size: 18px; margin: 0 0 6px; }
.note { font-size: 13px; color: #555; margin: 0 0 12px; line-height: 1.6; }
.row { display: flex; gap: 16px; align-items: flex-start; }
figure { margin: 0; text-align: center; }
figcaption { font-size: 15px; font-weight: 600; margin-bottom: 4px; }
img { width: 380px; height: 380px; border: 1px solid #bbb; background: #fff;
      image-rendering: auto; }
.ctrl { margin: 14px 0; display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
button { font-size: 15px; padding: 8px 16px; border: 1px solid #888; border-radius: 6px;
         background: #fff; cursor: pointer; }
button.sel { background: #2b6cb0; color: #fff; border-color: #2b6cb0; }
button:focus-visible { outline: 3px solid #f6ad55; }
#status { font-size: 14px; color: #333; }
#done { display: none; margin-top: 12px; }
textarea { width: 100%; height: 120px; font-family: monospace; font-size: 12px; }
.legend { font-size: 13px; }
.legend span { display: inline-block; width: 12px; height: 12px; vertical-align: middle;
               margin: 0 3px 0 10px; }
</style>
</head>
<body>
<h1>结体盲评（共 __N__ 题）</h1>
<p class="note">只看结体：各部件之间的相对位置、大小比例、相对关系，以及部件的粗略形状。整字在格子里整体平移不算（J 不度量整字绝对位置），笔画粗细、锋芒、墨色也不作为依据。
判断 A、B 哪一张的结体更接近 GT；看不出差别就选"持平"。<br>
快捷键：1 = A 更接近，2 = 持平，3 = B 更接近，O = 切换叠加图，← / → = 上一题 / 下一题。
叠加图里<span style="color:#c00">红</span>是只有 GT 有墨，<span style="color:#00c">蓝</span>是只有生成图有墨，黑是两者重合。</p>
<div class="row">
  <figure><figcaption>GT</figcaption><img id="img_gt" alt="GT 下半字"></figure>
  <figure><figcaption>A</figcaption><img id="img_a" alt="候选 A"></figure>
  <figure><figcaption>B</figcaption><img id="img_b" alt="候选 B"></figure>
</div>
<div class="ctrl">
  <button id="prev" aria-label="上一题">← 上一题</button>
  <button id="ans_L" data-v="L">1 · A 更接近</button>
  <button id="ans_T" data-v="T">2 · 持平</button>
  <button id="ans_R" data-v="R">3 · B 更接近</button>
  <button id="next" aria-label="下一题">下一题 →</button>
  <button id="ovl" aria-pressed="false">O · 叠加图：关</button>
  <span id="status" role="status" aria-live="polite"></span>
</div>
<div id="done">
  <p>全部答完。点"下载结果"保存 JSON，或复制下面的文本交给分析脚本。</p>
  <button id="dl">下载结果</button>
  <textarea id="out" readonly aria-label="评测结果 JSON"></textarea>
</div>
<script>
const ITEMS = __ITEMS__;
const SESSION = "__SESSION__";
const STORE = "jieti_ab_" + SESSION;
let answers = {};
try { answers = JSON.parse(localStorage.getItem(STORE) || "{}"); } catch (e) { answers = {}; }
let cur = 0, overlay = false;
const $ = id => document.getElementById(id);
function save() { try { localStorage.setItem(STORE, JSON.stringify(answers)); } catch (e) {} }
function result() {
  return JSON.stringify({session: SESSION, n_items: ITEMS.length, answers: answers,
                         exported_at: new Date().toISOString()}, null, 1);
}
function render() {
  const it = ITEMS[cur];
  const sfx = overlay ? "_o" : "";
  $("img_gt").src = "data:image/png;base64," + it.gt;
  $("img_a").src = "data:image/png;base64," + it["a" + sfx];
  $("img_b").src = "data:image/png;base64," + it["b" + sfx];
  const v = answers[String(it.item)];
  for (const k of ["L", "T", "R"]) $("ans_" + k).classList.toggle("sel", v === k);
  const n = Object.keys(answers).length;
  $("status").textContent = "第 " + (cur + 1) + " / " + ITEMS.length + " 题，已答 " + n;
  $("ovl").textContent = "O · 叠加图：" + (overlay ? "开" : "关");
  $("ovl").setAttribute("aria-pressed", String(overlay));
  if (n === ITEMS.length) { $("done").style.display = "block"; $("out").value = result(); }
}
function answer(v) {
  answers[String(ITEMS[cur].item)] = v; save();
  if (cur < ITEMS.length - 1) cur++;
  render();
}
function move(d) { cur = Math.min(ITEMS.length - 1, Math.max(0, cur + d)); render(); }
for (const k of ["L", "T", "R"]) $("ans_" + k).onclick = () => answer(k);
$("prev").onclick = () => move(-1);
$("next").onclick = () => move(1);
$("ovl").onclick = () => { overlay = !overlay; render(); };
$("dl").onclick = () => {
  const blob = new Blob([result()], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = "jieti_ab_answers.json"; a.click();
};
document.addEventListener("keydown", e => {
  if (e.target.tagName === "TEXTAREA") return;
  if (e.key === "1") answer("L");
  else if (e.key === "2") answer("T");
  else if (e.key === "3") answer("R");
  else if (e.key === "o" || e.key === "O") { overlay = !overlay; render(); }
  else if (e.key === "ArrowLeft") move(-1);
  else if (e.key === "ArrowRight") move(1);
});
render();
</script>
</body>
</html>
"""


def build_html(items, images, session):
    """items: assign_blind 的输出；images[idx] = {"gt", "base", "ctrl"} 的 (H,W,3) uint8。

    页面里每题只带 item 编号和图片，不带模型名、idx、分层名。
    """
    payload = []
    for it in items:
        im = images[it["idx"]]
        a, b = im[it["left"]], im[it["right"]]
        payload.append({
            "item": it["item"],
            "gt": png_b64(im["gt"]),
            "a": png_b64(a), "b": png_b64(b),
            "a_o": png_b64(ink_overlay(im["gt"], a)),
            "b_o": png_b64(ink_overlay(im["gt"], b)),
        })
    html = HTML_TEMPLATE.replace("__N__", str(len(items)))
    html = html.replace("__SESSION__", session)
    return html.replace("__ITEMS__", json.dumps(payload))


# ---------------------------------------------------------------------------
# torch / 模型部分（仅在服务器运行）
# ---------------------------------------------------------------------------

def run_models(args):
    """固定 val 的 JT 样本上，两个 checkpoint 逐样本算 J（eval_s_baseline 口径）并取图。"""
    import torch

    from data.pairdataset import PairDataset
    from tools.eval_s_baseline import build_val_transform, load_model
    from util.jieti_loss import JietiLoss
    from util.masking_generator import MaskingGenerator

    torch.manual_seed(args.seed)
    coef = json.loads(Path(args.calibration_json).read_text())["coef"]
    # 构造参数与 eval_s_baseline / eval_band_error 逐项相同
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
    models = {"base": load_model(args.baseline_ckpt).to(args.device).eval(),
              "ctrl": load_model(args.ctrl_ckpt).to(args.device).eval()}
    jmod = JietiLoss(w_centroid=coef["centroid"], w_logsigma=coef["logsigma"],
                     w_shape=coef["shape"]).to(args.device)

    rows, images = [], {}
    with torch.no_grad():
        for idx in range(len(dataset)):
            image, target, mask, valid, voro, valid_parts, is_jt = dataset[idx]
            if not bool(is_jt.item()):
                continue
            if not mask[mask.shape[0] // 2:].all() or mask[:mask.shape[0] // 2].any():
                raise RuntimeError(f"sample {idx}: val mask 不是下半整字全遮")
            images_b = image.unsqueeze(0).to(args.device)
            targets = target.unsqueeze(0).to(args.device)
            bmp = torch.from_numpy(mask).bool().flatten()[None].to(args.device)
            voro_b = voro.unsqueeze(0).to(args.device)
            vp_b = valid_parts.unsqueeze(0).to(args.device)
            is_jt_b = is_jt.view(1).to(args.device).bool()
            hp = mask.shape[0] // 2 * 16

            ref_path = dataset.pairs[idx]["target_path"]
            q = dataset._fixed_pairs[ref_path]
            row = {"idx": idx, "writer": q["type"].replace("font_", "").replace("JT", ""),
                   "target_path": ref_path, "query_path": q["target_path"]}
            keep = {"gt": to_rgb_u8(target.double().numpy()[:, hp:, :])}
            for tag, model in models.items():
                latent = model.forward_encoder(images_b, targets, bmp)
                pred = model.forward_decoder(latent)
                # 与 eval_s_baseline 相同：mask 展开到像素，遮盖区用预测、可见区用 GT
                mask_px = bmp[:, :, None].repeat(1, 1, model.patch_size ** 2 * 3).float()
                mask_px = model.unpatchify(mask_px)
                composite = pred * mask_px + targets * (1 - mask_px)
                J, parts, shield = jmod(composite, targets, voro_b, vp_b, is_jt_b)
                row[f"{tag}_J"] = float(J.item())
                for k in J_PARTS:
                    row[f"{tag}_{k}"] = float(parts[k].item())
                row[f"{tag}_shield"] = int(shield.item())
                keep[tag] = to_rgb_u8(pred[0].double().cpu().numpy()[:, hp:, :])
            row["dJ"] = row["ctrl_J"] - row["base_J"]
            rows.append(row)
            images[idx] = keep
            if len(rows) % 20 == 0:
                print(f"[ab] {len(rows)} JT samples done", flush=True)
    return rows, images


def check_mean_j(rows, expect_base, expect_ctrl, rtol):
    """逐样本 J 的均值与工具口径结果对账；超出 rtol 就报错。"""
    got = {t: float(np.mean([r[f"{t}_J"] for r in rows])) for t in MODEL_TAGS}
    report = {}
    for tag, exp in (("base", expect_base), ("ctrl", expect_ctrl)):
        if exp is None:
            continue
        rel = abs(got[tag] - exp) / abs(exp)
        report[tag] = {"got": got[tag], "expect": exp, "rel_diff": rel}
        if rel > rtol:
            raise RuntimeError(f"{tag} 平均 J = {got[tag]:.5f}，工具口径 {exp:.5f}，"
                               f"相对差 {rel:.2e} > {rtol}")
    return got, report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline_ckpt", required=True)
    parser.add_argument("--ctrl_ckpt", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--val_json_path", nargs="+", required=True)
    parser.add_argument("--semantic_mask_dir", default=None)
    parser.add_argument("--fixed_pair_path", required=True)
    parser.add_argument("--calibration_json", required=True,
                        help="calibrate_jieti.py 的输出，取 coef 作三项系数（同 eval_s_baseline）")
    parser.add_argument("--expect_j", type=float, nargs=2, default=None,
                        metavar=("BASE", "CTRL"), help="工具口径的平均 J，用于对账")
    parser.add_argument("--expect_rtol", type=float, default=0.01)
    parser.add_argument("--n_top", type=int, default=30)
    parser.add_argument("--n_rand", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", default="outputs/jieti_ab")
    args = parser.parse_args()

    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(f"{out} 已存在，不覆盖")
    rows, images = run_models(args)
    eb, ec = (args.expect_j if args.expect_j else (None, None))
    got, report = check_mean_j(rows, eb, ec, args.expect_rtol)

    selected = select_pairs(rows, args.n_top, args.n_rand, args.seed)
    items = assign_blind(selected, args.seed)
    session = f"s{args.seed}_n{len(items)}"

    key_dir = out / "_key"
    key_dir.mkdir(parents=True)
    by_idx = {r["idx"]: r for r in rows}
    key = {"session": session, "config": vars(args), "mean_J": got, "reconcile": report,
           "n_jt": len(rows),
           "items": [{**it, "dJ": by_idx[it["idx"]]["dJ"],
                      "query_path": by_idx[it["idx"]]["query_path"],
                      "writer": by_idx[it["idx"]]["writer"]} for it in items]}
    (key_dir / "jieti_ab_key.json").write_text(
        json.dumps(key, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(key_dir / "per_sample_J.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out / "jieti_ab.html").write_text(build_html(items, images, session), encoding="utf-8")
    n_neg = sum(r["dJ"] < 0 for r in rows)
    print(json.dumps({"n_jt": len(rows), "mean_J": got, "reconcile": report,
                      "n_dJ_negative": n_neg, "n_items": len(items)},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
