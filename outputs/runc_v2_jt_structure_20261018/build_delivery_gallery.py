"""Build a portable fixed-sample stage-I review without selecting good examples."""

import hashlib
import html
import json
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path("outputs/runc_v2_jt_structure_20261018/clean_run2")
MODELS = Path("models/runc_v2_jt_structure_20261018/clean_run2")
INDICES = (0, 2, 7, 10, 16, 24, 29, 34, 38, 39, 40, 76, 84)
OUT = ROOT / "coarse_comparison"
OUT.mkdir(exist_ok=True)
report = json.loads((MODELS / "ib/evaluation-step-000320/evaluation.json").read_text())
rows = {row["index"]: row for row in report["rows"]}
sections, manifest = [], []
for index in INDICES:
    row = rows[index]
    character = row.get("character", "")
    author = row.get("author", row.get("calligrapher", ""))
    title = f"{index:03d} / {author} / {character}"
    cards = []
    for stage, step in (("ia1", 400), ("ib", 80), ("ib", 320)):
        source = MODELS / stage / f"evaluation-step-{step:06d}/jt-{index:03d}.png"
        target = OUT / f"{stage}-{step:03d}-{index:03d}.png"
        shutil.copyfile(source, target)
        with Image.open(target) as image:
            image.verify()
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        manifest.append({"index": index, "author": author, "character": character,
                         "stage": stage, "step": step, "file": target.name,
                         "source": str(source), "sha256": digest})
        cards.append(
            f"<h3>{stage.upper()} step {step}</h3><a href='{target.name}'>"
            f"<img loading='lazy' src='{target.name}' alt='{html.escape(title)}'></a>"
        )
    sections.append(f"<section><h2>{html.escape(title)}</h2>{''.join(cards)}</section>")
page = """<!doctype html><meta charset='utf-8'><title>Run C v2 固定13字</title>
<style>body{font-family:system-ui;margin:24px;line-height:1.6}section{margin:36px 0;
border-top:1px solid #bbb}img{width:100%;height:auto}p{max-width:1000px}
a{color:#16599b}</style><h1>Clean Run C v2：固定13字，九书家</h1>
<p>同一组样本显示 IA1 step400、IB step80 与 step320。每条图从左到右为
source / GT（仅对照）/ JT GT occupancy / predicted S / D_GT oracle /
O_GT oracle / predicted D0 / predicted O0 / generated P0。点击图片查看原尺寸。</p>
<p>结论：阶段一结构技术审查未通过。均值尺度改善仍伴随细点消失、部件过厚与
内部空白合并。用户最终人工审阅尚未完成；不启动 BF 正式长训。</p>
""" + "".join(sections)
(OUT / "index.html").write_text(page, encoding="utf-8")
(OUT / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
assert len(manifest) == 39
print(json.dumps({"samples": len(INDICES), "images": len(manifest),
                  "authors": sorted({item["author"] for item in manifest})}))
