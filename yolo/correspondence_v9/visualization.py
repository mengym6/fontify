"""BF/JT rows with GT anchors, gen reference, gen predictions; no raw columns."""

import colorsys
import numpy as np
from PIL import Image, ImageDraw
from .predict import decode_masks


def preview(images, target, output, threshold=0.5):
    size = 256
    header = 28
    canvas = Image.new("RGB", (size * 3, (size + header) * 2 + 28), "white")
    scores = output["match_logits"].detach().sigmoid().cpu().numpy()
    predicted = (
        decode_masks(output["mask_logits"].detach(), images.shape[-1]).cpu().numpy()
    )
    groups = target["groups"].cpu().numpy()
    labels = target["matchable"].cpu().numpy()
    for group, name in enumerate(("BF", "JT")):
        rejected = []
        for column in range(3):
            side = 0 if column == 0 else 1
            image = (
                Image.fromarray(
                    (images[side].detach().cpu().permute(1, 2, 0).numpy() * 255)
                    .clip(0, 255)
                    .astype("uint8")
                )
                .resize((size, size))
                .convert("RGBA")
            )
            for i in range(len(groups)):
                if groups[i] != group:
                    continue
                if column == 0:
                    mask = target["gt_masks"][i].cpu().numpy() >= 0.5
                elif column == 1:
                    if labels[i] != 1:
                        continue
                    mask = target["gen_masks"][i].cpu().numpy() >= 0.5
                else:
                    if scores[i] < threshold:
                        rejected.append(str(i))
                        continue
                    mask = predicted[i]
                if not mask.any():
                    continue
                rgb = tuple(
                    round(x * 255)
                    for x in colorsys.hsv_to_rgb(i * 0.618 % 1, 0.85, 0.8)
                )
                if column == 0 and scores[i] < threshold:
                    rgb = (220, 115, 0)
                alpha = Image.fromarray((mask * 45).astype("uint8")).resize(
                    (size, size), Image.Resampling.NEAREST
                )
                layer = Image.new("RGBA", image.size, rgb + (0,))
                layer.putalpha(alpha)
                image = Image.alpha_composite(image, layer)
                y, x = np.nonzero(np.asarray(alpha))
                if not len(x):
                    # A tiny region may disappear when reduced to the preview size.
                    continue
                ImageDraw.Draw(image).text(
                    (int(x.mean()), int(y.mean())),
                    str(i),
                    fill=rgb + (255,),
                    stroke_width=1,
                    stroke_fill=(255, 255, 255, 230),
                )
            canvas.paste(
                image.convert("RGB"), (column * size, group * (size + header) + header)
            )
            title = (
                name
                + " | "
                + ["GT anchors", "Gen reference", "Gen matched prediction"][column]
            )
            ImageDraw.Draw(canvas).text(
                (column * size + 4, group * (size + header) + 7), title, fill="black"
            )
        ImageDraw.Draw(canvas).text(
            (4, group * (size + header) + header + size - 13),
            "Unmatched GT IDs: " + (",".join(rejected) or "none"),
            fill=(190, 80, 0),
        )
    ImageDraw.Draw(canvas).text(
        (4, canvas.height - 18),
        "IDs are GT anchors. Orange GT regions = predicted unmatched; no invented gen box.",
        fill="black",
    )
    return np.asarray(canvas)
