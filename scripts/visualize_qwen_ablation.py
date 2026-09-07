#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from birdsong_detect_distill.qwen import context, image


VARIANTS = {
    "direct": "1. Direct — no private reasoning",
    "reasoning": "2. + Private reasoning",
    "self_review": "3. + Self-review",
    "shifted_review": "4. + Independent shifted review",
    "full": "5. + Conditional adjudication",
}
COLORS = {
    "target_vocalization": "#39ff88",
    "uncertain_vocalization": "#ffd43b",
    "chorus": "#ff4fd8",
    "non_target_biological": "#45caff",
    "anthropogenic_noise": "#ff7849",
    "environmental_noise": "#c4c8d0",
}
SHORT = {
    "target_vocalization": "target",
    "uncertain_vocalization": "uncertain",
    "chorus": "chorus",
    "non_target_biological": "non-target bio",
    "anthropogenic_noise": "anthro noise",
    "environmental_noise": "environment noise",
}


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)


def read(path):
    rows = {}
    for line in path.open():
        row = json.loads(line)
        if row.get("status") != "ok":
            continue
        tile = row["tile"]
        key = row["recording"], tile["ownership_start_timebin"], tile["ownership_end_timebin"]
        rows[key] = row
    return rows


def annotate(picture, events):
    output = picture.copy()
    draw = ImageDraw.Draw(output)
    width, height = output.size
    label_font = font(17, True)
    for event in events:
        left, top, right, bottom = event["bbox_2d"]
        box = left * width / 1000, top * height / 1000, right * width / 1000, bottom * height / 1000
        color = COLORS[event["label"]]
        draw.rectangle(box, outline="black", width=7)
        draw.rectangle(box, outline=color, width=4)
        label = SHORT[event["label"]]
        bounds = draw.textbbox((0, 0), label, font=label_font, stroke_width=1)
        text_width, text_height = bounds[2] - bounds[0], bounds[3] - bounds[1]
        x, y = max(0, box[0]), max(0, box[1] - text_height - 7)
        draw.rectangle((x, y, x + text_width + 10, y + text_height + 7), fill="#080b10dc")
        draw.text((x + 5, y + 2), label, fill=color, font=label_font, stroke_width=1, stroke_fill="black")
    return output


def main():
    parser = argparse.ArgumentParser(description="Render matched Qwen ablation annotations on identical spectrograms.")
    parser.add_argument("--annotations", type=Path, default=Path("data/annotations/xcl/ablation_10"))
    parser.add_argument("--spec-dir", type=Path, default=Path("data/xcl"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    variants = {name: read(args.annotations / f"{name}.jsonl") for name in VARIANTS}
    keys = sorted(set.intersection(*(set(rows) for rows in variants.values())))
    args.out.mkdir(parents=True, exist_ok=True)
    title_font, row_font, detail_font = font(27, True), font(23, True), font(17)
    width, title_height, row_header, panel_height, legend_height = 2048, 86, 43, 512, 65

    for index, key in enumerate(keys, 1):
        reference = variants["direct"][key]
        source, tile = reference["source"], reference["tile"]
        qwen_tile = (reference["recording"], Path(source["shard"]).name, source["start"], source["end"],
            tile["ownership_start_timebin"], tile["ownership_end_timebin"])
        spec, _, _ = context(args.spec_dir, qwen_tile, tile["start_timebin"], tile["end_timebin"])
        clean = image(spec)
        canvas = Image.new("RGB", (width, title_height + len(VARIANTS) * (row_header + panel_height) + legend_height), "#11151c")
        draw = ImageDraw.Draw(canvas)
        onset, offset = tile["ownership_onset_ms"] / 1000, tile["ownership_offset_ms"] / 1000
        draw.text((22, 13), f"{reference['recording']}  |  {onset:.2f}–{offset:.2f} s", fill="white", font=title_font)
        draw.text((22, 52), f"{Path(source['shard']).name}  •  identical spectrogram in every row",
            fill="#aeb7c5", font=detail_font)

        y = title_height
        for name, label in VARIANTS.items():
            row = variants[name][key]
            events = row["events"]
            suffix = ""
            if name == "shifted_review":
                suffix = "  •  reviewers agreed" if row.get("reviewers_agreed") else "  •  disagreement; primary retained"
            elif name == "full":
                suffix = "  •  adjudicated" if row.get("adjudicated") else "  •  reviewers agreed"
            draw.rectangle((0, y, width, y + row_header), fill="#202733")
            draw.text((20, y + 7), f"{label}  •  {len(events)} boxes{suffix}", fill="white", font=row_font)
            canvas.paste(annotate(clean, events), (0, y + row_header))
            y += row_header + panel_height

        x = 20
        draw.text((x, y + 20), "Box colors:", fill="white", font=detail_font)
        x += 105
        for label in COLORS:
            text = SHORT[label]
            draw.rectangle((x, y + 18, x + 20, y + 38), outline=COLORS[label], width=4)
            draw.text((x + 27, y + 19), text, fill=COLORS[label], font=detail_font)
            x += 27 + draw.textlength(text, font=detail_font) + 28

        filename = f"{index:02d}_{reference['recording']}_{tile['ownership_start_timebin']:05d}.png"
        canvas.save(args.out / filename, optimize=True)

    (args.out / "README.txt").write_text(
        "Each PNG repeats the identical Qwen spectrogram across five rows so annotation changes can be compared directly.\n"
        "Rows are cumulative: direct, private reasoning, self-review, shifted independent review, and final adjudication.\n"
        "Green=target, yellow=uncertain, magenta=chorus, cyan=non-target biological, orange=anthropogenic noise, gray=environmental noise.\n")
    print(f"wrote {len(keys)} comparison images to {args.out}")


if __name__ == "__main__":
    main()
