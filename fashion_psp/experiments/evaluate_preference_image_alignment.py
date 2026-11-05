"""Inspect frozen preference/image pairs without loading GT or changing selection.

The independent OpenAI CLIP ViT-L/14 reports text/image cosine and the paired
preference's rank among the eight pre-generated alternatives. Neither is a
calibrated correctness probability. This is a diagnostic, not a reward gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[2]


def prompt(payload, category):
    return f"A product photo of a {category}. " + ". ".join(
        f"{key}: {value}" for key, value in payload.items()
    )


def make_panel(output, manifest, metadata, report):
    from PIL import Image, ImageDraw, ImageFont
    font_path = Path("C:/Windows/Fonts/arial.ttf")
    font = ImageFont.truetype(str(font_path), 21) if font_path.is_file() else ImageFont.load_default()
    small = ImageFont.truetype(str(font_path), 17) if font_path.is_file() else font
    winner = manifest["rewrites"][metadata["selected_rewrite_id"]]
    payload = json.loads(winner["preference_text"])
    canvas = Image.new("RGB", (1200, 730), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((22, 20), f"{manifest['uid']}_{manifest['oid']} | Rewrite-PSP@8 | Fixed Personal reward", fill="black", font=font)
    with Image.open(output / "selected.png") as image:
        canvas.paste(image.convert("RGB").resize((512, 512)), (20, 75))
    y = 85
    draw.text((560, y), "Preference used BEFORE generation", fill="black", font=font)
    y += 45
    for key, value in payload.items():
        draw.text((560, y), key, fill="#555555", font=small)
        y += 25
        for line in textwrap.wrap(value, width=52):
            draw.text((560, y), line, fill="black", font=font)
            y += 26
        y += 18
    row = next(r for r in report["pairs"] if r["phase"] == "terminal")
    draw.text((20, 610), f"Personal: {metadata['selected_personal_terminal']:.3f} | rewrite {winner['rewrite_id']}", fill="black", font=font)
    score = row["paired_text_image_cosine_x100"]
    if score is not None:
        draw.text((20, 647), f"Independent CLIP text/image cosine x100: {score:.3f} | paired preference rank: {row['paired_rank_of_8']}/8", fill="black", font=small)
    draw.text((20, 682), "CLIP is a proxy; inspect color, material appearance, shape and details separately. No GT comparison.", fill="#555555", font=small)
    canvas.save(output / "preference_image_pair.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=Path.home() / ".cache/clip/ViT-L-14.pt")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    import csv
    import open_clip
    import torch
    from PIL import Image

    output = args.output_dir.resolve()
    manifest = json.loads((output / "rewrite_manifest.json").read_text(encoding="utf-8"))
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    # Category sentence is generator input, not a label inferred from the image.
    category = manifest["sentence_text"].split("fashion ", 1)[1].split(" item", 1)[0]
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained=None, device=args.device, force_quick_gelu=True
    )
    open_clip.load_checkpoint(model, str(args.weights), strict=False, weights_only=False, device=args.device)
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-L-14")
    payloads = [json.loads(row["preference_text"]) for row in manifest["rewrites"]]
    prompts = [prompt(p, category) for p in payloads]

    def encode_text(texts):
        # Tokenizers truncate by default: explicitly detect it before scoring.
        valid = [len(tokenizer.encode(text)) <= model.context_length - 2 for text in texts]
        if not all(valid):
            return None, valid
        features = model.encode_text(tokenizer(texts).to(args.device)).float()
        return torch.nn.functional.normalize(features, dim=-1), valid

    records = list(csv.DictReader((output / "scores.csv").open(newline="", encoding="utf-8")))
    records = [r for r in records if r["phase"] == "psp"]
    records.append({"phase":"terminal", "step":28, "rewrite_id":metadata["selected_rewrite_id"],
                    "image_path":str(output / "selected.png")})
    pairs = []
    with torch.inference_mode():
        # Rank is unavailable if any full candidate description would truncate.
        text_features, valid_prompts = encode_text(prompts)
        for record in records:
            rid = int(record["rewrite_id"])
            with Image.open(record["image_path"]) as image:
                z = model.encode_image(preprocess(image.convert("RGB")).unsqueeze(0).to(args.device)).float()
            z = torch.nn.functional.normalize(z, dim=-1)
            scores = (100 * z @ text_features.T)[0].cpu().tolist() if text_features is not None else None
            paired_score = scores[rid] if scores is not None else None
            if paired_score is None and valid_prompts[rid]:
                f, _ = encode_text([prompts[rid]])
                paired_score = float((100 * z @ f.T).item())
            fields = {}
            for key, value in payloads[rid].items():
                f, valid = encode_text([prompt({key:value}, category)])
                fields[key] = float((100 * z @ f.T).item()) if all(valid) else None
            rank = 1 + sum(s > scores[rid] for s in scores) if scores is not None else None
            pairs.append({"phase":record["phase"], "step":int(record["step"]), "rewrite_id":rid,
                          "image_path":record["image_path"], "conditioning_preference":payloads[rid],
                          "paired_text_image_cosine_x100":paired_score,
                          "paired_rank_of_8":rank, "all_candidate_scores":scores,
                          "field_text_image_cosine_x100":fields})
    report = {"model":"independent frozen OpenAI CLIP ViT-L/14", "weights":str(args.weights.resolve()),
              "uses_gt":False, "affects_selection":False, "full_prompt_untruncated":valid_prompts,
              "limitation":"Cosine/rank are text-image alignment proxies, not correctness probabilities; material and fine detail need visual review.",
              "pairs":pairs}
    (output / "preference_alignment.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    make_panel(output, manifest, metadata, report)
    selected = next(row for row in pairs if row["phase"] == "terminal")
    print(json.dumps(selected, ensure_ascii=False, indent=2))
    print(output / "preference_image_pair.png")


if __name__ == "__main__":
    main()
