"""Create spatial Personal occlusion maps for selected FITB cases.

The Personal reward is a global CLIP-history cosine, so it does not expose
native pixel attention.  This script uses a faithful perturbation diagnostic:
mask one image patch at a time, recompute Personal, and visualize the score
drop.  Positive drops are regions that the frozen Personal scorer needs most.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont, ImageFilter


STEPS = (7, 14, 28)
WEIGHTS = Path(r"C:\Users\Administrator\.cache\huggingface\hub\models--laion--CLIP-ViT-H-14-laion2B-s32B-b79K\snapshots\1c2b8495b28150b8a4922ee1c8edee224c284c0c\open_clip_model.safetensors")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--analysis-root", type=Path, default=Path(r"E:\difashion-PSP\DualFashion-main\output\ifashion_fitb_timestep200_personal_analysis"))
    p.add_argument("--data-root", type=Path, default=Path(r"C:\tmp\ifashion"))
    p.add_argument("--task-indices", type=int, nargs="+", default=[98, 115, 33])
    p.add_argument("--weights", type=Path, default=WEIGHTS)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--grid-size", type=int, default=7)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output-root", type=Path, default=None)
    return p.parse_args()


def font(size: int):
    for name in ("arial.ttf", "segoeui.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def read_candidates(path: Path) -> dict[tuple[int, int], list[dict[str, str]]]:
    out: dict[tuple[int, int], list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            out.setdefault((int(row["task_index"]), int(row["step"])), []).append(row)
    return out


def load_personal_clip(weights: Path, device: torch.device):
    import open_clip
    from safetensors.torch import load_file

    if not weights.is_file():
        raise FileNotFoundError(weights)
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-H-14", pretrained=None)
    state = load_file(str(weights), device="cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Personal CLIP checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}")
    model = model.to(device).eval()
    if device.type == "cuda":
        model = model.half()
    return model, preprocess


@torch.inference_mode()
def score_pils(model, preprocess, images: list[Image.Image], history: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    target = F.normalize(torch.as_tensor(history, dtype=torch.float32, device=device), dim=0)
    values = []
    dtype = next(model.parameters()).dtype
    for start in range(0, len(images), batch_size):
        pixels = torch.stack([preprocess(image) for image in images[start:start + batch_size]]).to(device, dtype=dtype)
        features = F.normalize(model.encode_image(pixels).float(), dim=-1)
        values.append((100.0 * (features @ target)).detach().cpu().numpy())
    return np.concatenate(values)


def make_occlusions(image: Image.Image, grid_size: int) -> list[Image.Image]:
    image = image.convert("RGB")
    width, height = image.size
    # Match the white product background while preserving the original edges.
    occlusions = []
    for row in range(grid_size):
        for col in range(grid_size):
            masked = image.copy()
            left = int(col * width / grid_size)
            right = int((col + 1) * width / grid_size)
            top = int(row * height / grid_size)
            bottom = int((row + 1) * height / grid_size)
            draw = ImageDraw.Draw(masked)
            draw.rectangle((left, top, right, bottom), fill=(255, 255, 255))
            occlusions.append(masked)
    return occlusions


def overlay_heatmap(image: Image.Image, drops: np.ndarray, grid_size: int) -> Image.Image:
    image = image.convert("RGB")
    positive = np.maximum(drops, 0.0).reshape(grid_size, grid_size)
    if positive.max() > 0:
        positive = positive / positive.max()
    small = Image.fromarray(np.uint8(positive * 255), mode="L")
    heat = small.resize(image.size, Image.Resampling.BICUBIC)
    # Red alpha is proportional to Personal score drop.
    rgba = Image.new("RGBA", image.size, (235, 55, 45, 0))
    rgba.putalpha(heat.point(lambda value: int(185 * value / 255)))
    return Image.alpha_composite(image.convert("RGBA"), rgba).convert("RGB")


def build_case(task: int, candidates: dict[tuple[int, int], list[dict[str, str]]], history_vector: np.ndarray, uid: int, output: Path, model, preprocess, device, grid_size: int, batch_size: int) -> None:
    cell = 280
    left = 106
    top = 68
    row_gap = 28
    panel_width = left + 2 * cell + 20
    panel_height = top + 3 * cell + 2 * row_gap + 42
    canvas = Image.new("RGB", (panel_width, panel_height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    title_font, label_font, small_font = font(22), font(16), font(14)
    draw.text((16, 15), f"FITB task {task:04d} | Personal occlusion map", fill=(25, 25, 25), font=title_font)
    draw.text((16, 43), "Red = larger Personal score drop after masking that region", fill=(70, 70, 70), font=small_font)
    rows_for_task = [candidates[(task, step)] for step in STEPS]
    for index, (step, values) in enumerate(zip(STEPS, rows_for_task)):
        selected = max(values, key=lambda row: float(row["personal"]))
        original = Image.open(selected["image_path"]).convert("RGB")
        baseline = score_pils(model, preprocess, [original], history_vector, device, batch_size)[0]
        masked = make_occlusions(original, grid_size)
        occluded_scores = score_pils(model, preprocess, masked, history_vector, device, batch_size)
        drops = baseline - occluded_scores
        heatmap = overlay_heatmap(original, drops, grid_size)
        y = top + index * (cell + row_gap)
        for x, image in ((left, original), (left + cell, heatmap)):
            image.thumbnail((cell, cell), Image.Resampling.LANCZOS)
            canvas.paste(image, (x, y))
            draw.rectangle((x, y, x + cell - 1, y + cell - 1), outline=(45, 45, 45), width=2)
        draw.text((16, y + 8), f"step {step}", fill=(25, 25, 25), font=label_font)
        draw.text((left, y + cell + 7), f"best candidate | Personal {baseline:.1f}", fill=(45, 45, 45), font=small_font)
        draw.text((left + cell, y + cell + 7), "occlusion Personal", fill=(45, 45, 45), font=small_font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> None:
    parsed = parse_args()
    try:
        import open_clip  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("open_clip is required in the dualfashion environment") from exc
    device = torch.device(parsed.device)
    candidates = read_candidates(parsed.analysis_root / "candidate_metrics_with_gt.csv")
    fitb = np.load(parsed.data_root / "processed" / "fitb_test.npy", allow_pickle=True).item()
    history = np.load(parsed.data_root / "processed" / "test_history_clipembs.npy", allow_pickle=True).item()
    model, preprocess = load_personal_clip(parsed.weights, device)
    output_root = (parsed.output_root or parsed.analysis_root / "mechanism_analysis" / "case_panels_step7_14_28").resolve()
    panel_paths = []
    for task in parsed.task_indices:
        uid = int(fitb["uids"][task])
        incomplete = [int(value) for value in fitb["outfits"][task]]
        blank = incomplete.index(0)
        category = int(fitb["category"][task][blank])
        user_history = history[uid]
        history_vector = user_history.get(category, user_history.get(str(category)))
        if history_vector is None:
            raise KeyError(f"missing Personal history vector for uid={uid}, category={category}")
        panel_path = output_root / f"task_{task:04d}_personal_occlusion.png"
        build_case(task, candidates, np.asarray(history_vector), uid, panel_path, model, preprocess, device, parsed.grid_size, parsed.batch_size)
        panel_paths.append(panel_path)
    panels = [Image.open(path).convert("RGB") for path in panel_paths]
    if panels:
        combined = Image.new("RGB", (max(image.width for image in panels), sum(image.height for image in panels) + 24 * (len(panels) - 1)), (250, 250, 250))
        y = 0
        for index, image in enumerate(panels):
            combined.paste(image, (0, y))
            y += image.height
            if index + 1 < len(panels):
                y += 24
        combined.save(output_root / "selected_personal_occlusion_step7_14_28.png")
    print(output_root)


if __name__ == "__main__":
    main()
