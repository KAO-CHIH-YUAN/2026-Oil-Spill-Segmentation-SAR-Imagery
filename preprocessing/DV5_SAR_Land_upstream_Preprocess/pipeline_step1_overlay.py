#!/usr/bin/env python3
"""
Step 1 (Overlay) — Generate inspection overlay JPGs from TIF + Land JSON

For every land JSON produced by Step 1:
  - Load the TIF at reduced resolution (SCALE × size, capped at MAX_DIM on longest side)
  - Overlay land shapes in blue (20% opacity)
  - Save as JPG (quality JPEG_QUALITY) to OVERLAY_ROOT

Resizing ensures no image exceeds JPEG dimension limits even for very large TIFs.

File naming: all relative path parts joined with '_'
  e.g.  EventA/20230301/file_8b.tif  →  EventA_20230301_file_8b.jpg
Folder hierarchy is preserved.
"""

import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

# ─────────────────────────────────────────────────────────────
# Config — edit these paths before running
BASE = Path("/home/yuan/Yuan/OIL_Project_12_7/dataset/DV5_reorder/DV5_SAR_Preprocess")

TIF_INPUT_ROOT  = BASE / "DV5_All_TIF"
LAND_JSON_ROOT  = BASE / "Preprocess_Output" / "Step1_LandJSON"
OVERLAY_ROOT    = BASE / "Preprocess_Output" / "Step1_LandOverlay"

SCALE        = 0.5
MAX_DIM      = 4096          # hard cap on longest side after scaling
JPEG_QUALITY = 50
LAND_COLOR   = (0, 0, 255, int(255 * 0.2))  # blue, 20% opacity
# ─────────────────────────────────────────────────────────────


def compute_target_size(orig_h: int, orig_w: int) -> tuple[int, int]:
    new_h = max(1, int(orig_h * SCALE))
    new_w = max(1, int(orig_w * SCALE))
    longest = max(new_h, new_w)
    if longest > MAX_DIM:
        factor = MAX_DIM / longest
        new_h = max(1, int(new_h * factor))
        new_w = max(1, int(new_w * factor))
    return new_h, new_w


def tif_to_rgb_scaled(tif_path: Path) -> tuple[np.ndarray, float]:
    """
    Read TIF at target resolution using rasterio out_shape.
    Returns (H×W×3 uint8 array, actual_scale factor).
    """
    with rasterio.open(tif_path) as src:
        orig_h, orig_w = src.height, src.width
        new_h, new_w = compute_target_size(orig_h, orig_w)
        actual_scale = new_w / orig_w

        n_bands = src.count
        read_bands = [1, 2, 3] if n_bands >= 3 else [1, 1, 1]

        def read_band(idx: int) -> np.ndarray:
            return src.read(
                idx,
                out_shape=(new_h, new_w),
                resampling=rasterio.enums.Resampling.lanczos,
            ).astype(np.float32)

        def norm(ch: np.ndarray) -> np.ndarray:
            valid = ch[np.isfinite(ch)]
            if valid.size == 0:
                return np.zeros(ch.shape, dtype=np.uint8)
            p2, p98 = np.percentile(valid, [2, 98])
            if p98 == p2:
                return np.zeros(ch.shape, dtype=np.uint8)
            return np.clip((ch - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)

        channels = [read_band(b) for b in read_bands]

        if src.dtypes[0] == "uint8":
            rgb = np.stack([c.astype(np.uint8) for c in channels], axis=-1)
        else:
            rgb = np.stack([norm(c) for c in channels], axis=-1)

    return rgb, actual_scale


def draw_land_overlay(base_rgba: Image.Image, shapes: list, scale: float) -> Image.Image:
    """Draw land polygons onto base_rgba with LAND_COLOR fill."""
    overlay = Image.new("RGBA", base_rgba.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for shape in shapes:
        if shape.get("label") != "land":
            continue
        pts = shape.get("points", [])
        if len(pts) < 3:
            continue
        scaled_pts = [(x * scale, y * scale) for x, y in pts]
        draw.polygon(scaled_pts, fill=LAND_COLOR)
    return Image.alpha_composite(base_rgba, overlay)


def make_output_stem(rel: Path) -> str:
    parts = list(rel.parent.parts) + [rel.stem]
    return "_".join(p for p in parts if p)


def process_one(tif_path: Path) -> dict:
    rel  = tif_path.relative_to(TIF_INPUT_ROOT)
    stem = make_output_stem(rel)
    out_dir = rel.parent

    json_path = LAND_JSON_ROOT / rel.parent / (rel.stem + ".json")
    jpg_path  = OVERLAY_ROOT   / out_dir    / (stem + ".jpg")

    result = {"tif_rel": str(rel), "status": "", "warning": ""}

    if not json_path.exists():
        result["status"]  = "NO_JSON"
        result["warning"] = f"找不到陸地JSON: {json_path.relative_to(BASE)}"
        return result

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        shapes = data.get("shapes", [])

        rgb, actual_scale = tif_to_rgb_scaled(tif_path)
        base_img  = Image.fromarray(rgb, "RGB")
        base_rgba = base_img.convert("RGBA")
        del base_img

        composited = draw_land_overlay(base_rgba, shapes, actual_scale)
        del base_rgba

        jpg_path.parent.mkdir(parents=True, exist_ok=True)
        composited.convert("RGB").save(jpg_path, "JPEG", quality=JPEG_QUALITY)
        del composited

        result["status"] = "OK"
    except Exception as e:
        result["status"]  = f"FAILED: {e}"
        result["warning"] = str(e)

    return result


def main():
    OVERLAY_ROOT.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(set(
        list(TIF_INPUT_ROOT.rglob("*.tif")) +
        list(TIF_INPUT_ROOT.rglob("*.tiff"))
    ))
    print(f"找到 {len(tif_files)} 個 TIF 檔案\n{'─'*60}")

    counts = {"ok": 0, "no_json": 0, "failed": 0}
    warnings = []

    for tif_path in tqdm(tif_files, desc="生成overlay"):
        result = process_one(tif_path)
        s = result["status"]

        if s == "OK":
            counts["ok"] += 1
        elif s == "NO_JSON":
            counts["no_json"] += 1
            warnings.append(result)
            tqdm.write(f"[無JSON ] {result['tif_rel']}")
        else:
            counts["failed"] += 1
            warnings.append(result)
            tqdm.write(f"[錯誤  ] {result['tif_rel']}: {s}")

    print(f"\n{'─'*60}")
    print(f"成功    : {counts['ok']} 個")
    print(f"無JSON  : {counts['no_json']} 個")
    print(f"錯誤    : {counts['failed']} 個")

    if warnings:
        print("\n警告詳情:")
        for r in warnings:
            print(f"  [{r['status']}] {r['tif_rel']}: {r['warning']}")


if __name__ == "__main__":
    main()
