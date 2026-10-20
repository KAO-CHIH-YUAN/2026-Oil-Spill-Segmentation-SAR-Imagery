#!/usr/bin/env python3
"""
Step 3 — TIF + JSON → PNG image + color mask PNG + YOLO TXT

開關 (在 Config 區塊設定):
  GENERATE_PNG  = True/False   — TIF → RGB PNG
  GENERATE_MASK = True/False   — JSON → 彩色 mask PNG
  GENERATE_TXT  = True/False   — JSON → YOLO polygon .txt (與 mask 同資料夾)

For every .tif found recursively in TIF_INPUT_ROOT:

  PNG:
    - Read TIF bands → uint8 RGB
    - uint8 TIF: bands 1-3 used directly
    - other dtypes: normalize per-band to [0, 255] using global min/max
    - Saved to PNG_OUTPUT_ROOT preserving folder hierarchy
    - Filename: all relative path parts joined with '_'
      e.g.  EventA/20230301/file_8b.tif  →  EventA_20230301_file_8b.png

  Mask:
    - Looks for first .json in JSON_INPUT_ROOT at same relative directory
    - Draws filled polygons on black (0,0,0) canvas:
        label "land"  → Green  (0, 153, 0)
        label "0"     → White  (255, 255, 255)
    - If no JSON found: skip mask, write WARNING to log
    - Saved to MASK_OUTPUT_ROOT with the same filename scheme

  TXT (YOLO polygon format):
    - Saved alongside mask in MASK_OUTPUT_ROOT
    - Only labels in LABEL_CLASS_ID are written (land 不輸出)
    - Format per line: <class_id> x1/W y1/H x2/W y2/H ...

  Log:
    - CSV written to PNG_OUTPUT_ROOT/convert_log.csv
    - Fields: tif_rel, png_status, mask_status, txt_status, warning
"""

import csv
import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image, ImageDraw
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None

# ─────────────────────────────────────────────────────────────
# Config — 由 preprocess_config.yaml 統一設定 (見 _pp_config.py)
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pp_config import CFG

TIF_INPUT_ROOT   = CFG.inputs["tif_root"]
JSON_INPUT_ROOT  = CFG.out["step2_merged"]
PNG_OUTPUT_ROOT  = CFG.out["step3_png"]
MASK_OUTPUT_ROOT = CFG.out["step3_mask"]

# ── 開關：True = 生成，False = 跳過 ──────────────────────────
GENERATE_PNG  = CFG.step3["generate_png"]
GENERATE_MASK = CFG.step3["generate_mask"]
GENERATE_TXT  = CFG.step3["generate_txt"]

# ── Mask 顏色設定 ─────────────────────────────────────────────
LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "land": (0, 153, 0),      # green
    "0":    (255, 255, 255),  # white
}

# ── YOLO TXT：label → class_id（不在此 dict 的 label 不輸出）──
LABEL_CLASS_ID: dict[str, int] = {
    "0":    0,  # oil spill
    "land": 1,  # land
}
# ─────────────────────────────────────────────────────────────


def make_output_stem(rel: Path) -> str:
    parts = list(rel.parent.parts) + [rel.stem]
    return "_".join(p for p in parts if p)


def tif_to_rgb(tif_path: Path) -> np.ndarray:
    with rasterio.open(tif_path) as src:
        h, w    = src.height, src.width
        dtype   = src.dtypes[0]
        n_bands = src.count

        read_bands = [1, 2, 3] if n_bands >= 3 else [1, 1, 1]
        out = np.zeros((h, w, 3), dtype=np.uint8)

        if dtype == "uint8":
            for ch_idx, band_idx in enumerate(read_bands):
                for _, window in src.block_windows(band_idx):
                    data = src.read(band_idx, window=window)
                    rs = window.row_off
                    re = window.row_off + window.height
                    cs = window.col_off
                    ce = window.col_off + window.width
                    out[rs:re, cs:ce, ch_idx] = data
        else:
            stats: dict[int, tuple[float, float]] = {}
            for band_idx in set(read_bands):
                mn, mx = float("inf"), float("-inf")
                for _, window in src.block_windows(band_idx):
                    block = src.read(band_idx, window=window).astype(np.float32)
                    valid = block[np.isfinite(block)]
                    if valid.size:
                        mn = min(mn, float(valid.min()))
                        mx = max(mx, float(valid.max()))
                stats[band_idx] = (
                    mn if mn != float("inf") else 0.0,
                    mx if mx != float("-inf") else 0.0,
                )

            for ch_idx, band_idx in enumerate(read_bands):
                mn, mx = stats[band_idx]
                for _, window in src.block_windows(band_idx):
                    block = src.read(band_idx, window=window).astype(np.float32)
                    if mx > mn:
                        block = np.clip((block - mn) / (mx - mn) * 255.0, 0.0, 255.0)
                    else:
                        block = np.zeros_like(block)
                    rs = window.row_off
                    re = window.row_off + window.height
                    cs = window.col_off
                    ce = window.col_off + window.width
                    out[rs:re, cs:ce, ch_idx] = block.astype(np.uint8)

    return out


def generate_mask(shapes: list, h: int, w: int) -> Image.Image:
    mask = Image.new("RGB", (w, h), (0, 0, 0))
    draw = ImageDraw.Draw(mask)
    for shape in shapes:
        color = LABEL_COLORS.get(shape.get("label", ""))
        if color is None:
            continue
        pts = shape.get("points", [])
        if len(pts) < 3:
            continue
        draw.polygon([(float(x), float(y)) for x, y in pts], fill=color)
    return mask


def generate_yolo_txt(shapes: list, h: int, w: int, dst: Path):
    lines = []
    for shape in shapes:
        class_id = LABEL_CLASS_ID.get(shape.get("label", ""))
        if class_id is None:
            continue
        pts = shape.get("points", [])
        if len(pts) < 3:
            continue
        coords = []
        for x, y in pts:
            coords.append(f"{min(max(float(x) / w, 0.0), 1.0):.6f}")
            coords.append(f"{min(max(float(y) / h, 0.0), 1.0):.6f}")
        lines.append(f"{class_id} " + " ".join(coords))
    dst.write_text("\n".join(lines), encoding="utf-8")


def process_one(tif_path: Path) -> dict:
    rel     = tif_path.relative_to(TIF_INPUT_ROOT)
    stem    = make_output_stem(rel)
    out_dir = rel.parent

    png_path  = PNG_OUTPUT_ROOT  / out_dir / (stem + ".png")
    mask_path = MASK_OUTPUT_ROOT / out_dir / (stem + ".png")
    txt_path  = MASK_OUTPUT_ROOT / out_dir / (stem + ".txt")

    result = {
        "tif_rel":     str(rel),
        "png_status":  "SKIP",
        "mask_status": "SKIP",
        "txt_status":  "SKIP",
        "warning":     "",
    }

    h, w = None, None

    # ── TIF → PNG ────────────────────────────────────────────
    if GENERATE_PNG:
        try:
            png_path.parent.mkdir(parents=True, exist_ok=True)
            rgb = tif_to_rgb(tif_path)
            h, w = rgb.shape[:2]
            Image.fromarray(rgb).save(png_path, "PNG")
            result["png_status"] = "OK"
        except Exception as e:
            result["png_status"] = f"FAILED: {e}"
            result["warning"]    = str(e)
            if not (GENERATE_MASK or GENERATE_TXT):
                return result

    # ── 若跳過 PNG 但需要 mask/txt，先讀取圖像尺寸 ──────────
    if (GENERATE_MASK or GENERATE_TXT) and h is None:
        try:
            with rasterio.open(tif_path) as src:
                h, w = src.height, src.width
        except Exception as e:
            msg = f"無法讀取TIF尺寸: {e}"
            if GENERATE_MASK:
                result["mask_status"] = f"FAILED: {msg}"
            if GENERATE_TXT:
                result["txt_status"]  = f"FAILED: {msg}"
            result["warning"] = msg
            return result

    # ── 找 JSON（目錄比對，取第一個）────────────────────────
    if GENERATE_MASK or GENERATE_TXT:
        json_dir        = JSON_INPUT_ROOT / rel.parent
        json_candidates = sorted(json_dir.glob("*.json"))

        if not json_candidates:
            msg = f"找不到對應 JSON: {json_dir.relative_to(BASE)}"
            if GENERATE_MASK:
                result["mask_status"] = "NO_JSON"
            if GENERATE_TXT:
                result["txt_status"]  = "NO_JSON"
            result["warning"] = msg
            return result

        try:
            with open(json_candidates[0], "r", encoding="utf-8") as f:
                shapes = json.load(f).get("shapes", [])
        except Exception as e:
            msg = str(e)
            if GENERATE_MASK:
                result["mask_status"] = f"FAILED: {msg}"
            if GENERATE_TXT:
                result["txt_status"]  = f"FAILED: {msg}"
            result["warning"] = msg
            return result

        # ── JSON → Mask ──────────────────────────────────────
        if GENERATE_MASK:
            try:
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                generate_mask(shapes, h, w).save(mask_path, "PNG")
                result["mask_status"] = "OK"
            except Exception as e:
                result["mask_status"] = f"FAILED: {e}"
                result["warning"]     = str(e)

        # ── JSON → YOLO TXT ──────────────────────────────────
        if GENERATE_TXT:
            try:
                txt_path.parent.mkdir(parents=True, exist_ok=True)
                generate_yolo_txt(shapes, h, w, txt_path)
                result["txt_status"] = "OK"
            except Exception as e:
                result["txt_status"] = f"FAILED: {e}"
                result["warning"]    = str(e)

    return result


def main():
    active = [n for n, v in [("PNG", GENERATE_PNG), ("Mask", GENERATE_MASK), ("TXT", GENERATE_TXT)] if v]
    print(f"啟用輸出: {', '.join(active) if active else '全部關閉'}")

    PNG_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    MASK_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(set(
        list(TIF_INPUT_ROOT.rglob("*.tif")) +
        list(TIF_INPUT_ROOT.rglob("*.tiff"))
    ))
    print(f"找到 {len(tif_files)} 個 TIF 檔案\n{'─'*60}")

    log_path   = PNG_OUTPUT_ROOT / "convert_log.csv"
    log_fields = ["tif_rel", "png_status", "mask_status", "txt_status", "warning"]

    rows   = []
    counts = {"ok_png": 0, "ok_mask": 0, "ok_txt": 0,
              "no_json": 0, "png_err": 0, "mask_err": 0, "txt_err": 0}

    for tif_path in tqdm(tif_files, desc="轉換中"):
        result = process_one(tif_path)
        rows.append(result)

        if GENERATE_PNG:
            if result["png_status"] == "OK":
                counts["ok_png"] += 1
            elif result["png_status"] != "SKIP":
                counts["png_err"] += 1
                tqdm.write(f"[PNG錯誤 ] {result['tif_rel']}: {result['png_status']}")

        if GENERATE_MASK:
            ms = result["mask_status"]
            if ms == "OK":
                counts["ok_mask"] += 1
            elif ms == "NO_JSON":
                counts["no_json"] += 1
                tqdm.write(f"[無JSON  ] {result['tif_rel']}")
            elif ms != "SKIP":
                counts["mask_err"] += 1
                tqdm.write(f"[Mask錯誤] {result['tif_rel']}: {ms}")

        if GENERATE_TXT:
            ts = result["txt_status"]
            if ts == "OK":
                counts["ok_txt"] += 1
            elif ts not in ("SKIP", "NO_JSON"):
                counts["txt_err"] += 1
                tqdm.write(f"[TXT錯誤 ] {result['tif_rel']}: {ts}")

    with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=log_fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'─'*60}")
    if GENERATE_PNG:
        print(f"PNG 成功  : {counts['ok_png']} 個  |  錯誤 {counts['png_err']} 個")
    if GENERATE_MASK:
        print(f"Mask 成功 : {counts['ok_mask']} 個  |  錯誤 {counts['mask_err']} 個  |  無JSON {counts['no_json']} 個")
    if GENERATE_TXT:
        print(f"TXT 成功  : {counts['ok_txt']} 個  |  錯誤 {counts['txt_err']} 個")
    print(f"Log       : {log_path}")


if __name__ == "__main__":
    main()
