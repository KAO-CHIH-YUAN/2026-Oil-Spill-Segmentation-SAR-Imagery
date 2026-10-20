#!/usr/bin/env python3
"""
Step 1 — TIF → Land JSON + Inspection Overlay JPG (OSM Land Polygons)

For every .tif found recursively in TIF_INPUT_ROOT:
  1. Get WGS84 bounding box from TIF metadata
  2. Query OSM land polygons that intersect the extent
  3. Rasterize polygons into TIF pixel space
  4. Vectorize back to pixel-coordinate rings
  5. Write a labelme-style JSON with label="land" shapes (imageData=null)
  6. Write an overlay JPG for visual inspection (land in blue, 20% opacity,
     scaled to 50% with MAX_DIM cap to stay within JPEG size limits)

Output mirrors the TIF_INPUT_ROOT folder hierarchy in both output roots.
File naming for overlay: relative path parts joined with '_'
  e.g.  EventA/20230301/file_8b.tif  →  EventA_20230301_file_8b.jpg
"""

import json
import zipfile
from pathlib import Path

import fiona
import numpy as np
import requests
import rasterio
from PIL import Image, ImageDraw
from rasterio.crs import CRS
from rasterio.features import rasterize, shapes
from rasterio.warp import transform_bounds, transform_geom
from shapely.geometry import box as sh_box, shape as sh_shape, mapping

Image.MAX_IMAGE_PIXELS = None

# ─────────────────────────────────────────────────────────────
# Config — 由 preprocess_config.yaml 統一設定 (見 _pp_config.py)
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pp_config import CFG

TIF_INPUT_ROOT = CFG.inputs["tif_root"]
OUTPUT_ROOT    = CFG.out["step1_landjson"]
OVERLAY_ROOT   = CFG.out["step1_overlay"]
OSM_CACHE      = CFG.inputs["osm_cache"]

OSM_URL     = CFG.step1["osm_url"]
OSM_SHP_REL = CFG.step1["osm_shp_rel"]

MIN_LAND_PIXELS = CFG.step1["min_land_pixels"]
CRS_4326        = CRS.from_epsg(4326)

# Overlay settings
OVERLAY_SCALE  = CFG.step1["overlay_scale"]
OVERLAY_MAXDIM = CFG.step1["overlay_maxdim"]
JPEG_QUALITY   = CFG.step1["jpeg_quality"]
LAND_COLOR     = tuple(CFG.step1["land_color"])
# ─────────────────────────────────────────────────────────────


def ensure_osm_land() -> Path:
    """Download and cache OSM land polygons shapefile; return path to .shp."""
    shp = OSM_CACHE / OSM_SHP_REL
    if shp.exists():
        print(f"[OSM] 使用快取: {shp}")
        return shp

    OSM_CACHE.mkdir(parents=True, exist_ok=True)
    zip_path = OSM_CACHE / "land-polygons-complete-4326.zip"

    print(f"[OSM] 下載中: {OSM_URL}")
    r = requests.get(OSM_URL, stream=True, timeout=120)
    r.raise_for_status()
    total = int(r.headers.get("Content-Length", 0))
    downloaded = 0
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 // total
                print(f"\r  {pct:3d}%  ({downloaded / 1e6:.0f} / {total / 1e6:.0f} MB)",
                      end="", flush=True)
    print(f"\n[OSM] 下載完成 ({zip_path.stat().st_size / 1e6:.0f} MB)")

    print("[OSM] 解壓縮中 ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(OSM_CACHE)
    zip_path.unlink()
    print(f"[OSM] 完成: {shp}")
    return shp


def vectorize_land(land_raster: np.ndarray) -> list[list]:
    """Convert binary land raster (1=land) to pixel-coordinate polygon rings [[col, row], ...]."""
    result = []
    for geom_dict, val in shapes(land_raster.astype(np.uint8), connectivity=8):
        if val != 1:
            continue
        poly = sh_shape(geom_dict)
        if poly.area < MIN_LAND_PIXELS:
            continue
        if poly.geom_type == "Polygon":
            result.append([[c, r] for c, r in list(poly.exterior.coords)[:-1]])
        elif poly.geom_type == "MultiPolygon":
            for part in poly.geoms:
                if part.area >= MIN_LAND_PIXELS:
                    result.append([[c, r] for c, r in list(part.exterior.coords)[:-1]])
    return result


def _overlay_target_size(orig_h: int, orig_w: int) -> tuple[int, int]:
    new_h = max(1, int(orig_h * OVERLAY_SCALE))
    new_w = max(1, int(orig_w * OVERLAY_SCALE))
    longest = max(new_h, new_w)
    if longest > OVERLAY_MAXDIM:
        factor = OVERLAY_MAXDIM / longest
        new_h = max(1, int(new_h * factor))
        new_w = max(1, int(new_w * factor))
    return new_h, new_w


def _tif_to_rgb_scaled(tif_path: Path, new_h: int, new_w: int) -> np.ndarray:
    """Read TIF at (new_h, new_w) and return H×W×3 uint8 RGB array."""
    with rasterio.open(tif_path) as src:
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
            return np.stack([c.astype(np.uint8) for c in channels], axis=-1)
        return np.stack([norm(c) for c in channels], axis=-1)


def _write_overlay(tif_path: Path, rel: Path,
                   land_shapes: list, tif_h: int, tif_w: int):
    """Generate and save the inspection overlay JPG."""
    stem = "_".join(p for p in list(rel.parent.parts) + [rel.stem] if p)
    jpg_dst = OVERLAY_ROOT / rel.parent / (stem + ".jpg")

    new_h, new_w = _overlay_target_size(tif_h, tif_w)
    actual_scale = new_w / tif_w

    rgb = _tif_to_rgb_scaled(tif_path, new_h, new_w)
    base_rgba = Image.fromarray(rgb, "RGB").convert("RGBA")
    del rgb

    overlay_layer = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay_layer)
    for shape in land_shapes:
        pts = [(x * actual_scale, y * actual_scale) for x, y in shape["points"]]
        if len(pts) >= 3:
            draw.polygon(pts, fill=LAND_COLOR)

    composited = Image.alpha_composite(base_rgba, overlay_layer)
    del base_rgba, overlay_layer

    jpg_dst.parent.mkdir(parents=True, exist_ok=True)
    composited.convert("RGB").save(jpg_dst, "JPEG", quality=JPEG_QUALITY)
    del composited


def process(tif_path: Path, rel: Path, land_shp: Path) -> int:
    """
    Generate land JSON and overlay JPG for one TIF.
    Returns number of land shapes written.
    Overlay errors are printed as warnings but do not abort JSON generation.
    """
    with rasterio.open(tif_path) as src:
        tif_tfm   = src.transform
        tif_crs   = src.crs
        tif_h     = src.height
        tif_w     = src.width
        tif_shape = (tif_h, tif_w)

        if tif_crs.to_epsg() == 4326:
            b = src.bounds
            bounds_4326 = (b.left, b.bottom, b.right, b.top)
        else:
            bounds_4326 = transform_bounds(tif_crs, CRS_4326, *src.bounds)

    minx, miny, maxx, maxy = bounds_4326
    extent_box = sh_box(minx, miny, maxx, maxy)

    # Query OSM polygons within bounding box
    land_geoms = []
    with fiona.open(str(land_shp), "r") as shp_src:
        for feat in shp_src.filter(bbox=(minx, miny, maxx, maxy)):
            geom = sh_shape(feat["geometry"])
            if not geom.is_valid:
                geom = geom.buffer(0)
            clipped = geom.intersection(extent_box)
            if not clipped.is_empty:
                land_geoms.append(clipped)

    # Reproject geometries to TIF CRS if needed
    if tif_crs.to_epsg() != 4326 and land_geoms:
        land_geoms = [
            sh_shape(transform_geom("EPSG:4326", tif_crs, mapping(g)))
            for g in land_geoms
        ]

    land_shapes = []
    if land_geoms:
        land_raster = rasterize(
            [(g, 1) for g in land_geoms],
            out_shape=tif_shape,
            transform=tif_tfm,
            fill=0,
            dtype=np.uint8,
            all_touched=False,
        )
        for pts in vectorize_land(land_raster):
            land_shapes.append({
                "label":       "land",
                "points":      pts,
                "group_id":    None,
                "description": "",
                "shape_type":  "polygon",
                "flags":       {},
                "mask":        None,
            })

    # ── Write JSON ──
    data = {
        "version":     "5.3.1",
        "flags":       {},
        "shapes":      land_shapes,
        "imagePath":   tif_path.name,
        "imageData":   None,
        "imageHeight": tif_h,
        "imageWidth":  tif_w,
    }
    dst = OUTPUT_ROOT / rel.parent / (rel.stem + ".json")
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # ── Write overlay JPG ──
    try:
        _write_overlay(tif_path, rel, land_shapes, tif_h, tif_w)
    except Exception as e:
        print(f"  [overlay警告] {rel}: {e}")

    return len(land_shapes)


def main():
    land_shp = ensure_osm_land()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OVERLAY_ROOT.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(set(
        list(TIF_INPUT_ROOT.rglob("*.tif")) +
        list(TIF_INPUT_ROOT.rglob("*.tiff"))
    ))
    print(f"找到 {len(tif_files)} 個 TIF 檔案\n{'─'*60}")

    ok, errors = 0, []

    for tif_path in tif_files:
        rel = tif_path.relative_to(TIF_INPUT_ROOT)
        try:
            n = process(tif_path, rel, land_shp)
            print(f"[OK +{n:4d}land] {rel}")
            ok += 1
        except Exception as e:
            errors.append((rel, str(e)))
            print(f"[錯誤  ] {rel}: {e}")

    print(f"\n{'─'*60}")
    print(f"完成  : {ok} 個")
    print(f"錯誤  : {len(errors)} 個")
    if errors:
        print("\n錯誤詳情:")
        for p, e in errors:
            print(f"  {p}: {e}")


if __name__ == "__main__":
    main()
