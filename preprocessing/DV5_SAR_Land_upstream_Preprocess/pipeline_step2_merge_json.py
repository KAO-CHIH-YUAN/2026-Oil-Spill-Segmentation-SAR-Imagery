#!/usr/bin/env python3
"""
Step 2 — Merge Land JSON + Annotation JSON → Complete JSON

Matching is by FOLDER (relative directory), not filename.
Any JSON found in the same relative directory is treated as a pair.
Output filename uses the ANNOT_JSON_ROOT filename as the authoritative name.

Merge rules:
  land + annot both in same dir  → use annot as base, strip existing land
                                   shapes, append fresh land shapes;
                                   output filename = annot JSON filename
  only land dir exists           → copy land JSON as-is
  only annot dir exists          → copy annot JSON as-is

Output mirrors the same folder hierarchy in OUTPUT_ROOT.
"""

import json
import shutil
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Config — 由 preprocess_config.yaml 統一設定 (見 _pp_config.py)
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pp_config import CFG

LAND_JSON_ROOT  = CFG.out["step1_landjson"]
ANNOT_JSON_ROOT = CFG.inputs["annotation_json_root"]
OUTPUT_ROOT     = CFG.out["step2_merged"]
# ─────────────────────────────────────────────────────────────


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data: dict, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Build dir-level index: relative_dir → json_path
    land_dirs: dict[Path, Path] = {}
    for p in LAND_JSON_ROOT.rglob("*.json"):
        rel_dir = p.parent.relative_to(LAND_JSON_ROOT)
        land_dirs.setdefault(rel_dir, p)   # keep first found per dir

    annot_dirs: dict[Path, Path] = {}
    if ANNOT_JSON_ROOT.exists():
        for p in ANNOT_JSON_ROOT.rglob("*.json"):
            rel_dir = p.parent.relative_to(ANNOT_JSON_ROOT)
            annot_dirs.setdefault(rel_dir, p)

    all_dirs = sorted(set(land_dirs) | set(annot_dirs))

    print(f"陸地JSON 資料夾 : {len(land_dirs)} 個")
    print(f"標註JSON 資料夾 : {len(annot_dirs)} 個")
    print(f"合計資料夾      : {len(all_dirs)} 個\n{'─'*60}")

    stats = {"both": 0, "land_only": 0, "annot_only": 0, "errors": []}

    for rel_dir in all_dirs:
        land_path  = land_dirs.get(rel_dir)
        annot_path = annot_dirs.get(rel_dir)

        try:
            if land_path and annot_path:
                # Output filename = annotation JSON filename
                dst = OUTPUT_ROOT / rel_dir / annot_path.name

                land_data  = load_json(land_path)
                annot_data = load_json(annot_path)

                annot_shapes = [s for s in annot_data.get("shapes", [])
                                if s.get("label") != "land"]
                land_shapes  = [s for s in land_data.get("shapes", [])
                                if s.get("label") == "land"]

                merged = dict(annot_data)
                merged["shapes"] = annot_shapes + land_shapes
                write_json(merged, dst)
                print(f"[合併  ] {rel_dir / annot_path.name}"
                      f"  (標註{len(annot_shapes)} + 陸地{len(land_shapes)})")
                stats["both"] += 1

            elif land_path:
                dst = OUTPUT_ROOT / rel_dir / land_path.name
                copy_file(land_path, dst)
                print(f"[僅陸地] {land_path.relative_to(LAND_JSON_ROOT)}")
                stats["land_only"] += 1

            else:  # annot only
                dst = OUTPUT_ROOT / rel_dir / annot_path.name
                copy_file(annot_path, dst)
                print(f"[僅標註] {rel_dir / annot_path.name}")
                stats["annot_only"] += 1

        except Exception as e:
            stats["errors"].append((rel_dir, str(e)))
            print(f"[錯誤  ] {rel_dir}: {e}")

    print(f"\n{'─'*60}")
    print(f"合併    : {stats['both']} 個")
    print(f"僅陸地  : {stats['land_only']} 個")
    print(f"僅標註  : {stats['annot_only']} 個")
    print(f"錯誤    : {len(stats['errors'])} 個")
    if stats["errors"]:
        print("\n錯誤詳情:")
        for p, e in stats["errors"]:
            print(f"  {p}: {e}")


if __name__ == "__main__":
    main()
