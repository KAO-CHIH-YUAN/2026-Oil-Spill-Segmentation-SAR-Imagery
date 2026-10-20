#!/usr/bin/env python3
"""
Step 4 — 依照 DV5_SAR_All 資料集結構，從 Step 3 輸出複製組裝新資料集

邏輯：
  1. 讀取 PNG_JSON_Match_Report.xlsx 建立對應表
       DV5_SAR_All_reorganized_png  (原始資料集檔名)
       DV5_All_TIF_CorrectName-JSON (Step3 輸出 stem)
  2. 走訪 DV5_SAR_All/images/** 取得每個 PNG 的路徑（split/category/name）
  3. 依 stem 查表找到 Step3 的 PNG / Mask / TXT
  4. 複製到 OUTPUT_ROOT，維持相同目錄階層；label 檔名與 image 檔名 stem 一致

輸出結構：
  Step4_Dataset/
  ├── images/{split}/{category}/{original_name}.png   ← Step3 PNG
  └── labels/{split}/{category}/{original_stem}.png   ← Step3 Mask
                                  {original_stem}.txt   ← Step3 YOLO TXT

原始 DV5_SAR_All 不做任何修改。
"""

import shutil
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# Config — 由 preprocess_config.yaml 統一設定 (見 _pp_config.py)
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pp_config import CFG

SOURCE_DATASET  = CFG.inputs["source_dataset"]
EXCEL_PATH      = CFG.inputs["excel_path"]
STEP3_PNG_ROOT  = CFG.out["step3_png"]
STEP3_MASK_ROOT = CFG.out["step3_mask"]
OUTPUT_ROOT     = CFG.out["step4_dataset"]

COL_TARGET = CFG.step4["col_target"]   # Excel 欄：原始資料集檔名
COL_STEP3  = CFG.step4["col_step3"]    # Excel 欄：Step3 stem（含 .json 副檔名）
# ─────────────────────────────────────────────────────────────


def build_index(root: Path, suffix: str) -> dict[str, Path]:
    """遞迴搜尋 root 下所有 suffix 檔案，回傳 {stem: path}。stem 重複時保留先找到的。"""
    index: dict[str, Path] = {}
    duplicates = []
    for p in root.rglob(f"*{suffix}"):
        if p.stem in index:
            duplicates.append((p.stem, index[p.stem], p))
        else:
            index[p.stem] = p
    if duplicates:
        print(f"  [警告] {suffix} 有 {len(duplicates)} 個重複 stem（保留先找到的）：")
        for stem, first, second in duplicates[:5]:
            print(f"    {stem}")
            print(f"      保留: {first}")
            print(f"      略過: {second}")
    return index


def main():
    # ── 讀取 Excel ────────────────────────────────────────────
    df = pd.read_excel(EXCEL_PATH)
    df[COL_TARGET] = df[COL_TARGET].astype(str).str.strip()
    df[COL_STEP3]  = df[COL_STEP3].astype(str).str.strip()

    # 原始檔名 stem → Step3 stem
    lookup: dict[str, str] = {}
    for _, row in df.iterrows():
        target_stem = Path(row[COL_TARGET]).stem
        step3_stem  = Path(row[COL_STEP3]).stem
        if target_stem and step3_stem and target_stem != "nan":
            lookup[target_stem] = step3_stem

    print(f"Excel 對應表: {len(lookup)} 筆")

    # ── 建立 Step3 索引 ───────────────────────────────────────
    print("建立 Step3 索引中...")
    idx_png  = build_index(STEP3_PNG_ROOT,  ".png")
    idx_mask = build_index(STEP3_MASK_ROOT, ".png")
    idx_txt  = build_index(STEP3_MASK_ROOT, ".txt")
    print(f"  Step3 PNG : {len(idx_png)} 個")
    print(f"  Step3 Mask: {len(idx_mask)} 個")
    print(f"  Step3 TXT : {len(idx_txt)} 個")

    # ── 走訪來源結構 ──────────────────────────────────────────
    image_files = sorted((SOURCE_DATASET / "images").rglob("*.png"))
    print(f"\n來源圖片: {len(image_files)} 個\n{'─'*60}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    stats  = {"ok": 0, "no_excel": 0, "no_step3": 0, "copy_err": 0}
    errors = []

    for img_path in tqdm(image_files, desc="複製中"):
        # rel = images/{split}/{category}/{name}.png
        rel           = img_path.relative_to(SOURCE_DATASET)
        original_stem = img_path.stem
        # split_cat = {split}/{category}（可能有更多層）
        split_cat     = Path(*rel.parts[1:-1])

        # ── Excel 查表 ────────────────────────────────────────
        step3_stem = lookup.get(original_stem)
        if step3_stem is None:
            tqdm.write(f"[無Excel ] {rel}")
            stats["no_excel"] += 1
            errors.append({"rel": str(rel), "reason": "Excel 中無對應"})
            continue

        # ── 確認 Step3 檔案存在 ───────────────────────────────
        missing = [name for name, idx in [("PNG", idx_png), ("Mask", idx_mask), ("TXT", idx_txt)]
                   if step3_stem not in idx]
        if missing:
            msg = f"Step3 缺少 {', '.join(missing)}  (stem={step3_stem})"
            tqdm.write(f"[缺Step3 ] {rel}  → {msg}")
            stats["no_step3"] += 1
            errors.append({"rel": str(rel), "reason": msg})
            continue

        # ── 複製 ─────────────────────────────────────────────
        try:
            dst_png  = OUTPUT_ROOT / "images" / split_cat / (step3_stem + ".png")
            dst_mask = OUTPUT_ROOT / "labels" / split_cat / (step3_stem + ".png")
            dst_txt  = OUTPUT_ROOT / "labels" / split_cat / (step3_stem + ".txt")

            dst_png.parent.mkdir(parents=True, exist_ok=True)
            dst_mask.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(idx_png[step3_stem],  dst_png)
            shutil.copy2(idx_mask[step3_stem], dst_mask)
            shutil.copy2(idx_txt[step3_stem],  dst_txt)

            stats["ok"] += 1

        except Exception as e:
            tqdm.write(f"[複製錯誤] {rel}: {e}")
            stats["copy_err"] += 1
            errors.append({"rel": str(rel), "reason": str(e)})

    # ── 統計 ─────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"成功      : {stats['ok']} 個")
    print(f"無Excel   : {stats['no_excel']} 個  (Excel 表中找不到此檔名)")
    print(f"缺Step3   : {stats['no_step3']} 個  (Step3 輸出中找不到 PNG/Mask/TXT)")
    print(f"複製錯誤  : {stats['copy_err']} 個")
    print(f"輸出      : {OUTPUT_ROOT}")

    if errors:
        print(f"\n錯誤詳情 ({len(errors)} 個):")
        for e in errors[:30]:
            print(f"  [{e['reason']}] {e['rel']}")
        if len(errors) > 30:
            print(f"  ... 還有 {len(errors) - 30} 個，請查看上方 tqdm 輸出")


if __name__ == "__main__":
    main()
