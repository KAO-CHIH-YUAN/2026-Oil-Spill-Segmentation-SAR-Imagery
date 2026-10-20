"""
Ours（整幅版）：大圖直接做 RGB windowed + LandNeutral close，再交由 patch_method 裁切。
= conver_origin_img_to_RGB_LandMasked 的 windowed RGB 編碼（完全重用其函式，保證一致）
  + 在整幅層級把 (陸地 ∪ 偽nodata ∪ 0/255 ∪ close河流) 且非油 的區域，填成該場景海面 RGB 均值。

與舊 patch 層級 build_rgb_landneutral_close_source.py 的差異：
  - 填充區直接用 label+偽nodata 算（不再繞 VV_neutral != VV_orig），數學等價。
  - 海值/河流在整幅一次算（不跨 patch 聚合、無 patch 重疊重複計數）→ 更乾淨。
labels 一律鏡像複製（供後續 patch_method 讀取）。

輸入/輸出可用環境變數覆寫：
  RGB_CLOSE_INPUT   (預設 Step4_Dataset)
  RGB_CLOSE_OUTPUT  (預設 Step4_Dataset_RGB_LandNeutral_close)
"""
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))
import sys
import shutil
import gc
import cv2
import numpy as np
from pathlib import Path
from scipy import ndimage
import concurrent.futures
import multiprocessing
from tqdm import tqdm

# 重用 LandMasked 的 windowed RGB 編碼 + 遮罩/偽nodata 函式（保證編碼完全一致）
sys.path.insert(0, str(Path(__file__).resolve().parent))
import conver_origin_img_to_RGB_LandMasked as base

cv2.setNumThreads(0)

INPUT_ROOT  = os.environ.get("RGB_CLOSE_INPUT",
    "/home/yuan/Yuan/OIL_Project_12_7/OIL_Project_Handoff/dataset/DV5_SAR_Preprocess_Output/Step4_Dataset")
OUTPUT_ROOT = os.environ.get("RGB_CLOSE_OUTPUT",
    "/home/yuan/Yuan/OIL_Project_12_7/OIL_Project_Handoff/dataset/DV5_SAR_Preprocess_Output/Step4_Dataset_RGB_LandNeutral_close")

LAND_MIN, LAND_MAX = base.LAND_GRAY_MIN, base.LAND_GRAY_MAX   # 20, 128
K_RIVERS   = 81           # close 河流 kernel（同 patch 版）
RIVER_MAX_PX = 1_500_000_000  # 超大場景跳過河流 morphology（同 patch 版）

MAX_WORKERS = base.get_safe_max_workers(estimated_mem_per_worker_gb=22.0)  # 保守：整幅需同時 RGB+VV+label+多個 mask


def compute_close(vv_gray, label, rgb_bgr):
    """整幅計算填充區與海值，回傳填好的 rgb_bgr（就地修改）。"""
    shp = vv_gray.shape
    if label is not None and label.shape != shp:
        label = cv2.resize(label, (shp[1], shp[0]), interpolation=cv2.INTER_NEAREST)
    if label is not None:
        land = (label >= LAND_MIN) & (label <= LAND_MAX)
        oil  = (label > 127)
    else:
        land = np.zeros(shp, bool); oil = np.zeros(shp, bool)

    fake = base.detect_uniform_fake_nodata(vv_gray)
    neutral_fill = (land | (vv_gray == 0) | (vv_gray == 255) | fake) & (~oil)
    del fake

    # close 河流（整幅 morphology；超大場景跳過）
    rivers = np.zeros(shp, bool)
    if land.any() and land.size <= RIVER_MAX_PX:
        closed = cv2.morphologyEx(land.astype(np.uint8), cv2.MORPH_CLOSE,
                                  np.ones((K_RIVERS, K_RIVERS), np.uint8))
        rivers = ndimage.binary_fill_holes(closed.astype(bool))
        del closed
    del land

    sea = (~neutral_fill) & (~oil) & (~rivers)
    if sea.any():
        sea_val = rgb_bgr[sea].mean(axis=0).round().astype(np.uint8)
    else:
        sea_val = np.array([120, 120, 120], np.uint8)
    del sea

    fill = (neutral_fill | rivers) & (~oil)
    del neutral_fill, rivers, oil
    rgb_bgr[fill] = sea_val
    del fill
    return rgb_bgr, sea_val


def process_one_file(args):
    src, in_root, out_root = args
    os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))
    cv2.setNumThreads(0)
    src_p = Path(src)
    try:
        rel = src_p.relative_to(Path(in_root))
    except ValueError:
        return f"[SKIP] path: {src}"
    dst = Path(out_root) / rel
    dst.parent.mkdir(parents=True, exist_ok=True)

    try:
        if "labels" in src_p.parts:
            shutil.copy2(src, dst)               # label 鏡像複製（供 patch 讀取）
            return None
        if "images" not in src_p.parts or src_p.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
            shutil.copy2(src, dst)
            return None
        if dst.exists() and dst.stat().st_size > 1024:
            return None                          # resume

        rgb = base.sar_to_rgb_tiled(str(src))    # 完全重用 LandMasked windowed 編碼
        if rgb is None:
            return f"[FAIL] rgb None: {src}"
        vv = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        lbl_path = base.find_label_for_image(str(src))
        label = cv2.imread(str(lbl_path), cv2.IMREAD_GRAYSCALE) if lbl_path is not None else None
        rgb, _ = compute_close(vv, label, rgb)
        del vv, label
        cv2.imwrite(str(dst), rgb, [cv2.IMWRITE_PNG_COMPRESSION, 4])
        del rgb; gc.collect()
        return None
    except Exception as e:
        gc.collect()
        return f"[ERROR] {src}: {e}"


def main():
    inp, outp = Path(INPUT_ROOT), Path(OUTPUT_ROOT)
    print(f"[整幅 RGB LandNeutral close]")
    print(f"  INPUT : {INPUT_ROOT}")
    print(f"  OUTPUT: {OUTPUT_ROOT}")
    print(f"  workers: {MAX_WORKERS}")
    files = []
    for root, _, fs in os.walk(inp):
        for f in fs:
            files.append(os.path.join(root, f))
    tasks = [(f, str(inp), str(outp)) for f in files]
    print(f"  共 {len(tasks)} 檔案")
    errors = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for res in tqdm(ex.map(process_one_file, tasks), total=len(tasks), desc="RGB LandNeutral close"):
            if res is not None:
                errors.append(res)
    if errors:
        print(f"\n{len(errors)} 個錯誤/警告：")
        for e in errors[:20]:
            print(" ", e)
    else:
        print("\n全部完成，無錯誤。")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
