"""
[File] m4d_ourmethod_rgb_neutral_fullres.py
[目的] M4D「最新方法 = 陸地轉海洋平均值(LandNeutral)」的 binary 原尺寸中間產物。
       對齊 DV5 RGB_LandNeutral_close 的精神：陸地不塗白/黑，而是填海值(neutral)。
       海值定義 = 去除「陸地(4) + 油汙(1)」以外所有像素(class 0/2/3)的各通道均值。
       M4D 為乾淨資料(無 fake-nodata / 河流)，故 neutral 僅需 land(=4) → 該海值。

輸出: dataset/M4Ds/M4D-ourMethod/M4D_RGBours__landNeutral__fullres/
        images/{train,test}/img_XXXX.png   你的VV→RGB transform + 陸地填海值, 原尺寸1250×650
        labels/{train,test}/img_XXXX.png   binary 0/255 (油 labels_1D==1 →255, 其餘→0)
       + README.txt
供 m4d_ourmethod_patch_binary_ov128.py 切 512 patch (把該檔 SOURCE_BASE_DIR 指到此處)，
之後接 HNM 篩選(HNM 版) 與 merge(全部版)。

與 landWhite 版唯一差別: 陸地填色 White(255) → 該圖海面均值(neutral)。其餘流程一致。
"""
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))
import cv2
import numpy as np
import importlib.util
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import multiprocessing
cv2.setNumThreads(0)

M4D_ROOT = Path("/home/yuan/Yuan/OIL_Project_12_7/dataset/M4Ds/M4D")
OUT      = Path("/home/yuan/Yuan/OIL_Project_12_7/dataset/M4Ds/M4D-ourMethod/M4D_RGBours__landNeutral__fullres")
OIL_CLASS  = 1
LAND_CLASS = 4
MAX_WORKERS = 6
FALLBACK_SEA = (120, 120, 120)   # 萬一整張只有陸地+油時的後備值(BGR)

_spec = importlib.util.spec_from_file_location(
    "conv_rgb",
    "/home/yuan/Yuan/OIL_Project_12_7/preprocessing/RGB_Transformation/conver_origin_img_to_RGB.py")
_conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_conv)
sar_to_rgb_tiled = _conv.sar_to_rgb_tiled


def process_one(args):
    img_path, split = args
    name = Path(img_path).stem
    lbl_path = M4D_ROOT / split / "labels_1D" / f"{name}.png"
    try:
        m = cv2.imread(str(lbl_path), cv2.IMREAD_GRAYSCALE)
        if m is None:
            return f"[FAIL nolabel] {name}"
        rgb = sar_to_rgb_tiled(str(img_path))      # BGR 3ch
        if rgb is None:
            return f"[FAIL transform] {name}"
        rgb = rgb.copy()

        land = (m == LAND_CLASS)
        oil  = (m == OIL_CLASS)
        # neutral 填值 = 去除「陸地 + 油汙」以外所有像素(海/疑似/船 = class 0/2/3)的各通道均值
        ref = (~land) & (~oil)
        if ref.any():
            sea_val = rgb[ref].mean(axis=0).round().astype(np.uint8)
        else:
            sea_val = np.array(FALLBACK_SEA, dtype=np.uint8)
        if land.any():
            rgb[land] = sea_val

        bm = np.where(m == OIL_CLASS, 255, 0).astype(np.uint8)
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(OUT / "images" / split / f"{name}.png"), rgb, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(str(OUT / "labels" / split / f"{name}.png"), bm, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        return None
    except Exception as e:
        return f"[ERROR] {name}: {e}"


def main():
    tasks = []
    for split in ["train", "test"]:
        for f in sorted((M4D_ROOT / split / "images").glob("*.jpg")):
            tasks.append((str(f), split))
    print(f"M4D 你的RGB + 陸地→海值(neutral) 原尺寸中間產物: {len(tasks)} 張 → {OUT}")
    errors = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for r in tqdm(ex.map(process_one, tasks), total=len(tasks), desc="RGBours neutral fullres"):
            if r is not None:
                errors.append(r)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "README.txt").write_text(
        "M4D 最新方法(LandNeutral) binary 切割來源。你的VV→RGB transform + 陸地(labels_1D=4)填"
        "『去除陸地+油汙以外所有像素(class 0/2/3)的各通道均值』(neutral,非塗白)。原尺寸1250×650不pad不resize。"
        "binary油255。供 m4d_ourmethod_patch_binary_ov128.py 切512 patch → HNM/merge。\n",
        encoding="utf-8")
    if errors:
        print(f"\n{len(errors)} 錯誤:"); [print(" ", e) for e in errors[:10]]
    else:
        print("\n完成無錯誤。")
    for split in ["train", "test"]:
        print(f"  {split}: {len(list((OUT/'images'/split).glob('*.png')))} 張")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
