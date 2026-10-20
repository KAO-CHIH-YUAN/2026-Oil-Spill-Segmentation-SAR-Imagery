"""
[File] m4d_ourmethod_mc_rgb_fullres.py
[目的] M4D 5-class「我們的方法」原尺寸中間產物，供 512 patch 切割 (原生解析度,不resize不變形)。
       VV→RGB transform (你的方法) + 保留 5-class labels_1D (不改陸地顏色, 陸地=class4)。
輸出 (結構對齊 M4D raw, 供 patch_method 讀取):
  dataset/M4Ds/M4D-ourMethod/M4D_mc__RGBours__fullres/
    {train,test}/images/img_XXXX.png     你的RGB transform, 原尺寸 1250×650
    {train,test}/labels_1D/img_XXXX.png   5-class ID mask (0-4)
"""
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))
import cv2, importlib.util
from pathlib import Path
from tqdm import tqdm
import concurrent.futures, multiprocessing
cv2.setNumThreads(0)

M4D_ROOT = Path("/home/yuan/Yuan/OIL_Project_12_7/dataset/M4Ds/M4D")
OUT      = Path("/home/yuan/Yuan/OIL_Project_12_7/dataset/M4Ds/M4D-ourMethod/M4D_mc__RGBours__fullres")
MAX_WORKERS = 6

_spec = importlib.util.spec_from_file_location(
    "conv_rgb", "/home/yuan/Yuan/OIL_Project_12_7/preprocessing/RGB_Transformation/conver_origin_img_to_RGB.py")
_conv = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_conv)
sar_to_rgb_tiled = _conv.sar_to_rgb_tiled


def process_one(args):
    img_path, split = args
    name = Path(img_path).stem
    lbl = M4D_ROOT / split / "labels_1D" / f"{name}.png"
    try:
        m = cv2.imread(str(lbl), cv2.IMREAD_GRAYSCALE)
        if m is None: return f"[FAIL nolabel] {name}"
        rgb = sar_to_rgb_tiled(str(img_path))
        if rgb is None: return f"[FAIL transform] {name}"
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels_1D").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(OUT / split / "images" / f"{name}.png"), rgb, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        cv2.imwrite(str(OUT / split / "labels_1D" / f"{name}.png"), m, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        return None
    except Exception as e:
        return f"[ERROR] {name}: {e}"


def main():
    tasks = [(str(f), sp) for sp in ["train", "test"]
             for f in sorted((M4D_ROOT / sp / "images").glob("*.jpg"))]
    print(f"M4D 5-class RGBours 原尺寸: {len(tasks)} 張 → {OUT}")
    errs = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for r in tqdm(ex.map(process_one, tasks), total=len(tasks), desc="mc RGBours fullres"):
            if r: errs.append(r)
    print("完成" if not errs else f"{len(errs)} 錯誤: {errs[:5]}")
    for sp in ["train", "test"]:
        print(f"  {sp}: {len(list((OUT/sp/'images').glob('*.png')))}")


if __name__ == "__main__":
    multiprocessing.freeze_support(); main()
