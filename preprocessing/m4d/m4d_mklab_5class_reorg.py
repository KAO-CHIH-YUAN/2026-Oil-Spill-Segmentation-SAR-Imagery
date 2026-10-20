"""
[File] m4d_mklab_5class_reorg.py
[目的] 把原始 M4D (=MKLab, Krestenitis 2019) 重組成 pipeline 期望的格式，供
       FA-MobileUNet 5 類復刻使用。整張影像，不裁切 (resize 到 352 由 dataset transform 做)。

來源: dataset/M4Ds/M4D/{train,test}/images/*.jpg
      dataset/M4Ds/M4D/{train,test}/labels_1D/*.png   (整數類別 0=海 1=油 2=疑似 3=船 4=陸)
輸出: dataset/M4Ds/M4D_MKLab_5class/
        images/{train,test}/img_XXXX.jpg   (symlink → 原 jpg)
        labels/{train,test}/img_XXXX.png   (symlink → 原 labels_1D png, 保留 0-4)
      + README.txt
全部用 symlink (省空間)。train=1002 / test=110 (官方固定切分)。
"""
import os
from pathlib import Path

SRC = Path("/home/yuan/Yuan/OIL_Project_12_7/dataset/M4Ds/M4D")
OUT = Path("/home/yuan/Yuan/OIL_Project_12_7/dataset/M4Ds/M4D_MKLab_5class")


def link(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src.resolve())


def main():
    counts = {}
    for split in ("train", "test"):
        img_src = SRC / split / "images"
        lbl_src = SRC / split / "labels_1D"
        imgs = sorted(img_src.glob("*.jpg"))
        n = 0
        for f in imgs:
            lbl = lbl_src / f"{f.stem}.png"
            if not lbl.exists():
                print(f"  [skip] no label for {f.name}")
                continue
            link(f, OUT / "images" / split / f.name)             # img_XXXX.jpg
            link(lbl, OUT / "labels" / split / f"{f.stem}.png")  # img_XXXX.png
            n += 1
        counts[split] = n
        print(f"  {split}: {n} pairs linked")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "README.txt").write_text(
        "原始 M4D (MKLab, Krestenitis 2019) 5 類復刻用。整張影像 (1250x650)，不裁切。\n"
        "images/{train,test}: SAR 灰階三通道 jpg (symlink)。\n"
        "labels/{train,test}: labels_1D png, 整數 0海/1油/2疑似/3船/4陸 (symlink)。\n"
        "resize 到 352x352 由 FA-MobileUNet adapter 的 dataset transform 完成 (對齊論文)。\n"
        f"train={counts.get('train')} / test={counts.get('test')} (官方固定切分)。\n",
        encoding="utf-8")
    print(f"完成 → {OUT}")
    print(f"  train={counts.get('train')} test={counts.get('test')}")


if __name__ == "__main__":
    main()
