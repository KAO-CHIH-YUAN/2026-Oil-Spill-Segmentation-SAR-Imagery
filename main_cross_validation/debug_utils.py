"""
[File] debug_utils.py
[Purpose] Debugging Utilities for Dataset Verification.
[Description]
    - Verifies YOLO dataset integrity and format.
    - Checks for image corruption and missing labels.
    - Provides visualized verification of bounding boxes and masks.
[Author] Assistant (GitHub Copilot)
"""
import yaml
from pathlib import Path
import cv2

def debug_verify_dataset(dataset_path, image_subdir, label_subdir):
    """
    一個用於驗證 YOLO 資料集完整性和格式的除錯函式。
    - 遍歷圖片，檢查對應的標籤檔案是否存在且格式正確。
    - 嘗試讀取圖片以確保其未損壞。
    """
    print(f"\n--- [DEBUG] Starting dataset verification for: {image_subdir} ---")
    image_dir = Path(dataset_path) / image_subdir
    label_dir = Path(dataset_path) / label_subdir

    if not image_dir.is_dir():
        print(f"  [ERROR] Image directory not found: {image_dir}")
        return False
    if not label_dir.is_dir():
        print(f"  [ERROR] Label directory not found: {label_dir}")
        return False

    image_files = list(image_dir.glob('*.png'))
    if not image_files:
        print(f"  [WARNING] No images found in {image_dir}")
        return True

    is_ok = True
    for img_path in image_files:
        # 1. 驗證圖片檔案本身
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [ERROR] Failed to read image (file might be corrupt): {img_path.name}")
                is_ok = False
                continue
        except Exception as e:
            print(f"  [ERROR] Exception while reading image {img_path.name}: {e}")
            is_ok = False
            continue

        # 2. 驗證標籤檔案
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.exists() or label_path.stat().st_size == 0:
            # 標籤檔案不存在或為空，這代表是背景圖片，是合法情況。
            print(f"  [OK] Verified (background image): {img_path.name}")
            continue

        # 3. 驗證標籤檔案內容
        with open(label_path, 'r') as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 3 or len(parts) % 2 == 0:
                    print(f"  [ERROR] Invalid number of values in {label_path.name}, line {i+1}")
                    is_ok = False
                    break
                
                try:
                    # 檢查 class_id
                    class_id = int(parts[0])
                    # 檢查座標值
                    coords = [float(p) for p in parts[1:]]
                    if not all(0.0 <= c <= 1.0 for c in coords):
                        print(f"  [ERROR] Coordinate out of range [0, 1] in {label_path.name}, line {i+1}")
                        is_ok = False
                        break
                except ValueError:
                    print(f"  [ERROR] Non-numeric value found in {label_path.name}, line {i+1}")
                    is_ok = False
                    break
            
            if not is_ok:
                break # 如果內部迴圈出錯，跳出外部迴圈

        if is_ok:
            print(f"  [OK] Verified: {img_path.name}")

    if is_ok:
        print(f"--- [DEBUG] Dataset verification finished successfully for: {image_subdir} ---\n")
    else:
        print(f"--- [DEBUG] Dataset verification failed for: {image_subdir} ---\n")
        
    return is_ok
