"""
[File] patch_method_multi_processing_stable.py
[Purpose] Adaptive Image Patching with Multiprocessing (Stable Memory Version)
"""
import os
import cv2
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm
import random
import pandas as pd
from datetime import datetime
import concurrent.futures
import gc  # 引入垃圾回收機制

# 解除 PIL 對大圖的保護限制
Image.MAX_IMAGE_PIXELS = None
# 允許讀「不完整 / truncated」的 PNG (有些 600M+ 的 RGB output 會被誤判 truncated)
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==========================================
# ★ 核心防崩潰設定 ★
# 在 main() 中根據每張影像尺寸與系統即時記憶體動態派發任務
# ==========================================

import time

# 動態記憶體檢查函式 (改為單純評估)
def estimate_image_ram(image_path):
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            # 遙測大圖預估佔用 GB 數：寬 * 高 * 3(RGB) * Overhead(提升到20倍)
            return max(1.0, (w * h * 3 * 20) / (1024**3)) # 就算是很小的圖，也至少算它 1GB 的佔用，防範大量小圖海淹沒系統
    except:
        return 12.0 # 預設給一個較大的保護值

def get_total_ram_gb():
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemTotal' in line:
                    return int(line.split()[1]) / (1024**2)
    except:
        pass
    return 64.0 # 預設值

# --- 來源與輸出設定 (可用環境變數覆寫，方便同一支腳本跑不同裁切) ---
# 交接包預設路徑；換機器改這兩個預設或用 env 覆寫。
SOURCE_BASE_DIR = os.environ.get("PATCH_SOURCE_BASE",
    "/home/yuan/Yuan/OIL_Project_12_7/OIL_Project_Handoff/dataset/DV5_SAR_Preprocess_Output")
OUTPUT_BASE_DIR = os.environ.get("PATCH_OUTPUT_BASE",
    "/home/yuan/Yuan/OIL_Project_12_7/OIL_Project_Handoff/dataset/DV5_SAR_Land_Patch")
# 要切的 whole-image 資料夾(逗號分隔)。VV 用 "Step4_Dataset"；RGB 底用 "Step4_Dataset_RGB_LandMasked"
CATEGORIES = os.environ.get("PATCH_CATEGORIES", "Step4_Dataset").split(",")
SPLITS = os.environ.get("PATCH_SPLITS", "train,test").split(",")  # 本資料集只有 train/test

# --- Patching 參數設定 ---
PATCH_SIZE = int(os.environ.get("PATCH_SIZE", "2048"))

# === Mode 1: 雙軌 (預設) ===
# Train/Val 用 adaptive：pos 密集 (stride=PATCH_SIZE-OVERLAP_TRAIN_VAL)、neg 不重疊 (stride=PATCH_SIZE)
# Test 用 standard：全圖 stride=PATCH_SIZE-OVERLAP_TEST，pos+neg 同密度
OVERLAP_TRAIN_VAL = 512 # stride = PATCH_SIZE - OVERLAP_TRAIN_VAL
OVERLAP_TEST = 512      # stride = PATCH_SIZE - OVERLAP_TEST

# === Mode 2: 統一 stride (給 nested 5-fold CV 用) ===
# True 時忽略上面兩個 OVERLAP_*，改用 UNIFORM_OVERLAP 對 train/val/test 全部 split、pos+neg 全部用同一 stride
# 因為 nested CV 下 train 的影像會輪流變成其他 fold 的 test，stride 不一致會造成 metric 不公平
UNIFORM_OVERLAP_MODE = True
UNIFORM_OVERLAP      = int(os.environ.get("PATCH_UNIFORM_OVERLAP", "512"))   # stride = PATCH_SIZE - UNIFORM_OVERLAP

RANDOM_SEED = 42

# --- 背景樣本保留比例設定 ---
# (僅作用於 Train/Val 的背景樣本)
BACKGROUND_KEEP_RATIO = 1

SEPARATE_OUTPUT = True 
 
# --- PNG 轉 TXT 參數設定 ---
TXT_GENERATION_PARAMS = {
    "target_pixel_value": 255,   # 正樣本判定用（只看油污）
    "epsilon_factor": 0.002,
    "min_contour_area": 1.0,
    "classes": [
        {"class_id": 0, "pixel_value": 255},  # oil spill
        {"class_id": 1, "pixel_value": 90},   # land
    ],
}

# --- 儲存壓縮設定 (新增) ---
# optimize=True 會花多一點時間 CPU 計算，但能顯著減少 PNG 體積
# compress_level=9 為最高壓縮率 (0-9)，也會比較慢，但體積最小
PNG_SAVE_OPTIONS = {
    "optimize": False,
    "compress_level": 6
}

# ===================================================================

def check_target_pixels_exist(mask_segment_pil, target_value):
    segment_array = np.array(mask_segment_pil)
    return np.any(segment_array == target_value)

def slice_image_standard_generator(image_pil, mask_pil, patch_size, overlap, target_pixel_val):
    image_width, image_height = image_pil.size
    pad_color_image = (255, 255, 255) if image_pil.mode == 'RGB' else 255
    pad_value_mask = 0 
    stride = patch_size - overlap
    
    for y in range(0, image_height, stride):
        for x in range(0, image_width, stride):
            crop_box = (x, y, min(x + patch_size, image_width), min(y + patch_size, image_height))
            patch_img = image_pil.crop(crop_box)
            patch_mask = mask_pil.crop(crop_box)
            
            padded_img = Image.new(image_pil.mode, (patch_size, patch_size), pad_color_image)
            padded_seg = Image.new(mask_pil.mode, (patch_size, patch_size), pad_value_mask)
            padded_img.paste(patch_img, (0, 0))
            padded_seg.paste(patch_mask, (0, 0))
            
            is_positive = check_target_pixels_exist(patch_mask, target_pixel_val)
            yield padded_img, padded_seg, x, y, is_positive

def slice_image_adaptive_generator(image_pil, mask_pil, patch_size, positive_overlap, target_pixel_val):
    image_width, image_height = image_pil.size
    pad_color_image = (255, 255, 255) if image_pil.mode == 'RGB' else 255
    pad_value_mask = 0 

    # --- 密集掃描 (正樣本) ---
    stride_pos = patch_size - positive_overlap
    for y in range(0, image_height, stride_pos):
        for x in range(0, image_width, stride_pos):
            crop_box = (x, y, min(x + patch_size, image_width), min(y + patch_size, image_height))
            patch_mask = mask_pil.crop(crop_box)
            
            if check_target_pixels_exist(patch_mask, target_pixel_val):
                patch_img = image_pil.crop(crop_box)
                padded_img = Image.new(image_pil.mode, (patch_size, patch_size), pad_color_image)
                padded_seg = Image.new(mask_pil.mode, (patch_size, patch_size), pad_value_mask)
                padded_img.paste(patch_img, (0, 0))
                padded_seg.paste(patch_mask, (0, 0))
                yield padded_img, padded_seg, x, y, True

    # --- 稀疏掃描 (負樣本) ---
    stride_neg = patch_size 
    for y in range(0, image_height, stride_neg):
        for x in range(0, image_width, stride_neg):
            crop_box = (x, y, min(x + patch_size, image_width), min(y + patch_size, image_height))
            patch_mask = mask_pil.crop(crop_box)
            
            if not check_target_pixels_exist(patch_mask, target_pixel_val):
                patch_img = image_pil.crop(crop_box)
                padded_img = Image.new(image_pil.mode, (patch_size, patch_size), pad_color_image)
                padded_seg = Image.new(mask_pil.mode, (patch_size, patch_size), pad_value_mask)
                padded_img.paste(patch_img, (0, 0))
                padded_seg.paste(patch_mask, (0, 0))
                yield padded_img, padded_seg, x, y, False

# 直接接收 Numpy 陣列，大幅減少硬碟讀寫
def convert_mask_array_to_yolo_txt(mask_array, output_txt_path, params):
    try:
        height, width = mask_array.shape
        yolo_format_lines = []

        for cls in params["classes"]:
            class_id   = cls["class_id"]
            pixel_val  = cls["pixel_value"]

            binary_mask = np.zeros_like(mask_array, dtype=np.uint8)
            binary_mask[mask_array == pixel_val] = 255
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) < params["min_contour_area"]: continue
                epsilon = params["epsilon_factor"] * cv2.arcLength(contour, True)
                approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
                if len(approx_polygon) >= 3:
                    normalized_points = []
                    for point_wrapper in approx_polygon:
                        point = point_wrapper[0]
                        norm_x = max(0.0, min(1.0, point[0] / width))
                        norm_y = max(0.0, min(1.0, point[1] / height))
                        normalized_points.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])
                    if normalized_points:
                        yolo_format_lines.append(f"{class_id} {' '.join(normalized_points)}")

        with open(output_txt_path, 'w') as f:
            if yolo_format_lines:
                f.write("\n".join(yolo_format_lines) + "\n")
                return "contours_written"
            else:
                return "empty_file_written"
    except Exception as e:
        print(f"  [ERROR] YOLO 轉換時發生錯誤: {e}")
        return "error"

def process_single_image(job_args):
    # Worker subprocess 內重新確保 PIL 大圖 / truncated 設定（某些 OS 子行程不繼承）
    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    (
        image_filename, source_img_dir, source_mask_dir, main_output_path,
        category, split, patch_size, overlap, effective_bg_keep_ratio,
        separate_output, txt_generation_params, rel_dir
    ) = job_args

    # rel_dir != '.' 時，輸出也放到對應的事件子資料夾
    rel_suffix = (rel_dir,) if rel_dir != '.' else ()

    if separate_output:
        output_img_dir_pos = os.path.join(main_output_path, "images", f"{split}_pos", *rel_suffix)
        output_label_dir_pos = os.path.join(main_output_path, "labels", f"{split}_pos", *rel_suffix)
        output_img_dir_neg = os.path.join(main_output_path, "images", f"{split}_neg", *rel_suffix)
        output_label_dir_neg = os.path.join(main_output_path, "labels", f"{split}_neg", *rel_suffix)
        for d in [output_img_dir_pos, output_label_dir_pos, output_img_dir_neg, output_label_dir_neg]:
            os.makedirs(d, exist_ok=True)
    else:
        output_img_dir = os.path.join(main_output_path, "images", split, *rel_suffix)
        output_label_dir = os.path.join(main_output_path, "labels", split, *rel_suffix)
        for d in [output_img_dir, output_label_dir]:
            os.makedirs(d, exist_ok=True)

    patch_count_oil = 0
    patch_count_bg = 0
    img_width, img_height = 0, 0
    base_filename = os.path.splitext(image_filename)[0]
    image_path = os.path.join(source_img_dir, image_filename)
    mask_path = os.path.join(source_mask_dir, base_filename + ".png")
    
    if not os.path.exists(mask_path):
        return image_filename, 0, 0, 0, 0
        
    try:
        with Image.open(image_path) as img_pil_raw, Image.open(mask_path) as mask_pil_raw:
            img_width, img_height = img_pil_raw.size
            img_pil = img_pil_raw.convert("RGB")
            mask_pil = mask_pil_raw.convert("L")
            
            if UNIFORM_OVERLAP_MODE:
                # 統一 stride 模式：所有 split、pos+neg 都用同一密度
                generated_patches = slice_image_standard_generator(
                    img_pil, mask_pil, patch_size, overlap,
                    txt_generation_params["target_pixel_value"]
                )
            elif split == 'test':
                generated_patches = slice_image_standard_generator(
                    img_pil, mask_pil, patch_size, overlap,
                    txt_generation_params["target_pixel_value"]
                )
            else:
                generated_patches = slice_image_adaptive_generator(
                    img_pil, mask_pil, patch_size, overlap,
                    txt_generation_params["target_pixel_value"]
                )
            
            for patch_img, patch_mask, x, y, is_positive in generated_patches:
                if split != 'test' and not is_positive and (random.random() > effective_bg_keep_ratio):
                    continue

                patch_base_name = f"{base_filename}_patch_x{x}_y{y}"

                if separate_output:
                    current_img_dir, current_label_dir = (output_img_dir_pos, output_label_dir_pos) if is_positive else (output_img_dir_neg, output_label_dir_neg)
                else:
                    current_img_dir, current_label_dir = output_img_dir, output_label_dir
                
                output_img_png_path = os.path.join(current_img_dir, f"{patch_base_name}.png") 
                output_png_path = os.path.join(current_label_dir, f"{patch_base_name}.png")
                output_txt_path = os.path.join(current_label_dir, f"{patch_base_name}.txt")
                
                # 存檔圖片與遮罩 (啟用高壓縮率，無損)
                patch_img.save(output_img_png_path, **PNG_SAVE_OPTIONS)
                patch_mask.save(output_png_path, **PNG_SAVE_OPTIONS)

                # 存檔標註（不論正負樣本都呼叫，讓陸地也寫入 class 1）
                mask_array = np.array(patch_mask)
                convert_mask_array_to_yolo_txt(mask_array, output_txt_path, txt_generation_params)
                if is_positive:
                    patch_count_oil += 1
                else:
                    patch_count_bg += 1
                    
            # 確保大圖的變數被徹底清除
            del img_pil
            del mask_pil
            
    except Exception as e:
        print(f"\n  [ERROR] 處理檔案 '{image_filename}' 時發生嚴重錯誤: {e}")
        return image_filename, img_width, img_height, -1, -1 # 回傳 -1 代表失敗
    finally:
        # ★ 強制記憶體回收：每處理完一張大圖，立刻清理記憶體垃圾
        gc.collect()

    return image_filename, img_width, img_height, patch_count_oil, patch_count_bg

def main():
    random.seed(RANDOM_SEED)

    if UNIFORM_OVERLAP_MODE:
        base_dir_name = f"P{PATCH_SIZE}_OUni{UNIFORM_OVERLAP}_BG{int(BACKGROUND_KEEP_RATIO*100)}"
    else:
        base_dir_name = f"P{PATCH_SIZE}_TrainO{OVERLAP_TRAIN_VAL}_TestO{OVERLAP_TEST}_BG{int(BACKGROUND_KEEP_RATIO*100)}"

    if UNIFORM_OVERLAP_MODE:
        print(f"模式: 統一 stride (Uniform) — 給 nested CV 用")
        print(f"  - All splits, pos+neg 同 stride: {UNIFORM_OVERLAP} (= PATCH_SIZE - {UNIFORM_OVERLAP})")
    else:
        print(f"模式: 雙軌策略 (Hybrid Strategy) [v1.5_Stable]")
        print(f"  - Train/Val Overlap: {OVERLAP_TRAIN_VAL} (僅正樣本)")
        print(f"  - Test Overlap     : {OVERLAP_TEST} (標準全圖)")
    print(f"[安全機制] 將在處理各資料夾時，動態偵測每張影像尺寸與系統記憶體來派發任務...")
    print(f"[Categories] 將依序處理: {CATEGORIES}")

    current_run_stats = []
    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for category in CATEGORIES:
        # ── 每個 category 一個獨立輸出資料夾，加入 category suffix 避免互相覆蓋 ──
        # 例：CATEGORIES=["Step4_Dataset_RGB","Step4_Dataset_RGB_LandMasked"] →
        #     P2048_OUni512_BG100_RGB_Split / P2048_OUni512_BG100_RGB_LandMasked_Split
        # 規則：把 category 名稱去掉 "Step4_Dataset" / "Step4_Dataset_" 前綴後當 suffix
        cat_suffix = category.replace("Step4_Dataset_", "").replace("Step4_Dataset", "").strip("_")
        output_dir_name = base_dir_name + (f"_{cat_suffix}" if cat_suffix else "")
        if SEPARATE_OUTPUT:
            output_dir_name += "_Split"
        main_output_path = os.path.join(OUTPUT_BASE_DIR, output_dir_name)
        print(f"\n========== Category '{category}' → 輸出到: {main_output_path} ==========")

        for split in SPLITS:
            print(f"\n--- 開始處理: Category '{category}', Split '{split}' ---")
            
            source_img_dir = os.path.join(SOURCE_BASE_DIR, category, "images", split)
            source_mask_dir = os.path.join(SOURCE_BASE_DIR, category, "labels", split)
            
            if not os.path.isdir(source_img_dir): continue

            if SEPARATE_OUTPUT:
                os.makedirs(os.path.join(main_output_path, "images", f"{split}_pos"), exist_ok=True)
                os.makedirs(os.path.join(main_output_path, "images", f"{split}_neg"), exist_ok=True)
                os.makedirs(os.path.join(main_output_path, "labels", f"{split}_pos"), exist_ok=True)
                os.makedirs(os.path.join(main_output_path, "labels", f"{split}_neg"), exist_ok=True)
            else:
                os.makedirs(os.path.join(main_output_path, "images", split), exist_ok=True)
                os.makedirs(os.path.join(main_output_path, "labels", split), exist_ok=True)

            effective_bg_keep_ratio = BACKGROUND_KEEP_RATIO
            if split == 'test': effective_bg_keep_ratio = 1.0

            # 遞迴掃描圖片，支援事件子資料夾結構（如 images/train/{事件名}/img.png）
            image_entries = []  # (unique_key, filename, actual_img_dir, actual_mask_dir, rel_dir)
            for dirpath, dirnames, filenames in os.walk(source_img_dir):
                dirnames.sort()  # 保證跨平台順序一致
                rel_dir = os.path.relpath(dirpath, source_img_dir)
                actual_mask_dir = os.path.join(source_mask_dir, rel_dir) if rel_dir != '.' else source_mask_dir
                for fname in sorted(filenames):
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        ukey = fname if rel_dir == '.' else os.path.join(rel_dir, fname)
                        image_entries.append((ukey, fname, dirpath, actual_mask_dir, rel_dir))

            source_image_count = len(image_entries)
            if source_image_count == 0: continue

            if UNIFORM_OVERLAP_MODE:
                current_overlap = UNIFORM_OVERLAP
            elif split == 'test':
                current_overlap = OVERLAP_TEST
            else:
                current_overlap = OVERLAP_TRAIN_VAL

            # --- 建立/讀取 Checkpoint 機制 ---
            checkpoint_path = os.path.join(main_output_path, f"checkpoint_{split}.xlsx")
            checkpoint_records = []
            done_images = set()

            if os.path.exists(checkpoint_path):
                try:
                    df_cp = pd.read_excel(checkpoint_path)
                    checkpoint_records = df_cp.to_dict('records')
                    done_images = {str(r['ImageName']) for r in checkpoint_records if str(r.get('Status', '')) == 'Done'}
                    # 舊版 checkpoint 只存 basename；新版存 rel_dir/filename。兩種都要能比對。
                    done_basenames = {os.path.basename(k) for k in done_images}
                    print(f"[*] 解析 {split} 的 Checkpoint，已完成影像數量: {len(done_images)}")
                except Exception as e:
                    print(f"[警告] 讀取 {checkpoint_path} 失敗: {e}，將建立新的。")
                    done_basenames = set()
            else:
                done_basenames = set()

            # jobs 存 (job_tuple, unique_key)；unique_key 含子資料夾路徑以確保唯一性
            jobs = []
            for ukey, image_filename, img_subdir, mask_subdir, rel_dir in image_entries:
                if ukey in done_images or image_filename in done_basenames:
                    continue
                jobs.append((
                    (
                        image_filename, img_subdir, mask_subdir, main_output_path,
                        category, split, PATCH_SIZE, current_overlap, effective_bg_keep_ratio,
                        SEPARATE_OUTPUT, TXT_GENERATION_PARAMS, rel_dir
                    ),
                    ukey
                ))
            
            if len(jobs) == 0:
                print(f"  [Info] 所有 {split} 的影像都已切割完成，直接跳過！")
                total_oil = sum(r['PatchesWithObject'] for r in checkpoint_records if r.get('Status') == 'Done')
                total_bg = sum(r['BackgroundPatchesKept'] for r in checkpoint_records if r.get('Status') == 'Done')
                print(f"  結果 (From Checkpoint): 正樣本 {total_oil} 張, 負樣本 {total_bg} 張")
                continue

            total_oil, total_bg = 0, 0
            # 加入斷點續傳前已完成的計數
            total_oil += sum(r['PatchesWithObject'] for r in checkpoint_records if r.get('Status') == 'Done')
            total_bg += sum(r['BackgroundPatchesKept'] for r in checkpoint_records if r.get('Status') == 'Done')

            # 使用 ProcessPoolExecutor 發送任務，遙測大圖處理極吃記憶體與 I/O
            # 這裡強制鎖死最多只能開 3 個 Worker，不管 CPY 有多少核心！避免多圖齊開爆發
            max_cores = min(3, max(1, (os.cpu_count() or 4) - 1))

            # 準備動態監聽的隊列與靜態配額參數
            total_ram_gb = get_total_ram_gb()
            system_reserve_gb = 10.0 # 保留 16GB 給系統及其它程式，非常保守的安全值
            active_futures = {} # 記錄 active future -> estimated_gb
            future_to_key = {}  # 記錄 future -> unique_key（含子資料夾路徑）
            progress_bar = tqdm(total=len(jobs), desc=f"Processing {split}")

            with concurrent.futures.ProcessPoolExecutor(max_workers=max_cores) as executor:
                # 逐張發送任務
                for job_args, ukey in jobs:
                    image_filename = job_args[0]
                    img_path = os.path.join(job_args[1], image_filename)  # job_args[1] = actual_img_subdir
                    
                    # 估計該張影像所需的記憶體
                    estimated_gb = estimate_image_ram(img_path)
                    
                    # 開始循環檢查是否可以送出任務
                    while True:
                        # 1. 清理已經完成的任務，回收佔用狀態
                        done_futures = [fut for fut in active_futures.keys() if fut.done()]
                        
                        if done_futures:
                            for df in done_futures:
                                try:
                                    img_name, w, h, o, b = df.result()
                                    actual_key = future_to_key.get(df, img_name)
                                    if o != -1: # 不是失敗狀態
                                        total_oil += o
                                        total_bg += b
                                        # 記錄單張圖
                                        checkpoint_records.append({
                                            "ImageName": actual_key,
                                            "Width": w,
                                            "Height": h,
                                            "PatchesWithObject": o,
                                            "BackgroundPatchesKept": b,
                                            "TotalPatches": o + b,
                                            "PatchSize": PATCH_SIZE,
                                            "Overlap_TrainVal": OVERLAP_TRAIN_VAL,
                                            "Overlap_Test": OVERLAP_TEST,
                                            "Status": "Done"
                                        })
                                        # 即時更新 Excel
                                        pd.DataFrame(checkpoint_records).to_excel(checkpoint_path, index=False)
                                    else:
                                        print(f"\n  [WARNING] 影像 {actual_key} 發生錯誤未完成，不會標記 Checkpoint。")
                                except Exception as e:
                                    print(f"\n  [ERROR] Worker 發生錯誤或被終止: {e}")
                                active_futures.pop(df, None)
                            progress_bar.update(len(done_futures))
                        
                        # 2. 評估可不可以發送新任務
                        # 計算目前「正在跑的所有任務」總共會宣稱占用多少記憶體
                        running_estimated_gb = sum(active_futures.values())
                        
                        # [條件]:
                        # A. 運算任務尚未超過 CPU 核心限制
                        # B. 當前"已分配"的記憶體 + 這張新圖需佔用的記憶體 < (系統物理總記憶體 - 系統保護區)
                        # (若隊列是空，代表系統再滿也得硬擠出資源跑這唯一一張，避免卡死)
                        can_run = (len(active_futures) < max_cores) and ((running_estimated_gb + estimated_gb) < (total_ram_gb - system_reserve_gb))
                        
                        if can_run or len(active_futures) == 0:
                            break # 跳出 while，執行 submit
                            
                        # 如果不符合條件，強制等待已發送任務至少其中一部分做完
                        if active_futures:
                            concurrent.futures.wait(active_futures.keys(), return_when=concurrent.futures.FIRST_COMPLETED, timeout=2.0)
                        else:
                            time.sleep(1)
                            
                    # 正式發送任務
                    future = executor.submit(process_single_image, job_args)
                    active_futures[future] = estimated_gb
                    future_to_key[future] = ukey
                    
                # 等待最後一批任務全部完成
                if active_futures:
                    for future in concurrent.futures.as_completed(active_futures.keys()):
                        try:
                            img_name, w, h, o, b = future.result()
                            actual_key = future_to_key.get(future, img_name)
                            if o != -1:
                                total_oil += o
                                total_bg += b
                                checkpoint_records.append({
                                    "ImageName": actual_key,
                                    "Width": w,
                                    "Height": h,
                                    "PatchesWithObject": o,
                                    "BackgroundPatchesKept": b,
                                    "TotalPatches": o + b,
                                    "PatchSize": PATCH_SIZE,
                                    "Overlap_TrainVal": OVERLAP_TRAIN_VAL,
                                    "Overlap_Test": OVERLAP_TEST,
                                    "Status": "Done"
                                })
                                pd.DataFrame(checkpoint_records).to_excel(checkpoint_path, index=False)
                            else:
                                print(f"\n  [WARNING] 影像 {actual_key} 發生錯誤未完成。")
                        except Exception as e:
                            print(f"\n  [ERROR] Worker 發生錯誤或被終止: {e}")
                        progress_bar.update(1)
                        
            progress_bar.close()

            print(f"  結果: 正樣本 {total_oil} 張, 負樣本 {total_bg} 張")
            
            current_run_stats.append({
                "RunTimestamp": run_timestamp,
                "Category": category, 
                "Split": split,
                "SourceImageCount": source_image_count, 
                "PatchesWithObject": total_oil, 
                "BackgroundPatchesKept": total_bg,
                "TotalPatches": total_oil + total_bg, 
                "PatchSize": PATCH_SIZE,                
                "Overlap_TrainVal": OVERLAP_TRAIN_VAL,  
                "Overlap_Test": OVERLAP_TEST,           
                "Method": "Hybrid_v1.7_Adaptive_Stable" 
            })

    if current_run_stats:
        excel_log_path = os.path.join(OUTPUT_BASE_DIR, "patch_generation_log.xlsx")
        new_df = pd.DataFrame(current_run_stats)
        if os.path.exists(excel_log_path):
            pd.concat([pd.read_excel(excel_log_path), new_df], ignore_index=True).to_excel(excel_log_path, index=False)
        else:
            new_df.to_excel(excel_log_path, index=False)
            
    print("\n所有任務已安穩完成！")

if __name__ == "__main__":
    main()