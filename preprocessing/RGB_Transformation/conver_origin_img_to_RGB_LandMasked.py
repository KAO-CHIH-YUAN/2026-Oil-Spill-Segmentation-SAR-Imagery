"""
[File] conver_origin_img_to_RGB_LandMasked.py

跟 conver_origin_img_to_RGB.py 同樣的 VV→RGB 流程 (Kussul 2025 ISPRS 風格)，
差別：**計算 G/B 統計時把 OSM 陸地像素也視為「不存在」**，避免陸地影響 mean/std/local pooling。
陸地像素在最終 RGB 輸出仍然可見（用無偏 stats 算出的值，不會被強制 0）。

對應的 label PNG 用 path mirror 找：images/.../<file>.png → labels/.../<file>.png

陸地像素值約定（沿用本專案其他模組）：
    0  = 背景 / 海洋 / 無效
   90  = 陸地 (OSM MASK_LAND_PIXEL)
  255  = 油污 / 無效亮值

統計用遮罩 (stats_mask):  排除 0, 255 (IGNORE_VALUES) + 陸地 (灰階 20~128)
顯示用遮罩 (display_mask): 排除 0, 255 (跟原檔一致)；陸地保留可見

用法跟原檔一樣（多進程平行 + 自動 worker 數 + tile-based）：
    python conver_origin_img_to_RGB_LandMasked.py
輸出 path: {INPUT_ROOT 改 Step4_Dataset → Step4_Dataset_RGB_LandMasked}
"""
import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))

import cv2
import numpy as np
import shutil
import gc
from tqdm import tqdm
from pathlib import Path
import concurrent.futures
import multiprocessing

# ================= 設定區域 =================
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))
cv2.setNumThreads(0)

INPUT_ROOT  = r"/home/yuan/Yuan/OIL_Project_12_7/OIL_Project_Handoff/dataset/DV5_SAR_Preprocess_Output/Step4_Dataset"
OUTPUT_ROOT = r"/home/yuan/Yuan/OIL_Project_12_7/OIL_Project_Handoff/dataset/DV5_SAR_Preprocess_Output/Step4_Dataset_RGB_LandMasked"

TILE_SIZE = 1280

# 顯示用：排除 0 (純黑無效) 和 255 (純白無效)。陸地 (90) 不排除 → 仍可見。
IGNORE_VALUES = [0, 255]

# 陸地像素灰階範圍 (用於 stats_mask)
# 嚴格：== 90；保險：20~128 (避免 PNG 壓縮造成 ±幾個像素的偏移)
LAND_GRAY_MIN = 20
LAND_GRAY_MAX = 128

# === 偽 no-data 偵測 (對抗 SAR 切邊填值 217 / 244 / 254 等非極值 fill) ===
# True：用 local std 偵測連片均勻極值區域 → 同時排除於 stats + display (塗黑)
DETECT_FAKE_NODATA       = True
FAKE_NODATA_BRIGHT_THRESH = 200
FAKE_NODATA_DARK_THRESH   = 15
FAKE_NODATA_UNIFORM_STD   = 5.0
FAKE_NODATA_WINDOW        = 9

# === G band std floor (對抗 calm sea 大圖過飽和) ===
# 物理常數，非 data-driven。當 local std < 此值時，G band 用此值代替，
# 避免 arctan(z) 在 calm sea (std≈5) 場景被噪聲推進 saturate。
# 依據：~Sentinel-1 GRDH 在典型入射角下的 speckle noise floor 估計值
#       (低於此 contrast level 視為訊號統計上不可分辨於 speckle)。
# 不從 dataset 算 P10 等百分位數，避免 test 統計洩漏進 preprocessing。
G_BAND_STD_FLOOR = 10.0

# === G band 統計尺度：windowed local (取代原 image-level) ===
# 從「全圖一個 mean/std」改成「windowed local mean/std field」(per-pixel)
# 視窗在 original-resolution 大小 = G_BAND_WINDOW pixel
# Sentinel-1 10m/pixel × 2048 ≈ 20 km regional context，正好填 B band local(~2km) 跟
# image-level(~100+km) 之間的尺度缺口。
#
# 理由：
#   - Terrain Correction + Calibration 已吸收 scene-level drift，per-image stats 已無
#     額外資訊；改 windowed 不會丟掉論文真正想要的東西。
#   - PoC (windowed_g_poc/gulf_cross_region_strip.png) 顯示不同 sea-state region
#     local_mean 可差 130+ 點，per-image G 把這個差異強行套單一基準 → 油污訊號跨
#     region 不一致；windowed 讓「比在地暗多少」跨 region 一致 → 改善 generalization。
#
# 實作：在 stride=32 downsampled scale 跑 cv2.boxFilter(window=64)，等效於
# full-res 2048 box 但記憶體省。下採樣不影響 windowed 結果，因 2048 視窗的 stats
# 本就變化緩慢。
G_BAND_WINDOW = 2048   # 在 original-resolution 視窗大小 (pixel)
# ===========================================


def get_safe_max_workers(estimated_mem_per_worker_gb=14.0):
    try:
        total_ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        total_ram_gb = total_ram / (1024**3)
        reserved_ram = max(8.0, total_ram_gb * 0.2)
        available_for_workers = total_ram_gb - reserved_ram
        mem_workers = int(available_for_workers / estimated_mem_per_worker_gb)
        cpu_workers = multiprocessing.cpu_count() - 2
        safe_workers = max(1, min(mem_workers, cpu_workers, 8))
        print(f"[系統資訊] 總記憶體: {total_ram_gb:.1f} GB, 預留: {reserved_ram:.1f} GB")
        print(f"[安全模式] 預估每行程耗用: {estimated_mem_per_worker_gb} GB -> 記憶體允許 {mem_workers} 個")
        print(f"[最終決定] 使用 {safe_workers} 個平行程序 (已限制上限為 8)")
        return safe_workers
    except Exception as e:
        print(f"無法偵測記憶體，使用保守設定: 2 Workers. ({e})")
        return 2


MAX_WORKERS = get_safe_max_workers(estimated_mem_per_worker_gb=14.0)


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def find_label_for_image(image_path):
    """Mirror image path to label: images/ → labels/，副檔名強制 .png。找不到回傳 None。"""
    p = Path(image_path)
    parts = list(p.parts)
    try:
        idx = parts.index('images')
    except ValueError:
        return None
    new_parts = list(parts)
    new_parts[idx] = 'labels'
    candidate = Path(*new_parts).with_suffix('.png')
    return candidate if candidate.exists() else None


def detect_uniform_fake_nodata(img):
    """偵測「均勻極值」no-data fill 區域 (例如 SAR 切邊填 217 / 244 / 254 等)。"""
    if not DETECT_FAKE_NODATA:
        return np.zeros(img.shape, dtype=bool)
    img_f = img.astype(np.float32)
    lm = cv2.boxFilter(img_f, -1, (FAKE_NODATA_WINDOW, FAKE_NODATA_WINDOW))
    lvar = cv2.boxFilter(img_f**2, -1, (FAKE_NODATA_WINDOW, FAKE_NODATA_WINDOW)) - lm**2
    lvar[lvar < 0] = 0
    lstd = np.sqrt(lvar)
    bright = (img > FAKE_NODATA_BRIGHT_THRESH) & (lstd < FAKE_NODATA_UNIFORM_STD)
    dark   = (img < FAKE_NODATA_DARK_THRESH)   & (lstd < FAKE_NODATA_UNIFORM_STD)
    return bright | dark


def load_land_mask_bool(image_path, img_shape):
    """讀對應 label，回傳 (H, W) bool array，True = 陸地像素。

    沒找到 label / 讀失敗 / 尺寸對不上 都會回傳全 False（= 無陸地，行為等價原檔）。
    """
    h, w = img_shape
    label_path = find_label_for_image(image_path)
    if label_path is None:
        return np.zeros((h, w), dtype=bool)
    try:
        mask = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return np.zeros((h, w), dtype=bool)
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return (mask >= LAND_GRAY_MIN) & (mask <= LAND_GRAY_MAX)
    except Exception:
        return np.zeros((h, w), dtype=bool)


def get_display_mask(img, fake_nodata_bool=None):
    """顯示遮罩：排除 IGNORE_VALUES (0 / 255) + 偽 no-data。陸地保留為 255 (可見)。"""
    mask = np.full(img.shape, 255, dtype=np.uint8)
    for val in IGNORE_VALUES:
        mask[img == val] = 0
    if fake_nodata_bool is not None and fake_nodata_bool.any():
        mask[fake_nodata_bool] = 0
    return mask


def get_stats_mask(img, land_mask_bool, fake_nodata_bool=None):
    """統計遮罩：排除 IGNORE_VALUES + 陸地 + 偽 no-data。"""
    mask = np.full(img.shape, 255, dtype=np.uint8)
    for val in IGNORE_VALUES:
        mask[img == val] = 0
    if land_mask_bool is not None and land_mask_bool.any():
        mask[land_mask_bool] = 0
    if fake_nodata_bool is not None and fake_nodata_bool.any():
        mask[fake_nodata_bool] = 0
    return mask


def get_global_stats_and_background(img, land_mask_bool, fake_nodata_bool=None):
    """計算抗陸地+抗無效+抗偽 no-data 的:
       - global mean/std/min/max (R band 用 min/max；mean/std 供 fallback)
       - B channel local pooling field (background_mean / background_std at small scale)
       - G channel windowed local field (g_local_mean / g_local_std at small scale)
       後兩組以 stride=32 downsampled small scale 回傳，由 sar_to_rgb_tiled 在 tile loop 中 remap 回 tile 大小。"""
    stats_mask = get_stats_mask(img, land_mask_bool, fake_nodata_bool)
    valid_pixel_count = cv2.countNonZero(stats_mask)
    if valid_pixel_count == 0:
        zero = np.zeros_like(img, dtype=np.float32)
        return 0, 1, 0, 1, zero, zero, zero.copy(), zero.copy() + G_BAND_STD_FLOOR

    # Masked Mean / Std (排除無效 + 陸地)
    mean_val, std_val = cv2.meanStdDev(img, mask=stats_mask)
    mean_vv = mean_val[0][0]
    std_vv = std_val[0][0]

    # Masked Min / Max (排除無效 + 陸地)
    min_val, max_val, _, _ = cv2.minMaxLoc(img, mask=stats_mask)

    h, w = img.shape
    stride = 32
    small_h = max(1, h // stride)
    small_w = max(1, w // stride)

    # small 尺度共用：原始 downsampled image + mask
    img_small_clean = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA).astype(np.float32)
    mask_small = cv2.resize(stats_mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    valid_small_f = (mask_small > 0).astype(np.float32)

    # === B band local pooling (沿用原本) ===
    img_small = img_small_clean.copy()
    img_small[mask_small == 0] = mean_vv  # 無效 (含陸地) 區域 → 用 global mean 填

    img_sq_small = cv2.resize(img.astype(np.float32) ** 2, (small_w, small_h), interpolation=cv2.INTER_AREA)
    img_sq_small[mask_small == 0] = (mean_vv ** 2) + (std_vv ** 2)

    diff = img_sq_small - (img_small ** 2)
    diff[diff < 0] = 0
    std_small = np.sqrt(diff)

    background_mean = cv2.GaussianBlur(img_small, (7, 7), 0)
    background_std  = cv2.GaussianBlur(std_small, (7, 7), 0)

    # === G band windowed local stats (新) ===
    # window 在 small scale = G_BAND_WINDOW / stride，等效原 G_BAND_WINDOW pixel
    # valid mask 加權 (排陸地+0/255+fake_nodata 不貢獻)
    g_kernel = max(3, G_BAND_WINDOW // stride)
    if g_kernel % 2 == 0:
        g_kernel += 1
    g_kernel = min(g_kernel, max(small_h, small_w))  # 不能比 small image 大

    sum_w_g  = cv2.boxFilter(valid_small_f,                          -1, (g_kernel, g_kernel),
                              normalize=False, borderType=cv2.BORDER_REFLECT)
    sum_v_g  = cv2.boxFilter(img_small_clean * valid_small_f,        -1, (g_kernel, g_kernel),
                              normalize=False, borderType=cv2.BORDER_REFLECT)
    sum_v2_g = cv2.boxFilter((img_small_clean ** 2) * valid_small_f, -1, (g_kernel, g_kernel),
                              normalize=False, borderType=cv2.BORDER_REFLECT)
    safe_g = np.maximum(sum_w_g, 1.0)
    g_local_mean_small = sum_v_g / safe_g
    g_local_var_small  = sum_v2_g / safe_g - g_local_mean_small ** 2
    g_local_var_small  = np.maximum(g_local_var_small, 0)
    g_local_std_small  = np.sqrt(g_local_var_small)
    g_local_std_small  = np.maximum(g_local_std_small, G_BAND_STD_FLOOR)

    # 邊角無 valid 鄰居 → fallback 用全圖 stats
    too_few = sum_w_g < 0.5
    if too_few.any():
        g_local_mean_small[too_few] = mean_vv
        g_local_std_small[too_few]  = max(std_vv, G_BAND_STD_FLOOR)

    return (mean_vv, std_vv, min_val, max_val,
            background_mean, background_std,
            g_local_mean_small.astype(np.float32), g_local_std_small.astype(np.float32))


def process_tile(vv_tile, g_mean_tile, g_std_tile, global_min, global_max, bg_mean_tile, bg_std_tile, fake_nodata_tile=None):
    """處理單一圖塊。陸地像素仍會被計算 R/G/B 值（用無偏 stats，陸地保留可見）。
    R band: image-wide min/max scaling
    G band: windowed local stats (per-pixel field，已套 STD_FLOOR)
    B band: 32-stride local pooling (per-pixel field)
    偽 no-data 區域會被塗黑 (跟 IGNORE_VALUES 同等處理)。"""
    epsilon = 1e-6
    vv = vv_tile.astype(np.float32)

    # 顯示用遮罩：陸地保留可見；偽 no-data 一併塗黑
    display_mask = get_display_mask(vv_tile, fake_nodata_bool=fake_nodata_tile)

    # --- 1. R Channel: Min-Max Scaling (image-wide) ---
    r_float = vv_tile.astype(np.float32)
    if global_max - global_min > 0:
        r_norm = (r_float - global_min) / (global_max - global_min)
    else:
        r_norm = np.zeros_like(r_float)
    R = (np.clip(r_norm, 0, 1) * 255).astype(np.uint8)

    # --- 2. G Channel: arctan z-score 用 windowed local stats (per-pixel field) ---
    # g_std_tile 已在 small scale 套 STD_FLOOR，這裡直接除即可
    g_raw = np.arctan((vv - g_mean_tile) / (g_std_tile + epsilon))
    g_norm = (g_raw / np.pi) + 0.5
    G = (np.clip(g_norm, 0, 1) * 255).astype(np.uint8)

    # --- 3. B Channel: 局部 background normalization (bg 統計也已排除陸地) ---
    b_star = (vv - bg_mean_tile) / (bg_std_tile + epsilon)
    b_norm = (np.arctan(b_star) / np.pi) + 0.5
    B = (np.clip(b_norm, 0, 1) * 255).astype(np.uint8)

    # --- 4. 還原無效背景 (0/255) 為黑色，陸地不動 ---
    merged = cv2.merge([B, G, R])
    merged = cv2.bitwise_and(merged, merged, mask=display_mask)
    return merged


def sar_to_rgb_tiled(image_path):
    try:
        full_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print(f"Read Error: {e}")
        return None

    if full_img is None:
        return None

    if full_img.ndim == 3:
        full_img = full_img[:, :, 0]

    h_full, w_full = full_img.shape

    # 1a. 載入陸地遮罩 (沒有 label 就全 False)
    land_mask_bool = load_land_mask_bool(image_path, full_img.shape)
    # 1b. 偵測偽 no-data 區域
    fake_nodata_full = detect_uniform_fake_nodata(full_img)

    # 2. 全域 stats + B band local field + G band windowed local field (都排 IGNORE + 陸地 + 偽 no-data)
    (g_mean, g_std, g_min, g_max,
     bg_mean_small, bg_std_small,
     g_local_mean_small, g_local_std_small) = get_global_stats_and_background(
        full_img, land_mask_bool, fake_nodata_bool=fake_nodata_full
    )
    small_h, small_w = bg_mean_small.shape

    if g_max == 0 and g_min == 0:
        return np.zeros((h_full, w_full, 3), dtype=np.uint8)

    try:
        result_img = np.zeros((h_full, w_full, 3), dtype=np.uint8)
    except MemoryError:
        print(f"記憶體不足 ({h_full}x{w_full})，跳過。")
        return None

    del land_mask_bool  # 大圖層級已用完，省 RAM (fake_nodata_full 還要 tile 切片)

    # 3. 分塊迴圈
    for y in range(0, h_full, TILE_SIZE):
        for x in range(0, w_full, TILE_SIZE):
            y_end = min(y + TILE_SIZE, h_full)
            x_end = min(x + TILE_SIZE, w_full)
            curr_h = y_end - y
            curr_w = x_end - x

            patch_vv = full_img[y:y_end, x:x_end]
            patch_fake = fake_nodata_full[y:y_end, x:x_end]

            x_indices = np.arange(x, x + curr_w, dtype=np.float32) * (small_w / w_full)
            y_indices = np.arange(y, y + curr_h, dtype=np.float32) * (small_h / h_full)
            mesh_x, mesh_y = np.meshgrid(x_indices, y_indices)

            bg_mean_final = cv2.remap(bg_mean_small, mesh_x, mesh_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            bg_std_final  = cv2.remap(bg_std_small,  mesh_x, mesh_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

            # G band windowed local field (per-pixel, upsample from small scale)
            g_mean_final = cv2.remap(g_local_mean_small, mesh_x, mesh_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            g_std_final  = cv2.remap(g_local_std_small,  mesh_x, mesh_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

            processed_patch = process_tile(patch_vv, g_mean_final, g_std_final, g_min, g_max, bg_mean_final, bg_std_final, fake_nodata_tile=patch_fake)
            result_img[y:y_end, x:x_end] = processed_patch

            del patch_vv, patch_fake, processed_patch, bg_mean_final, bg_std_final, g_mean_final, g_std_final, mesh_x, mesh_y

    del full_img, fake_nodata_full, g_mean, g_std, g_min, g_max, bg_mean_small, bg_std_small, g_local_mean_small, g_local_std_small
    gc.collect()
    return result_img


def process_one_file(args):
    src_file_path, input_root_str, output_root_str = args
    os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))
    cv2.setNumThreads(0)

    src_path_obj = Path(src_file_path)
    input_path = Path(input_root_str)
    output_path = Path(output_root_str)

    try:
        relative_path = src_path_obj.relative_to(input_path)
    except ValueError:
        return f"[SKIP] Path error: {src_file_path}"

    dest_file_path = output_path / relative_path

    try:
        if not dest_file_path.parent.exists():
            dest_file_path.parent.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass

    try:
        if "labels" in src_path_obj.parts:
            # label 一律 copy (跟原檔一樣，這份 label 也用於 RGB stage 抓陸地)
            shutil.copy2(src_file_path, dest_file_path)
            return None

        elif "images" in src_path_obj.parts:
            if src_path_obj.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                # Resume：已存在且 > 1KB 就 skip
                if dest_file_path.exists() and dest_file_path.stat().st_size > 1024:
                    return None

                processed_img = sar_to_rgb_tiled(str(src_file_path))
                if processed_img is not None:
                    ext = dest_file_path.suffix.lower()
                    params = []
                    if ext == '.png':
                        params = [cv2.IMWRITE_PNG_COMPRESSION, 4]
                    elif ext in ['.tif', '.tiff']:
                        params = [cv2.IMWRITE_TIFF_COMPRESSION, 5]
                    elif ext in ['.jpg', '.jpeg']:
                        params = [cv2.IMWRITE_JPEG_QUALITY, 90]

                    cv2.imwrite(str(dest_file_path), processed_img, params)
                    del processed_img
                    gc.collect()
                    return None
                else:
                    return f"[FAIL] Processing returned None: {src_file_path}"
            else:
                shutil.copy2(src_file_path, dest_file_path)
                return None
        else:
            shutil.copy2(src_file_path, dest_file_path)
            return None

    except Exception as e:
        gc.collect()
        return f"[ERROR] {src_file_path}: {e}"


def process_dataset():
    input_path = Path(INPUT_ROOT)
    output_path = Path(OUTPUT_ROOT)

    print(f"開始處理資料集 (LandMasked 版 — 陸地不參與 G/B stats)...")
    print(f"原始路徑: {INPUT_ROOT}")
    print(f"輸出路徑: {OUTPUT_ROOT}")
    print(f"忽略背景值 (顯示+stats): {IGNORE_VALUES}")
    print(f"陸地灰階範圍 (僅排除於 stats，顯示保留): [{LAND_GRAY_MIN}, {LAND_GRAY_MAX}]")
    print(f"平行 Worker 數量: {MAX_WORKERS}")

    file_list = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            file_list.append(os.path.join(root, file))

    tasks = [(f, str(input_path), str(output_path)) for f in file_list]
    print(f"共發現 {len(tasks)} 個檔案，開始處理...")

    errors = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(process_one_file, tasks), total=len(tasks), desc="Processing (Parallel, LandMasked)"))
        for res in results:
            if res is not None:
                errors.append(res)

    if errors:
        print(f"\n處理完成，共有 {len(errors)} 個檔案發生錯誤或警告:")
        for err in errors[:20]:
            print(err)
        if len(errors) > 20:
            print(f"... (還有 {len(errors)-20} 個錯誤未顯示)")
    else:
        print("\n全部處理完成，無錯誤。")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    process_dataset()
