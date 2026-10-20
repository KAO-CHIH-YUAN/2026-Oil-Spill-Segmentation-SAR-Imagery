import os
# [FIX] 設定 OpenCV 最大像素限制，必須在 import cv2 之前設定環境變數才有效
# 但有些 OpenCV 版本可能需要透過 cv2.setNumThreads(0) 或其他方式
# 最保險的方式是直接設定環境變數
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
# [加速關鍵] 在多行程模式下，關閉 OpenCV 內部的多執行緒，避免互搶資源
cv2.setNumThreads(0)

INPUT_ROOT = r"/home/yuan/Yuan/OIL_Project_12_7/dataset/DV5_reorder/DV5_SAR_Land/Preprocess_Output/Step4_Dataset"
OUTPUT_ROOT = r"/home/yuan/Yuan/OIL_Project_12_7/dataset/DV5_reorder/DV5_SAR_Land/Preprocess_Output/Step4_Dataset_RGB"

TILE_SIZE = 1280 # 私有資料集必須2048以上，這裡改為 1280 以適應 M4D 的圖片尺寸 (672x1280)，同時減少記憶體壓力

# ================= 自動計算最佳 Worker 數量 (防止記憶體炸裂) =================
def get_safe_max_workers(estimated_mem_per_worker_gb=14.0):
    """
    根據系統總記憶體計算安全的 Worker 數量
    estimated_mem_per_worker_gb: 預估每個處理程序會吃掉多少 RAM (GB)。
    [修正] 提升至 6.0GB 以應對極端峰值與 OOM Killer
    """
    try:
        # Linux 取得總記憶體 (Bytes)
        total_ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        total_ram_gb = total_ram / (1024**3)
        
        # 保留 20% 或 8GB 給系統 (增加保留量)
        reserved_ram = max(8.0, total_ram_gb * 0.2)
        available_for_workers = total_ram_gb - reserved_ram
        
        # 計算可容納數量
        mem_workers = int(available_for_workers / estimated_mem_per_worker_gb)
        
        # CPU 限制
        cpu_workers = multiprocessing.cpu_count() - 2
        
        # [安全上限] 強制不超過 8，避免過多程序同時搶 IO 或瞬間峰值疊加
        safe_workers = max(1, min(mem_workers, cpu_workers, 8))
        
        print(f"[系統資訊] 總記憶體: {total_ram_gb:.1f} GB, 預留: {reserved_ram:.1f} GB")
        print(f"[安全模式] 預估每行程耗用: {estimated_mem_per_worker_gb} GB -> 記憶體允許 {mem_workers} 個")
        print(f"[最終決定] 使用 {safe_workers} 個平行程序 (已限制上限為 8)")
        
        return safe_workers
    except Exception as e:
        print(f"無法偵測記憶體，使用保守設定: 2 Workers. ({e})")
        return 2

# 設定最大平行處理數量
MAX_WORKERS = get_safe_max_workers(estimated_mem_per_worker_gb=14.0) 
# =========================================================================

# 設定要忽略的背景值
# 如果您的無效背景只有純黑，設為 [0]
# 如果有純黑也有純白，設為 [0, 255]
IGNORE_VALUES = [0, 255]

# === 偽 no-data 偵測 (對抗 SAR 切邊填值 217 / 244 / 254 等非極值 fill) ===
# True：用 local std 偵測連片均勻極值區域 → 同時排除於 stats + display (塗黑)
# 偵測條件：local_std < UNIFORM_STD AND (img > BRIGHT_THRESH OR img < DARK_THRESH)
DETECT_FAKE_NODATA       = True
FAKE_NODATA_BRIGHT_THRESH = 200
FAKE_NODATA_DARK_THRESH   = 15
FAKE_NODATA_UNIFORM_STD   = 5.0
FAKE_NODATA_WINDOW        = 9
# ===========================================

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def detect_uniform_fake_nodata(img):
    """偵測「均勻極值」no-data fill 區域 (例如 SAR 切邊填 217 / 244 / 254 等)。
    回傳 bool array (H, W)，True 表偽 no-data。"""
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


def get_valid_mask(img, fake_nodata_bool=None):
    """
    建立有效像素遮罩，排除 IGNORE_VALUES 中定義的背景值 + 偽 no-data (若提供)
    """
    mask = np.full(img.shape, 255, dtype=np.uint8)
    for val in IGNORE_VALUES:
        mask[img == val] = 0
    if fake_nodata_bool is not None and fake_nodata_bool.any():
        mask[fake_nodata_bool] = 0
    return mask

def get_global_stats_and_background(img, fake_nodata_bool=None):
    """
    計算抗干擾的全域統計數據 (排除背景值 + 偽 no-data)
    """
    # 1. 建立遮罩
    mask = get_valid_mask(img, fake_nodata_bool=fake_nodata_bool)
    
    # 檢查是否整張圖都是無效值 (避免 crash)
    valid_pixel_count = cv2.countNonZero(mask)
    if valid_pixel_count == 0:
        # 假如全黑，返回預設值，避免除以零
        return 0, 1, 0, 1, np.zeros_like(img, dtype=np.float32), np.zeros_like(img, dtype=np.float32)

    # 2. Masked Mean/Std (只計算有效區域)
    mean_val, std_val = cv2.meanStdDev(img, mask=mask)
    mean_vv = mean_val[0][0]
    std_vv = std_val[0][0]

    # 3. Masked Min/Max (只計算有效區域)
    min_val, max_val, _, _ = cv2.minMaxLoc(img, mask=mask)
    
    # 4. B Channel 背景表面處理 (Inpainting 策略)
    # 論文建議：用平均值填充無效區域，以避免模糊時產生邊緣誤差 
    
    h, w = img.shape
    stride = 32
    small_h, small_w = h // stride, w // stride
    if small_h == 0: small_h = 1
    if small_w == 0: small_w = 1

    # 縮小圖像
    img_small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA).astype(np.float32)
    mask_small = cv2.resize(mask, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
    
    # [關鍵步驟] 將小圖中的無效區域 (mask==0) 替換為全域平均值 (mean_vv)
    # 這樣 GaussianBlur 就不會被 0 拉低數值
    img_small[mask_small == 0] = mean_vv

    # 計算局部標準差
    img_sq_small = cv2.resize(img.astype(np.float32)**2, (small_w, small_h), interpolation=cv2.INTER_AREA)
    # 同樣填充平方圖的無效區域
    img_sq_small[mask_small == 0] = (mean_vv ** 2) + (std_vv ** 2) # E[X^2] approximation

    diff = img_sq_small - (img_small ** 2)
    diff[diff < 0] = 0
    std_small = np.sqrt(diff)

    # Gaussian Blur
    background_mean = cv2.GaussianBlur(img_small, (7, 7), 0)
    background_std = cv2.GaussianBlur(std_small, (7, 7), 0)

    return mean_vv, std_vv, min_val, max_val, background_mean, background_std

def process_tile(vv_tile, global_mean, global_std, global_min, global_max, bg_mean_tile, bg_std_tile, fake_nodata_tile=None):
    """
    處理單一圖塊，並保留原本的無效區域為 0 (黑色)
    fake_nodata_tile: 偽 no-data 區域 bool array，同樣會被塗黑
    """
    epsilon = 1e-6
    vv = vv_tile.astype(np.float32)

    # 建立該 Tile 的遮罩 (含 IGNORE + 偽 no-data)
    tile_mask = get_valid_mask(vv_tile, fake_nodata_bool=fake_nodata_tile)

    # --- 1. R Channel ---
    # r = np.log10(vv + epsilon)
    # g_r_min = np.log10(global_min + epsilon)
    # g_r_max = np.log10(global_max + epsilon)
    
    # if g_r_max - g_r_min > 0:
    #     r_norm = (r - g_r_min) / (g_r_max - g_r_min)
    # else:
    #     r_norm = np.zeros_like(r)
    
    # R = (np.clip(r_norm, 0, 1) * 255).astype(np.uint8)


    # 如果輸入已經是 0-255 且適合觀看，不要再取 log10
    # 直接做 Min-Max 確保對比度拉滿即可
    # 這裡我們假設 vv_tile 已經是 0-255 的數值
    # 方案 A: 如果原圖對比度已經很好，直接用
    # R = vv_tile.astype(np.uint8) 
    
    # 方案 B (推薦): 為了保險，還是做一次 Min-Max Scaling (但不做 Log)
    # 這是為了確保 R 通道能佔滿 0-255 的動態範圍，符合論文 "scale to [0, 255]" 的精神
    r_float = vv_tile.astype(np.float32)
    # 使用傳入的全域 Min/Max (注意：傳入的 global_min/max 也要是基於 0-255 的)
    if global_max - global_min > 0:
        r_norm = (r_float - global_min) / (global_max - global_min)
    else:
        r_norm = np.zeros_like(r_float)
    
    R = (np.clip(r_norm, 0, 1) * 255).astype(np.uint8)

    # --- 2. G Channel ---
    g_raw = np.arctan((vv - global_mean) / (global_std + epsilon))
    g_norm = (g_raw / np.pi) + 0.5 
    G = (np.clip(g_norm, 0, 1) * 255).astype(np.uint8)

    # --- 3. B Channel ---
    b_star = (vv - bg_mean_tile) / (bg_std_tile + epsilon)
    b_norm = (np.arctan(b_star) / np.pi) + 0.5
    B = (np.clip(b_norm, 0, 1) * 255).astype(np.uint8)

    # --- 4. Post-processing: 還原背景 ---
    # 如果原圖是 0 (背景)，轉換後應該保持為 0 (黑色)，而不是被數學公式算出一個灰色值
    # 這一步對於可視化和訓練都很重要
    
    merged = cv2.merge([B, G, R])
    # 使用 mask 將背景強制設回 0
    merged = cv2.bitwise_and(merged, merged, mask=tile_mask)
    
    return merged

def sar_to_rgb_tiled(image_path):
    try:
        full_img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    except Exception as e:
        print(f"Read Error: {e}")
        return None
    
    if full_img is None: return None

    if full_img.ndim == 3:
        full_img = full_img[:, :, 0]
    
    h_full, w_full = full_img.shape

    # 1a. 偵測偽 no-data (連片均勻極值區域，例如 SAR 切邊填 217/244/254)
    fake_nodata_full = detect_uniform_fake_nodata(full_img)

    # 1b. 計算全域參數 (Masked Statistics — 含偽 no-data 排除)
    g_mean, g_std, g_min, g_max, bg_mean_small, bg_std_small = get_global_stats_and_background(
        full_img, fake_nodata_bool=fake_nodata_full
    )
    small_h, small_w = bg_mean_small.shape
    
    # 如果整張圖都是空的 (極端情況)
    if g_max == 0 and g_min == 0:
         return np.zeros((h_full, w_full, 3), dtype=np.uint8)

    # 2. 準備輸出
    try:
        result_img = np.zeros((h_full, w_full, 3), dtype=np.uint8)
    except MemoryError:
        print(f"記憶體不足 ({h_full}x{w_full})，跳過。")
        return None

    # 3. 分塊迴圈
    for y in range(0, h_full, TILE_SIZE):
        for x in range(0, w_full, TILE_SIZE):
            y_end = min(y + TILE_SIZE, h_full)
            x_end = min(x + TILE_SIZE, w_full)
            curr_h = y_end - y
            curr_w = x_end - x
            
            patch_vv = full_img[y:y_end, x:x_end]
            patch_fake_nodata = fake_nodata_full[y:y_end, x:x_end]

            # 背景 Remap
            x_indices = np.arange(x, x + curr_w, dtype=np.float32) * (small_w / w_full)
            y_indices = np.arange(y, y + curr_h, dtype=np.float32) * (small_h / h_full)
            mesh_x, mesh_y = np.meshgrid(x_indices, y_indices)

            bg_mean_final = cv2.remap(bg_mean_small, mesh_x, mesh_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            bg_std_final = cv2.remap(bg_std_small, mesh_x, mesh_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

            # 核心處理
            processed_patch = process_tile(patch_vv, g_mean, g_std, g_min, g_max, bg_mean_final, bg_std_final, fake_nodata_tile=patch_fake_nodata)

            result_img[y:y_end, x:x_end] = processed_patch

            del patch_vv, patch_fake_nodata, processed_patch, bg_mean_final, bg_std_final, mesh_x, mesh_y

    del full_img, fake_nodata_full, g_mean, g_std, g_min, g_max, bg_mean_small, bg_std_small
    gc.collect()
    
    return result_img

def process_one_file(args):
    """
    單一檔案處理函數 (供多行程呼叫)
    args: (src_file_path, input_root_str, output_root_str)
    """
    src_file_path, input_root_str, output_root_str = args
    
    # 在子行程中重新設定 (雖通常繼承，但保險起見)
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
    
    # 確保目錄存在 (多行程同時建立目錄通常安全，或會拋出 FileExistsError 可忽略)
    try:
        if not dest_file_path.parent.exists():
            dest_file_path.parent.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        pass

    # 邏輯分流
    try:
        if "labels" in src_path_obj.parts:
            shutil.copy2(src_file_path, dest_file_path)
            return None
        
        elif "images" in src_path_obj.parts:
            # 檢查副檔名是否為影像
            if src_path_obj.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
                # 若目標已存在且大小正常，或許可以跳過？ (這裡預設覆蓋)
                # [FIX] 如果檔案已存在且大於 1KB，直接跳過，實現 Resume (回復) 功能
                if dest_file_path.exists() and dest_file_path.stat().st_size > 1024:
                    return None # 已經有了，跳過
                
                processed_img = sar_to_rgb_tiled(str(src_file_path))
                if processed_img is not None:
                    # [優化] 加入壓縮參數以減少檔案體積 (解決 32G->100G 問題)
                    ext = dest_file_path.suffix.lower()
                    params = []
                    if ext == '.png':
                        # PNG 壓縮等級 0-9 (9最壓縮但最慢，4是平衡點)
                        params = [cv2.IMWRITE_PNG_COMPRESSION, 4] 
                    elif ext in ['.tif', '.tiff']:
                        # TIFF LZW 壓縮
                        params = [cv2.IMWRITE_TIFF_COMPRESSION, 5] 
                    elif ext in ['.jpg', '.jpeg']:
                        # JPG 品質 0-100
                        params = [cv2.IMWRITE_JPEG_QUALITY, 90]

                    cv2.imwrite(str(dest_file_path), processed_img, params)
                    
                    del processed_img
                    gc.collect()
                    return None
                else:
                    return f"[FAIL] Processing returned None: {src_file_path}"
            else:
                # 非影像檔案 (如 xml, json 等) 直接複製
                shutil.copy2(src_file_path, dest_file_path)
                return None
        else:
            # 其他結構直接複製
            shutil.copy2(src_file_path, dest_file_path)
            return None

    except Exception as e:
        gc.collect()
        return f"[ERROR] {src_file_path}: {e}"

def process_dataset():
    input_path = Path(INPUT_ROOT)
    output_path = Path(OUTPUT_ROOT)

    print(f"開始處理資料集 (平行加速優化版)...")
    print(f"原始路徑: {INPUT_ROOT}")
    print(f"輸出路徑: {OUTPUT_ROOT}")
    print(f"忽略背景值: {IGNORE_VALUES}")
    print(f"平行 Worker 數量: {MAX_WORKERS}")
    
    file_list = []
    # os.walk 收集所有檔案路徑
    for root, dirs, files in os.walk(input_path):
        for file in files:
            file_list.append(os.path.join(root, file))
            
    # 準備 Tasks (將參數打包成 Tuple)
    # 格式: (檔案絕對路徑, 輸入根目錄字串, 輸出根目錄字串)
    tasks = [(f, str(input_path), str(output_path)) for f in file_list]
    
    print(f"共發現 {len(tasks)} 個檔案，開始處理...")

    # 使用 ProcessPoolExecutor 進行多行程平行處理
    # chunksize 設定稍微大一點可以減少 IPC 開銷，但對於大型影像處理，任務時間長，預設 1 也可以
    errors = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 使用 tqdm 顯示進度條
        results = list(tqdm(executor.map(process_one_file, tasks), total=len(tasks), desc="Processing (Parallel)"))
        
        # 收集錯誤訊息
        for res in results:
            if res is not None:
                errors.append(res)

    # 顯示錯誤報告
    if errors:
        print(f"\n處理完成，共有 {len(errors)} 個檔案發生錯誤或警告:")
        for err in errors[:20]:
            print(err)
        if len(errors) > 20:
            print(f"... (還有 {len(errors)-20} 個錯誤未顯示)")
    else:
        print("\n全部處理完成，無錯誤。")

if __name__ == "__main__":
    # Windows 系統必須要有 freeze_support()，Linux 雖然不需要但加上無妨
    multiprocessing.freeze_support()
    process_dataset()