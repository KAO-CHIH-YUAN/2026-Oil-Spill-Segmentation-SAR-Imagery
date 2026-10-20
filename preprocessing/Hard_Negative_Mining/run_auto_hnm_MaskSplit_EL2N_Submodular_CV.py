import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import timm
from PIL import Image
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil
import random
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
import psutil
import threading
import time
from sklearn.model_selection import StratifiedGroupKFold
from concurrent.futures import ThreadPoolExecutor

"""
與舊版 (run_auto_hnm_cluster_Classifier_global_EL2N_Submodular_Optimization_CV.py) 相比，
本版本的核心改動只在 Stage 0：

  舊版 Stage 0 → VGG16 特徵提取 + MiniBatchKMeans 無監督分群 → land / water
  新版 Stage 0 → 直接讀取 mask PNG，按陸地像素比例分類  → land / coast / ocean

  mask 像素定義：
      0  = 背景 / 海洋
     90  = 陸地 (MASK_LAND_PIXEL)
    255  = 油污 (理論上負樣本不應含此值)

  分類規則 (可用 LAND_DOMINANT_THRESH / COAST_PRESENT_THRESH 調整)：
      land_ratio >= LAND_DOMINANT_THRESH  → "land"
      land_ratio >= COAST_PRESENT_THRESH  → "coast"
      otherwise                           → "ocean"

Stage 1 / 2 / 3 邏輯與舊版相同，但 Phase 3 抽樣由 land/water 2 類
擴展為 land/coast/ocean 3 類，並新增對應的比例設定參數。
"""

# --- ⭐⭐⭐ 全域設定 (User Config) ⭐⭐⭐ ---
import sys as _sys
# 讓 HNM 能 import main_cross_validation/cross_validation_utils（nested split 用）
_sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "main_cross_validation"))
_PATCH_BASE = os.environ.get("PATCH_BASE", "/home/yuan/Yuan/OIL_Project_12_7/OIL_Project_Handoff/dataset/DV5_SAR_Land_Patch")
_PATCH_TAG  = os.environ.get("PATCH_TAG", "P2048_OUni512_BG100")
# HNM 輸入 patch 目錄；2048 主法用 VV(_Split)，1024/512 直接用 RGB close(_RGB_LandNeutral_close_Split)。用 env 覆寫。
SOURCE_PATCH_DIR    = Path(os.environ.get("HNM_SOURCE", f"{_PATCH_BASE}/{_PATCH_TAG}_Split"))
HNM_PIPELINE_BASE_DIR = Path(os.environ.get("HNM_OUTPUT", f"{_PATCH_BASE}/HNM_{_PATCH_TAG}_nested"))

K_FOLDS       = 3
TARGET_RATIOS = [1.5]
RANDOM_SEED   = 42

# Phase 3 file linking 策略：True=symlink (省 disk，多 fold nested 模式必開)；False=shutil.copy
USE_SYMLINK = True

# ── 是否跑 inner K-fold (Phase 1) ─────────────────────────────────────────────
# True (預設)：完整原行為 — Phase 1 訓 K_FOLDS 個 fold classifier + Phase 1b 訓 1 個 full-data
#              Phase 2 用 fold classifiers 算 OOF EL2N (無 in-sample bias)
# False：跳過 Phase 1，只訓 Phase 1b 一個 full-data classifier；Phase 2 用 full-data 同時算
#        in-sample EL2N + embedding (省 K_FOLDS 倍訓練成本；EL2N 有 in-sample bias)
# 5-fold nested 下建議設 False，省 (5 × K_FOLDS) 次分類器訓練
ENABLE_INNER_KFOLD = False

# ── 5-fold Nested CV 模式 ────────────────────────────────────────────────────
# False (預設)：跑單一份 HNM，輸出到 HNM_PIPELINE_BASE_DIR (跟以前一樣)
# True：把 SOURCE_PATCH_DIR 重切成 K outer fold，每 fold 跑一次 HNM，
#       輸出成 HNM_PIPELINE_BASE_DIR/fold_0/Global_Dataset_Ratio_*,
#       HNM_PIPELINE_BASE_DIR/fold_1/Global_Dataset_Ratio_*, ...
#       訓練端在 yaml 設 cross_validation.mode=nested_folds + nested_folds_base_dir
#       就會自動 5 折輪流訓練。
ENABLE_5FOLD_NESTED_SPLIT     = True
NESTED_K_OUTER                = 5
NESTED_FIX_LAST_FOLD_AS_ORIG  = True   # fold_{K-1} 固定為原 test (保留 baseline 可比性)
NESTED_SCENE_LEVEL_EVENTS     = ["沉船 1487",
                                 "疑似油污",
                                 "艙底水排放",
                                 "似乎源自石油平台",
                                 "Taylor 平台",
                                 "破損管道",
                                 "天然滲漏點"]     # 列表中的事件按 scene 拆，其他按 event 綁定 # 範例：["艙底水排放", "Taylor 平台"]


def _link_or_copy(src, dst):
    """根據 USE_SYMLINK 決定用 symlink 還是 copy。symlink 失敗時自動 fallback copy。"""
    src_path = Path(src).resolve()
    dst_path = Path(dst)
    if dst_path.exists() or dst_path.is_symlink():
        dst_path.unlink()
    if USE_SYMLINK:
        try:
            os.symlink(src_path, dst_path)
            return
        except OSError:
            pass  # 不支援 symlink (如某些 NTFS) → fallback copy
    shutil.copy(src_path, dst_path)

# ── Stage 0: Mask-based 分類閾值 ────────────────────────────────────────────
MASK_LAND_PIXEL      = 90    # 陸地灰階值
LAND_DOMINANT_THRESH = 0.80  # land_ratio >= 80% → "land"
COAST_PRESENT_THRESH = 0.03  # land_ratio  3~80% → "coast"
                              # land_ratio < 3%   → "ocean"

# ── Phase 3: 三類別負樣本抽樣比例 (各值為佔 n_neg 的比例，需 <= 1.0) ────────
#   env 可覆寫 (預設=舊配額，2048 用此重現)：
#     HNM_HARD_ALLOC / HNM_COAST_ALLOC / HNM_LAND_ALLOC
#   crop 消融 1024/512 改用 coast=0.25 / land=0.00 (ocean 自動=0.25)。
HARD_ALLOC  = float(os.environ.get("HNM_HARD_ALLOC",  "0.50"))  # EL2N 最高分 (不分類別)
COAST_ALLOC = float(os.environ.get("HNM_COAST_ALLOC", "0.15"))  # Submodular from coast
LAND_ALLOC  = float(os.environ.get("HNM_LAND_ALLOC",  "0.15"))  # Submodular from land
# ocean gets the remainder: 1 - HARD - COAST - LAND

# ── Stage 0: 切邊無效 patch 偵測 ──────────────────────────────────────────────
# SAR 影像在切邊處會填入一個固定的亮值（每張圖不同，例如 217 或 244），
# 這種區域像素值極為均勻（std ≈ 0），與陸地（有自然紋理）明顯不同。
BORDER_BRIGHT_THRESH  = 200   # 大於此灰階值才算「亮色像素」
BORDER_DOMINANT_RATIO = 0.50  # 亮色像素佔比 > 50% 且均勻 → 切邊
BORDER_UNIFORMITY_STD = 5.0   # 亮色像素 std < 此值 → 視為均勻切邊填充

# ── Stage 0: 偽陸地偵測（OSM 標 land 實際是 no-data 黑色）────────────────────
# 原始 TIF 沒有 no-data 屬性時外圍會填純黑，而 OSM 是按地理座標標 land，
# 因此會出現「mask=land 但 SAR 像素全黑」的偽標。這類 patch 對訓練無意義。
BLACK_PIXEL_THRESH    = 5     # 灰階 < 此值視為 no-data 黑色
FAKE_LAND_BLACK_RATIO = 0.70  # land 標註區域中黑色佔比 > 70% → 視為偽標 invalid
FAKE_LAND_MIN_PIXELS  = 100   # land 像素少於此值不做檢查（無實質意義）

# ── Classifier ───────────────────────────────────────────────────────────────

CLASSIFIER_MODEL_NAME    = 'tf_efficientnetv2_s'
CLASSIFIER_IMG_SIZE      = 512
CLASSIFIER_BATCH_SIZE    = 48
CLASSIFIER_EPOCHS        = 70
CLASSIFIER_LR            = 0.0001
CLASSIFIER_WEIGHT_DECAY  = 0.01
EARLY_STOPPING_PATIENCE  = 15
EARLY_STOPPING_MIN_DELTA = 0.001
CLASSIFIER_NEG_RATIO     = 5

# ── 斷點接續開關 ──────────────────────────────────────────────────────────────
RESUME_FROM_PHASE_3 = False
RESUME_FROM_FOLD    = 0

# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# hw_logger 在這裡只建 logger（不綁 file handler）；FileHandler 在 run_hnm_pipeline()
# 內 lazy attach，這樣 orchestrator 每個 fold 覆寫 HNM_PIPELINE_BASE_DIR 時都會寫到正確路徑。
hw_logger = logging.getLogger("HardwareMonitor")
hw_logger.setLevel(logging.INFO)

def _attach_hw_logger(base_dir: Path):
    """Attach (or re-attach) the FileHandler to the current HNM_PIPELINE_BASE_DIR."""
    base_dir.mkdir(parents=True, exist_ok=True)
    # 移除舊 FileHandler（避免重複呼叫時 log 寫到舊路徑）
    for h in list(hw_logger.handlers):
        if isinstance(h, logging.FileHandler):
            hw_logger.removeHandler(h)
            try: h.close()
            except: pass
    handler = logging.FileHandler(base_dir / "hardware_monitor_log.txt")
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    hw_logger.addHandler(handler)

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML  = True
    gpu_count = pynvml.nvmlDeviceGetCount()
except ImportError:
    HAS_NVML  = False
    gpu_count = 0
except Exception as e:
    hw_logger.warning(f"Failed to initialize NVML: {e}")
    HAS_NVML  = False
    gpu_count = 0

def monitor_hardware(interval=20, stop_event=None):
    while not stop_event.is_set():
        try:
            cpu_percent = psutil.cpu_percent()
            ram = psutil.virtual_memory()
            log_str = f"CPU: {cpu_percent}% | RAM: {ram.percent}% ({ram.used/(1024**3):.1f}GB)"
            if HAS_NVML:
                for i in range(gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util   = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem    = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    temp   = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    try:   power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    except: power = 0.0
                    log_str += f" || GPU {i}: Util {util.gpu}% | VRAM {mem.used/(1024**3):.1f}GB | Temp {temp}C | Power {power:.1f}W"
            elif torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    log_str += f" || GPU {i} (torch): VRAM {torch.cuda.memory_allocated(i)/(1024**3):.1f}GB"
            hw_logger.info(log_str)
        except Exception as e:
            hw_logger.error(f"Error in hardware monitor: {e}")
        stop_event.wait(interval)

monitor_stop_event = threading.Event()
monitor_thread = threading.Thread(target=monitor_hardware, args=(20, monitor_stop_event), daemon=True)
monitor_thread.start()

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CURRENT_FOLD_DIR    = None
CLASSIFIER_MODEL_DIR  = None
CLASSIFIER_MODEL_PATH = None
MINED_BASE_DIR      = None

def setup_fold_globals(fold_dir):
    global CURRENT_FOLD_DIR, CLASSIFIER_MODEL_DIR, CLASSIFIER_MODEL_PATH, MINED_BASE_DIR
    CURRENT_FOLD_DIR    = Path(fold_dir)
    CLASSIFIER_MODEL_DIR  = CURRENT_FOLD_DIR / "hnm_1_classifier_model"
    CLASSIFIER_MODEL_PATH = CLASSIFIER_MODEL_DIR / "hnm_classifier.pt"
    MINED_BASE_DIR      = CURRENT_FOLD_DIR / "hnm_2_mined"
    ensure_clean_dir(MINED_BASE_DIR)

def ensure_clean_dir(d):
    if d.exists(): shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

# ── K-Fold ────────────────────────────────────────────────────────────────────
def parse_scene_id(filename: str) -> str:
    parts = filename.split('_')
    if len(parts) >= 4 and 'patch' in filename:
        patch_i = parts.index('patch')
        return "_".join(parts[:patch_i])
    return filename

def _collect_files(base_dir, split_type):
    records = []
    img_dir = base_dir / "images"
    lbl_dir = base_dir / "labels"
    for cat in ['pos', 'neg']:
        cat_img_dir = img_dir / f"{split_type}_{cat}"
        cat_lbl_dir = lbl_dir / f"{split_type}_{cat}"
        if not cat_img_dir.exists(): continue
        for img_path in cat_img_dir.rglob("*.*"):
            if img_path.suffix.lower() not in ['.png', '.jpg']: continue
            scene_id   = parse_scene_id(img_path.name)
            rel_parent = img_path.parent.relative_to(cat_img_dir)
            lbl_png = cat_lbl_dir / rel_parent / f"{img_path.stem}.png"
            lbl_txt = cat_lbl_dir / rel_parent / f"{img_path.stem}.txt"
            t_lbl = lbl_png if lbl_png.exists() else (lbl_txt if lbl_txt.exists() else None)
            records.append({
                'img_path': img_path, 'lbl_path': t_lbl,
                'is_pos': (cat == 'pos'), 'scene_id': scene_id
            })
    return pd.DataFrame(records)

def create_virtual_kfold(original_dir, base_save_dir, n_splits=3):
    df_train_val = pd.concat([_collect_files(original_dir, 'train'), _collect_files(original_dir, 'val')], ignore_index=True)
    df_test      = _collect_files(original_dir, 'test')
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    X = df_train_val['img_path'].values
    y = df_train_val['is_pos'].values.astype(int)
    groups = df_train_val['scene_id'].values
    fold_paths = []
    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        fold_dir = base_save_dir / f"fold_{fold_idx}"
        fold_paths.append(fold_dir)
        ensure_clean_dir(fold_dir)
        df_train = df_train_val.iloc[train_idx].copy(); df_train['split'] = 'train'
        df_val   = df_train_val.iloc[val_idx].copy();   df_val['split']   = 'val'
        df_test_c = df_test.copy();                      df_test_c['split'] = 'test'
        fold_df = pd.concat([df_train, df_val, df_test_c], ignore_index=True)
        fold_df['img_path'] = fold_df['img_path'].astype(str)
        fold_df['lbl_path'] = fold_df['lbl_path'].apply(lambda x: str(x) if pd.notna(x) and x is not None else "")
        fold_df.to_excel(fold_dir / "virtual_fold_info.xlsx", index=False)
    return fold_paths

# ── Stage 0: Mask-based Pre-split ────────────────────────────────────────────
def is_border_patch(img_path: Path) -> bool:
    """
    判斷 SAR patch 是否為切邊產生的無效區域。

    切邊特徵：影像邊緣被填入某個固定亮值（每張圖可能不同，如 217 或 244），
    這些亮色像素佔絕大多數且極為均勻（std 接近 0）。
    陸地雖然也偏亮，但有自然 SAR 紋理，std 明顯大。
    """
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return True
    flat = img.flatten().astype(np.float32)
    bright = flat[flat >= BORDER_BRIGHT_THRESH]
    # 亮色像素不夠多 → 不是切邊
    if len(bright) / flat.size < BORDER_DOMINANT_RATIO:
        return False
    # 亮色像素夠多 → 再看是否均勻（切邊 std 極低，陸地 std 較大）
    return float(np.std(bright)) < BORDER_UNIFORMITY_STD

def has_fake_land(img_path: Path, lbl_path_str: str) -> bool:
    """
    偵測 OSM land 標註是否實際覆蓋到 SAR 影像的純黑 no-data 區域。
    這通常發生在原始 TIF 沒有 no-data 屬性、外圍被填成純黑時，
    但 OSM 按地理座標仍會將該區域標為 land。

    回傳 True 表示應視為 invalid patch。
    """
    if not lbl_path_str or lbl_path_str in ("", "None", "nan"):
        return False
    lbl_path = Path(lbl_path_str)
    if not lbl_path.exists() or lbl_path.suffix.lower() != ".png":
        return False

    mask = cv2.imread(str(lbl_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return False

    land_pixels = (mask == MASK_LAND_PIXEL)
    n_land = int(np.sum(land_pixels))
    if n_land < FAKE_LAND_MIN_PIXELS:
        return False  # land 像素太少，沒實質意義

    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False

    # 在 land 標註區域內，多少比例是純黑 no-data？
    land_pixel_values = img[land_pixels]
    n_black = int(np.sum(land_pixel_values < BLACK_PIXEL_THRESH))
    black_ratio_in_land = n_black / n_land

    return black_ratio_in_land > FAKE_LAND_BLACK_RATIO

def classify_patch_by_mask(lbl_path_str: str) -> str:
    """
    讀取 mask PNG，按陸地像素比例回傳 'land' / 'coast' / 'ocean'。
    找不到 mask 或讀取失敗時保守回傳 'ocean'。
    """
    if not lbl_path_str or lbl_path_str in ("", "None", "nan"):
        return "ocean"
    lbl_path = Path(lbl_path_str)
    if not lbl_path.exists() or lbl_path.suffix.lower() != ".png":
        return "ocean"
    mask = cv2.imread(str(lbl_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return "ocean"
    land_ratio = float(np.sum(mask == MASK_LAND_PIXEL)) / mask.size
    if land_ratio >= LAND_DOMINANT_THRESH:
        return "land"
    elif land_ratio >= COAST_PRESENT_THRESH:
        return "coast"
    else:
        return "ocean"

def run_global_phase_0_mask(source_dir, base_save_dir):
    """
    Stage 0 (Mask版本) — 直接用 mask 檔案的陸地像素比例將負樣本分為
    land / coast / ocean，不需要 GPU 或 VGG 特徵提取。
    輸出格式與舊版相同 (global_stage0_file_map.xlsx + 預覽 JPG)，
    以確保 Phase 1/2/3 介面不變。
    """
    logging.info("=== Global Phase 0 [Mask-based]: Land / Coast / Ocean Pre-split ===")
    global_stage0_dir = base_save_dir / "hnm_0_global_staging"
    ensure_clean_dir(global_stage0_dir)

    df_train = _collect_files(source_dir, 'train')
    df_val   = _collect_files(source_dir, 'val')
    df_all   = pd.concat([df_train, df_val], ignore_index=True)
    neg_records = df_all[df_all['is_pos'] == False].copy().reset_index(drop=True)

    if neg_records.empty:
        logging.warning("No negative files found for phase 0.")
        return

    logging.info(f"共 {len(neg_records)} 個負樣本，開始切邊過濾 + 偽陸地過濾 + mask 分類...")

    # 有效類別 + invalid 細分為 border / fake_land 兩個子資料夾方便稽核
    for cat in ['land', 'coast', 'ocean']:
        (global_stage0_dir / cat / "images").mkdir(parents=True, exist_ok=True)
    for sub in ['border', 'fake_land']:
        (global_stage0_dir / 'invalid' / sub).mkdir(parents=True, exist_ok=True)

    stage0_records  = []
    conversion_tasks = []

    for _, row in tqdm(neg_records.iterrows(), total=len(neg_records), desc="Stage 0"):
        img_path    = Path(row['img_path'])
        lbl_path_str = str(row['lbl_path']) if (pd.notna(row['lbl_path']) and row['lbl_path'] is not None) else ""

        # 先檢測切邊（最便宜）→ 再檢測偽陸地 → 最後做 mask 分類
        invalid_reason = ""
        if is_border_patch(img_path):
            cat = 'invalid'
            invalid_reason = 'border'
        elif has_fake_land(img_path, lbl_path_str):
            cat = 'invalid'
            invalid_reason = 'fake_land'
        else:
            cat = classify_patch_by_mask(lbl_path_str)

        if cat == 'invalid':
            tgt_jpg = global_stage0_dir / 'invalid' / invalid_reason / (img_path.stem + ".jpg")
        else:
            tgt_jpg = global_stage0_dir / cat / "images" / (img_path.stem + ".jpg")

        conversion_tasks.append((img_path, tgt_jpg))
        stage0_records.append({
            'land_water_class':    cat,
            'invalid_reason':      invalid_reason,
            'original_img_path':   str(img_path),
            'original_lbl_path':   lbl_path_str,
            'staging_jpg_path':    str(tgt_jpg),
        })

    def convert_worker(task):
        src, dst = task
        im = cv2.imread(str(src))
        if im is not None:
            cv2.imwrite(str(dst), im, [cv2.IMWRITE_JPEG_QUALITY, 50])

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(tqdm(executor.map(convert_worker, conversion_tasks),
                  total=len(conversion_tasks), desc="Save Preview JPGs"))

    df_out = pd.DataFrame(stage0_records)

    # 統計各類別
    total = len(df_out)
    for cat in ['land', 'coast', 'ocean']:
        cnt = int((df_out['land_water_class'] == cat).sum())
        logging.info(f"  {cat:10s}: {cnt:6d} patches ({cnt/total*100:.1f}%)")
    # invalid 細分顯示
    for reason in ['border', 'fake_land']:
        cnt = int(((df_out['land_water_class'] == 'invalid') & (df_out['invalid_reason'] == reason)).sum())
        logging.info(f"  invalid/{reason:9s}: {cnt:6d} patches ({cnt/total*100:.1f}%) ← 已排除，請至 invalid/{reason}/ 人工確認")

    # 只將有效 patch 寫入 map（invalid 不進 Phase 1/2/3）
    df_valid = df_out[df_out['land_water_class'] != 'invalid'].copy()
    df_valid.to_excel(global_stage0_dir / "global_stage0_file_map.xlsx", index=False)
    # 全部紀錄（含 invalid）另存供稽核
    df_out.to_excel(global_stage0_dir / "global_stage0_file_map_all.xlsx", index=False)
    logging.info(f"有效 patch: {len(df_valid)}/{total}，invalid 預覽: {global_stage0_dir / 'invalid'}")

# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, reduction='mean', label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma
        self.reduction = reduction; self.label_smoothing = label_smoothing
    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        with torch.no_grad(): pt = torch.exp(-F.cross_entropy(inputs, targets, reduction='none'))
        fl = (1 - pt) ** self.gamma * ce
        if self.alpha is not None: fl = self.alpha[targets] * fl
        return fl.mean() if self.reduction == 'mean' else fl.sum()

def get_virtual_pos_neg_files(fold_root, split_name):
    df_path = fold_root / "virtual_fold_info.xlsx"
    if not df_path.exists(): return [], []
    df = pd.read_excel(df_path)
    df_split = df[df['split'] == split_name]
    pos_files = [Path(p) for p in df_split[df_split['is_pos'] == True]['img_path']]
    neg_files = [Path(p) for p in df_split[df_split['is_pos'] == False]['img_path']]
    return pos_files, neg_files

# ── Stage 1: Classifier ───────────────────────────────────────────────────────
class ClassifierDS(Dataset):
    def __init__(self, samples, tf): self.samples = samples; self.tf = tf
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p, l = self.samples[i]
        im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        if self.tf: im = self.tf(image=im)['image']
        return im, torch.tensor(l, dtype=torch.long)

def run_phase_1(fold_root):
    logging.info("=== Phase 1: Train Classifier ===")
    ensure_clean_dir(CLASSIFIER_MODEL_DIR)

    pos_files, neg_files = get_virtual_pos_neg_files(fold_root, 'train')
    max_negatives = len(pos_files) * CLASSIFIER_NEG_RATIO

    all_pos = [(str(p), 1) for p in pos_files]
    all_neg = [(str(p), 0) for p in neg_files]
    random.shuffle(all_pos); random.shuffle(all_neg)

    val_pos_size = max(int(len(all_pos) * 0.35), 10)
    val_neg_size = max(int(len(all_pos) * 0.35 * CLASSIFIER_NEG_RATIO), 10)
    sub_val_s = all_pos[:val_pos_size] + all_neg[:val_neg_size]
    tr_pos    = all_pos[val_pos_size:]
    tr_neg_pool = all_neg[val_neg_size:]

    tf_base = [A.Resize(CLASSIFIER_IMG_SIZE, CLASSIFIER_IMG_SIZE),
               A.Normalize(mean=[0.417]*3, std=[0.268]*3), ToTensorV2()]
    tf_aug  = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)] + tf_base)
    tf_val  = A.Compose(tf_base)

    val_l = DataLoader(ClassifierDS(sub_val_s, tf_val), batch_size=CLASSIFIER_BATCH_SIZE,
                       shuffle=False, num_workers=4)
    m = timm.create_model(CLASSIFIER_MODEL_NAME, pretrained=True, num_classes=2).to(device)
    opt = optim.AdamW(m.parameters(), lr=CLASSIFIER_LR, weight_decay=CLASSIFIER_WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    crit   = FocalLoss(alpha=torch.tensor([0.05, 0.95]).to(device)).to(device)

    best_acc = 0; patience_counter = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for ep in range(CLASSIFIER_EPOCHS):
        cur_neg = random.sample(tr_neg_pool, max_negatives) if len(tr_neg_pool) > max_negatives else tr_neg_pool
        tr_s = tr_pos + cur_neg; random.shuffle(tr_s)
        tr_l = DataLoader(ClassifierDS(tr_s, tf_aug), batch_size=CLASSIFIER_BATCH_SIZE,
                          shuffle=True, num_workers=4)
        m.train(); rl = 0; tr_corr = 0
        for x, y in tqdm(tr_l, desc=f"Ep{ep+1}"):
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast('cuda', enabled=True):
                out = m(x); loss = crit(out, y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            rl += loss.item() * x.size(0)
            tr_corr += (out.argmax(1) == y).sum().item()
        train_losses.append(rl / len(tr_s)); train_accs.append(tr_corr / len(tr_s))

        m.eval(); corr = 0; val_rl = 0
        with torch.no_grad():
            for x, y in val_l:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast('cuda', enabled=True):
                    out = m(x); val_rl += crit(out, y).item() * x.size(0)
                corr += (out.argmax(1) == y).sum().item()
        acc = corr / len(sub_val_s)
        val_losses.append(val_rl / len(sub_val_s)); val_accs.append(acc)

        if acc > best_acc + EARLY_STOPPING_MIN_DELTA:
            best_acc = acc; patience_counter = 0
            torch.save(m.state_dict(), CLASSIFIER_MODEL_PATH)
            logging.info(f" Ep{ep+1} Sub-Val Acc: {acc:.4f} (Best: {best_acc:.4f}) - Saved")
        else:
            patience_counter += 1
            logging.info(f" Ep{ep+1} Sub-Val Acc: {acc:.4f} (Best: {best_acc:.4f}) - Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logging.info(f"Early stopping at epoch {ep+1}"); break

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(train_losses, label='Train'); ax1.plot(val_losses, label='Val', color='orange')
    ax1.set_title('Loss'); ax1.legend()
    ax2.plot(train_accs, label='Train', color='blue'); ax2.plot(val_accs, label='Val', color='orange')
    ax2.set_title('Accuracy'); ax2.legend()
    plt.tight_layout(); plt.savefig(CLASSIFIER_MODEL_DIR / "training_curves.png"); plt.close()

    logging.info("Evaluating best model on True Val Set...")
    m.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=device)); m.eval()
    val_pos, val_neg = get_virtual_pos_neg_files(fold_root, 'val')
    true_val_s = [(str(p), 1) for p in val_pos] + [(str(p), 0) for p in val_neg]
    if not true_val_s:
        logging.warning("True Val Set empty."); return 0.0
    true_val_l = DataLoader(ClassifierDS(true_val_s, tf_val), batch_size=CLASSIFIER_BATCH_SIZE,
                             shuffle=False, num_workers=4)
    true_corr = 0
    with torch.no_grad():
        for x, y in true_val_l:
            true_corr += (m(x.to(device)).argmax(1) == y.to(device)).sum().item()
    true_acc = true_corr / len(true_val_s)
    logging.info(f"Final True-Val Accuracy: {true_acc:.4f}")
    return true_acc

# ── Phase 1b: Full-Data Classifier (for unified Embedding) ───────────────────
def run_phase_1b_full_data_classifier():
    """
    用「全部 train+val」資料訓練一個分類器，僅供 Phase 2 抽取統一特徵空間的 embedding。
    EL2N 計算仍由 3 個 fold 分類器負責（保持 OOF），這個 model 只是為了讓所有負樣本
    的 embedding 落在同一個特徵空間裡，後續 Submodular 才能算出有意義的 cosine similarity。

    與舊版差異：舊版 Phase 2 直接用 3 個 fold model 各自抽 embedding，結果 3 組向量
    其實住在 3 個不同特徵空間，Submodular 在跨 fold 之間比相似度是不對等的。
    """
    logging.info("=== Phase 1b: Train Full-Data Classifier (for unified Embedding only) ===")
    full_model_dir  = HNM_PIPELINE_BASE_DIR / "hnm_1_full_data_classifier"
    ensure_clean_dir(full_model_dir)
    full_model_path = full_model_dir / "hnm_classifier_full.pt"

    df_train  = _collect_files(SOURCE_PATCH_DIR, 'train')
    df_val    = _collect_files(SOURCE_PATCH_DIR, 'val')
    df_all    = pd.concat([df_train, df_val], ignore_index=True)
    pos_files = df_all[df_all['is_pos'] == True ]['img_path'].tolist()
    neg_files = df_all[df_all['is_pos'] == False]['img_path'].tolist()
    logging.info(f"Full-Data Pool: pos={len(pos_files)}, neg={len(neg_files)}")

    max_negatives = len(pos_files) * CLASSIFIER_NEG_RATIO
    all_pos = [(str(p), 1) for p in pos_files]
    all_neg = [(str(p), 0) for p in neg_files]
    random.shuffle(all_pos); random.shuffle(all_neg)

    # 內部驗證集僅供 early-stopping (10%)，遠小於 fold model 的 35%
    val_pos_size = max(int(len(all_pos) * 0.10), 10)
    val_neg_size = max(int(len(all_pos) * 0.10 * CLASSIFIER_NEG_RATIO), 10)
    sub_val_s    = all_pos[:val_pos_size] + all_neg[:val_neg_size]
    tr_pos       = all_pos[val_pos_size:]
    tr_neg_pool  = all_neg[val_neg_size:]

    tf_base = [A.Resize(CLASSIFIER_IMG_SIZE, CLASSIFIER_IMG_SIZE),
               A.Normalize(mean=[0.417]*3, std=[0.268]*3), ToTensorV2()]
    tf_aug  = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                         A.RandomRotate90(p=0.5)] + tf_base)
    tf_val  = A.Compose(tf_base)

    val_l = DataLoader(ClassifierDS(sub_val_s, tf_val), batch_size=CLASSIFIER_BATCH_SIZE,
                       shuffle=False, num_workers=4)
    m = timm.create_model(CLASSIFIER_MODEL_NAME, pretrained=True, num_classes=2).to(device)
    opt = optim.AdamW(m.parameters(), lr=CLASSIFIER_LR, weight_decay=CLASSIFIER_WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda', enabled=True)
    crit   = FocalLoss(alpha=torch.tensor([0.05, 0.95]).to(device)).to(device)

    best_acc = 0; patience_counter = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for ep in range(CLASSIFIER_EPOCHS):
        cur_neg = random.sample(tr_neg_pool, max_negatives) if len(tr_neg_pool) > max_negatives else tr_neg_pool
        tr_s = tr_pos + cur_neg; random.shuffle(tr_s)
        tr_l = DataLoader(ClassifierDS(tr_s, tf_aug), batch_size=CLASSIFIER_BATCH_SIZE,
                          shuffle=True, num_workers=4)
        m.train(); rl = 0; tr_corr = 0
        for x, y in tqdm(tr_l, desc=f"Full Ep{ep+1}"):
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast('cuda', enabled=True):
                out = m(x); loss = crit(out, y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            rl += loss.item() * x.size(0)
            tr_corr += (out.argmax(1) == y).sum().item()
        train_losses.append(rl / len(tr_s)); train_accs.append(tr_corr / len(tr_s))

        m.eval(); corr = 0; val_rl = 0
        with torch.no_grad():
            for x, y in val_l:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast('cuda', enabled=True):
                    out = m(x); val_rl += crit(out, y).item() * x.size(0)
                corr += (out.argmax(1) == y).sum().item()
        acc = corr / len(sub_val_s)
        val_losses.append(val_rl / len(sub_val_s)); val_accs.append(acc)

        if acc > best_acc + EARLY_STOPPING_MIN_DELTA:
            best_acc = acc; patience_counter = 0
            torch.save(m.state_dict(), full_model_path)
            logging.info(f" Full Ep{ep+1} Internal-Val Acc: {acc:.4f} (Best: {best_acc:.4f}) - Saved")
        else:
            patience_counter += 1
            logging.info(f" Full Ep{ep+1} Internal-Val Acc: {acc:.4f} (Best: {best_acc:.4f}) - Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logging.info(f"Early stopping at epoch {ep+1}"); break

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(train_losses, label='Train'); ax1.plot(val_losses, label='Val', color='orange')
    ax1.set_title('Loss (Full-Data)'); ax1.legend()
    ax2.plot(train_accs, label='Train', color='blue'); ax2.plot(val_accs, label='Val', color='orange')
    ax2.set_title('Accuracy (Full-Data)'); ax2.legend()
    plt.tight_layout(); plt.savefig(full_model_dir / "training_curves.png"); plt.close()
    logging.info(f"Full-Data Classifier saved -> {full_model_path}")
    return full_model_path

# ── Stage 2: OOF Inference ────────────────────────────────────────────────────
class MineDS(Dataset):
    def __init__(self, files, tf): self.files, self.tf = files, tf
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        return self.tf(image=cv2.cvtColor(cv2.imread(str(self.files[i])), cv2.COLOR_BGR2RGB))['image'], str(self.files[i])

def run_phase_2_oof():
    """
    Phase 2 計算 EL2N 難度分數 + 抽 embedding。

    ENABLE_INNER_KFOLD=True (預設)：
      Pass 1: 用 K_FOLDS 個 fold classifier 各自在 val 集上算 OOF EL2N (無 in-sample bias)
      Pass 2: 用 Phase 1b 的 full-data classifier 抽 embedding

    ENABLE_INNER_KFOLD=False：
      單一 pass — 用 full-data classifier 對所有負樣本同時算
      (1) in-sample EL2N (probability of positive class，有 in-sample bias 但仍有排序意義)
      (2) embedding
    """
    tf = A.Compose([A.Resize(CLASSIFIER_IMG_SIZE, CLASSIFIER_IMG_SIZE),
                    A.Normalize(mean=[0.417]*3, std=[0.268]*3), ToTensorV2()])

    global_stage0_map_path = HNM_PIPELINE_BASE_DIR / "hnm_0_global_staging" / "global_stage0_file_map.xlsx"
    if not global_stage0_map_path.exists():
        logging.error(f"Global stage0 map not found at {global_stage0_map_path}."); return

    df_global0 = pd.read_excel(global_stage0_map_path)
    global_path_map = {str(r.original_img_path): r for r in df_global0.itertuples()}

    el2n_by_path = {}
    if ENABLE_INNER_KFOLD:
        logging.info("=== Phase 2: OOF EL2N (fold models) + Full-Data Embedding ===")
        # ── Pass 1: 各 fold 模型計算 OOF EL2N ───────────────────────────────
        for f_idx in range(K_FOLDS):
            fold_root = HNM_PIPELINE_BASE_DIR / f"fold_{f_idx}"
            m_path    = fold_root / "hnm_1_classifier_model" / "hnm_classifier.pt"
            if not m_path.exists():
                logging.warning(f"Fold {f_idx}: model missing. Skipping."); continue

            logging.info(f"OOF EL2N for Fold {f_idx}...")
            m = timm.create_model(CLASSIFIER_MODEL_NAME, pretrained=False, num_classes=2).to(device)
            m.load_state_dict(torch.load(m_path, map_location=device)); m.eval()

            df_info   = pd.read_excel(fold_root / "virtual_fold_info.xlsx")
            val_neg_df = df_info[(df_info['split'] == 'val') & (df_info['is_pos'] == False)]
            val_neg_paths = [p for p in val_neg_df['img_path'].astype(str).tolist() if p in global_path_map]

            loader = DataLoader(MineDS(val_neg_paths, tf), batch_size=CLASSIFIER_BATCH_SIZE,
                                shuffle=False, num_workers=4)
            with torch.no_grad():
                for imgs, paths in tqdm(loader, desc=f"Fold {f_idx} EL2N"):
                    imgs  = imgs.to(device)
                    probs = torch.softmax(m(imgs), dim=1)[:, 1].cpu().numpy()
                    el2n_scores = np.sqrt(2) * probs
                    for i, p in enumerate(paths):
                        el2n_by_path[p] = float(el2n_scores[i])

            del m
            if device.type == 'cuda': torch.cuda.empty_cache()

        if not el2n_by_path:
            logging.error("No EL2N collected from any fold. Cannot proceed."); return
        logging.info(f"Collected EL2N for {len(el2n_by_path)} unique negative patches.")

    # ── 載入 Full-Data classifier (任一模式都需要) ──────────────────────────
    full_model_path = HNM_PIPELINE_BASE_DIR / "hnm_1_full_data_classifier" / "hnm_classifier_full.pt"
    if not full_model_path.exists():
        logging.error(f"Full-Data classifier not found at {full_model_path}. "
                      f"Run Phase 1b first."); return

    m_full = timm.create_model(CLASSIFIER_MODEL_NAME, pretrained=False, num_classes=2).to(device)
    m_full.load_state_dict(torch.load(full_model_path, map_location=device)); m_full.eval()

    if ENABLE_INNER_KFOLD:
        logging.info("Extracting unified embeddings with Full-Data Classifier...")
        target_paths = list(el2n_by_path.keys())   # 用 Pass 1 收集到的 paths
        compute_el2n_inline = False
    else:
        logging.info("=== Phase 2 (no inner K-fold): Single-pass EL2N + Embedding (Full-Data) ===")
        target_paths = [str(r.original_img_path) for r in df_global0.itertuples()]  # 全部有效負樣本
        compute_el2n_inline = True

    loader_full = DataLoader(MineDS(target_paths, tf), batch_size=CLASSIFIER_BATCH_SIZE,
                             shuffle=False, num_workers=4)
    oof_recs = []; oof_embs = []
    with torch.no_grad():
        for imgs, paths in tqdm(loader_full,
                                 desc="Full-Data EL2N+Emb" if compute_el2n_inline else "Full-Data Embedding"):
            imgs  = imgs.to(device)
            feats = m_full.forward_features(imgs)
            if compute_el2n_inline:
                # forward_features 已產出 feature map，分類 logits 需單獨 forward
                logits = m_full(imgs)
                probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                el2n_scores = np.sqrt(2) * probs
            if feats.ndim == 4: feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)
            feats = feats.cpu().numpy()
            for i, p in enumerate(paths):
                info = global_path_map[p]
                if compute_el2n_inline:
                    el2n_by_path[p] = float(el2n_scores[i])
                oof_recs.append({
                    'original_img_path': p,
                    'original_lbl_path': info.original_lbl_path,
                    'staging_jpg_path':  info.staging_jpg_path,
                    'land_water_class':  info.land_water_class,   # land/coast/ocean
                    'el2n': el2n_by_path[p],
                    'emb_idx': len(oof_embs)
                })
                oof_embs.append(feats[i])

    del m_full
    if device.type == 'cuda': torch.cuda.empty_cache()

    global_mined_dir = HNM_PIPELINE_BASE_DIR / "global_oof_mined"
    ensure_clean_dir(global_mined_dir)
    if not oof_recs:
        logging.error("No OOF records. Cannot proceed."); return

    df_recs = pd.DataFrame(oof_recs)
    df_recs.to_excel(global_mined_dir / "oof_mined_file_map.xlsx", index=False)
    np.save(str(global_mined_dir / "oof_embeddings.npy"), np.array(oof_embs))

    # 視覺化分級
    logging.info("生成視覺化分級目錄...")
    df_sorted = df_recs.sort_values(by='el2n', ascending=False).reset_index(drop=True)
    n = len(df_sorted)
    if n > 0:
        vis_dir = global_mined_dir / "OOF_Visual_Difficulty"
        ensure_clean_dir(vis_dir)
        folders = {k: vis_dir / k for k in ['Hard', 'Medium', 'Easy']}
        for f in folders.values(): f.mkdir(parents=True, exist_ok=True)
        p1, p2 = n // 3, 2 * (n // 3)
        def save_vis(df_part, folder, mx=100):
            for i, row in enumerate(df_part.itertuples()):
                if i >= mx: break
                shutil.copy(row.staging_jpg_path, folder / f"{row.el2n:.4f}_{Path(row.staging_jpg_path).name}")
        save_vis(df_sorted.iloc[:p1], folders['Hard'])
        save_vis(df_sorted.iloc[p1:p2], folders['Medium'])
        save_vis(df_sorted.iloc[p2:], folders['Easy'])

# ── Submodular ────────────────────────────────────────────────────────────────
def greedy_facility_location(features, k, batch_size=2000):
    N = features.shape[0]
    if k >= N or k <= 0: return list(range(N))
    feats_t = torch.tensor(features, dtype=torch.float32, device=device)
    norms = torch.norm(feats_t, dim=1, keepdim=True); norms[norms == 0] = 1
    feats_t = feats_t / norms
    mean_feat = torch.mean(feats_t, dim=0)
    first_idx = int(torch.argmax(torch.matmul(feats_t, mean_feat)).item())
    selected = [first_idx]
    max_sim = torch.matmul(feats_t, feats_t[first_idx])
    for _ in tqdm(range(1, k), desc=f"Submodular (k={k})"):
        best_gain, best_idx = -1, -1
        for s in range(0, N, batch_size):
            e = min(N, s + batch_size)
            sim_b = torch.matmul(feats_t, feats_t[s:e].T)
            gains = torch.sum(torch.clamp(sim_b - max_sim.unsqueeze(1), min=0), dim=0)
            for j in range(e - s):
                if s + j in selected: gains[j] = -float('inf')
            lv, li = torch.max(gains, dim=0)
            if lv.item() > best_gain:
                best_gain = lv.item(); best_idx = s + li.item()
        selected.append(best_idx)
        max_sim = torch.max(max_sim, torch.matmul(feats_t, feats_t[best_idx]))
    return selected

# ── Stage 3: Assembly ─────────────────────────────────────────────────────────
def run_phase_3():
    logging.info("=== Phase 3: Assembly (3-class OOF-based Sampling) ===")
    global_mined_dir = HNM_PIPELINE_BASE_DIR / "global_oof_mined"
    oof_map_path  = global_mined_dir / "oof_mined_file_map.xlsx"
    oof_embs_path = global_mined_dir / "oof_embeddings.npy"
    if not oof_map_path.exists() or not oof_embs_path.exists():
        logging.error("OOF map/embeddings missing. Run Phase 2 first."); return

    df   = pd.read_excel(oof_map_path)
    embs = np.load(str(oof_embs_path))

    all_pos_files  = set()
    all_test_files = set()
    file_info_map  = {}

    if ENABLE_INNER_KFOLD:
        # 既有邏輯：從 inner fold 的 virtual_fold_info.xlsx 收集
        for f_idx in range(K_FOLDS):
            fold_root = HNM_PIPELINE_BASE_DIR / f"fold_{f_idx}"
            df_info   = pd.read_excel(fold_root / "virtual_fold_info.xlsx")
            for _, row in df_info.iterrows():
                file_info_map[str(row['img_path'])] = {
                    'split':    str(row['split']),
                    'is_pos':   row['is_pos'],
                    'lbl_path': str(row['lbl_path'])
                }
            all_pos_files.update(df_info[df_info['is_pos'] == True]['img_path'].apply(str))
            all_test_files.update(df_info[df_info['split'] == 'test']['img_path'].apply(str))
    else:
        # 無 inner kfold：直接從 SOURCE_PATCH_DIR 走 train / val / test 三 split 收集
        # 注意：nested 模式下 source 只有 train + test，沒 val (val 切分在訓練端 main_runner 做)
        for split_type in ['train', 'val', 'test']:
            df_s = _collect_files(SOURCE_PATCH_DIR, split_type)
            if df_s.empty:
                continue
            for _, row in df_s.iterrows():
                p = str(row['img_path'])
                file_info_map[p] = {
                    'split':    split_type,
                    'is_pos':   bool(row['is_pos']),
                    'lbl_path': str(row['lbl_path']) if row['lbl_path'] is not None else '',
                }
                if row['is_pos']:
                    all_pos_files.add(p)
                if split_type == 'test':
                    all_test_files.add(p)

    n_pos = len(all_pos_files)
    logging.info(f"[Phase 3] file_info_map 收齊 {len(file_info_map)} 個檔案，"
                  f"pos={n_pos}, test={len(all_test_files)}")

    for r in TARGET_RATIOS:
        out = HNM_PIPELINE_BASE_DIR / f"Global_Dataset_Ratio_1_{r}"
        ensure_clean_dir(out)
        d_i = out / 'images'; d_l = out / 'labels'
        d_i.mkdir(parents=True, exist_ok=True); d_l.mkdir(parents=True, exist_ok=True)

        def copy_file_and_label(img_path):
            str_p = str(img_path)
            if not Path(img_path).exists() or str_p not in file_info_map: return
            info = file_info_map[str_p]
            split_type = info['split']
            img_p = Path(img_path)
            images_dir = SOURCE_PATCH_DIR / "images"
            try:
                orig_split_cat = img_p.relative_to(images_dir).parts[0]
                cat_img_dir    = images_dir / orig_split_cat
                rel_parent     = img_p.parent.relative_to(cat_img_dir)
            except (ValueError, IndexError):
                rel_parent = Path('.')
            rel_parent_str = str(rel_parent)
            if rel_parent_str != '.':
                tgt_img_dir = d_i / split_type / rel_parent_str
                tgt_lbl_dir = d_l / split_type / rel_parent_str
            else:
                tgt_img_dir = d_i / split_type
                tgt_lbl_dir = d_l / split_type
            tgt_img_dir.mkdir(parents=True, exist_ok=True)
            tgt_lbl_dir.mkdir(parents=True, exist_ok=True)
            _link_or_copy(img_path, tgt_img_dir / img_p.name)
            lbl_str = info['lbl_path']
            if lbl_str and lbl_str != "None":
                lbl_p = Path(lbl_str)
                if lbl_p.exists():
                    _link_or_copy(lbl_p, tgt_lbl_dir / lbl_p.name)
                    if lbl_p.suffix == '.png' and lbl_p.with_suffix('.txt').exists():
                        _link_or_copy(lbl_p.with_suffix('.txt'), tgt_lbl_dir / lbl_p.with_suffix('.txt').name)

        for p in all_test_files:  copy_file_and_label(p)
        for p in all_pos_files:   copy_file_and_label(p)

        # ── 3-class 抽樣 ────────────────────────────────────────────────────
        n_neg   = int(n_pos * r)
        t_hard  = int(n_neg * HARD_ALLOC)
        t_coast = int(n_neg * COAST_ALLOC)
        t_land  = int(n_neg * LAND_ALLOC)
        t_ocean = max(n_neg - t_hard - t_coast - t_land, 0)

        logging.info(f"Ratio 1:{r} | n_pos={n_pos} n_neg={n_neg} "
                     f"=> hard={t_hard} coast={t_coast} land={t_land} ocean={t_ocean}")

        df_sort   = df.sort_values(by='el2n', ascending=False)
        hard_recs = df_sort.head(t_hard).to_dict('records')
        rem_df    = df_sort.iloc[t_hard:]

        coast_df = rem_df[rem_df['land_water_class'] == 'coast']
        land_df  = rem_df[rem_df['land_water_class'] == 'land']
        ocean_df = rem_df[rem_df['land_water_class'] == 'ocean']

        def _submod_select(sub_df, k):
            # k<=0：該類別配額 0 → 不選 (注意 greedy_facility_location 對 k<=0 會回傳全部，
            #        故必須在此短路，否則 LAND_ALLOC=0 會誤納全部 land patch)
            if len(sub_df) == 0 or k <= 0: return []
            if len(sub_df) <= k: return sub_df.to_dict('records')
            idx = greedy_facility_location(embs[sub_df['emb_idx'].values], k)
            return sub_df.iloc[idx].to_dict('records')

        coast_recs = _submod_select(coast_df, t_coast)
        land_recs  = _submod_select(land_df,  t_land)
        ocean_recs = _submod_select(ocean_df, t_ocean)

        # ── 視覺化 ──────────────────────────────────────────────────────────
        vis_dir = global_mined_dir / f"Selected_Visuals_Ratio_1_{r}"
        vis_folders = {
            'land_hard':   vis_dir / "Land_Hard",
            'coast_hard':  vis_dir / "Coast_Hard",
            'ocean_hard':  vis_dir / "Ocean_Hard",
            'land_easy':   vis_dir / "Land_Easy",
            'coast_easy':  vis_dir / "Coast_Easy",
            'ocean_easy':  vis_dir / "Ocean_Easy",
        }
        for d in vis_folders.values(): d.mkdir(parents=True, exist_ok=True)

        def copy_to_vis(recs, folder):
            for rec in recs:
                ip = Path(rec['original_img_path'])
                if ip.exists():
                    try:
                        im = cv2.imread(str(ip))
                        if im is not None:
                            cv2.imwrite(str(folder / f"{ip.stem}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    except: pass

        copy_to_vis([x for x in hard_recs if x.get('land_water_class') == 'land'],  vis_folders['land_hard'])
        copy_to_vis([x for x in hard_recs if x.get('land_water_class') == 'coast'], vis_folders['coast_hard'])
        copy_to_vis([x for x in hard_recs if x.get('land_water_class') == 'ocean'], vis_folders['ocean_hard'])
        copy_to_vis(land_recs,  vis_folders['land_easy'])
        copy_to_vis(coast_recs, vis_folders['coast_easy'])
        copy_to_vis(ocean_recs, vis_folders['ocean_easy'])

        for rec in hard_recs + coast_recs + land_recs + ocean_recs:
            copy_file_and_label(rec['original_img_path'])

        # ── [Manifest Excel] 紀錄最終 dataset 內所有檔案的來源、split、HNM bucket ─
        manifest_rows = []
        for p in all_pos_files:
            info = file_info_map.get(str(p), {})
            manifest_rows.append({
                'original_img_path': str(p),
                'split': info.get('split', 'unknown'),
                'is_pos': True,
                'hnm_bucket': 'positive',
                'land_water_class': '',
            })
        for p in all_test_files:
            info = file_info_map.get(str(p), {})
            manifest_rows.append({
                'original_img_path': str(p),
                'split': info.get('split', 'test'),
                'is_pos': bool(info.get('is_pos', False)),
                'hnm_bucket': 'test_passthrough',
                'land_water_class': '',
            })
        for bucket_name, recs in [('hard', hard_recs), ('coast', coast_recs),
                                   ('land', land_recs), ('ocean', ocean_recs)]:
            for rec in recs:
                info = file_info_map.get(rec['original_img_path'], {})
                manifest_rows.append({
                    'original_img_path': rec['original_img_path'],
                    'split': info.get('split', 'train'),
                    'is_pos': False,
                    'hnm_bucket': bucket_name,
                    'land_water_class': rec.get('land_water_class', ''),
                })
        df_manifest = pd.DataFrame(manifest_rows)
        manifest_path = out / "dataset_manifest.xlsx"
        df_manifest.to_excel(manifest_path, index=False)
        logging.info(f"[Manifest] dataset 內容清單寫入: {manifest_path} ({len(df_manifest)} entries)")


# ── Pipeline entry ────────────────────────────────────────────────────────────
def run_hnm_pipeline():
    """Dispatcher：根據本檔頂部開關決定跑單一份 HNM 還是 5-fold nested。

    所有設定都從本檔頂部 USER CONFIG 區塊讀取 (SOURCE_PATCH_DIR、TARGET_RATIOS、
    ENABLE_5FOLD_NESTED_SPLIT、...) ── 想改行為就改頂部的常數，不用傳參數。
    """
    if ENABLE_5FOLD_NESTED_SPLIT:
        _run_5fold_nested()
    else:
        _attach_hw_logger(HNM_PIPELINE_BASE_DIR)
        _run_single_hnm()


def _run_single_hnm():
    """執行一次完整 HNM (跟以前單獨跑的行為完全一樣)。直接讀取 module globals。"""
    if RESUME_FROM_PHASE_3:
        logging.info("[Resume] Skipping Phase 0-1, running Phase 1b (if missing) + Phase 2 & 3.")
        full_model_path = HNM_PIPELINE_BASE_DIR / "hnm_1_full_data_classifier" / "hnm_classifier_full.pt"
        if not full_model_path.exists():
            logging.info("[Resume] Full-Data classifier missing, training it now...")
            run_phase_1b_full_data_classifier()
        else:
            logging.info(f"[Resume] Full-Data classifier found at {full_model_path}, skipping Phase 1b.")
        run_phase_2_oof()
        run_phase_3()
        logging.info("HNM pipeline (Phase 1b + 2 + 3 Resume) done.")
        return

    # 每個 phase 都做 idempotent check：output 存在就跳過
    # (Global_Dataset_Ratio_* 一律重產，因為它是 Phase 3 結果且抽樣很快)
    HNM_PIPELINE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    for pat in ['Global_Dataset_Ratio_*']:
        for d in HNM_PIPELINE_BASE_DIR.glob(pat):
            if d.is_dir(): shutil.rmtree(d)

    # Phase 0
    phase0_out = HNM_PIPELINE_BASE_DIR / "hnm_0_global_staging" / "global_stage0_file_map.xlsx"
    if phase0_out.exists():
        logging.info(f"[Skip] Phase 0 已完成 ({phase0_out.name} 存在)")
    else:
        logging.info("[CV Step 1.5] Global Phase 0: Mask-based pre-split...")
        run_global_phase_0_mask(SOURCE_PATCH_DIR, HNM_PIPELINE_BASE_DIR)

    # Phase 1 (Inner K-fold classifiers, OOF EL2N)
    fold_accuracies = {}
    if ENABLE_INNER_KFOLD:
        if RESUME_FROM_FOLD == 0:
            logging.info(f"[CV Step 1] Splitting into {K_FOLDS} Folds...")
            fold_paths = create_virtual_kfold(SOURCE_PATCH_DIR, HNM_PIPELINE_BASE_DIR, n_splits=K_FOLDS)
        else:
            logging.info(f"[Resume] Resuming from Fold {RESUME_FROM_FOLD}")
            fold_paths = [HNM_PIPELINE_BASE_DIR / f"fold_{i}" for i in range(K_FOLDS)]
            if not (HNM_PIPELINE_BASE_DIR / "fold_0" / "virtual_fold_info.xlsx").exists():
                fold_paths = create_virtual_kfold(SOURCE_PATCH_DIR, HNM_PIPELINE_BASE_DIR, n_splits=K_FOLDS)

        for fold_idx, str_path in enumerate(fold_paths):
            fold_root = Path(str_path)
            setup_fold_globals(fold_root)
            if fold_idx < RESUME_FROM_FOLD:
                logging.info(f"Skipping Fold {fold_idx+1} (resuming from {RESUME_FROM_FOLD})")
                m_path = CLASSIFIER_MODEL_PATH
                fold_acc = 0.0
                if m_path.exists():
                    tf_val = A.Compose([A.Resize(CLASSIFIER_IMG_SIZE, CLASSIFIER_IMG_SIZE),
                                        A.Normalize(mean=[0.417]*3, std=[0.268]*3), ToTensorV2()])
                    m = timm.create_model(CLASSIFIER_MODEL_NAME, pretrained=False, num_classes=2).to(device)
                    m.load_state_dict(torch.load(m_path, map_location=device)); m.eval()
                    val_pos, val_neg = get_virtual_pos_neg_files(fold_root, 'val')
                    true_val_s = [(str(p), 1) for p in val_pos] + [(str(p), 0) for p in val_neg]
                    if true_val_s:
                        tl = DataLoader(ClassifierDS(true_val_s, tf_val), batch_size=CLASSIFIER_BATCH_SIZE,
                                        shuffle=False, num_workers=4)
                        corr = 0
                        with torch.no_grad():
                            for x, y in tl:
                                corr += (m(x.to(device)).argmax(1) == y.to(device)).sum().item()
                        fold_acc = corr / len(true_val_s)
                fold_accuracies[f"Fold {fold_idx+1}"] = fold_acc
                continue
            logging.info(f"\n{'='*60}\n Phase 1 for Fold {fold_idx+1}/{K_FOLDS}\n{'='*60}")
            fold_accuracies[f"Fold {fold_idx+1}"] = run_phase_1(fold_root)
    else:
        logging.info(f"\n{'='*60}\n ⚠️ ENABLE_INNER_KFOLD=False → 跳過 Phase 1 (no inner CV)\n"
                      f" Phase 2 EL2N 將用 Phase 1b 的 full-data classifier (in-sample bias)\n"
                      f"{'='*60}")

    # Phase 1b: 訓練 full-data 分類器（無 inner K-fold 時也用它算 EL2N）
    phase1b_out = HNM_PIPELINE_BASE_DIR / "hnm_1_full_data_classifier" / "hnm_classifier_full.pt"
    if phase1b_out.exists():
        logging.info(f"[Skip] Phase 1b 已完成 (hnm_classifier_full.pt 存在)")
    else:
        logging.info(f"\n{'='*60}\n Phase 1b: Full-Data Classifier\n{'='*60}")
        run_phase_1b_full_data_classifier()

    # Phase 2: OOF EL2N + embedding
    phase2_map = HNM_PIPELINE_BASE_DIR / "global_oof_mined" / "oof_mined_file_map.xlsx"
    phase2_emb = HNM_PIPELINE_BASE_DIR / "global_oof_mined" / "oof_embeddings.npy"
    if phase2_map.exists() and phase2_emb.exists():
        logging.info(f"[Skip] Phase 2 已完成 (oof_mined + embeddings 存在)")
    else:
        run_phase_2_oof()

    # Phase 3 一律重跑 (只是 file linking + 抽樣，便宜)
    run_phase_3()
    logging.info("HNM pipeline completed.")
    for f, acc in fold_accuracies.items():
        logging.info(f"  {f} Val Acc: {acc:.4f}")


def _path_to_event(file_path, source_patch_dir):
    """從 file_path 推 event name。
    路徑結構假設: source_patch_dir / images / <split>_<cat> / <event_name> / file.png
    """
    try:
        rel = Path(file_path).resolve().relative_to(Path(source_patch_dir).resolve())
        # rel.parts = ('images', '<split>_<cat>', '<event>', 'file.png')
        if len(rel.parts) >= 4 and rel.parts[0] == 'images':
            return rel.parts[2]
    except ValueError:
        pass
    return ""


def _write_5fold_manifest(folds, out_xlsx: Path, source_patch_dir: Path):
    """寫 5fold_split_manifest.xlsx — fold loop 前的 split 紀錄

    Sheets:
      Summary        — 每 fold 的 train/test pos/neg/total 數量
      EventBreakdown — 每 (fold, event, role) 檔案數 + pos/neg 分布
      Files_All      — 全部檔案清單 (fold_idx, role, is_pos, event_name, path)
    """
    summary_rows = []
    file_rows    = []
    for f in folds:
        i = f['fold_idx']
        n_tr_pos = sum(f['train_is_pos']); n_tr_neg = len(f['train_is_pos']) - n_tr_pos
        n_te_pos = sum(f['test_is_pos']);  n_te_neg = len(f['test_is_pos'])  - n_te_pos
        summary_rows.append({
            'fold': i,
            'train_pos': n_tr_pos, 'train_neg': n_tr_neg, 'train_total': n_tr_pos + n_tr_neg,
            'test_pos':  n_te_pos, 'test_neg':  n_te_neg, 'test_total':  n_te_pos + n_te_neg,
            'description': f['description'],
        })
        for p, is_pos in zip(f['test_files'], f['test_is_pos']):
            file_rows.append({
                'fold_idx': i, 'role': 'test', 'is_pos': bool(is_pos),
                'event_name': _path_to_event(p, source_patch_dir),
                'path': str(p),
            })
        for p, is_pos in zip(f['train_files'], f['train_is_pos']):
            file_rows.append({
                'fold_idx': i, 'role': 'train', 'is_pos': bool(is_pos),
                'event_name': _path_to_event(p, source_patch_dir),
                'path': str(p),
            })

    df_files = pd.DataFrame(file_rows)
    if not df_files.empty:
        grp = df_files.groupby(['fold_idx', 'event_name', 'role'], dropna=False)
        event_df = grp.agg(n_files=('path', 'count'),
                           n_pos=('is_pos', 'sum')).reset_index()
        event_df['n_neg'] = event_df['n_files'] - event_df['n_pos']
        event_df = event_df[['fold_idx', 'event_name', 'role', 'n_files', 'n_pos', 'n_neg']]
    else:
        event_df = pd.DataFrame()

    with pd.ExcelWriter(out_xlsx) as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name='Summary', index=False)
        event_df.to_excel(writer, sheet_name='EventBreakdown', index=False)
        df_files.to_excel(writer, sheet_name='Files_All', index=False)
    logging.info(f"[5fold] split manifest 寫入: {out_xlsx} ({len(df_files)} 檔案總計)")


def _append_5fold_post_hnm_stats(orig_base: Path, n_folds: int, manifest_xlsx: Path):
    """fold loop 跑完後讀各 fold 的 stage 0 + 最終 HNM dataset 結果，附加到 manifest xlsx

    新增 sheets:
      Stage0_Stats    — 每 fold 的 land/coast/ocean/invalid (含 border/fake_land 細分) 計數
      Final_HNM_Stats — 每 fold Global_Dataset_Ratio_* dataset_manifest 的 pos/neg/bucket 統計
    """
    stage0_rows = []
    final_rows  = []

    for i in range(n_folds):
        fold_dir = orig_base / f'fold_{i}'
        if not fold_dir.exists():
            continue

        # ── Stage 0 stats (含 invalid 細分) ────────────────────────────────
        stage0_xlsx = fold_dir / 'hnm_0_global_staging' / 'global_stage0_file_map_all.xlsx'
        if stage0_xlsx.exists():
            try:
                df = pd.read_excel(stage0_xlsx)
                row = {'fold_idx': i, 'total_neg': len(df)}
                for cat in ['land', 'coast', 'ocean', 'invalid']:
                    row[f'n_{cat}'] = int((df['land_water_class'] == cat).sum())
                if 'invalid_reason' in df.columns:
                    for reason in ['border', 'fake_land']:
                        row[f'n_invalid_{reason}'] = int(
                            ((df['land_water_class'] == 'invalid') &
                             (df['invalid_reason'] == reason)).sum()
                        )
                stage0_rows.append(row)
            except Exception as e:
                logging.warning(f"[5fold-stats] 讀 {stage0_xlsx} 失敗: {e}")

        # ── Final HNM dataset stats ────────────────────────────────────────
        for ratio_dir in sorted(fold_dir.glob('Global_Dataset_Ratio_*')):
            manifest_path = ratio_dir / 'dataset_manifest.xlsx'
            if not manifest_path.exists():
                continue
            try:
                df = pd.read_excel(manifest_path)
                row = {'fold_idx': i, 'ratio_dir': ratio_dir.name, 'total': len(df)}
                if 'is_pos' in df.columns:
                    row['n_pos'] = int(df['is_pos'].sum())
                    row['n_neg'] = int((~df['is_pos'].astype(bool)).sum())
                if 'hnm_bucket' in df.columns:
                    for bucket in sorted(df['hnm_bucket'].dropna().unique()):
                        row[f'bucket_{bucket}'] = int((df['hnm_bucket'] == bucket).sum())
                if 'land_water_class' in df.columns:
                    for cat in ['land', 'coast', 'ocean']:
                        row[f'lwclass_{cat}'] = int((df['land_water_class'] == cat).sum())
                final_rows.append(row)
            except Exception as e:
                logging.warning(f"[5fold-stats] 讀 {manifest_path} 失敗: {e}")

    if not stage0_rows and not final_rows:
        logging.info(f"[5fold-stats] 沒有可彙整的 stage 0 / final stats，跳過")
        return

    with pd.ExcelWriter(manifest_xlsx, mode='a', engine='openpyxl',
                         if_sheet_exists='replace') as writer:
        if stage0_rows:
            pd.DataFrame(stage0_rows).to_excel(writer, sheet_name='Stage0_Stats', index=False)
        if final_rows:
            pd.DataFrame(final_rows).to_excel(writer, sheet_name='Final_HNM_Stats', index=False)
    logging.info(f"[5fold-stats] Stage0_Stats + Final_HNM_Stats 已附加: {manifest_xlsx}")


def _run_5fold_nested():
    """5-fold outer split → 對每個 outer fold 物化 symlink + 跑單一份 HNM。

    輸出結構：
        HNM_PIPELINE_BASE_DIR/
          5fold_split_manifest.xlsx
          fold_0/
            source_patch_symlinks/   (symlink tree → orig SOURCE_PATCH_DIR)
            hnm_0_global_staging/
            hnm_1_full_data_classifier/
            ... (跟單一 HNM 輸出一樣的內容)
            Global_Dataset_Ratio_1_<X>/
          fold_1/ ...  fold_4/

    後續訓練：在 yaml 設 cross_validation.mode=nested_folds +
    nested_folds_base_dir = HNM_PIPELINE_BASE_DIR + folds_subpath = Global_Dataset_Ratio_1_<X>
    """
    global SOURCE_PATCH_DIR, HNM_PIPELINE_BASE_DIR, USE_SYMLINK

    # 從 main_cross_validation/ import helpers (HNM 與 main_cross_validation 在 sibling 層級)
    project_root = Path(__file__).resolve().parents[4]   # /home/yuan/Yuan/OIL_Project_12_7
    helpers_dir  = project_root / "main_cross_validation"
    if str(helpers_dir) not in sys.path:
        sys.path.insert(0, str(helpers_dir))
    from cross_validation_utils import prepare_5fold_nested_split, materialize_fold_as_symlinks

    orig_source  = SOURCE_PATCH_DIR
    orig_base    = HNM_PIPELINE_BASE_DIR
    orig_symlink = USE_SYMLINK
    orig_base.mkdir(parents=True, exist_ok=True)

    logging.info(f"\n{'#'*70}")
    logging.info(f"# [5-fold Nested HNM] outer K={NESTED_K_OUTER}, "
                  f"fix_last_fold={NESTED_FIX_LAST_FOLD_AS_ORIG}")
    logging.info(f"# source = {orig_source}")
    logging.info(f"# output = {orig_base}")
    logging.info(f"{'#'*70}\n")

    folds = prepare_5fold_nested_split(
        source_patch_dir=orig_source,
        k_outer=NESTED_K_OUTER,
        fix_last_fold_as_orig_test=NESTED_FIX_LAST_FOLD_AS_ORIG,
        scene_level_events=NESTED_SCENE_LEVEL_EVENTS,
        random_state=RANDOM_SEED,
    )
    manifest_xlsx = orig_base / '5fold_split_manifest.xlsx'
    _write_5fold_manifest(folds, manifest_xlsx, orig_source)

    # nested 模式強制 symlink (節省 99% disk)
    USE_SYMLINK = True

    for fold in folds:
        i = fold['fold_idx']
        fold_dir    = orig_base / f'fold_{i}'
        fold_source = fold_dir / 'source_patch_symlinks'
        fold_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"\n{'='*70}")
        logging.info(f"=== [Outer Fold {i+1}/{NESTED_K_OUTER}] 物化 symlinks → {fold_source}")
        logging.info(f"{'='*70}")
        materialize_fold_as_symlinks(fold, fold_source, orig_source)

        # 覆寫 globals 給 _run_single_hnm()，跑完還原
        SOURCE_PATCH_DIR      = fold_source
        HNM_PIPELINE_BASE_DIR = fold_dir
        _attach_hw_logger(HNM_PIPELINE_BASE_DIR)

        logging.info(f"\n=== [Outer Fold {i+1}/{NESTED_K_OUTER}] 跑單一 HNM → {fold_dir}")
        _run_single_hnm()

    # 還原 globals (萬一同進程後續還會用到)
    SOURCE_PATCH_DIR      = orig_source
    HNM_PIPELINE_BASE_DIR = orig_base
    USE_SYMLINK           = orig_symlink

    # 彙整 Stage 0 + Final HNM 統計 → 附加到 manifest xlsx
    _append_5fold_post_hnm_stats(orig_base, NESTED_K_OUTER, manifest_xlsx)

    logging.info(f"\n{'#'*70}")
    logging.info(f"# 5-fold Nested HNM pipeline 完成")
    logging.info(f"# 輸出根目錄: {orig_base}")
    logging.info(f"# Manifest:   {manifest_xlsx}")
    logging.info(f"#   sheets: Summary / EventBreakdown / Files_All / Stage0_Stats / Final_HNM_Stats")
    logging.info(f"# 下一步: yaml 設 cross_validation.mode=nested_folds + "
                  f"nested_folds_base_dir={orig_base}")
    logging.info(f"{'#'*70}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        # 不帶參數 → 用本檔頂部 USER CONFIG 區塊的設定
        run_hnm_pipeline()
    except Exception as e:
        logging.exception(f"Error: {e}")
    finally:
        logging.info("Stopping hardware monitor...")
        monitor_stop_event.set()
        monitor_thread.join(timeout=5)
        if HAS_NVML:
            try: pynvml.nvmlShutdown()
            except: pass
