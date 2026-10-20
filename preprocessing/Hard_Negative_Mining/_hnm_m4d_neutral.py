import sys
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
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from concurrent.futures import ThreadPoolExecutor

"""
[本版本] NoPhase0 版本 — 移除 Phase 0 陸地/水域分離步驟

負樣本抽樣比例：
  困難樣本（Hard EL2N）：佔總負樣本的 50%
    直接取 EL2N 分數最高的前 50%
  多樣性樣本（Submodular Easy）：佔總負樣本的 50%
    對剩餘全部負樣本做 Greedy Facility Location 選出最具代表性的 50%
    (不再區分陸地/海洋，全部一起做 submodular)
"""

# --- ⭐⭐⭐ 全域設定 (User Config) ⭐⭐⭐ ---
SOURCE_PATCH_DIR = Path("/home/yuan/Yuan/OIL_Project_12_7/dataset/M4Ds/M4D-ourMethod/M4D_neutral__patch512ov128")
HNM_PIPELINE_BASE_DIR = Path("/home/yuan/Yuan/OIL_Project_12_7/dataset/M4Ds/M4D-ourMethod/M4D_neutral__patch512ov128__HNM_ratio1.5")

K_FOLDS = 3
TARGET_RATIOS = [1.5]
RANDOM_SEED = 42

VALID_PIXEL_THRESHOLD = 0.2

CLASSIFIER_MODEL_NAME = 'tf_efficientnetv2_s'
CLASSIFIER_IMG_SIZE = 512
CLASSIFIER_BATCH_SIZE = 48
CLASSIFIER_EPOCHS = 70
CLASSIFIER_LR = 0.00001
CLASSIFIER_WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 0.001
CLASSIFIER_NEG_RATIO = 5

# --- 斷點接續開關 ---
RESUME_FROM_PHASE_3 = False
RESUME_FROM_FOLD = 0

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

HARDWARE_LOG_FILE = HNM_PIPELINE_BASE_DIR / "hardware_monitor_log.txt"
HNM_PIPELINE_BASE_DIR.mkdir(parents=True, exist_ok=True)
hw_logger = logging.getLogger("HardwareMonitor")
hw_logger.setLevel(logging.INFO)
hw_handler = logging.FileHandler(HARDWARE_LOG_FILE)
hw_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
hw_logger.addHandler(hw_handler)

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_NVML = True
    gpu_count = pynvml.nvmlDeviceGetCount()
except ImportError:
    HAS_NVML = False
    gpu_count = 0
except Exception as e:
    hw_logger.warning(f"Failed to initialize NVML: {e}")
    HAS_NVML = False
    gpu_count = 0

def monitor_hardware(interval=10, stop_event=None):
    while not stop_event.is_set():
        try:
            cpu_percent = psutil.cpu_percent()
            ram = psutil.virtual_memory()
            log_str = f"CPU: {cpu_percent}% | RAM: {ram.percent}% ({ram.used/(1024**3):.1f}GB)"
            if HAS_NVML:
                for i in range(gpu_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    try:
                        power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                    except:
                        power = 0.0
                    log_str += f" || GPU {i}: Util {util.gpu}% | VRAM {mem.used/(1024**3):.1f}GB | Temp {temp}C | Power {power:.1f}W"
            hw_logger.info(log_str)
        except Exception as e:
            hw_logger.error(f"Error in hardware monitor loop: {e}")
        stop_event.wait(interval)

monitor_stop_event = threading.Event()
monitor_thread = threading.Thread(target=monitor_hardware, args=(20, monitor_stop_event), daemon=True)
monitor_thread.start()
hw_logger.info("Hardware monitoring started.")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CURRENT_FOLD_DIR = None
CLASSIFIER_MODEL_DIR = None
CLASSIFIER_MODEL_PATH = None
MINED_BASE_DIR = None

def setup_fold_globals(fold_dir):
    global CURRENT_FOLD_DIR, CLASSIFIER_MODEL_DIR, CLASSIFIER_MODEL_PATH, MINED_BASE_DIR
    CURRENT_FOLD_DIR = Path(fold_dir)
    CLASSIFIER_MODEL_DIR = CURRENT_FOLD_DIR / "hnm_1_classifier_model"
    CLASSIFIER_MODEL_PATH = CLASSIFIER_MODEL_DIR / "hnm_classifier.pt"
    MINED_BASE_DIR = CURRENT_FOLD_DIR / "hnm_2_mined"
    ensure_clean_dir(MINED_BASE_DIR)

def ensure_clean_dir(d):
    if d.exists(): shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)

# ----------------- 自建 K-Fold -----------------
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
            scene_id = parse_scene_id(img_path.name)
            rel_parent = img_path.parent.relative_to(cat_img_dir)
            lbl_png_path = cat_lbl_dir / rel_parent / f"{img_path.stem}.png"
            lbl_txt_path = cat_lbl_dir / rel_parent / f"{img_path.stem}.txt"
            t_lbl_path = None
            if lbl_png_path.exists(): t_lbl_path = lbl_png_path
            elif lbl_txt_path.exists(): t_lbl_path = lbl_txt_path
            records.append({
                'img_path': img_path, 'lbl_path': t_lbl_path,
                'is_pos': (cat == 'pos'), 'scene_id': scene_id
            })
    return pd.DataFrame(records)

def create_virtual_kfold(original_dir, base_save_dir, n_splits=3):
    df_train_val = pd.concat([_collect_files(original_dir, 'train'), _collect_files(original_dir, 'val')], ignore_index=True)
    df_test = _collect_files(original_dir, 'test')
    sgkf = StratifiedGroupKFold(n_splits=n_splits)
    X = df_train_val['img_path'].values
    y = df_train_val['is_pos'].values.astype(int)
    groups = df_train_val['scene_id'].values
    fold_paths = []
    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        fold_dir = base_save_dir / f"fold_{fold_idx}"
        fold_paths.append(fold_dir)
        ensure_clean_dir(fold_dir)
        df_train = df_train_val.iloc[train_idx].copy()
        df_val   = df_train_val.iloc[val_idx].copy()
        df_test_copy = df_test.copy()
        df_train['split'] = 'train'
        df_val['split']   = 'val'
        df_test_copy['split'] = 'test'
        fold_df = pd.concat([df_train, df_val, df_test_copy], ignore_index=True)
        fold_df['img_path'] = fold_df['img_path'].astype(str)
        fold_df['lbl_path'] = fold_df['lbl_path'].apply(lambda x: str(x) if pd.notna(x) and x is not None else "")
        fold_df.to_excel(fold_dir / "virtual_fold_info.xlsx", index=False)
    return fold_paths

# ----------------- 輔助函式 -----------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.5, reduction='mean', label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.reduction = reduction; self.label_smoothing = label_smoothing
    def forward(self, inputs, targets):
        ce_loss_smooth = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        with torch.no_grad(): pt = torch.exp(-F.cross_entropy(inputs, targets, reduction='none'))
        focal_loss = (1 - pt) ** self.gamma * ce_loss_smooth
        if self.alpha is not None: focal_loss = self.alpha[targets] * focal_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

def get_virtual_pos_neg_files(fold_root, split_name):
    df_path = fold_root / "virtual_fold_info.xlsx"
    if not df_path.exists(): return [], []
    df = pd.read_excel(df_path)
    df_split = df[df['split'] == split_name]
    pos_files = [Path(p) for p in df_split[df_split['is_pos'] == True]['img_path']]
    neg_files  = [Path(p) for p in df_split[df_split['is_pos'] == False]['img_path']]
    return pos_files, neg_files

# ----------------- Stage 0 Bypass -----------------
def run_global_stage0_bypass(source_dir, base_save_dir):
    """
    Phase 0 旁路：不做任何陸地/水域分離。
    只掃描所有 train+val 的負樣本，建立 global_stage0_file_map.xlsx，
    land_water_class 全部填 'all'，供 Phase 2 查詢路徑使用。
    """
    logging.info("=== Global Stage 0 (Bypass): 建立負樣本路徑索引，跳過陸地/水域分離 ===")
    global_stage0_dir = base_save_dir / "hnm_0_global_staging"
    global_stage0_dir.mkdir(parents=True, exist_ok=True)

    df_train = _collect_files(source_dir, 'train')
    df_val   = _collect_files(source_dir, 'val')
    df_all   = pd.concat([df_train, df_val], ignore_index=True)
    neg_df   = df_all[df_all['is_pos'] == False].copy()

    if neg_df.empty:
        logging.warning("No negative files found.")
        return

    records = []
    skipped = 0
    for _, row in neg_df.iterrows():
        img_path = Path(row['img_path'])
        lbl_path = row['lbl_path']

        # 過濾有效像素比例不足的殘破 patch（邊緣切割、大片黑邊）
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            skipped += 1
            continue
        valid_ratio = np.count_nonzero((img > 5) & (img < 250)) / img.size
        if valid_ratio < VALID_PIXEL_THRESHOLD:
            skipped += 1
            continue

        records.append({
            'land_water_class' : 'all',
            'original_img_path': str(img_path),
            'original_lbl_path': str(lbl_path) if lbl_path and str(lbl_path) != 'None' else "",
            'staging_jpg_path' : str(img_path),
        })

    pd.DataFrame(records).to_excel(
        global_stage0_dir / "global_stage0_file_map.xlsx", index=False)
    logging.info(f"  Stage 0 bypass 完成：{len(records)} 筆有效，{skipped} 筆因有效像素不足跳過。")

# ----------------- Stage 1 -----------------
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

    sub_val_s  = all_pos[:val_pos_size] + all_neg[:val_neg_size]
    tr_pos     = all_pos[val_pos_size:]
    tr_neg_pool= all_neg[val_neg_size:]

    tf_base = [A.Resize(CLASSIFIER_IMG_SIZE, CLASSIFIER_IMG_SIZE), A.Normalize(mean=[0.417]*3, std=[0.268]*3), ToTensorV2()]
    tf_aug  = A.Compose([A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)] + tf_base)
    tf_val  = A.Compose(tf_base)

    val_l = DataLoader(ClassifierDS(sub_val_s, tf_val), batch_size=CLASSIFIER_BATCH_SIZE, shuffle=False, num_workers=4)

    m     = timm.create_model(CLASSIFIER_MODEL_NAME, pretrained=True, num_classes=2).to(device)
    opt   = optim.AdamW(m.parameters(), lr=CLASSIFIER_LR, weight_decay=CLASSIFIER_WEIGHT_DECAY)
    scaler= torch.amp.GradScaler('cuda', enabled=True)
    crit  = FocalLoss(alpha=torch.tensor([0.05, 0.95]).to(device)).to(device)

    best_acc = 0; patience_counter = 0
    train_losses = []; val_losses = []; train_accuracies = []; val_accuracies = []

    for ep in range(CLASSIFIER_EPOCHS):
        cur_neg = random.sample(tr_neg_pool, max_negatives) if len(tr_neg_pool) > max_negatives else tr_neg_pool
        tr_s = tr_pos + cur_neg; random.shuffle(tr_s)
        tr_l = DataLoader(ClassifierDS(tr_s, tf_aug), batch_size=CLASSIFIER_BATCH_SIZE, shuffle=True, num_workers=4)

        m.train(); rl = 0; tr_corr = 0
        for x, y in tqdm(tr_l, desc=f"Ep{ep+1}"):
            x, y = x.to(device), y.to(device)
            with torch.amp.autocast('cuda', enabled=True):
                out = m(x); loss = crit(out, y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            rl += loss.item() * x.size(0)
            tr_corr += (out.argmax(1) == y).sum().item()

        avg_loss = rl / len(tr_s)
        train_losses.append(avg_loss)
        train_accuracies.append(tr_corr / len(tr_s))

        m.eval(); corr = 0; val_rl = 0
        with torch.no_grad():
            for x, y in val_l:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast('cuda', enabled=True):
                    out = m(x); val_loss = crit(out, y)
                val_rl += val_loss.item() * x.size(0)
                corr += (out.argmax(1) == y).sum().item()

        avg_val_loss = val_rl / len(sub_val_s)
        val_losses.append(avg_val_loss)
        acc = corr / len(sub_val_s)
        val_accuracies.append(acc)

        if acc > best_acc + EARLY_STOPPING_MIN_DELTA:
            best_acc = acc; patience_counter = 0
            torch.save(m.state_dict(), CLASSIFIER_MODEL_PATH)
            logging.info(f" Ep{ep+1} Sub-Val Acc: {acc:.4f} (Best: {best_acc:.4f}) - Model Saved")
        else:
            patience_counter += 1
            logging.info(f" Ep{ep+1} Sub-Val Acc: {acc:.4f} (Best: {best_acc:.4f}) - Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logging.info(f"Early stopping triggered at epoch {ep+1}"); break

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    ax1.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss', color='orange')
    ax1.set_title('Loss per Epoch'); ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend()
    ax2.plot(range(1, len(train_accuracies)+1), train_accuracies, label='Train Accuracy', color='blue')
    ax2.plot(range(1, len(val_accuracies)+1), val_accuracies, label='Validation Accuracy', color='orange')
    ax2.set_title('Accuracy per Epoch'); ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.legend()
    plt.tight_layout()
    plt.savefig(CLASSIFIER_MODEL_DIR / "training_curves.png"); plt.close()

    logging.info("Evaluating best model on True Val Set...")
    m.load_state_dict(torch.load(CLASSIFIER_MODEL_PATH, map_location=device)); m.eval()
    val_pos, val_neg = get_virtual_pos_neg_files(fold_root, 'val')
    true_val_s = [(str(p), 1) for p in val_pos] + [(str(p), 0) for p in val_neg]
    if len(true_val_s) == 0:
        logging.warning("True Val Set is empty!"); return 0.0

    true_val_l = DataLoader(ClassifierDS(true_val_s, tf_val), batch_size=CLASSIFIER_BATCH_SIZE, shuffle=False, num_workers=4)
    true_corr = 0
    with torch.no_grad():
        for x, y in true_val_l:
            out = m(x.to(device))
            true_corr += (out.argmax(1) == y.to(device)).sum().item()
    true_acc = true_corr / len(true_val_s)
    logging.info(f"Final True-Val Accuracy for this Fold: {true_acc:.4f}")
    return true_acc

# ----------------- Stage 2 -----------------
class MineDS(Dataset):
    def __init__(self, files, tf): self.files, self.tf = files, tf
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        return self.tf(image=cv2.cvtColor(cv2.imread(str(self.files[i])), cv2.COLOR_BGR2RGB))['image'], str(self.files[i])

def run_phase_2_oof():
    logging.info("=== Phase 2: Out-of-Fold (OOF) Inference & EL2N Extraction ===")
    oof_recs = []; oof_embs = []
    tf = A.Compose([A.Resize(CLASSIFIER_IMG_SIZE, CLASSIFIER_IMG_SIZE), A.Normalize(mean=[0.417]*3, std=[0.268]*3), ToTensorV2()])

    global_stage0_map_path = HNM_PIPELINE_BASE_DIR / "hnm_0_global_staging" / "global_stage0_file_map.xlsx"
    if not global_stage0_map_path.exists():
        logging.error(f"Global stage0 map not found: {global_stage0_map_path}")
        return

    df_global0 = pd.read_excel(global_stage0_map_path)
    global_path_map = {str(r.original_img_path): r for r in df_global0.itertuples()}

    for f_idx in range(K_FOLDS):
        current_fold_root = HNM_PIPELINE_BASE_DIR / f"fold_{f_idx}"
        m_path = current_fold_root / "hnm_1_classifier_model" / "hnm_classifier.pt"
        if not m_path.exists():
            logging.warning(f"Fold {f_idx}: Model missing. Skipping."); continue

        logging.info(f"Processing OOF for Fold {f_idx}...")
        m = timm.create_model(CLASSIFIER_MODEL_NAME, pretrained=False, num_classes=2).to(device)
        m.load_state_dict(torch.load(m_path, map_location=device)); m.eval()

        df_fold_info = pd.read_excel(current_fold_root / "virtual_fold_info.xlsx")
        val_neg_df   = df_fold_info[(df_fold_info['split'] == 'val') & (df_fold_info['is_pos'] == False)]
        val_neg_paths= val_neg_df['img_path'].astype(str).tolist()
        valid_paths  = [p for p in val_neg_paths if p in global_path_map]

        loader = DataLoader(MineDS(valid_paths, tf), batch_size=CLASSIFIER_BATCH_SIZE, shuffle=False, num_workers=4)
        with torch.no_grad():
            for imgs, paths in tqdm(loader, desc=f"Fold {f_idx} OOF"):
                imgs = imgs.to(device)
                probs = torch.softmax(m(imgs), dim=1)[:, 1].cpu().numpy()
                feats = m.forward_features(imgs)
                if feats.ndim == 4: feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)
                feats = feats.cpu().numpy()
                el2n_scores = np.sqrt(2) * probs
                for i, p in enumerate(paths):
                    info = global_path_map[p]
                    oof_recs.append({
                        'original_img_path': p,
                        'original_lbl_path': info.original_lbl_path,
                        'staging_jpg_path' : info.staging_jpg_path,
                        'el2n'             : float(el2n_scores[i]),
                        'emb_idx'          : len(oof_embs),
                    })
                    oof_embs.append(feats[i])

    global_mined_dir = HNM_PIPELINE_BASE_DIR / "global_oof_mined"
    ensure_clean_dir(global_mined_dir)

    if len(oof_recs) == 0:
        logging.error("No OOF records gathered."); return

    df_recs = pd.DataFrame(oof_recs)
    df_recs.to_excel(global_mined_dir / "oof_mined_file_map.xlsx", index=False)
    np.save(str(global_mined_dir / "oof_embeddings.npy"), np.array(oof_embs))

    # 視覺化：難/中/易 三個分級目錄
    logging.info("生成 OOF 視覺化分級目錄...")
    df_sorted = df_recs.sort_values(by='el2n', ascending=False).reset_index(drop=True)
    total_imgs = len(df_sorted)
    if total_imgs > 0:
        vis_dir = global_mined_dir / "OOF_Visual_Difficulty"
        ensure_clean_dir(vis_dir)
        folders = {'Hard': vis_dir/'Hard', 'Medium': vis_dir/'Medium', 'Easy': vis_dir/'Easy'}
        for f in folders.values(): f.mkdir(parents=True, exist_ok=True)
        p1 = total_imgs // 3; p2 = 2 * (total_imgs // 3)

        def save_visual_img(df_part, target_folder, max_num=100):
            count = 0
            for row in df_part.itertuples():
                if count >= max_num: break
                src = row.staging_jpg_path
                if Path(src).exists():
                    shutil.copy(src, target_folder / f"{row.el2n:.4f}_{Path(src).name}")
                    count += 1

        save_visual_img(df_sorted.iloc[:p1], folders['Hard'])
        save_visual_img(df_sorted.iloc[p1:p2], folders['Medium'])
        save_visual_img(df_sorted.iloc[p2:], folders['Easy'])

# ----------------- Submodular -----------------
def greedy_facility_location(features, k, batch_size=2000):
    N = features.shape[0]
    if k >= N or k <= 0: return list(range(N))
    features_t = torch.tensor(features, dtype=torch.float32, device=device)
    norms = torch.norm(features_t, dim=1, keepdim=True)
    norms[norms == 0] = 1
    features_t = features_t / norms
    mean_feat = torch.mean(features_t, dim=0)
    first_idx = int(torch.argmax(torch.matmul(features_t, mean_feat)).item())
    selected = [first_idx]
    max_sim_to_selected = torch.matmul(features_t, features_t[first_idx])
    for _ in tqdm(range(1, k), desc=f"Submodular (k={k})"):
        best_gain = -1; best_idx = -1
        for start_idx in range(0, N, batch_size):
            end_idx = min(N, start_idx + batch_size)
            sim_batch = torch.matmul(features_t, features_t[start_idx:end_idx].T)
            gain_batch = torch.clamp(sim_batch - max_sim_to_selected.unsqueeze(1), min=0)
            total_gains = torch.sum(gain_batch, dim=0)
            for idx_in_batch in range(end_idx - start_idx):
                if start_idx + idx_in_batch in selected:
                    total_gains[idx_in_batch] = -float('inf')
            local_best_val, local_best_idx = torch.max(total_gains, dim=0)
            if local_best_val.item() > best_gain:
                best_gain = local_best_val.item()
                best_idx = start_idx + local_best_idx.item()
        selected.append(best_idx)
        max_sim_to_selected = torch.max(max_sim_to_selected, torch.matmul(features_t, features_t[best_idx]))
    return selected

# ----------------- Stage 3 -----------------
def run_phase_3():
    logging.info("=== Phase 3: Assembly (Hard 50% + Submodular_All 50%) ===")

    global_mined_dir = HNM_PIPELINE_BASE_DIR / "global_oof_mined"
    oof_map_path  = global_mined_dir / "oof_mined_file_map.xlsx"
    oof_embs_path = global_mined_dir / "oof_embeddings.npy"

    if not oof_map_path.exists() or not oof_embs_path.exists():
        logging.error("OOF mined map or embeddings not found. Run Phase 2 first."); return

    df   = pd.read_excel(oof_map_path)
    embs = np.load(str(oof_embs_path))

    all_pos_files  = set()
    all_test_files = set()
    file_info_map  = {}

    for f_idx in range(K_FOLDS):
        fold_root = HNM_PIPELINE_BASE_DIR / f"fold_{f_idx}"
        df_info   = pd.read_excel(fold_root / "virtual_fold_info.xlsx")
        for _, row in df_info.iterrows():
            str_path = str(row['img_path'])
            file_info_map[str_path] = {
                'split'   : str(row['split']),
                'is_pos'  : row['is_pos'],
                'lbl_path': str(row['lbl_path']),
            }
        pos_df  = df_info[df_info['is_pos'] == True]
        all_pos_files.update(pos_df['img_path'].apply(str).tolist())
        test_df = df_info[df_info['split'] == 'test']
        all_test_files.update(test_df['img_path'].apply(str).tolist())

    n_pos = len(all_pos_files)

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
                cat_img_dir = images_dir / orig_split_cat
                rel_parent  = img_p.parent.relative_to(cat_img_dir)
            except (ValueError, IndexError):
                rel_parent = Path('.')
            rel_parent_str = str(rel_parent)
            if rel_parent_str != '.':
                target_img_dir = d_i / split_type / rel_parent_str
                target_lbl_dir = d_l / split_type / rel_parent_str
            else:
                target_img_dir = d_i / split_type
                target_lbl_dir = d_l / split_type
            target_img_dir.mkdir(parents=True, exist_ok=True)
            target_lbl_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_path, target_img_dir / img_p.name)
            lbl_str = info['lbl_path']
            if lbl_str and lbl_str != "None":
                lbl_p = Path(lbl_str)
                if lbl_p.exists():
                    shutil.copy(lbl_p, target_lbl_dir / lbl_p.name)
                    if lbl_p.suffix == '.png' and lbl_p.with_suffix('.txt').exists():
                        shutil.copy(lbl_p.with_suffix('.txt'), target_lbl_dir / lbl_p.with_suffix('.txt').name)

        # 1. 測試集 (100% 保留)
        for p in all_test_files:
            copy_file_and_label(p)

        # 2. 正樣本
        for p in all_pos_files:
            copy_file_and_label(p)

        # 3. 負樣本抽樣：Hard 50% + Submodular_All 50%
        n_neg  = int(n_pos * r)
        t_hard = int(n_neg * 0.5)
        t_easy = n_neg - t_hard

        df_sort   = df.sort_values(by='el2n', ascending=False)
        hard_recs = df_sort.head(t_hard).to_dict('records')
        rem_df    = df_sort.iloc[t_hard:].reset_index(drop=True)

        logging.info(f"[Ratio 1:{r}] n_neg={n_neg}  hard={t_hard}  submodular_easy={t_easy}  pool={len(rem_df)}")

        if len(rem_df) > t_easy:
            easy_indices = greedy_facility_location(embs[rem_df['emb_idx'].values], t_easy)
            easy_recs    = rem_df.iloc[easy_indices].to_dict('records')
        else:
            easy_recs = rem_df.to_dict('records')

        # 視覺化目錄
        vis_dir      = global_mined_dir / f"Selected_Visuals_Ratio_1_{r}"
        vis_hard_dir = vis_dir / "Hard_EL2N"
        vis_easy_dir = vis_dir / "Easy_Submodular"
        for d in [vis_hard_dir, vis_easy_dir]: d.mkdir(parents=True, exist_ok=True)

        def copy_to_vis(recs, target_vis_dir, max_num=200):
            count = 0
            for rec in recs:
                if count >= max_num: break
                img_path = Path(rec['original_img_path'])
                if img_path.exists():
                    try:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            tgt = target_vis_dir / f"{rec['el2n']:.4f}_{img_path.stem}.jpg"
                            cv2.imwrite(str(tgt), img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                            count += 1
                    except:
                        pass

        copy_to_vis(hard_recs, vis_hard_dir)
        copy_to_vis(easy_recs, vis_easy_dir)

        for rec in hard_recs + easy_recs:
            copy_file_and_label(rec['original_img_path'])

        logging.info(f"[Ratio 1:{r}] 完成，輸出：{out}")

if __name__ == "__main__":
    try:
        if RESUME_FROM_PHASE_3:
            logging.info("=== [Resume Mode] 直接執行 Phase 2 & 3 ===")
            run_phase_2_oof()
            run_phase_3()
            logging.info("HNM pipeline (Phase 2 & 3 Resume) completed successfully!")
        else:
            if RESUME_FROM_FOLD == 0:
                ensure_clean_dir(HNM_PIPELINE_BASE_DIR)
                logging.info(f"=== [Step 1] 切分 {K_FOLDS} Folds ===")
                fold_paths = create_virtual_kfold(SOURCE_PATCH_DIR, HNM_PIPELINE_BASE_DIR, n_splits=K_FOLDS)
                logging.info("=== [Step 1.5] Stage 0 Bypass（建立路徑索引，不做陸地/水域分離）===")
                run_global_stage0_bypass(SOURCE_PATCH_DIR, HNM_PIPELINE_BASE_DIR)
            else:
                logging.info(f"=== [Resume] 從 Fold {RESUME_FROM_FOLD} 繼續 ===")
                fold_paths = [HNM_PIPELINE_BASE_DIR / f"fold_{i}" for i in range(K_FOLDS)]
                if not (HNM_PIPELINE_BASE_DIR / "fold_0" / "virtual_fold_info.xlsx").exists():
                    fold_paths = create_virtual_kfold(SOURCE_PATCH_DIR, HNM_PIPELINE_BASE_DIR, n_splits=K_FOLDS)
                if not (HNM_PIPELINE_BASE_DIR / "hnm_0_global_staging" / "global_stage0_file_map.xlsx").exists():
                    run_global_stage0_bypass(SOURCE_PATCH_DIR, HNM_PIPELINE_BASE_DIR)

            fold_accuracies = {}
            for fold_idx, str_path in enumerate(fold_paths):
                fold_root = Path(str_path)
                setup_fold_globals(fold_root)

                if fold_idx < RESUME_FROM_FOLD:
                    logging.info(f"Skipping Fold {fold_idx+1} (resume mode)")
                    m_path = CLASSIFIER_MODEL_PATH
                    if m_path.exists():
                        tf_val = A.Compose([A.Resize(CLASSIFIER_IMG_SIZE, CLASSIFIER_IMG_SIZE), A.Normalize(mean=[0.417]*3, std=[0.268]*3), ToTensorV2()])
                        m = timm.create_model(CLASSIFIER_MODEL_NAME, pretrained=False, num_classes=2).to(device)
                        m.load_state_dict(torch.load(m_path, map_location=device)); m.eval()
                        val_pos, val_neg = get_virtual_pos_neg_files(fold_root, 'val')
                        true_val_s = [(str(p), 1) for p in val_pos] + [(str(p), 0) for p in val_neg]
                        if len(true_val_s) > 0:
                            true_val_l = DataLoader(ClassifierDS(true_val_s, tf_val), batch_size=CLASSIFIER_BATCH_SIZE, shuffle=False, num_workers=4)
                            true_corr = 0
                            with torch.no_grad():
                                for x, y in true_val_l:
                                    out = m(x.to(device))
                                    true_corr += (out.argmax(1) == y.to(device)).sum().item()
                            fold_acc = true_corr / len(true_val_s)
                        else: fold_acc = 0.0
                    else:
                        fold_acc = 0.0
                    fold_accuracies[f"Fold {fold_idx+1}"] = fold_acc
                    continue

                logging.info(f"\n{'='*60}\n======== Phase 1: Fold {fold_idx+1}/{K_FOLDS} ========\n{'='*60}")
                fold_acc = run_phase_1(fold_root)
                fold_accuracies[f"Fold {fold_idx+1}"] = fold_acc

            logging.info("======== Phase 2: OOF Extraction ========")
            run_phase_2_oof()
            logging.info("======== Phase 3: Global Assembly ========")
            run_phase_3()
            logging.info("HNM pipeline completed successfully!")
            logging.info("=== CV Classifier Results ===")
            for f, acc in fold_accuracies.items():
                logging.info(f"  {f}: {acc:.4f}")

    except Exception as e:
        logging.exception(f"Error: {e}")
    finally:
        logging.info("Stopping hardware monitor...")
        monitor_stop_event.set()
        monitor_thread.join(timeout=5)
        if HAS_NVML:
            try: pynvml.nvmlShutdown()
            except: pass
