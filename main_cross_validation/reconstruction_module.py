"""
[File] reconstruction_module.py
[Purpose] Image Reconstruction and Patch Assembly.
[Description]
    - Reconstructs full-sized images from processed patches.
    - Handles overlap blending and padding removal.
    - Supports adaptive strategies for non-standard image sizes.
    - Essential for converting patch-level predictions back to original image space.
[Author] Assistant (GitHub Copilot)
"""
import os
# [FIX] 放寬 OpenCV 讀圖像素上限 (預設 ~2^30)。fold_4 等折含超大原始 SAR 影像，
# reconstruction 讀整張原圖時會觸發 CV_IO_MAX_IMAGE_PIXELS assertion 而整折評估失敗 (指標 NaN)。
# 必須在 import cv2 之前設定才生效。與 RGB transform / VV blackout script 一致。
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(pow(2, 40))

import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
cv2.setNumThreads(0)
import pandas as pd
from pathlib import Path
from collections import defaultdict
import re
import sys
from tqdm import tqdm
from torchvision.ops import nms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.spatial import cKDTree


def compute_hd95(pred_mask, gt_mask):
    """真正的 HD95（95th-percentile Hausdorff Distance，單位: 像素）。
    做法: 取兩 mask 的邊界像素 → 雙向 KDTree 最近距離 → 取 95 百分位 (對齊 medpy.hd95)。
    相較舊的 max(directed_hausdorff)：
      (1) 真 95 百分位，對離群雜點穩健（HD 會被單一雜點炸到上百 px）；
      (2) 只用邊界點，DV5 全圖(8 億像素)約 4s 完成，全像素法則 >3 分鐘。
    邊界情況: 兩者皆空→0.0；僅一方有前景→該圖對角線長(最大可能距離)當懲罰。
    pred_mask/gt_mask: 2D，非零視為前景。"""
    pred_b = (pred_mask > 0).astype(np.uint8)
    gt_b = (gt_mask > 0).astype(np.uint8)
    ps, gs = int(pred_b.sum()), int(gt_b.sum())
    if ps == 0 and gs == 0:
        return 0.0
    h, w = pred_b.shape[:2]
    diag = float(np.hypot(h, w))
    if ps == 0 or gs == 0:
        return diag  # 一方全空：給最大可能距離(對角線)當懲罰
    # [限制安全] cv2.erode 對 >2^31 像素整圖會 segfault → 超大場景改用 scipy(numpy int64 索引)。
    if pred_b.size > 1_500_000_000:
        from scipy.ndimage import binary_erosion as _be
        pbm = (pred_b > 0); gbm = (gt_b > 0)
        pb = np.argwhere(pbm & (~_be(pbm)))  # pred 邊界像素
        gb = np.argwhere(gbm & (~_be(gbm)))  # gt 邊界像素
    else:
        k = np.ones((3, 3), np.uint8)
        pb = np.argwhere((pred_b > 0) & (cv2.erode(pred_b, k) == 0))  # pred 邊界像素
        gb = np.argwhere((gt_b > 0) & (cv2.erode(gt_b, k) == 0))      # gt 邊界像素
    if len(pb) == 0 or len(gb) == 0:
        return diag
    # [記憶體安全] 巨大場景模型 flood→邊界數百萬點, KDTree 爆 RAM(fold_4 OOM 主因)。
    # 邊界抽樣到 CAP: 95 百分位是分布統計, 抽樣後幾乎不變。固定 seed 保可重現。
    _CAP = 300000
    if len(pb) > _CAP or len(gb) > _CAP:
        _rng = np.random.default_rng(0)
        if len(pb) > _CAP: pb = pb[_rng.choice(len(pb), _CAP, replace=False)]
        if len(gb) > _CAP: gb = gb[_rng.choice(len(gb), _CAP, replace=False)]
    d_pg = cKDTree(gb).query(pb)[0]  # 每個 pred 邊界點 → 最近 gt 邊界
    d_gp = cKDTree(pb).query(gb)[0]  # 每個 gt 邊界點 → 最近 pred 邊界
    return float(np.percentile(np.concatenate([d_pg, d_gp]), 95))


def encode_mask_to_rgb(mask_idx, num_classes=5):
    """
    將 Multiclass Index Mask 轉換為 RGB 視覺化圖像 (BGR 格式供 cv2 使用)。
    """
    # 定義顏色表 (BGR)
    # 0: Background (Black)
    # 1: Oil (Yellow: 0, 255, 255)
    # 2: Look-alike (Red: 0, 0, 255)
    # 3: Land (Green: 0, 255, 0)
    # 4: Ship (Blue: 255, 0, 0)
    colors = [
        (0, 0, 0),       
        (0, 255, 255),   
        (0, 0, 255),     
        (0, 255, 0),     
        (255, 0, 0)      
    ]
    # 如果類別更多，隨機生成顏色
    while len(colors) < num_classes:
        colors.append(np.random.randint(0, 255, 3).tolist())
    
    h, w = mask_idx.shape
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(1, num_classes): # 跳過背景
        vis[mask_idx == c] = colors[c]
    return vis

def generate_gaussian_window(h, w, sigma_factor=4.0):
    """生成高斯權重矩陣，用於平滑拼接"""
    y = np.arange(h) - (h - 1) / 2.0
    x = np.arange(w) - (w - 1) / 2.0
    y_grid, x_grid = np.meshgrid(y, x, indexing='ij')
    # 防止 sigma 為 0 的情況
    sigma_h = max(h / sigma_factor, 1e-3)
    sigma_w = max(w / sigma_factor, 1e-3)
    gaussian = np.exp(-(x_grid**2 / (2 * sigma_w**2) + y_grid**2 / (2 * sigma_h**2)))
    return gaussian.astype(np.float32)

def stitch_masks(original_size, patch_predictions, patch_size, num_classes=1):
    """
    【Soft Voting (Probability Averaging)】
    將來自多個 overlapping patch 的分割機率圖拼接到原始大圖：
    直接累加機率 (而非二值化後的 0/1 票)，最後對 overlap 區域取平均機率。

    為什麼用 Soft 不用 Hard：
      Hard voting 在 50% overlap 下要求 >= 3/4 個 patch 同時 > 0.5 才會被標為正類，
      對 conservative model 來說會系統性壓制 borderline 預測（model 抓得到但被投票抹掉）。
      Soft voting 保留 prob 中的細節資訊，避免上述問題。

    Args:
        original_size: tuple (h, w) 原始大圖尺寸
        patch_predictions: list of (prob_map, x, y)
                           prob_map shape: (C, H_p, W_p) or (H_p, W_p)
        patch_size: int, Patch 的原始大小 (用於邊界計算參考)
        num_classes: 類別數量
    """
    h_orig, w_orig = original_size

    if num_classes == 1:
        score_map = np.zeros((h_orig, w_orig), dtype=np.float32)
    else:
        score_map = np.zeros((num_classes, h_orig, w_orig), dtype=np.float32)

    weight_map_accum = np.zeros((h_orig, w_orig), dtype=np.float32)

    for prob, x, y in patch_predictions:
        if num_classes > 1 and prob.ndim == 2:
            prob = np.expand_dims(prob, axis=0)

        if prob.ndim == 3:
            c, h_p, w_p = prob.shape
        else:
            h_p, w_p = prob.shape

        h_valid = min(h_p, h_orig - y)
        w_valid = min(w_p, w_orig - x)
        if h_valid <= 0 or w_valid <= 0:
            continue

        weight = 1.0

        if num_classes > 1:
            # Soft voting：直接累加各類 softmax prob，而非 argmax 後的 one-hot
            score_map[:, y:y+h_valid, x:x+w_valid] += prob[:, :h_valid, :w_valid].astype(np.float32) * weight
        else:
            # Soft voting：累加 sigmoid prob 本身，不做 > 0.5 二值化
            if prob.ndim == 3:
                patch_prob = prob[0, :h_valid, :w_valid].astype(np.float32)
            else:
                patch_prob = prob[:h_valid, :w_valid].astype(np.float32)
            score_map[y:y+h_valid, x:x+w_valid] += patch_prob * weight

        weight_map_accum[y:y+h_valid, x:x+w_valid] += weight
            
    # 計算得票率 (避免除以 0)
    weight_map_accum[weight_map_accum == 0] = 1.0
    
    if num_classes > 1:
        # score_map 目前為累加 softmax prob，除以 overlap 次數即為平均機率圖
        score_map /= weight_map_accum # (C, H, W)
        final_mask = np.argmax(score_map, axis=0).astype(np.uint8)
        return final_mask, score_map
    else:
        # score_map 目前為累加 sigmoid prob，除以 overlap 次數即為平均機率
        score_map /= weight_map_accum
        final_mask = (score_map > 0.5).astype(np.uint8)  # 平均機率 > 0.5 即判為 1
        return final_mask, score_map


class _GpuStitcher:
    """
    [Optimization ⑤] GPU 加速的增量式 mask 拼接器。
    記憶體優化：不累積所有 patch 後再拼接，而是邊推論邊累加到 score_map。

    與 CPU 版 (stitch_masks) 數值對應性：
      - score_map / weight_map_accum 完全相同（torch float32 reduction order 可能讓
        個別像素值差 ~1e-7，僅在 score == 0.5 邊界像素可能 flip，IoU 偏差通常 < 1e-3）。
    """
    def __init__(self, h, w, num_classes, device):
        self.h, self.w = h, w
        self.num_classes = num_classes
        self.device = device
        if num_classes == 1:
            self.score_map = torch.zeros((h, w), dtype=torch.float32, device=device)
        else:
            self.score_map = torch.zeros((num_classes, h, w), dtype=torch.float32, device=device)
        self.weight_map = torch.zeros((h, w), dtype=torch.float32, device=device)

    def add_batch(self, batch_prob, coords):
        """batch_prob: GPU tensor (B, C, H_p, W_p); coords: list of (x, y)

        [Soft Voting] 直接累加 prob，與 CPU 版 stitch_masks 行為一致。
        """
        for b, (x, y) in enumerate(coords):
            prob = batch_prob[b]
            if prob.ndim == 2:
                h_p, w_p = prob.shape
            else:
                _, h_p, w_p = prob.shape
            h_valid = min(h_p, self.h - y)
            w_valid = min(w_p, self.w - x)
            if h_valid <= 0 or w_valid <= 0:
                continue

            if self.num_classes > 1:
                # Soft voting：累加 softmax prob 而非 argmax 後的 one-hot
                self.score_map[:, y:y+h_valid, x:x+w_valid] += prob[:, :h_valid, :w_valid].float()
            else:
                # Soft voting：累加 sigmoid prob 本身
                if prob.ndim == 3:
                    patch_prob = prob[0, :h_valid, :w_valid].float()
                else:
                    patch_prob = prob[:h_valid, :w_valid].float()
                self.score_map[y:y+h_valid, x:x+w_valid] += patch_prob

            self.weight_map[y:y+h_valid, x:x+w_valid] += 1.0

    def finalize(self):
        """回傳 (final_mask: uint8 numpy, score_avg: float32 numpy)"""
        wm = torch.where(self.weight_map == 0,
                         torch.ones_like(self.weight_map),
                         self.weight_map)
        if self.num_classes > 1:
            avg = self.score_map / wm.unsqueeze(0)
            final = torch.argmax(avg, dim=0).to(torch.uint8).cpu().numpy()
        else:
            avg = self.score_map / wm
            final = (avg > 0.5).to(torch.uint8).cpu().numpy()
        avg_np = avg.cpu().numpy()
        del self.score_map, self.weight_map, avg, wm
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        return final, avg_np

def _save_jpg_downsized(path, img, max_dim, quality=70):
    """
    儲存 JPEG 前若超過 max_dim 先降採樣，加速大圖視覺化輸出。
    視覺化用途下，10000+ px 與 4096 px 對人眼幾乎無差異，但寫檔時間差 ~10x。
    """
    h, w = img.shape[:2]
    cur_max = max(h, w)
    if cur_max > max_dim:
        scale = max_dim / cur_max
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

def calculate_iou(mask1, mask2):
    """計算兩個二值化遮罩的 IoU。"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / (union + 1e-9)

def calculate_multiclass_iou(pred_mask, gt_mask, num_classes=5):
    """計算 Multiclass 每類 IoU"""
    iou_per_class = []
    for c in range(num_classes):
        p = (pred_mask == c)
        g = (gt_mask == c)
        intersection = np.logical_and(p, g).sum()
        union = np.logical_or(p, g).sum()
        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0 # Define as 1 if both empty? Or NaN. Usually ignore. 
            # 這裡為了簡單，若都沒有該類，視為 1 (預測正確地沒有該類)
            # 但這會拉高平均，通常 IoU 計算會忽略 union=0 的類別
            iou = np.nan 
        else:
            iou = intersection / union
        iou_per_class.append(iou)
    return iou_per_class

# ─────────────────────────────────────────────────────────────
# Test-Time Augmentation (TTA) — 推論期翻轉/旋轉平均，預設關閉。
#   tta='none' → 單次前向，行為與舊版完全一致 (ops=[identity])。
#   tta='flip' → 4 向 (原圖/水平/垂直/水平+垂直翻轉)；翻轉皆自反，逆=自身。
#   tta='d4'   → 8 向 dihedral (rot90×4 ⊗ 水平翻轉)；逆= 先解翻轉再 rot90(-k)，正方 patch 安全。
#   平均在「機率」上做 (softmax/sigmoid 後)，再 stitch。
# ─────────────────────────────────────────────────────────────
def _tta_ops(mode):
    """回傳 [(forward, inverse), ...]，作用於 (B,C,H,W) tensor 的空間維 (2,3)。"""
    import torch
    if mode == 'flip':
        return [
            (lambda t: t, lambda t: t),
            (lambda t: torch.flip(t, [3]), lambda t: torch.flip(t, [3])),       # 水平
            (lambda t: torch.flip(t, [2]), lambda t: torch.flip(t, [2])),       # 垂直
            (lambda t: torch.flip(t, [2, 3]), lambda t: torch.flip(t, [2, 3])), # 水平+垂直 (=180°)
        ]
    if mode == 'd4':
        ops = []
        for a in (0, 1):              # a: 是否水平翻轉
            for k in (0, 1, 2, 3):    # k: rot90 次數
                def make(k, a):
                    def fwd(t):
                        if k: t = torch.rot90(t, k, [2, 3])
                        if a: t = torch.flip(t, [3])
                        return t
                    def inv(t):
                        if a: t = torch.flip(t, [3])
                        if k: t = torch.rot90(t, -k, [2, 3])
                        return t
                    return (fwd, inv)
                ops.append(make(k, a))
        return ops
    return [(lambda t: t, lambda t: t)]


def _logits_to_prob(out, num_classes):
    """把模型輸出 (dict/logits/tensor) 轉成機率 (B,C,H,W)。"""
    import torch
    if isinstance(out, dict):
        if 'main' in out:     out = out['main']
        elif 'out' in out:    out = out['out']
        elif 'logits' in out: out = out['logits']
        else:                 out = next(iter(out.values()))
    elif hasattr(out, 'logits'):
        out = out.logits
    return torch.softmax(out, dim=1) if num_classes > 1 else torch.sigmoid(out)


def _forward_probs(model_adapter, batch_tensor, num_classes, tta_mode='none'):
    """前向 + (可選)TTA：對每個變換推論、逆變換還原、平均機率。tta='none' 時為單次前向。"""
    ops = _tta_ops(tta_mode)
    acc = None
    for fwd, inv in ops:
        out = model_adapter.model(fwd(batch_tensor))
        prob = inv(_logits_to_prob(out, num_classes))
        acc = prob if acc is None else acc + prob
    return acc / len(ops)


def run_reconstruction_evaluation(model_adapter, test_image_dir, original_data_root, results_path, imgsz, vis_params):
    """
    執行 Patch 重組評估 (Multiclass 修正版)。
    """
    print("\n--- [Info] Start: Running Patch Reconstruction Evaluation ---")
    
    architecture = model_adapter.config.get('architecture', 'unknown')
    num_classes = model_adapter.config.get('dataset', {}).get('nc', 1)
    print(f"  - [Info] Reconstruction running for architecture: '{architecture}', Num Classes: {num_classes}")
    
    # nested CV 下 fold 切分是 patch 層級，原始大圖可能落在 images/train/ 或 images/test/，
    # 跟 CV fold 無關，所以 lookup 要掃整個 images/ 跟 labels/ root (不限 train/test)。
    images_root = original_data_root / 'images'
    labels_root = original_data_root / 'labels'
    if not images_root.is_dir() or not labels_root.is_dir():
        print(f"  [ERROR] Reconstruction failed: Cannot find original images/labels root at:")
        print(f"  - {images_root}")
        print(f"  - {labels_root}")
        print(f"  - 請檢查 'reconstruction.original_data_root' 是否指向含 images/ 跟 labels/ 的資料集 root。")
        return {}

    all_patch_files = list(test_image_dir.rglob('*.jpg')) + list(test_image_dir.rglob('*.png'))

    patches_by_original = defaultdict(list)
    for patch_path in all_patch_files:
        # [DV5] 原命名 *_patch_x{N}_y{N}：DV5 一律先 match 這條，行為完全不變
        match = re.match(r"(.+)_patch_x(\d+)_y(\d+)", patch_path.stem)
        # [M4D] fallback：無 _patch token 的命名 img_0001_x0_y0（只有上面不中時才走）
        if not match:
            match = re.match(r"(.+)_x(\d+)_y(\d+)", patch_path.stem)
        if match:
            original_name, x, y = match.groups()
            patches_by_original[original_name].append((patch_path, int(x), int(y)))
    
    if not patches_by_original:
        print(f"  [ERROR] Reconstruction failed: No patches found in '{test_image_dir}'")
        return {}

    overlay_vis_dir = results_path / "reconstruction_overlays"; overlay_vis_dir.mkdir(exist_ok=True)
    label_vis_dir = results_path / "reconstruction_visuals"; label_vis_dir.mkdir(exist_ok=True)

    original_patch_size = vis_params.get('original_patch_size', imgsz)
    # [Optimization ①] 視覺化輸出最大邊長，預設 4096；可在 yaml 的 reconstruction 下覆寫
    vis_max_dim = int(vis_params.get('vis_max_dim', 4096))
    # [Optimization ⑤] GPU 加速拼接，預設關閉；開啟時 IoU 可能有 <1e-3 級數值差異
    gpu_stitch = bool(vis_params.get('gpu_stitch', False))
    if gpu_stitch:
        print(f"  [Info] GPU stitching ENABLED (numeric variance vs CPU < 1e-3 IoU expected).")
    # [Vis Layout] composite 視覺化是否包含 Original 那一格 (預設包含)
    # 為了省時間可在 yaml 設 vis_include_original: false → 只剩 (Pred | Overlay) 2 格
    vis_include_original = bool(vis_params.get('vis_include_original', True))
    print(f"  [Info] Composite layout = {'Original | Pred | Overlay' if vis_include_original else 'Pred | Overlay'}")
    # [HD95] yaml 開關：reconstruction 下設 compute_hd95: false 可關閉全圖 HD95 (DV5 約 +0.7s/張)。預設開。
    compute_hd95_flag = bool(vis_params.get('compute_hd95', True))
    print(f"  [Info] HD95 (whole-image oil) = {'ON' if compute_hd95_flag else 'OFF'}")
    # [TTA] yaml 開關：reconstruction 下設 tta: 'flip'/'d4' 開啟推論期增強 (預設 'none' = 行為不變)。
    tta_mode = str(vis_params.get('tta', 'none')).lower()
    if tta_mode not in ('none', 'flip', 'd4'):
        print(f"  [Warning] 未知 tta='{tta_mode}'，改用 'none'"); tta_mode = 'none'
    if tta_mode != 'none':
        print(f"  [Info] TTA = {tta_mode.upper()} ({'4' if tta_mode=='flip' else '8'} 向翻轉/旋轉，機率平均)")

    # determines real number of classes to track (Binary=2: 0,1; Multiclass=N)
    tracker_classes = 2 if num_classes == 1 else num_classes

    # 全域指標累積
    total_intersections = np.zeros(tracker_classes, dtype=np.float64)
    total_unions = np.zeros(tracker_classes, dtype=np.float64)
    hd95_list = []  # [HD95] 每張全圖 oil HD95，最後取平均
    hd95_tp_list = []  # [HD95_TP] 只統計「有 TP(pred∩gt>0)」的影像; 完全漏判(無TP)的影像剔除
    # [Threshold Sweep] binary 才啟用; vis_params['threshold_sweep'] = [0.3,0.4,...]; 累積各門檻 TP/FP/FN
    _sweep_ts = vis_params.get('threshold_sweep', None)
    sweep_acc = {float(t): [0, 0, 0] for t in _sweep_ts} if (_sweep_ts and num_classes == 1) else None
    if sweep_acc is not None:
        print(f"  [Info] Threshold Sweep ENABLED: {sorted(sweep_acc)} (純推論, 量 FP/IoU vs 門檻)")

    # Pre-build lookup dicts once to avoid repeated rglob inside the loop
    # 同時掃 train/ + test/ (rglob 從 root 出發)，因 CV fold 的 patch 對應原始圖可能在任一邊。
    _img_exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    orig_img_lookup = {p.stem: p for p in images_root.rglob('*') if p.suffix.lower() in _img_exts}
    orig_gt_lookup  = {p.stem: p for p in labels_root.rglob('*.png')}
    print(f"  [Info] Lookup tables built (train+test 合掃): {len(orig_img_lookup)} original images, {len(orig_gt_lookup)} GT masks.")

    if sys.stdout.isatty():
        iterator = tqdm(patches_by_original.items(), desc="Reconstructing and Visualizing")
    else:
        iterator = patches_by_original.items()

    log_interval = max(1, int(len(patches_by_original) * 0.2))

    for i, (original_name, patches) in enumerate(iterator):
        if not sys.stdout.isatty() and (i + 1) % log_interval == 0:
            print(f"  Reconstructing {i+1}/{len(patches_by_original)} ({((i+1)/len(patches_by_original))*100:.0f}%)")
        print(f"  [DBG SCENE-START {i+1}] {original_name[:50]} (patch={len(patches)})", flush=True)

        original_img_path = orig_img_lookup.get(original_name)
        gt_mask_path      = orig_gt_lookup.get(original_name)

        if not original_img_path or not original_img_path.exists(): continue
        if not gt_mask_path or not gt_mask_path.exists(): continue

        # 先用 PIL header 讀尺寸(不解碼)判斷大小, 決定載入方式
        from PIL import Image as _PILImage
        _PILImage.MAX_IMAGE_PIXELS = None
        try:
            _pw, _ph = _PILImage.open(str(gt_mask_path)).size; _spx = _pw * _ph
        except Exception:
            _spx = 0
        # [限制安全] cv2.imread 對 >2^31 像素整圖會 segfault(2.68GP 場景 = fold_4 評估真死點) →
        #   超大 GT 改用 PIL 載入(numpy int64 索引, 不過 cv2)。
        if _spx > 1_500_000_000:
            gt_mask_cv = np.asarray(_PILImage.open(str(gt_mask_path)).convert('L'))
        else:
            gt_mask_cv = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE) # 0-4 labels or 0/255
        if gt_mask_cv is None: continue
        h, w = gt_mask_cv.shape[:2]

        # [記憶體/限制安全] 原圖(H,W,3, 2.6GP≈8GB)僅供 overlay 視覺化; 超大場景(>600MP)既跳過 vis,
        #   cv2 又無法載超大圖 → 不載原圖。
        if (h * w) > 600_000_000:
            original_image = None
        else:
            original_image = cv2.imread(str(original_img_path))
            if original_image is None: continue

        # [FIX] Handle binary masks stored as 0/255
        if np.max(gt_mask_cv) > (num_classes if num_classes > 1 else 1):
            if num_classes == 1:
                # Binary Case: 255 -> 1
                gt_mask_cv = (gt_mask_cv > 127).astype(np.uint8)
            else:
                # Multiclass stored as large values? Not expected based on user input, but safe check
                pass

        gt_mask_multiclass = gt_mask_cv
        
        import torch.nn.functional as F
        batch_size = model_adapter.config.get('train', {}).get('batch_size', 2)
        model_adapter.model.eval()

        # [Optimization ⑤] GPU 路徑：邊推論邊拼接，免去 patch 列表累積；
        #                   CPU 路徑：保持原邏輯（累積後一次性 stitch）
        # [記憶體安全] 超大場景(>600MP, 如 2.6GP): GPU stitch 會 GPU OOM(10GB score_map),
        #   CPU collect-all 又囤 all_mask_inputs(~10GB)+score/weight(~21GB) → RAM OOM(fold_4 死因)。
        #   → 改用 _GpuStitcher 但 device='cpu' 串流: 逐 batch 移 cpu 累加, 不囤 patch、不爆 GPU。
        _giant = (h * w) > 600_000_000
        _stream_cpu = False
        if _giant:
            stitcher = _GpuStitcher(h, w, num_classes, torch.device('cpu')); all_mask_inputs = None; _stream_cpu = True
            print(f"    [記憶體安全] 場景 {h}x{w}={h*w/1e6:.0f}MP > 600MP → CPU 串流 stitch", flush=True)
        elif gpu_stitch:
            stitcher = _GpuStitcher(h, w, num_classes, model_adapter.device)
            all_mask_inputs = None
        else:
            stitcher = None
            all_mask_inputs = []

        has_any_patch = False
        for batch_start in range(0, len(patches), batch_size):
            batch_patches = patches[batch_start:batch_start+batch_size]
            batch_inputs = []
            valid_coords = []

            _nc_in = getattr(model_adapter, 'in_channels',
                             model_adapter.config.get('architecture_cfg', {}).get('in_channels', 3))
            for patch_path, x, y in batch_patches:
                if _nc_in == 4:
                    patch_img = cv2.imread(str(patch_path), cv2.IMREAD_UNCHANGED)  # BGRA
                    if patch_img is None: continue
                    if patch_img.ndim == 2:
                        patch_img = cv2.cvtColor(patch_img, cv2.COLOR_GRAY2BGRA)
                    elif patch_img.shape[2] == 3:
                        patch_img = cv2.cvtColor(patch_img, cv2.COLOR_BGR2BGRA)
                    patch_img = cv2.cvtColor(patch_img, cv2.COLOR_BGRA2RGBA)  # (H,W,4)
                else:
                    patch_img = cv2.imread(str(patch_path))
                    if patch_img is None: continue
                    patch_img = cv2.cvtColor(patch_img, cv2.COLOR_BGR2RGB)

                if hasattr(model_adapter, 'val_ds'):
                     aug = model_adapter.val_ds.transform(image=patch_img, mask=np.zeros(patch_img.shape[:2]))
                     inp = aug['image']
                else:
                     inp = torch.from_numpy(patch_img.transpose(2, 0, 1)).float() / 255.0

                batch_inputs.append(inp)
                valid_coords.append((x, y))

            if not batch_inputs: continue
            has_any_patch = True

            batch_tensor = torch.stack(batch_inputs).to(model_adapter.device)

            with torch.no_grad():
                # [TTA] tta_mode='none' 時為單次前向 (與舊版完全一致)；'flip'/'d4' 時多次推論平均機率
                batch_prob = _forward_probs(model_adapter, batch_tensor, num_classes, tta_mode)

                if batch_prob.shape[-1] != original_patch_size:
                    batch_prob = F.interpolate(batch_prob, size=(original_patch_size, original_patch_size),
                                               mode='bilinear', align_corners=False)

                if stitcher is not None:
                    # GPU stitch 直接加(免轉換); CPU 串流(_stream_cpu)先移到 cpu 再加
                    stitcher.add_batch(batch_prob.cpu() if _stream_cpu else batch_prob, valid_coords)
                else:
                    batch_prob_np = batch_prob.cpu().numpy()
                    for prob_map, (x, y) in zip(batch_prob_np, valid_coords):
                        all_mask_inputs.append((prob_map, x, y))

        if not has_any_patch: continue

        # Stitching ([Threshold Sweep] 接住 stitched 機率圖, 原本被丟成 _)
        if stitcher is not None:
            reconstructed_mask, prob_map = stitcher.finalize()  # finalize 內已 del score/weight map
            if _giant:  # 放掉 stitcher 物件讓 GC 回收
                stitcher = None
                import gc as _gc; _gc.collect()
        else:
            reconstructed_mask, prob_map = stitch_masks((h, w), all_mask_inputs, original_patch_size, num_classes=num_classes)
        filtered_mask = reconstructed_mask

        # --- Metrics Evaluation (Per Image) ---
        # Binary: num_classes=1 -> track 2 classes (0,1). Multiclass: num_classes=5 -> track 5
        eval_classes = 2 if num_classes == 1 else num_classes 
        
        ious = calculate_multiclass_iou(filtered_mask, gt_mask_multiclass, num_classes=eval_classes)
        # Oil is Class 1
        per_image_iou = ious[1] if len(ious) > 1 else ious[0]
        if np.isnan(per_image_iou): per_image_iou = 0.0

        # Accumulate for global
        for c in range(eval_classes):
             p = (filtered_mask == c)
             g = (gt_mask_multiclass == c)
             total_intersections[c] += np.logical_and(p, g).sum()
             total_unions[c] += np.logical_or(p, g).sum()

        # [Threshold Sweep] binary: 從 stitched 機率掃多門檻, 累積全域 TP/FP/FN
        #   (純推論、確定性 → 無訓練抽樣噪聲; 直接量「調高門檻能砍多少 FP / IoU 最佳門檻」)
        if sweep_acc is not None and prob_map is not None:
            g_oil = (gt_mask_multiclass == 1)
            for t in sweep_acc:
                pt = (prob_map > t)
                sweep_acc[t][0] += int(np.logical_and(pt, g_oil).sum())    # TP
                sweep_acc[t][1] += int(np.logical_and(pt, ~g_oil).sum())   # FP
                sweep_acc[t][2] += int(np.logical_and(~pt, g_oil).sum())   # FN

        # --- Visualization (Optimized) ---
        # 一次性計算 pred/gt mask 與 error_vis；合併三個輸出檔為單一 4-panel composite；
        # 視覺化前統一降採樣到 vis_max_dim，大幅減少 JPEG encode + I/O 時間。

        # 1. Identify Masks (Oil = Class 1) — 只算一次
        pred_oil = (filtered_mask == 1)
        gt_oil   = (gt_mask_multiclass == 1)

        # [HD95] 全圖 oil HD95（邊界KDTree 真95百分位），累積成全域平均；受 yaml 開關控制
        img_hd95 = None
        if compute_hd95_flag:
            img_hd95 = compute_hd95(pred_oil, gt_oil)
            hd95_list.append(img_hd95)
            # [HD95_TP] 該影像有 TP(pred 與 gt 油有重疊)才納入 TP-only 統計; 完全漏判的剔除
            if bool(np.logical_and(pred_oil, gt_oil).any()):
                hd95_tp_list.append(img_hd95)

        # [記憶體安全] 超大場景(>600MP, 如 2.6GP) overlay 視覺化會配 error_vis+overlay+composite(hstack)
        # 數十GB → OOM(fold_4 評估卡死/被 OOM-killer 砍的真主因)。metric(IoU)+HD95 已於上方算完, 跳過視覺化即可。
        if (h * w) > 600_000_000:
            print(f"    [記憶體安全] 場景 {h}x{w}={h*w/1e6:.0f}MP > 600MP, 跳過 overlay 視覺化(metric 不受影響)", flush=True)
        else:
            # 2. Error map (TP=Yellow, FP=Red, FN=Blue, BGR)
            error_vis = np.zeros_like(original_image)
            error_vis[np.logical_and(pred_oil, gt_oil)]                    = (0, 255, 255)  # TP
            error_vis[np.logical_and(pred_oil, np.logical_not(gt_oil))]    = (0, 0, 255)    # FP
            error_vis[np.logical_and(np.logical_not(pred_oil), gt_oil)]    = (255, 0, 0)    # FN

            # 3. Overlay (semi-transparent error on original) — 取代原本的第二次重複計算
            overlay_image = cv2.addWeighted(error_vis, vis_params.get('alpha', 0.5),
                                            original_image, 1 - vis_params.get('alpha', 0.5), 0)

            # 4. Pred visualization
            if num_classes > 1:
                pred_vis = encode_mask_to_rgb(filtered_mask, num_classes=eval_classes)
            else:
                pred_vis = error_vis  # binary 時直接用 error map（更含資訊）

            # 5. Composite：GT 拿掉（pred_vis 已含 TP/FP/FN 顏色資訊，足夠檢查）
            #    vis_include_original=True  → Original | Pred | Overlay
            #    vis_include_original=False → Pred | Overlay (更省時間)
            if vis_include_original:
                composite = np.hstack([original_image, pred_vis, overlay_image])
            else:
                composite = np.hstack([pred_vis, overlay_image])

            # 6. 單一檔輸出
            _save_jpg_downsized(
                overlay_vis_dir / f"{original_name}_result_iou_{per_image_iou:.4f}.jpg",
                composite, max_dim=vis_max_dim, quality=60
            )

            # 7. 獨立 pred 視覺化（保留給下游使用）
            _save_jpg_downsized(
                label_vis_dir / f"{original_name}.jpg",
                pred_vis, max_dim=vis_max_dim, quality=70
            )

        # 8. 進度紀錄：每張都印（方便逐張觀察）
        hd95_str = f", HD95: {img_hd95:.1f}px" if img_hd95 is not None else ""
        print(f"  - [{i+1}/{len(patches_by_original)}] {original_name} "
              f"({len(patches)} patches), IoU(Oil): {per_image_iou:.4f}{hd95_str}", flush=True)

    # --- Global Metrics ---
    iou_per_class = total_intersections / (total_unions + 1e-9)
    iou_oil = iou_per_class[1]
    mean_iou = np.mean(iou_per_class)
    
    # Background IoU (Class 0)
    iou_bg = iou_per_class[0]

    print(f"\n--- [Info] Reconstruction Evaluation Complete ---")
    print(f"    IoU (Oil - Class 1):        {iou_oil:.4f}")
    print(f"    IoU (Background - Class 0): {iou_bg:.4f}")
    print(f"    Mean IoU (mIoU):            {mean_iou:.4f}")

    result = {
        "reconstruction_iou_oil": f"{iou_oil:.4f}",
        "reconstruction_iou_bg": f"{iou_bg:.4f}",
        "reconstruction_mean_iou": f"{mean_iou:.4f}",
        # Dummy values for compatibility
        "reconstruction_accuracy": f"{mean_iou:.4f}",
        "reconstruction_f1_score": f"{iou_oil:.4f}"
    }
    # [HD95] 僅在開關開啟且有算到時才輸出 (越小越好，單位:像素)
    if compute_hd95_flag and hd95_list:
        mean_hd95 = float(np.mean(hd95_list))
        print(f"    HD95 (Oil, mean, 全影像):     {mean_hd95:.2f} px  (n={len(hd95_list)})")
        result["reconstruction_hd95"] = f"{mean_hd95:.2f}"
        # [HD95_TP] 剔除「完全無 TP(漏判)」影像後的 HD95: 只衡量有偵測到的影像的邊界精度
        if hd95_tp_list:
            mean_hd95_tp = float(np.mean(hd95_tp_list))
            n_drop = len(hd95_list) - len(hd95_tp_list)
            print(f"    HD95_TP (Oil, 僅有TP影像):    {mean_hd95_tp:.2f} px  (n={len(hd95_tp_list)}, 剔除無TP {n_drop} 張)")
            result["reconstruction_hd95_tp"] = f"{mean_hd95_tp:.2f}"
            result["reconstruction_hd95_tp_n"] = f"{len(hd95_tp_list)}"

    # [Threshold Sweep] 全域 IoU/Precision/Recall vs 門檻 (純推論; 找最佳門檻 / 量 FP 可砍程度)
    if sweep_acc is not None:
        print(f"\n--- [Threshold Sweep] 全域 Oil IoU/P/R vs 門檻 (pixel-pooled) ---")
        print(f"    {'thr':>5} {'IoU':>7} {'Prec':>7} {'Recall':>7} {'FP':>12} {'FN':>12}")
        best_t, best_iou = None, -1.0
        for t in sorted(sweep_acc):
            tp, fp, fn = sweep_acc[t]
            iou = tp / (tp + fp + fn + 1e-9)
            prec = tp / (tp + fp + 1e-9)
            rec = tp / (tp + fn + 1e-9)
            mark = ""
            if iou > best_iou: best_iou, best_t = iou, t
            print(f"    {t:>5.2f} {iou:>7.4f} {prec:>7.3f} {rec:>7.3f} {fp:>12,} {fn:>12,}")
        print(f"    → 最佳門檻 = {best_t:.2f} (Oil IoU={best_iou:.4f}); 預設 0.5 對照見上表")
        result["sweep_best_threshold"] = f"{best_t:.2f}"
        result["sweep_best_iou"] = f"{best_iou:.4f}"
    return result

def run_reconstruction_pipeline(model_adapter, dataset_config, save_dir, gradcam_config=None, reconstruction_config=None, explicit_original_root=None):
    """
    [V6.0 純淨版] 
    - 移除所有標準 Overlay 生成 (因為 run_reconstruction_evaluation 已做)。
    - 僅生成 vis_reconstruction_gradcam。
    """
    if not gradcam_config or not gradcam_config.get('enabled', False):
        return

    print("\n[Reconstruction] 開始執行 Grad-CAM 重建 (V6.0)...")
    save_dir = Path(save_dir)
    
    vis_grad_dir = save_dir / "vis_reconstruction_gradcam"
    vis_grad_dir.mkdir(parents=True, exist_ok=True)
    print(f"  [Info] Saving Grad-CAM outputs to: {vis_grad_dir}")

    try:
        from .gradcam_utils import GradCAM, compute_input_saliency
        gradcam_obj = GradCAM(model_adapter.model)
        enable_saliency = gradcam_config.get('saliency_map', False)
    except ImportError: return

    # --- 1. 準備路徑 ---
    dataset_path = Path(dataset_config['path'])
    test_img_dir = dataset_path / 'images' / 'test'
    if not test_img_dir.exists():
        test_img_dir = dataset_path / 'images' / 'val'
    if not test_img_dir.exists(): return

    # --- 2. 準備 Original Dataset 路徑 ---
    # 跟 run_reconstruction_evaluation 一致：直接掃 images/ 整層 (含 train/+test/)，
    # 因 nested CV 下 patch 對應原始圖可能落在任一邊。
    orig_img_root = None
    if explicit_original_root:
        cand = Path(explicit_original_root) / 'images'
        if cand.exists():
            orig_img_root = cand
        else:
            orig_img_root = Path(explicit_original_root)  # bare root fallback

    if not orig_img_root or not orig_img_root.exists():
        print("  [Error] Cannot find original images root for Grad-CAM. Skipping.")
        return

    # --- 3. 掃描 Patch ---
    all_patch_files = sorted(list(test_img_dir.rglob('*.jpg')) + list(test_img_dir.rglob('*.png')))
    patches_by_scene = defaultdict(list)
    
    for p_file in all_patch_files:
        # [DV5] 原命名 *_patch_x{N}_y{N}：DV5 一律先 match 這條，行為完全不變
        match = re.match(r"(.+)_patch_x(\d+)_y(\d+)", p_file.stem)
        # [M4D] fallback：無 _patch token 的命名 img_0001_x0_y0（只有上面不中時才走）
        if not match:
            match = re.match(r"(.+)_x(\d+)_y(\d+)", p_file.stem)
        if match:
            scene_id, x, y = match.groups()
            patches_by_scene[scene_id].append({'file': p_file, 'x': int(x), 'y': int(y)})
    
    if not patches_by_scene: return

    model_imgsz = 512
    if hasattr(model_adapter, 'config') and 'imgsz' in model_adapter.config:
        model_imgsz = model_adapter.config['imgsz']

    original_patch_size = None
    if reconstruction_config and 'original_patch_size' in reconstruction_config:
        original_patch_size = int(reconstruction_config['original_patch_size'])
        print(f"  [Info] Grad-CAM Reconstruction using original_patch_size: {original_patch_size}")

    if sys.stdout.isatty():
        iterator = tqdm(patches_by_scene.items(), desc="Reconstructing Grad-CAM")
    else:
        iterator = patches_by_scene.items()
    
    log_interval = max(1, int(len(patches_by_scene) * 0.2))

    for i, (scene_id, patch_list) in enumerate(iterator):
        if not sys.stdout.isatty() and (i + 1) % log_interval == 0:
            print(f"  Grad-CAM Reconstructing {i+1}/{len(patches_by_scene)} ({((i+1)/len(patches_by_scene))*100:.0f}%)")
        
        orig_file = next(orig_img_root.glob(f"{scene_id}.*"), None)
        if not orig_file: continue

        temp_img = cv2.imread(str(orig_file))
        if temp_img is None: continue
        max_h, max_w = temp_img.shape[:2]

        full_heatmap = np.zeros((max_h, max_w), dtype=np.float32)
        full_count_map = np.zeros((max_h, max_w), dtype=np.float32)
        full_saliency = None
        if enable_saliency:
            full_saliency = np.zeros((3, max_h, max_w), dtype=np.float32)

        for info in patch_list:
            p_file, x, y = info['file'], info['x'], info['y']
            patch_img_bgr = cv2.imread(str(p_file))
            if patch_img_bgr is None: continue
            
            if original_patch_size:
                target_h, target_w = original_patch_size, original_patch_size
            else:
                target_h, target_w = patch_img_bgr.shape[:2]

            h_valid = min(target_h, max_h - y)
            w_valid = min(target_w, max_w - x)
            if h_valid <= 0 or w_valid <= 0: continue

            full_count_map[y:y+h_valid, x:x+w_valid] += 1.0

            img_rgb = cv2.cvtColor(patch_img_bgr, cv2.COLOR_BGR2RGB)
            val_transform = A.Compose([
                A.Resize(model_imgsz, model_imgsz),
                A.Normalize(mean=(0.417, 0.417, 0.417), std=(0.267, 0.267, 0.267)),
                ToTensorV2()
            ])
            input_tensor = val_transform(image=img_rgb)['image'].unsqueeze(0).to(model_adapter.device)
            
            heatmap = gradcam_obj.generate_cam(input_tensor)
            if heatmap.ndim > 2: heatmap = np.mean(heatmap, axis=0)
            
            if heatmap.shape[:2] != (target_h, target_w): 
                heatmap = cv2.resize(heatmap, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            
            heatmap_valid = heatmap[0:h_valid, 0:w_valid]
            full_heatmap[y:y+h_valid, x:x+w_valid] += heatmap_valid
            
            if enable_saliency and full_saliency is not None:
                s_maps = compute_input_saliency(model_adapter.model, input_tensor)
                for c in range(min(3, s_maps.shape[0])):
                    sm = s_maps[c]
                    if sm.shape[:2] != (target_h, target_w): 
                        sm = cv2.resize(sm, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                    sm_valid = sm[0:h_valid, 0:w_valid]
                    full_saliency[c, y:y+h_valid, x:x+w_valid] += sm_valid

        full_count_map[full_count_map == 0] = 1.0
        full_heatmap /= full_count_map
        vis_h = (full_heatmap * 255).astype(np.uint8)
        vis_h = cv2.applyColorMap(vis_h, cv2.COLORMAP_JET)
        cv2.imwrite(str(vis_grad_dir / f"{scene_id}_GradCAM.jpg"), vis_h, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        
        if enable_saliency and full_saliency is not None:
            for c in range(3):
                full_saliency[c] /= full_count_map
                s_vis = (full_saliency[c] * 255).astype(np.uint8)
                s_vis = cv2.applyColorMap(s_vis, cv2.COLORMAP_JET)
                ch_name = ['Red', 'Green', 'Blue'][c]
                cv2.imwrite(str(vis_grad_dir / f"{scene_id}_Saliency_CH{c}_{ch_name}.jpg"), s_vis, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
    print(f"[Reconstruction Grad-CAM] 完成。")
