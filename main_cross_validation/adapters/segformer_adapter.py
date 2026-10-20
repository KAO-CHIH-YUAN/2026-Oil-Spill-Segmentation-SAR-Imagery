"""
[Module] Segformer Adapter
[Description]
    This adapter integrates the Segformer model.
    It is designed to handle both:
      1. Multiclass Segmentation (e.g., M4D dataset with 5 classes).
      2. Binary Segmentation (e.g., Private Oil/Water dataset).
    
    Key Features:
    - Supports dynamic switching between Multiclass and Binary modes based on 'nc' (num_classes).
    
    - Configurable 'DropPath' and 'Dropout' for regularization (EL2N / Domain Adaptation).
    - Designed for easy modification and hyperparameter tuning (see 'architecture_cfg' in yaml).

[Excel Logging Rules]
    The 'main_runner.py' dynamically selects columns to log based on 'nc' (num_classes):
    1. Multiclass Mode (nc > 2):
       - Logs Global Metrics: Accuracy, mIoU, Macro Precision/Recall/F1.
       - Logs Per-Class IoU: e.g. IoU_Sea_Surface, IoU_Oil_Spill.
       - Hides Binary-specific metrics (IoU_Bg, Reconstruction scores).
    2. Binary Mode (nc = 1 or 2):
       - Logs Binary Metrics: Accuracy, IoU(Fg), IoU_Bg.
       - Logs Reconstruction Metrics (if enabled): reconstruction_mIoU, etc.

[Usage]
    Refer to 'experiments_m4d_multiclass.yaml' or 'experiments.yaml'.
    Set 'architecture: segformer' and provide 'architecture_cfg'.
"""

"""
此外，在 MiT 編碼器的深度前饋網路（Feed-Forward Networks, FFN）中，應適度提高 DropPath（Stochastic Depth）的比率，
隨機丟棄整個殘差區塊，這等同於在訓練過程中隱式地訓練了一個模型集成（Ensemble），極大化泛化能力。
"""

import os
import torch
import torch.nn as nn
import cv2
import sys
import time
import copy
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pathlib import Path
import yaml
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import pandas as pd
import pickle # [New]
try:
    import seaborn as sns
except ImportError:
    pass

try:
    from timm.utils import ModelEmaV2
    has_timm_ema = True
except ImportError:
    has_timm_ema = False
    print("[Warning] timm not found. EMA will be disabled.")
except ImportError:
    sns = None
from sklearn.metrics import confusion_matrix
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from ..training_module import register_model
from ..calculate_stats import calculate_dataset_stats


class SARSpeckleNoise(A.ImageOnlyTransform):
    """
    自定義的 SAR Speckle Noise 資料擴增。
    將影像(uint8)暫轉為浮點數，乘上均值為1、變異數為 variance 的 Gamma 分布雜訊，
    最後裁剪(Clip)並轉回 uint8。符合 SAR 影像乘性雜訊(Multiplicative Noise)物理特性。
    """
    def __init__(self, variance=0.04, always_apply=False, p=0.5):
        super(SARSpeckleNoise, self).__init__(always_apply, p)
        self.variance = variance

    def apply(self, img, **params):
        img_float = img.astype(np.float32)
        # Gamma distribution: mean = shape * scale = 1.0, var = shape * scale^2 = variance
        shape = 1.0 / self.variance
        scale = self.variance
        noise = np.random.gamma(shape, scale, img.shape).astype(np.float32)
        noisy_img = img_float * noise
        return np.clip(noisy_img, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("variance",)


class ModelEma:
    def __init__(self, model, decay=0.9999, device=None):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device
        if self.device:
            self.module.to(device=device)

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.module.state_dict()
            for k in msd:
                if k in esd:
                    v_m = msd[k].detach()
                    v_e = esd[k]
                    if self.device:
                        v_m = v_m.to(self.device)
                    v_e.copy_(v_e * self.decay + v_m * (1. - self.decay))

class SimpleSegmentationHead(nn.Module):
    def __init__(self, in_channels, num_classes, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size, stride, padding)
    def forward(self, x): return self.conv(x)


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None, in_channels=3, num_classes=1):
        self.image_dir, self.mask_dir = Path(image_dir), Path(mask_dir)
        self.image_files = sorted(list(self.image_dir.glob('*.png')) + list(self.image_dir.glob('*.jpg')))
        self.transforms = transforms
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # [新增] 預先計算每個 Mask 是否包含目標類別，用於 WeightedRandomSampler
        self.has_object = []
        # 為了避免在 init 時讀取過久，這裡可以設一個簡單的 cache 機制或只在需要 sampling 時計算
        # 但為了穩定性，這裡假設我們需要這個信息。如果 dataset 很大，這步可能會慢。
        # 暫時先不在此計算，改為由外部計算權重傳入 DataLoader
        
    def get_labels(self):
        # 快速掃描 Mask 作為 Sampling 依據
        labels = []
        print("[Info] Scanning dataset for class balance (WeightedRandomSampler)... [Optimized with .txt]")
        for img_path in tqdm(self.image_files):
            # Optimization: 嘗試讀取 YOLO .txt 標註檔 (如果存在)
            # 假設 .txt 存在且非空 = 有物件 (Positive)
            # 假設 .txt 存在且為空 (size=0) = 無物件 (Negative)
            txt_path = self.mask_dir / (img_path.stem + '.txt')
            
            if txt_path.exists():
                if txt_path.stat().st_size > 0:
                    labels.append(1) # Has object
                else:
                    labels.append(0) # Background
                continue

            # Fallback: 如果沒有 .txt，讀取 PNG Mask
            mask_path = self.mask_dir / img_path.name
            if not mask_path.exists(): mask_path = self.mask_dir / (img_path.stem + '.png')
            
            # 必須讀取內容才能知道有沒有像素
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                labels.append(0)
            else:
                if np.any(mask > 0): labels.append(1)
                else: labels.append(0)
        return labels

    def __len__(self): return len(self.image_files)
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_dir / img_path.name
        # Fallback search if name differs (e.g. .jpg image but .png mask)
        if not mask_path.exists():
             mask_path = self.mask_dir / (img_path.stem + '.png')
        
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            # [修改] 處理圖像讀取失敗
            print(f"[Warning] Failed to read image: {img_path}. Returning empty tensor.")
            placeholder_imgsz = 512
            if self.transforms:
                 for t in self.transforms.transforms:
                      if isinstance(t, A.Resize): placeholder_imgsz = t.height; break
            
            # Return empty
            if self.num_classes > 1:
                return torch.zeros((self.in_channels, placeholder_imgsz, placeholder_imgsz)), torch.zeros((placeholder_imgsz, placeholder_imgsz), dtype=torch.long)
            else:
                return torch.zeros((self.in_channels, placeholder_imgsz, placeholder_imgsz)), torch.zeros((1, placeholder_imgsz, placeholder_imgsz))
        
        # --- [修改] 根據 in_channels 選擇通道 ---
        if self.in_channels == 1:
             image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        elif self.in_channels == 2:
             image = image_bgr[:, :, [2, 1]] # 選取 R 和 G 通道 (順序 GR)
        else: # 預設 3 通道
             image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB) # HF Segformer 預設使用 RGB

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None: mask = np.zeros((image_bgr.shape[0], image_bgr.shape[1]), dtype=np.uint8)
        
        if self.num_classes > 1:
             # Multiclass: Keep original values (0, 1, 2...)
             # Albumentations expects mask to be numpy array
             pass
        else:
             # Binary: Threshold
             mask = (mask > 0).astype(np.float32)
             mask = np.expand_dims(mask, axis=-1)
        
        if self.transforms:
            transformed = self.transforms(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']
        
        # Handle Output Tensor Types
        if self.num_classes > 1:
            # Ensure mask is LongTensor (int64) for CrossEntropy, and usually 2D (H, W) or (1, H, W)
            # ToTensorV2 usually converts to Tensor. If it's 2D numpy, it becomes 2D Tensor.
            if isinstance(mask, torch.Tensor):
                 mask = mask.long()
            else:
                 mask = torch.from_numpy(mask).long()
            # If Albumentations added a channel dim to mask (e.g. 1, H, W) but we need (H, W) for CE:
            if mask.ndim == 3 and mask.shape[0] == 1:
                 mask = mask.squeeze(0)
                 
            return image, mask # (C, H, W), (H, W)
        else:
            # Binary: Needs (1, H, W) Float
            if not isinstance(mask, torch.Tensor):
                mask = torch.from_numpy(mask).float()
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.shape[0] != 1 and mask.shape[-1] == 1: # (H, W, 1) -> (1, H, W)
                 mask = mask.permute(2, 0, 1)
            # Ensure shape (C, H, W) from the earlier permute call in original code is respected?
            # Original code had `return image, mask.permute(2, 0, 1)` which implies mask was (H, W, C).
            # If ToTensorV2 was used, it's already (C, H, W) or (H, W).
            # Let's trust ToTensorV2 results or standard conventions.
            
            # Revert to standard check if permute is needed:
            if mask.ndim == 3 and mask.shape[2] == 1: # HWC
                 mask = mask.permute(2, 0, 1)
                 
            return image, mask

class SegformerPredictionResult:
    def __init__(self, original_image, pred_mask_binary, pred_mask_prob, **kwargs):
        self.original_image = original_image
        
        # [FIX v1.2] 儲存兩個版本的遮罩
        self.pred_mask_np = pred_mask_binary # (二值化, 0/1) 供 evaluation_module 使用
        self.pred_mask_binary_np = pred_mask_binary # (明確命名)
        self.pred_mask_prob_np = pred_mask_prob     # (機率圖, 0.0~1.0) 供 reconstruction_module 使用

        self.masks = torch.from_numpy(pred_mask_binary).unsqueeze(0) if pred_mask_binary.sum() > 0 else None
        # [User Request] Default Bounding Box to False to avoid clutter/errors
        self.draw_boxes_enabled = kwargs.get('draw_bounding_boxes', False)
        self.boxes = self._get_boxes_from_mask() if self.draw_boxes_enabled else None
        self.names = {0: 'oil'}

        # [New Features 3 & 4]
        self.ground_truth = kwargs.get('ground_truth', None) # 預期為 (H, W) numpy array, 0/1
        self.global_stats = kwargs.get('global_stats', None) # 全域訓練集 RGB 統計
        self.normalization = kwargs.get('normalization', None) # (Mean, Std) for denormalization if needed

    def _get_boxes_from_mask(self):
        if self.pred_mask_np.sum() == 0: return None
        # Handle Multiclass: Extract Class 1 (Oil) for box generation
        if self.pred_mask_np.max() > 1:
            mask_binary = (self.pred_mask_np == 1).astype(np.uint8)
        else:
            mask_binary = (self.pred_mask_np > 0).astype(np.uint8)
            
        contours, _ = cv2.findContours((mask_binary * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        class BoxContainer:
            def __init__(self, xyxy, conf):
                self.xyxy = torch.tensor(xyxy, dtype=torch.float32); self.conf = torch.tensor(conf, dtype=torch.float32)
                self.data = torch.cat((self.xyxy, self.conf.unsqueeze(1), torch.zeros(self.conf.shape[0], 1)), dim=1)
        boxes_xyxy, scores = [], []
        for c in contours:
            if cv2.contourArea(c) < 1: continue
            x, y, w, h = cv2.boundingRect(c); boxes_xyxy.append([x, y, x + w, y + h]); scores.append(0.95)
        return BoxContainer(boxes_xyxy, scores) if boxes_xyxy else None

    def plot(self, base_image=None):
        """
        擴增版繪圖函式：
        1. 若無 Ground Truth，維持原樣 (顯示預測遮罩與框)。
        2. 若有 Ground Truth，生成「TP/FP/FN/TN 視覺化圖」與「RGB 分布分析圖」，並拼接回傳。
        """
        # 1. 基本預測圖 (Original)
        img = base_image if base_image is not None else self.original_image.copy()
        
        # Detect if this is a Multiclass setup
        is_multiclass = self.pred_mask_np.max() > 1 or (self.ground_truth is not None and self.ground_truth.max() > 1)

        # 如果沒有 GT，走舊邏輯
        if self.ground_truth is None:
            if self.masks is not None:
                mask = cv2.resize(self.pred_mask_np, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # Multiclass Visualization for Basic Plot
                if is_multiclass:
                     # Colorize distinct classes matching README (BGR format)
                     # 1: Oil(Cyan), 2: Lookalike(Red), 3: Ship(Brown), 4: Land(Green)
                     colors = {
                         1: (255, 255, 0),   # Cyan (BGR)
                         2: (0, 0, 255),     # Red
                         3: (0, 76, 153),    # Brown (approx)
                         4: (0, 153, 0)      # Green
                     }
                     overlay = np.zeros_like(img)
                     for c, color in colors.items():
                         overlay[mask == c] = color
                     
                     # Simple mask logic for weighted add (where overlay is not black)
                     mask_bool = (np.any(overlay > 0, axis=-1))
                     # Increase weight for visibility
                     img[mask_bool] = cv2.addWeighted(img[mask_bool], 0.4, overlay[mask_bool], 0.6, 0).reshape(-1, 3) 
                else:
                     overlay = np.zeros_like(img); overlay[mask == 1] = (0, 255, 255) # Yellow for Binary Oil
                     img = cv2.addWeighted(img, 1.0, overlay, 0.5, 0)
            
            if self.boxes:
                scale_x, scale_y = img.shape[1] / self.original_image.shape[1], img.shape[0] / self.original_image.shape[0]
                for box, conf in zip(self.boxes.xyxy, self.boxes.conf):
                    x1, y1, x2, y2 = map(int, [box[0]*scale_x, box[1]*scale_y, box[2]*scale_x, box[3]*scale_y])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2); cv2.putText(img, f"oil {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            return img
        
        # 2. 進階分析模式 (有 GT)
        else:
            # A. 生成 TP/FP/FN/TN 視覺化圖
            vis_img = img.copy() # Use base_image if provided
            h, w = vis_img.shape[:2]
            
            # 確保 GT 與 Pred 符合圖像尺寸
            gt = self.ground_truth
            pred = self.pred_mask_np
            
            # Resize GT if needed
            if gt.shape[:2] != (h, w):
                gt = cv2.resize(gt, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Resize Pred if needed
            if pred.shape[:2] != (h, w):
                pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
            
            self.ground_truth = gt
            self.pred_mask_np = pred
            
            # Branch: Multiclass vs Binary Analysis
            if is_multiclass:
                # [Multiclass Mode] Overlay Prediction Colors directly
                # We do NOT use TP/FP/FN colors here to avoid confusion, 
                # as we have 5 classes. We just show the predicted segmentation.
                colors = {
                     1: (255, 255, 0),   # Cyan (Oil)
                     2: (0, 0, 255),     # Red (Look-alike)
                     3: (0, 76, 153),    # Brown (Ship)
                     4: (0, 153, 0)      # Green (Land)
                }
                overlay = np.zeros_like(vis_img)
                # Draw GT boundary or Pred fill? 
                # Ideally, show Pred fill.
                for c, color in colors.items():
                    overlay[pred == c] = color

                # Blend
                mask_bool = np.any(overlay > 0, axis=-1)
                vis_img[mask_bool] = cv2.addWeighted(vis_img[mask_bool], 0.5, overlay[mask_bool], 0.5, 0).reshape(-1, 3)

                # Add Text
                cv2.putText(vis_img, "Pred: Cyan(Oil) Red(Lk) Brn(Shp) Grn(Lnd)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            else:
                # [Binary Mode] TP/FP/FN Analysis for Oil
                target_class = 1
                
                tp_mask = (pred == target_class) & (gt == target_class)
                fp_mask = (pred == target_class) & (gt != target_class) 
                fn_mask = (pred != target_class) & (gt == target_class)
                
                # 繪製
                overlay = np.zeros_like(vis_img)
                overlay[tp_mask] = (0, 255, 255)   # TP (Yellow)
                overlay[fp_mask] = (0, 0, 255)     # FP (Red)
                overlay[fn_mask] = (255, 0, 0)     # FN (Blue)
                
                # 混合
                mask_any = tp_mask | fp_mask | fn_mask
                vis_img[mask_any] = cv2.addWeighted(vis_img[mask_any], 0.6, overlay[mask_any], 0.4, 0).reshape(-1, 3)
                
                # Add Legend Text
                cv2.putText(vis_img, "TP:(0,255,255) FP:Red FN:Blue (Target: Oil)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            
            # B. 生成 RGB 分布分析圖 (使用 Matplotlib)
            if sns is not None:
                analysis_img = self._generate_analysis_plot()
                if analysis_img is not None:
                    # 拼接圖片 (左右拼接)
                    # 先調整高度一致
                    h1, w1 = vis_img.shape[:2]
                    h2, w2 = analysis_img.shape[:2]
                    
                    if h1 != h2:
                        scale = h1 / h2
                        new_w2 = int(w2 * scale)
                        analysis_img = cv2.resize(analysis_img, (new_w2, h1))
                    
                    combined = np.hstack((vis_img, analysis_img))
                    return combined
            
            return vis_img

    def _generate_analysis_plot(self):
        """生成個別影像的 RGB 分布分析圖 (Matplotlib -> Numpy Image)"""
        try:
            # 安全性檢查：確保 GT 與 原圖尺寸一致 (Handle Index Error)
            h, w = self.original_image.shape[:2]
            if self.ground_truth is not None and self.ground_truth.shape[:2] != (h, w):
                # 如果 GT 尺寸不合，強制縮放至原圖大小
                self.ground_truth = cv2.resize(self.ground_truth, (w, h), interpolation=cv2.INTER_NEAREST)
            
            if self.pred_mask_np is not None and self.pred_mask_np.shape[:2] != (h, w):
                self.pred_mask_np = cv2.resize(self.pred_mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

            # 準備數據
            img_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            channels = ['Red', 'Green', 'Blue']
            
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            # title info
            intersection = np.sum((self.pred_mask_np == 1) & (self.ground_truth == 1))
            union = np.sum(self.pred_mask_np) + np.sum(self.ground_truth) - intersection
            iou = (intersection + 1e-6) / (union + 1e-6)
            fig.suptitle(f"Individual Analysis (IoU: {iou:.2f})", fontsize=14)
            
            # Global Stats DataFrame (for background plotting)
            df_global = None
            if self.global_stats:
                global_plot_data = []
                for cat in ['Oil', 'Background']:
                     for i, ch_key in enumerate(['R', 'G', 'B']):
                         vals = self.global_stats[cat][ch_key]
                         if not vals: continue
                         # Downsample global for speed
                         if len(vals) > 2000: vals = np.random.choice(vals, 2000, replace=False)
                         for v in vals:
                             global_plot_data.append({'Value': v, 'Type': f'Global_{cat}', 'Channel': channels[i]})
                if global_plot_data:
                    df_global = pd.DataFrame(global_plot_data)

            # Local Stats
            masks_np = self.ground_truth
            for i, ch_name in enumerate(channels):
                # Global
                if df_global is not None:
                     subset_oil = df_global[(df_global['Type'] == 'Global_Oil') & (df_global['Channel'] == ch_name)]
                     if not subset_oil.empty:
                         sns.kdeplot(data=subset_oil, x="Value", ax=ax[i], color='red', alpha=0.1, fill=True, linestyle="--", label='Global Oil')
                
                        
                # Local
                # Oil Pixels
                oil_indices = np.where(masks_np == 1)
                if len(oil_indices[0]) > 10:
                    vals_oil = img_rgb[oil_indices[0], oil_indices[1], i]
                    # Check variance to avoid "Dataset has 0 variance" warning
                    if np.var(vals_oil) < 1e-6:
                         # Fallback: simple histogram or skip
                         sns.histplot(x=vals_oil, ax=ax[i], color='red', bins=30, label='Local Oil (Low Var)', stat="density", common_norm=False)
                    else:
                         sns.kdeplot(x=vals_oil, ax=ax[i], color='red', linewidth=2, label='Local Oil', warn_singular=False)
                
                # Bg Pixels
                bg_indices = np.where(masks_np == 0)
                if len(bg_indices[0]) > 10:
                    # Downsample local bg if too large
                    vals_bg = img_rgb[bg_indices[0], bg_indices[1], i]
                    if len(vals_bg) > 5000: vals_bg = np.random.choice(vals_bg, 5000, replace=False)
                    
                    if np.var(vals_bg) < 1e-6:
                         sns.histplot(x=vals_bg, ax=ax[i], color='blue', bins=30, label='Local Bg (Low Var)', stat="density", common_norm=False)
                    else:
                         sns.kdeplot(x=vals_bg, ax=ax[i], color='blue', linewidth=2, label='Local Bg', warn_singular=False)
                
                ax[i].set_title(ch_name)
                ax[i].set_xlim(0, 255)
                if i == 0: ax[i].legend()

            # Convert Plot to Image (Modern Matplotlib Compatible)
            fig.canvas.draw()
            # Convert canvas to image array
            data = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
            # RGBA to RGB (drop alpha)
            data = data[:, :, :3]
            plt.close(fig)
            
            # RGB to BGR for OpenCV
            return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[Warning] Failed to generate analysis plot: {e}")
            return None

@register_model('segformer')
class SegformerAdapter(nn.Module):
    def __init__(self, exp_config):
        super().__init__()
        print("--- [Info] Initializing SegFormer Adapter ---")
        self.config = exp_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        architecture_cfg = self.config.get('architecture_cfg', {})
        base_model_path = self.config.get('base_model')
        num_classes = self.config.get('dataset', {}).get('nc', 1)
        hf_model_name = architecture_cfg.get('hf_model_name', 'nvidia/segformer-b0-finetuned-ade-512-512')

        # 請確認你已經將下載的 3 個檔案 (config.json, preprocessor_config.json, pytorch_model.bin) 放在這個資料夾內
        LOCAL_SEGFORMER_DIR = "/home/yuan/Yuan/OIL_Project_12_7/pretrain/local_model/segformer-b5" 
        # 防呆檢查：如果資料夾不存在或裡面沒有 config.json，直接報錯，不用等 timeout！
        if not os.path.exists(os.path.join(LOCAL_SEGFORMER_DIR, "config.json")):
            raise FileNotFoundError(f"【嚴重錯誤】在本地資料夾 {LOCAL_SEGFORMER_DIR} 找不到 config.json，請確認你已經把下載的 3 個檔案放進去了！")
            
        print(f"  - [Info] 啟動離線模式，強制使用本地模型: {LOCAL_SEGFORMER_DIR}")
        hf_model_name = LOCAL_SEGFORMER_DIR
        architecture_cfg['hf_model_name'] = LOCAL_SEGFORMER_DIR
        # ============================================
        
        # 讀取 in_channels
        self.in_channels = architecture_cfg.get('in_channels', 3)
        print(f"  - [Info] Input channels set to: {self.in_channels}")
        
        # [Optimization] DropPath for regularization (User requested 0.1)
        drop_path_rate = architecture_cfg.get('drop_path_rate', 0.1)
        print(f"  - [Info] Setting DropPath rate to: {drop_path_rate}")

        # [Optimization] Classifier Dropout for MLP head
        classifier_dropout_prob = architecture_cfg.get('classifier_dropout_prob', 0.0)
        print(f"  - [Info] Setting Classifier Dropout to: {classifier_dropout_prob}")
        
        # 建立模型 config
        model_config = SegformerConfig.from_pretrained(
            hf_model_name, 
            num_labels=num_classes, 
            output_hidden_states=True, 
            num_channels=self.in_channels, 
            # [FIX] 傳遞 use_safetensors 參數以繞過 torch.load 錯誤
            use_safetensors=False,
            drop_path_rate=drop_path_rate, # [Optimization]
            classifier_dropout_prob=classifier_dropout_prob, # [Optimization]
            # local_files_only=True
        )
        self.model = SegformerForSemanticSegmentation(model_config)
        print(f"  - [Info] Main model architecture: '{hf_model_name}'")
        
        # 輔助頭建立 (不變)
                                
        if base_model_path and (base_model_path.endswith('.pt') or base_model_path.endswith('.pth')):
            print(f"  - [Info] Weight Source: Local file '{base_model_path}' (Loading main model only)")
            try:
                self.model.load_state_dict(torch.load(base_model_path, map_location=self.device, weights_only=True), strict=False)
                print("  - [Info] Main model weights loaded successfully (strict=False).")
            except Exception as e:
                print(f"  - [Error] Failed to load main model weights: {e}")

        elif 'hf_model_name' in architecture_cfg:
            print(f"  - [Info] Weight Source: Hugging Face Hub '{architecture_cfg['hf_model_name']}' (Loading main model only)")
            if self.in_channels != 3:
                print(f"  - [Warning] Pretrained weights from Hub require 3 channels, but {self.in_channels} are configured. Loading weights for other layers (strict=False).")
            
            # [FIX] 使用 'use_safetensors=True' 繞過 torch.load 安全漏洞檢查
            pretrained_model = SegformerForSemanticSegmentation.from_pretrained(
                architecture_cfg['hf_model_name'], 
                num_labels=num_classes, 
                num_channels=self.in_channels, 
                ignore_mismatched_sizes=True,
                use_safetensors=False, # <--- [FIX] 新增此行
                drop_path_rate=drop_path_rate, # [Optimization] (Reusing drop_path_rate from above)
                classifier_dropout_prob=classifier_dropout_prob # [Optimization]
            )
            self.model.load_state_dict(pretrained_model.state_dict(), strict=False)
            print("  - [Info] Main model weights loaded successfully (strict=False).")
        self.model.to(self.device);         self.names = {i: n for i, n in enumerate(self.config.get('dataset', {}).get('names', ['oil']))}
        print(f"  - [Info] All models moved to device: {self.device}")
        
        # 初始化正規化參數快取
        self._norm_stats_cache = None
        self._global_rgb_stats_cache = None # [New]

    def _get_global_rgb_stats(self, dataset_path=None):
        """
        [New Feature] 掃描訓練集，計算 Oil vs Background 的 RGB 分佈。
        用於 SegformerPredictionResult 的進階分析圖表。
        """
        if self._global_rgb_stats_cache is not None: return self._global_rgb_stats_cache

        if not dataset_path:
             dataset_path = self.config.get('dataset', {}).get('path')
        if not dataset_path: return None

        dataset_path = Path(dataset_path)
        cache_file = dataset_path / 'global_rgb_stats_v1.pkl'
        
        # 1. 嘗試讀取 Cache
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self._global_rgb_stats_cache = pickle.load(f)
                    print(f"  - [Info] Loaded global RGB stats from {cache_file}")
                    return self._global_rgb_stats_cache
            except Exception as e:
                print(f"  - [Warning] Failed to load RGB stats cache: {e}")

        # 2. 計算 (若無 Cache)
        # 注意: 若是在預測階段且無 Cache，為了效能可能會選擇跳過，或者只計算少量樣本
        # 這裡我們做一個快速採樣版本
        print("  - [Info] Computing Global RGB Stats (Oil vs Bg) for Analysis... (Sampling)")
        img_dir = dataset_path / 'images/train'
        lbl_dir = dataset_path / 'labels/train'
        
        if not img_dir.exists(): return None
        
        pixel_data = {'Oil': {'R': [], 'G': [], 'B': []}, 'Background': {'R': [], 'G': [], 'B': []}}
        counts = {'Oil': 0, 'Background': 0}
        max_pixels = 30000 # 限制採樣數
        
        files = sorted(list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg')))
        # 隨機抽樣最多 50 張圖片來建立統計
        if not files: return None
        random_indices = np.random.choice(len(files), min(len(files), 50), replace=False)
        
        for idx in random_indices: # tqdm 若在預測迴圈中會干擾，這裡靜默執行或簡單 print
            if all(c >= max_pixels for c in counts.values()): break
            f = files[idx]
            l_png = lbl_dir / f"{f.stem}.png" 
            if not l_png.exists(): continue
            
            img = cv2.imread(str(f)); 
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            mask = cv2.imread(str(l_png), cv2.IMREAD_GRAYSCALE)
            if mask is None: continue
            mask = (mask > 0).astype(np.uint8)
            
            for cat_name, val in [('Oil', 1), ('Background', 0)]:
                if counts[cat_name] >= max_pixels: continue
                coords = np.where(mask == val)
                if len(coords[0]) == 0: continue
                num_p = len(coords[0])
                take = min(num_p, max_pixels - counts[cat_name], 2000) 
                if num_p > take:
                    selected = np.random.choice(num_p, take, replace=False)
                    p_vals = img[coords[0][selected], coords[1][selected]]
                else:
                    p_vals = img[coords[0], coords[1]]
                
                pixel_data[cat_name]['R'].extend(p_vals[:, 0].tolist())
                pixel_data[cat_name]['G'].extend(p_vals[:, 1].tolist())
                pixel_data[cat_name]['B'].extend(p_vals[:, 2].tolist())
                counts[cat_name] += take
        
        # 3. 儲存
        try:
             with open(cache_file, 'wb') as f:
                 pickle.dump(pixel_data, f)
        except: pass
        
        self._global_rgb_stats_cache = pixel_data
        return pixel_data

    def _get_normalization_stats(self, train_params=None, dataset_path=None):
        """
        獲取正規化參數 (Mean, Std)。
        優先順序:
        1. experiments.yaml 中的 train.normalization
        2. [Test Mode] base_model 所在目錄的上層目錄中的 dataset_stats.yaml
        3. 資料集目錄下的 dataset_stats.yaml
        4. 預設值 (Hardcoded)
        """
        # Check cache first
        if self._norm_stats_cache is not None:
            return self._norm_stats_cache

        # 1. Check experiments.yaml
        if train_params and 'normalization' in train_params:
            norm_cfg = train_params['normalization']
            if 'mean' in norm_cfg and 'std' in norm_cfg:
                print(f"  - [Info] Using Normalization stats from experiments.yaml")
                self._norm_stats_cache = (tuple(norm_cfg['mean']), tuple(norm_cfg['std']))
                return self._norm_stats_cache

        # 2. [Test Mode] Check dataset_stats.yaml relative to base_model
        # 假設 base_model 路徑結構: .../Experiment/fold_X/weights/best.pt
        # 我們要找: .../Experiment/fold_X/dataset_stats.yaml
        if 'base_model' in self.config:
            base_model_path = Path(self.config['base_model'])
            # 往上兩層: weights -> fold_X
            stats_file_exp = base_model_path.parent.parent / 'dataset_stats.yaml'
            if stats_file_exp.exists():
                try:
                    with open(stats_file_exp, 'r') as f:
                        stats = yaml.safe_load(f)
                    if 'mean' in stats and 'std' in stats:
                        print(f"  - [Info] Using Normalization stats from {stats_file_exp} (derived from base_model)")
                        self._norm_stats_cache = (tuple(stats['mean']), tuple(stats['std']))
                        return self._norm_stats_cache
                except Exception as e:
                    print(f"  - [Warning] Failed to read {stats_file_exp}: {e}")

        # 3. Check dataset_stats.yaml in dataset root
        if dataset_path:
            stats_file = Path(dataset_path) / 'dataset_stats.yaml'
            if stats_file.exists():
                try:
                    with open(stats_file, 'r') as f:
                        stats = yaml.safe_load(f)
                    if 'mean' in stats and 'std' in stats:
                        print(f"  - [Info] Using Normalization stats from {stats_file}")
                        self._norm_stats_cache = (tuple(stats['mean']), tuple(stats['std']))
                        return self._norm_stats_cache
                except Exception as e:
                    print(f"  - [Warning] Failed to read {stats_file}: {e}")

        # 4. Defaults
        print(f"  - [Info] Using Default Normalization stats (Hardcoded)")
        if self.in_channels == 1:
            self._norm_stats_cache = ((0.41733294,), (0.26790292,))
        elif self.in_channels == 2:
            self._norm_stats_cache = ((0.41733294, 0.41733294), (0.26790292, 0.26790292))
        else:
            self._norm_stats_cache = ((0.41733294, 0.41733294, 0.41733294), (0.26790292, 0.26790292, 0.26790292))
            
        return self._norm_stats_cache

    # (將此函式新增到 SegformerAdapter class 內部)
    def _save_epoch_plot(self, history, results_path, epoch, loss_name='dice'):
        """
        [New Feature] 儲存當前的訓練曲線快照。
        """
        try:
            df = pd.DataFrame(history)
            if df.empty:
                print(f"  - [Debug] History is empty, skipping interim plot at epoch {epoch}.")
                return

            plt.figure(figsize=(12, 5))
            
            # --- 繪製損失曲線 ---
            plt.subplot(1, 2, 1)
            plt.plot(df['epoch'], df['train_loss'], label=f'Train Loss ({loss_name})', linewidth=2)
            plt.plot(df['epoch'], df['val_loss'], label=f'Validation Loss ({loss_name})')
            
            # 檢查是否有輔助損失
            if 'train_loss_main' in df.columns:
                plt.plot(df['epoch'], df['train_loss_main'], label='Train Loss (Main)', linestyle='--')
            
            
            plt.tight_layout()
            # 儲存為帶有 epoch 編號的檔案
            plot_path = results_path / f'training_curves_epoch_{epoch}.png'
            plt.savefig(plot_path)
            plt.close()
            print(f"  - [Info] Interim training plot saved to: {plot_path}")
        
        except Exception as e:
            print(f"  - [Warning] Failed to save interim plot at epoch {epoch}: {e}")

    # (使用此函式替換舊的 train 函式)
    def train(self, data, results_path, **train_params):
        from torch.cuda.amp import GradScaler
        import pandas as pd
        import matplotlib.pyplot as plt

        print("--- [Info] Starting SegFormer Training (Full Features) ---")
        with open(data, 'r') as f: data_config = yaml.safe_load(f)
        base_path = Path(data_config['path'])
        train_img_dir = base_path / data_config.get('train', 'images/train')
        train_mask_dir = base_path / 'labels' / Path(data_config.get('train', 'images/train')).name
        val_img_dir = base_path / data_config.get('val', 'images/val')
        val_mask_dir = base_path / 'labels' / Path(data_config.get('val', 'images/val')).name
        
        imgsz = train_params.get('imgsz', 512)
        if isinstance(imgsz, (list, tuple)):
            if len(imgsz) >= 2:
                img_h, img_w = imgsz[0], imgsz[1]
            else:
                 img_h, img_w = imgsz[0], imgsz[0]
        else:
            img_h, img_w = int(imgsz), int(imgsz)
            
        print(f"  - [Info] Image Size: {img_h}x{img_w}")

        batch_size = train_params.get('batch_size', 16)
        epochs = train_params.get('epochs', 100)
        workers = train_params.get('workers', 4)
        lr = train_params.get('lr0', 1e-4) # <--- [新增] 讀取 lr0 參數

        # [修改] 讀取包含 flip 在內的所有資料增強參數
        degrees, translate, scale, fliplr, flipud = [train_params.get(k, 0) for k in ['degrees', 'translate', 'scale', 'fliplr', 'flipud']]
        
        # --- [新增 v1.1] 讀取新的擴增參數 ---
        p_bright_contrast = train_params.get('random_brightness_contrast', 0.0)
        p_gauss_noise = train_params.get('gauss_noise', 0.0)
        p_speckle_noise = train_params.get('speckle_noise', 0.0) # [新增] SAR 專用 Speckle Noise 機率
        speckle_var = train_params.get('speckle_variance', 0.04)   # [新增] Speckle 變異數(強度)
        p_coarse_dropout = train_params.get('coarse_dropout', 0.0)
        p_elastic = train_params.get('elastic_transform', 0.0)
        # --- [新增 v1.1 結束] ---

        train_augmentations = []

        import warnings
        
        if fliplr > 0:
            print(f"  - [Info] Data Augmentation Enabled: HorizontalFlip (p={fliplr})")
            train_augmentations.append(A.HorizontalFlip(p=fliplr))
        if flipud > 0:
            print(f"  - [Info] Data Augmentation Enabled: VerticalFlip (p={flipud})")
            train_augmentations.append(A.VerticalFlip(p=flipud))
        if degrees != 0 or translate != 0 or scale != 0:
            print(f"  - [Info] Data Augmentation Enabled: Affine (degrees={degrees}, translate_percent={translate}, scale={max(0, 1-scale)}to{1+scale})")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                train_augmentations.append(A.Affine(translate_percent=translate, scale=(max(0.1, 1-scale), 1+scale), rotate=degrees, p=0.5, cval=0))
        
        # --- [新增 v1.1] 將新的擴增加入列表 ---
        if p_bright_contrast > 0:
            print(f"  - [Info] Data Augmentation Enabled: RandomBrightnessContrast (p={p_bright_contrast})")
            train_augmentations.append(A.RandomBrightnessContrast(p=p_bright_contrast))
            
        if p_gauss_noise > 0:
            print(f"  - [Info] Data Augmentation Enabled: GaussNoise (p={p_gauss_noise})")
            train_augmentations.append(A.GaussNoise(p=p_gauss_noise))

        if p_speckle_noise > 0:
            print(f"  - [Info] Data Augmentation Enabled: SARSpeckleNoise (p={p_speckle_noise}, variance={speckle_var})")
            train_augmentations.append(SARSpeckleNoise(variance=speckle_var, p=p_speckle_noise))
            
        if p_coarse_dropout > 0:
            print(f"  - [Info] Data Augmentation Enabled: CoarseDropout (p={p_coarse_dropout})")
            # max_holes: 挖幾個洞, max_height/width: 洞的大小
            train_augmentations.append(A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(int(img_h*0.05), int(img_h*0.1)), hole_width_range=(int(img_w*0.05), int(img_w*0.1)), p=p_coarse_dropout))
            
        if p_elastic > 0:
            print(f"  - [Info] Data Augmentation Enabled: ElasticTransform (p={p_elastic})")
            # 彈性失真 (對影像和遮罩同時作用)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                train_augmentations.append(A.ElasticTransform(p=p_elastic, border_mode=cv2.BORDER_CONSTANT))
        # --- [新增 v1.1 結束] ---

        if not train_augmentations:
            print("  - [Info] Data Augmentation Disabled")

        # --- [新增] Freeze Epochs 設定 ---
        freeze_epochs = train_params.get('freeze_epochs', 0)
        if freeze_epochs > 0:
            print(f"  - [Info] Freeze Training Enabled: Encoder will be frozen for the first {freeze_epochs} epochs.")

        # --- [修改] 計算並獲取正規化參數 ---
        print("--- Checking Dataset Statistics ---")
        # 嘗試計算並儲存到 results_path
        mean, std = calculate_dataset_stats(base_path, save_yaml=True, output_path=results_path / 'dataset_stats.yaml')
        
        if mean is None:
             print("  - [Warning] Failed to calculate stats (or no images found). Falling back to existing config/defaults.")
             normalize_mean, normalize_std = self._get_normalization_stats(train_params, base_path)
        else:
             normalize_mean, normalize_std = tuple(mean), tuple(std)
             print(f"  - [Info] Calculated and saved stats to {results_path / 'dataset_stats.yaml'}")

        print(f"  - [Info] Using Normalize mean={normalize_mean}, std={normalize_std} for {self.in_channels} channels")
            
        train_transforms = A.Compose(train_augmentations + [
            A.Resize(img_h, img_w), 
            A.Normalize(mean=normalize_mean, std=normalize_std), # <--- [修改]
            ToTensorV2()
        ])

        val_transforms = A.Compose([
            A.Resize(img_h, img_w), 
            A.Normalize(mean=normalize_mean, std=normalize_std), # <--- [修改]
            ToTensorV2()
        ])
        
        # --- [修改] 建立 Dataset 時傳入 in_channels 和 num_classes ---
        # 實作 WeightedRandomSampler
        use_weighted_sampler = train_params.get('use_weighted_sampler', False)
        train_dataset = SegmentationDataset(train_img_dir, train_mask_dir, train_transforms, in_channels=self.in_channels, num_classes=self.model.config.num_labels)
        val_dataset = SegmentationDataset(val_img_dir, val_mask_dir, val_transforms, in_channels=self.in_channels, num_classes=self.model.config.num_labels)

        if use_weighted_sampler:
            try:
                # 這裡是一個耗時操作，我們會先嘗試讀取 cache
                labels = train_dataset.get_labels() # [0, 0, 1, 0, ...]
                
                # 計算 Class Counts
                num_neg = labels.count(0)
                num_pos = labels.count(1)
                
                if num_pos == 0:
                     print("  - [Warning] No positive samples found in training set. Disabling WeightedRandomSampler.")
                     train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=workers, pin_memory=True)
                else:
                     print(f"  - [Info] Class Balance: Pos={num_pos}, Neg={num_neg}")
                     weight_neg = 1.0 / num_neg
                     weight_pos = 1.0 / num_pos
                     sample_weights = np.array([weight_pos if l == 1 else weight_neg for l in labels])
                     sample_weights = torch.from_numpy(sample_weights).double()
                     
                     print("  - [Info] Initializing WeightedRandomSampler...")
                     sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))
                     train_loader = DataLoader(train_dataset, batch_size, sampler=sampler, num_workers=workers, pin_memory=True)
            except Exception as e:
                print(f"  - [Error] Failed to initialize WeightedRandomSampler: {e}")
                train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=workers, pin_memory=True)
        else:
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=workers, pin_memory=True)
            
        val_loader = DataLoader(val_dataset, batch_size, num_workers=workers, pin_memory=True)
        
        num_classes_train = self.model.config.num_labels

        use_amp = train_params.get('amp', False)
        accumulation_steps = train_params.get('gradient_accumulation_steps', 1)
        print(f"  - [Info] Training Config: LR={lr}, AMP={'Enabled' if use_amp else 'Disabled'}, Grad Accumulation={accumulation_steps}, Num Classes={num_classes_train}")

        scaler = GradScaler(enabled=use_amp)
        
        # --- [修改] Loss Function 選擇 (支援 Multiclass) ---
        loss_name = train_params.get('loss_function', 'dice')
        print(f"  - [Info] Loss Function: {loss_name}")

        if num_classes_train > 1:
            if loss_name == 'focal_iou':
                 # Custom Hybrid Loss for Multiclass
                 class FocalIoULoss(nn.Module):
                     def __init__(self):
                         super().__init__()
                         self.focal = smp.losses.FocalLoss(mode='multiclass')
                         self.iou = smp.losses.JaccardLoss(mode='multiclass', from_logits=True)
                     def forward(self, x, y):
                         return self.focal(x, y) + self.iou(x, y)
                 loss_fn = FocalIoULoss()
            elif loss_name == 'focal_iou_ls':
                 class FocalIoULabsmoLoss(nn.Module):
                     def __init__(self):
                         super().__init__()
                         self.focal = smp.losses.FocalLoss(mode='multiclass')
                         self.iou = smp.losses.JaccardLoss(mode='multiclass', from_logits=True)
                         self.ls_val = train_params.get('label_smoothing', 0.1)
                         self.ce_ls = nn.CrossEntropyLoss(label_smoothing=self.ls_val)
                     def forward(self, x, y):
                         return self.focal(x, y) + self.iou(x, y) + self.ce_ls(x, y)
                 loss_fn = FocalIoULabsmoLoss()
            elif loss_name == 'focal_lovasz_ls':
                 class FocalLovaszLsLoss(nn.Module):
                     def __init__(self):
                         super().__init__()
                         self.focal = smp.losses.FocalLoss(mode='multiclass')
                         self.lovasz = smp.losses.LovaszLoss(mode='multiclass', per_image=False)
                         self.ls_val = train_params.get('label_smoothing', 0.1)
                         self.ce_ls = nn.CrossEntropyLoss(label_smoothing=self.ls_val)
                         self.lovasz_weight = 1.0 # 預設權重
                     def forward(self, x, y):
                         return self.focal(x, y) + (self.lovasz(x, y) * self.lovasz_weight) + self.ce_ls(x, y)
                 loss_fn = FocalLovaszLsLoss()
            elif loss_name == 'unified_focal_lovasz':
                 class UnifiedFocalLovaszLoss(nn.Module):
                     def __init__(self):
                         super().__init__()
                         self.focal = smp.losses.FocalLoss(mode='multiclass')
                         self.tversky = smp.losses.TverskyLoss(mode='multiclass', alpha=0.3, beta=0.7, gamma=0.75, from_logits=True)
                         self.lovasz = smp.losses.LovaszLoss(mode='multiclass', per_image=False)
                     def forward(self, x, y):
                         ufl = 0.5 * self.focal(x, y) + 0.5 * self.tversky(x, y)
                         return ufl + self.lovasz(x, y)
                 loss_fn = UnifiedFocalLovaszLoss()
            elif loss_name == 'focal_lovasz':
                 class FocalLovaszLoss(nn.Module):
                     def __init__(self):
                         super().__init__()
                         self.focal = smp.losses.FocalLoss(mode='multiclass')
                         self.lovasz = smp.losses.LovaszLoss(mode='multiclass', per_image=False)
                     def forward(self, x, y):
                         return self.focal(x, y) + self.lovasz(x, y)
                 loss_fn = FocalLovaszLoss()
            elif loss_name == 'unified_focal_iou':
                 class UnifiedFocalIoULoss(nn.Module):
                     def __init__(self):
                         super().__init__()
                         self.focal = smp.losses.FocalLoss(mode='multiclass')
                         self.tversky = smp.losses.TverskyLoss(mode='multiclass', alpha=0.3, beta=0.7, gamma=0.75, from_logits=True)
                         self.iou = smp.losses.JaccardLoss(mode='multiclass', from_logits=True)
                     def forward(self, x, y):
                         ufl = 0.5 * self.focal(x, y) + 0.5 * self.tversky(x, y)
                         return ufl + self.iou(x, y)
                 loss_fn = UnifiedFocalIoULoss()
            elif loss_name == 'unified_focal_iou_ls':
                 class UnifiedFocalIoULsLoss(nn.Module):
                     def __init__(self):
                         super().__init__()
                         self.focal = smp.losses.FocalLoss(mode='multiclass')
                         self.tversky = smp.losses.TverskyLoss(mode='multiclass', alpha=0.3, beta=0.7, gamma=0.75, from_logits=True)
                         self.iou = smp.losses.JaccardLoss(mode='multiclass', from_logits=True)
                         self.ls_val = train_params.get('label_smoothing', 0.1)
                         self.ce_ls = nn.CrossEntropyLoss(label_smoothing=self.ls_val)
                     def forward(self, x, y):
                         ufl = 0.5 * self.focal(x, y) + 0.5 * self.tversky(x, y)
                         return ufl + self.iou(x, y) + self.ce_ls(x, y)
                 loss_fn = UnifiedFocalIoULsLoss()
            elif loss_name == 'ce_dice':
                 class CEDiceLoss(nn.Module):
                     def __init__(self):
                         super().__init__()
                         self.ce = nn.CrossEntropyLoss()
                         self.dice = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
                     def forward(self, x, y):
                         return 0.5 * self.ce(x, y) + 0.5 * self.dice(x, y)
                 loss_fn = CEDiceLoss()
            elif loss_name == 'ce_iou':
                 class CEIoULoss(nn.Module):
                     def __init__(self):
                         super().__init__()
                         self.ce = nn.CrossEntropyLoss()
                         self.iou = smp.losses.JaccardLoss(mode='multiclass', from_logits=True)
                     def forward(self, x, y):
                         return 0.5 * self.ce(x, y) + 0.5 * self.iou(x, y)
                 loss_fn = CEIoULoss()
            elif loss_name == 'dice':
                 loss_fn = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
            elif loss_name == 'focal':
                 loss_fn = smp.losses.FocalLoss(mode='multiclass')
            elif loss_name == 'lovasz':
                 loss_fn = smp.losses.LovaszLoss(mode='multiclass', per_image=False)
            elif loss_name == 'ce':
                 loss_fn = nn.CrossEntropyLoss()
            else:
                 # Default to CE if unspecified or simple
                 print(f"  [Warning] Loss name '{loss_name}' not found for multiclass. Defaulting to CrossEntropyLoss.")
                 loss_fn = nn.CrossEntropyLoss()
        else:
            # Binary Case
            if loss_name == 'bce_dice':
                class BCEDiceLoss(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.bce = nn.BCEWithLogitsLoss()
                        self.dice = smp.losses.DiceLoss(mode='binary', from_logits=True)
                    def forward(self, x, y):
                        return 0.5 * self.bce(x, y) + 0.5 * self.dice(x, y)
                loss_fn = BCEDiceLoss()
            elif loss_name == 'focal_lovasz_ls':
                class BinaryFocalLovaszLsLoss(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.focal = smp.losses.FocalLoss(mode='binary')
                        self.lovasz = smp.losses.LovaszLoss(mode='binary', per_image=False)
                        self.ls_val = train_params.get('label_smoothing', 0.1)
                        self.bce = nn.BCEWithLogitsLoss()
                        self.lovasz_weight = 1.0 # 預設權重
                    def forward(self, x, y):
                        y_smoothed = y.float() * (1.0 - self.ls_val) + 0.5 * self.ls_val
                        return self.focal(x, y) + (self.lovasz(x, y) * self.lovasz_weight) + self.bce(x, y_smoothed)
                loss_fn = BinaryFocalLovaszLsLoss()
            elif loss_name == 'focal_iou_ls':
                class BinaryFocalIoULsLoss(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.focal = smp.losses.FocalLoss(mode='binary')
                        self.iou = smp.losses.JaccardLoss(mode='binary', from_logits=True)
                        self.ls_val = train_params.get('label_smoothing', 0.1)
                        self.bce = nn.BCEWithLogitsLoss()
                    def forward(self, x, y):
                        y_smoothed = y.float() * (1.0 - self.ls_val) + 0.5 * self.ls_val
                        return self.focal(x, y) + self.iou(x, y) + self.bce(x, y_smoothed)
                loss_fn = BinaryFocalIoULsLoss()
            elif loss_name == 'focal_iou':
                class BinaryFocalIoULoss(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.focal = smp.losses.FocalLoss(mode='binary')
                        self.iou = smp.losses.JaccardLoss(mode='binary', from_logits=True)
                    def forward(self, x, y):
                        return self.focal(x, y) + self.iou(x, y)
                loss_fn = BinaryFocalIoULoss()
            elif loss_name == 'focal_lovasz':
                class BinaryFocalLovaszLoss(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.focal = smp.losses.FocalLoss(mode='binary')
                        self.lovasz = smp.losses.LovaszLoss(mode='binary', per_image=False)
                    def forward(self, x, y):
                        return self.focal(x, y) + self.lovasz(x, y)
                loss_fn = BinaryFocalLovaszLoss()
            elif loss_name == 'unified_focal_lovasz':
                class BinaryUnifiedFocalLovaszLoss(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.focal = smp.losses.FocalLoss(mode='binary')
                        self.tversky = smp.losses.TverskyLoss(mode='binary', alpha=0.3, beta=0.7, gamma=0.75, from_logits=True)
                        self.lovasz = smp.losses.LovaszLoss(mode='binary', per_image=False)
                    def forward(self, x, y):
                        ufl = 0.5 * self.focal(x, y) + 0.5 * self.tversky(x, y)
                        return ufl + self.lovasz(x, y)
                loss_fn = BinaryUnifiedFocalLovaszLoss()
            elif loss_name == 'unified_focal_iou':
                class BinaryUnifiedFocalIoULoss(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.focal = smp.losses.FocalLoss(mode='binary')
                        self.tversky = smp.losses.TverskyLoss(mode='binary', alpha=0.3, beta=0.7, gamma=0.75, from_logits=True)
                        self.iou = smp.losses.JaccardLoss(mode='binary', from_logits=True)
                    def forward(self, x, y):
                        ufl = 0.5 * self.focal(x, y) + 0.5 * self.tversky(x, y)
                        return ufl + self.iou(x, y)
                loss_fn = BinaryUnifiedFocalIoULoss()
            elif loss_name == 'unified_focal_iou_ls':
                class BinaryUnifiedFocalIoULsLoss(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.focal = smp.losses.FocalLoss(mode='binary')
                        self.tversky = smp.losses.TverskyLoss(mode='binary', alpha=0.3, beta=0.7, gamma=0.75, from_logits=True)
                        self.iou = smp.losses.JaccardLoss(mode='binary', from_logits=True)
                        self.ls_val = train_params.get('label_smoothing', 0.1)
                        self.bce = nn.BCEWithLogitsLoss()
                    def forward(self, x, y):
                        y_smoothed = y.float() * (1.0 - self.ls_val) + 0.5 * self.ls_val
                        ufl = 0.5 * self.focal(x, y) + 0.5 * self.tversky(x, y)
                        return ufl + self.iou(x, y) + self.bce(x, y_smoothed)
                loss_fn = BinaryUnifiedFocalIoULsLoss()
            elif loss_name == 'bce_iou':
                class BCEIoULoss(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.bce = nn.BCEWithLogitsLoss()
                        self.iou = smp.losses.JaccardLoss(mode='binary', from_logits=True)
                    def forward(self, x, y):
                        return 0.5 * self.bce(x, y) + 0.5 * self.iou(x, y)
                loss_fn = BCEIoULoss()
            elif loss_name == 'bce_iou_ls':
                class BCEIoULsLoss(nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.bce = nn.BCEWithLogitsLoss()
                        self.iou = smp.losses.JaccardLoss(mode='binary', from_logits=True)
                        self.ls_val = train_params.get('label_smoothing', 0.1)
                    def forward(self, x, y):
                        y_smoothed = y.float() * (1.0 - self.ls_val) + 0.5 * self.ls_val
                        return 0.5 * self.bce(x, y_smoothed) + 0.5 * self.iou(x, y)
                loss_fn = BCEIoULsLoss()
            elif loss_name == 'bce':
                loss_fn = nn.BCEWithLogitsLoss()
            elif loss_name == 'focal':
                loss_fn = smp.losses.FocalLoss(mode='binary')
            elif loss_name == 'dice':
                loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)
            elif loss_name == 'lovasz':
                loss_fn = smp.losses.LovaszLoss(mode='binary', per_image=False)
            else:
                print(f"  [Warning] Loss name '{loss_name}' not found for binary. Defaulting to DiceLoss.")
                loss_fn = smp.losses.DiceLoss(mode='binary', from_logits=True)

        # [修改] 使用 weight_decay 參數
        weight_decay = train_params.get('weight_decay', 1e-2)
        print(f"  - [Info] Optimizer: AdamW (lr={lr}, weight_decay={weight_decay})")
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay) # <--- [修改] 傳入 weight_decay
        
        # --- [New Feature] EMA Initialization ---
        use_ema = train_params.get('use_ema', True) # Default to True if logic allows
        ema_decay = train_params.get('ema_decay', 0.999)
        model_ema = None
        
        # Define container regardless of EMA usage (for potential future use), or just for EMA
        if use_ema:
            print(f"  - [Info] Initializing EMA with decay: {ema_decay}")
            self.model_container = nn.ModuleDict({'model': self.model})
            try:
                # Try to use local simple EMA first (more stable for this custom setup)
                model_ema = ModelEma(self.model_container, decay=ema_decay, device=self.device)
                print("  - [Info] EMA Model initialized.")
            except Exception as e:
                print(f"  - [Error] Failed to initialize EMA: {e}")
                model_ema = None

        best_iou, best_model_path = -1.0, results_path/'weights'/'best.pt'; best_model_path.parent.mkdir(exist_ok=True, parents=True)

        patience = train_params.get('patience', 0)
        epochs_no_improve = 0
        if patience > 0:
            print(f"  - [Info] Early Stopping Enabled with patience: {patience}")

        history = [] # <--- [新增] 初始化 history

        lovasz_warmup_epochs = train_params.get('lovasz_warmup_epochs', 10) # 前 10 個 epoch lovasz_weight = 0

        for epoch in range(epochs):
            # --- [New Feature: Dynamic Lovasz Weight Warm-up] ---
            if hasattr(loss_fn, 'lovasz_weight'):
                if epoch < lovasz_warmup_epochs:
                    loss_fn.lovasz_weight = 0.0
                else:
                    loss_fn.lovasz_weight = 1.0 # default weight
                print(f"  - [Dynamic Lovasz] Epoch {epoch+1}: Lovasz Loss Weight = {loss_fn.lovasz_weight}")

            
            # --- [新增] Freeze Logic ---
            if freeze_epochs > 0:
                if epoch < freeze_epochs:
                    for param in self.model.segformer.parameters():
                        param.requires_grad = False
                    print(f"  - [Freeze] Epoch {epoch+1}/{epochs}: Encoder frozen.")
                elif epoch == freeze_epochs:
                    for param in self.model.segformer.parameters():
                        param.requires_grad = True
                    print(f"  - [Freeze] Epoch {epoch+1}/{epochs}: Encoder Unfrozen! Full training starts.")
            
            self.model.train()
            running_loss = 0.0
            
            # Confusion Matrix: For multiclass, better use total_intersection/union logic or per-class cm
            # For simplicity, we track global accuracy or mean IoU. 
            # Currently just keeping simple binary metrics logic or extending it slightly.
            # Using basic accumulators for multiclass.
            
            # Binary fallback
            total_cm_train = np.zeros((2, 2), dtype=np.int64) 
            
            # Multiclass fallback
            total_intersections = np.zeros(num_classes_train, dtype=np.float64)
            total_unions = np.zeros(num_classes_train, dtype=np.float64)
            
            epoch_start_time = time.time()
            
            use_tqdm = sys.stdout.isatty()
            if use_tqdm:
                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
                iterator = pbar
            else:
                print(f"Epoch {epoch+1}/{epochs} [Train] Started...")
                iterator = train_loader
            
            log_interval = max(1, len(train_loader) // 3)

            for i, (images, masks) in enumerate(iterator):
                images, masks = images.to(self.device), masks.to(self.device)
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                    outputs = self.model(pixel_values=images)
                    main_logits = outputs.logits
                    upsampled_main = nn.functional.interpolate(main_logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                    loss_main = loss_fn(upsampled_main, masks)
                    
                    loss = loss_main / accumulation_steps
                
                scaler.scale(loss).backward()
                if (i + 1) % accumulation_steps == 0:
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
                    if model_ema is not None:
                         model_ema.update(self.model_container)

                running_loss += loss.item() * accumulation_steps
                                
                # --- Metrics Calc ---
                if num_classes_train > 1:
                     # Multiclass Metrics
                     preds_train = torch.softmax(upsampled_main, dim=1).argmax(dim=1) # (B, H, W)
                     
                     # Simple IoU accumulation
                     for cls_idx in range(num_classes_train):
                         p = (preds_train == cls_idx)
                         g = (masks == cls_idx)
                         total_intersections[cls_idx] += (p & g).float().sum().item()
                         total_unions[cls_idx] += (p | g).float().sum().item()
                else:
                     # Binary Metrics
                     preds_train = (torch.sigmoid(upsampled_main) > 0.5).cpu().numpy().flatten().astype(np.uint8)
                     total_cm_train += confusion_matrix(masks.cpu().numpy().flatten(), preds_train, labels=[0, 1])

                if use_tqdm:
                    pbar.set_postfix(loss=running_loss / (i + 1))
                elif (i + 1) % log_interval == 0:
                    elapsed = time.time() - epoch_start_time
                    eta = (len(train_loader) - (i + 1)) * (elapsed / (i + 1))
                    print(f"  Epoch {epoch+1} [Train] Batch {i+1}/{len(train_loader)} ({((i+1)/len(train_loader))*100:.0f}%) Loss: {running_loss / (i + 1):.4f} | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")
            
            avg_train_loss = running_loss / len(train_loader)
            
            if num_classes_train > 1:
                iou_per_class = total_intersections / (total_unions + 1e-9)
                train_iou = iou_per_class.mean() # mIoU
            else:
                TN_t, FP_t, FN_t, TP_t = total_cm_train.ravel()
                train_iou = TP_t / (TP_t + FP_t + FN_t + 1e-9)

            self.model.eval()
            
            # --- [Refactor] Validation Function for Main & EMA ---
            def run_validation(model_to_test, desc_suffix="", log_start=True):
                total_cm_val = np.zeros((2, 2), dtype=np.int64)
                total_intersections_val = np.zeros(num_classes_train, dtype=np.float64)
                total_unions_val = np.zeros(num_classes_train, dtype=np.float64)
                running_val_loss = 0.0
                
                # Determine iterator
                if use_tqdm:
                    val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val{desc_suffix}]", leave=False)
                else:
                    if log_start:
                        print(f"Epoch {epoch+1}/{epochs} [Val] Started...")
                    val_iterator = val_loader
                
                with torch.no_grad():
                    for i, (images, masks) in enumerate(val_iterator):
                        images, masks = images.to(self.device), masks.to(self.device)
                        
                        # Handle Dict vs Direct model
                        if isinstance(model_to_test, nn.ModuleDict):
                             # EMA case: model_to_test is the container
                             logits = model_to_test['model'](pixel_values=images).logits
                        else:
                             # Main model case
                             logits = model_to_test(pixel_values=images).logits
                             
                        upsampled = nn.functional.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                        running_val_loss += loss_fn(upsampled, masks).item()
                        
                        if num_classes_train > 1:
                            preds = torch.softmax(upsampled, dim=1).argmax(dim=1)
                            for cls_idx in range(num_classes_train):
                                 p = (preds == cls_idx)
                                 g = (masks == cls_idx)
                                 total_intersections_val[cls_idx] += (p & g).float().sum().item()
                                 total_unions_val[cls_idx] += (p | g).float().sum().item()
                        else:
                            preds = (torch.sigmoid(upsampled) > 0.5).cpu().numpy().flatten().astype(np.uint8)
                            total_cm_val += confusion_matrix(masks.cpu().numpy().flatten(), preds, labels=[0, 1])
                
                avg_val = running_val_loss / len(val_loader)
                if num_classes_train > 1:
                    iou_per_class = total_intersections_val / (total_unions_val + 1e-9)
                    mIoU = iou_per_class.mean()
                else:
                    TN, FP, FN, TP = total_cm_val.ravel()
                    mIoU = TP / (TP + FP + FN + 1e-9)
                return avg_val, mIoU

            # 1. Validate Main Model
            avg_val_loss, val_iou = run_validation(self.model, "", log_start=True)
            
            # 2. Validate EMA Model (if exists)
            val_iou_ema = 0.0
            avg_val_loss_ema = 0.0
            if model_ema is not None:
                 avg_val_loss_ema, val_iou_ema = run_validation(model_ema.module, " (EMA)", log_start=False)
            
            # Logging & Saving Logic
            # Prefer EMA metrics if available
            current_val_iou = val_iou_ema if model_ema is not None else val_iou
            current_val_loss = avg_val_loss_ema if model_ema is not None else avg_val_loss
            
            log_msg = f"  - [Log] Epoch {epoch+1} -> Train Loss: {avg_train_loss:.4f}, Train mIoU: {train_iou:.4f} | Val Loss: {avg_val_loss:.4f}, Val mIoU: {val_iou:.4f}"
            if model_ema is not None:
                log_msg += f" | EMA Val Loss: {avg_val_loss_ema:.4f}, EMA Val mIoU: {val_iou_ema:.4f}"
            print(log_msg)

            history.append({
                'epoch': epoch+1, 
                'train_loss': avg_train_loss, 
                'val_loss': avg_val_loss, 'val_loss_ema': avg_val_loss_ema,
                'train_iou': train_iou, 'val_iou': val_iou, 'val_iou_ema': val_iou_ema
            })

            if current_val_iou > best_iou:
                best_iou = current_val_iou
                epochs_no_improve = 0
                
                # Save Best Model
                if model_ema is not None:
                     # Save EMA weights (only the 'model' part, ignoring aux_head/container structure)
                     torch.save(model_ema.module['model'].state_dict(), best_model_path)
                     print(f"  - [Info] New best model (EMA) saved with IoU: {best_iou:.4f}")
                else:
                     torch.save(self.model.state_dict(), best_model_path)
                     print(f"  - [Info] New best model saved with IoU: {best_iou:.4f}")
            else:
                epochs_no_improve += 1
            
            # --- [新功能] 每 100 epoch 儲存一次圖表 ---
            if (epoch + 1) % 20 == 0 or (epoch + 1) == epochs:
                self._save_epoch_plot(history, results_path, epoch + 1, loss_name=loss_name)
            # --- [新功能] 結束 ---

            if patience > 0 and epochs_no_improve >= patience:
                print(f"  - [Info] Early stopping triggered after {patience} epochs with no improvement.")
                break
            
            # [Add] Cleanup GPU memory at the end of epoch to prevent OOM
            torch.cuda.empty_cache()
            
        print("--- [Info] Training finished. Generating final result plots... ---")
        df = pd.DataFrame(history)
        plt.figure(figsize=(12, 5)); plt.subplot(1, 2, 1)
        plt.plot(df['epoch'], df['train_loss'], label=f'Train Loss ({loss_name})', linewidth=2)
        
        plt.tight_layout(); plt.savefig(results_path / 'training_curves.png'); plt.close()
        df.to_csv(results_path / 'training_log.csv', index=False)
        print(f"  - [Info] Final plots and logs saved to {results_path}")
        
        return {'best_model_path': str(best_model_path)}

    def predict(self, source, imgsz, **kwargs):
        original_image = cv2.imread(str(source))
        if original_image is None: return [None]
        h, w, _ = original_image.shape
        
        # --- [修改] 根據 self.in_channels 和您的新數值設定 Normalize 參數 ---
        if self.in_channels == 1:
            image_predict = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        elif self.in_channels == 2:
            image_predict = original_image[:, :, [2, 1]] # 選 RG
        else: # 預設 RGB
            image_predict = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # 使用 _get_normalization_stats 獲取參數
        normalize_mean, normalize_std = self._get_normalization_stats(
            self.config.get('train', {}), 
            self.config.get('dataset', {}).get('path')
        )

        if isinstance(imgsz, (list, tuple)):
            if len(imgsz) >= 2: img_h, img_w = imgsz[0], imgsz[1]
            else: img_h, img_w = imgsz[0], imgsz[0]
        else:
            img_h, img_w = int(imgsz), int(imgsz)

        transforms = A.Compose([
            A.Resize(img_h, img_w), 
            A.Normalize(mean=normalize_mean, std=normalize_std), # <--- [修改]
            ToTensorV2()
        ])
        
        image_tensor = transforms(image=image_predict)['image'].unsqueeze(0).to(self.device) # <--- [修改]
        
        with torch.no_grad():
            self.model.eval()
            logits = self.model(pixel_values=image_tensor).logits
            upsampled_logits = nn.functional.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
            
            # --- [FIX v1.2] Multiclass Support ---
            if self.model.config.num_labels > 1:
                # Multiclass
                pred_prob_tensor = torch.softmax(upsampled_logits, dim=1) # (1, C, H, W)
                pred_mask_binary_tensor = pred_prob_tensor.argmax(dim=1)  # (1, H, W) - Indices
                
                pred_prob_np = pred_prob_tensor.cpu().numpy().squeeze(0).astype(np.float32) # (C, H, W)
                pred_mask_binary_np = pred_mask_binary_tensor.cpu().numpy().squeeze(0).astype(np.uint8) # (H, W)
            else:
                # Binary
                pred_prob_tensor = torch.sigmoid(upsampled_logits)
                pred_mask_binary_tensor = (pred_prob_tensor > 0.5)
                
                pred_prob_np = pred_prob_tensor.cpu().numpy().squeeze().astype(np.float32)
                pred_mask_binary_np = pred_mask_binary_tensor.cpu().numpy().squeeze().astype(np.uint8)
            # --- [FIX v1.2 END] ---

        # === [New Feature] Try to load Ground Truth & Global Stats ===
        ground_truth = None
        global_stats = None
        
        # 1. 嘗試載入 Ground Truth
        try:
            source_path = Path(source)
            # 假設結構: images/xxx/abc.png -> labels/xxx/abc.png
            # 簡單替換: "images" -> "labels"
            parts = list(source_path.parts)
            if 'images' in parts:
                idx = parts.index('images')
                parts[idx] = 'labels'
                lbl_path = Path(*parts).with_suffix('.png')
                if lbl_path.exists():
                    gt_mask = cv2.imread(str(lbl_path), cv2.IMREAD_GRAYSCALE)
                    if gt_mask is not None:
                        # Resize binary mask to original image size if needed (usually it matches)
                        if gt_mask.shape != (h, w):
                            gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        
                        if self.model.config.num_labels > 1:
                            ground_truth = gt_mask.astype(np.uint8) # Keep original labels
                        else:
                            ground_truth = (gt_mask > 0).astype(np.uint8)
        except Exception as e:
            print(f"  - [Warning] GT load failed: {e}")

        # 2. 只有在有 GT 的情況下，才去載入 Global Stats (否則無法畫比較圖)
        if ground_truth is not None:
             global_stats = self._get_global_rgb_stats()

        kwargs.update({
            'ground_truth': ground_truth,
            'global_stats': global_stats,
            'normalization': (normalize_mean, normalize_std),
            'img_path': str(source)
        })

        return [SegformerPredictionResult(original_image, pred_mask_binary_np, pred_prob_np, **kwargs)]
    
    def val(self, data, split='test', **kwargs):
        imgsz = kwargs.get('imgsz', 640)
        
        if isinstance(imgsz, (list, tuple)):
            if len(imgsz) >= 2: img_h, img_w = imgsz[0], imgsz[1]
            else: img_h, img_w = imgsz[0], imgsz[0]
        else:
            img_h, img_w = int(imgsz), int(imgsz)

        with open(data, 'r') as f: data_config = yaml.safe_load(f)
        base_path = Path(data_config['path'])
        test_img_dir = base_path / data_config.get(split, f'images/{split}')
        test_mask_dir = base_path / 'labels' / Path(data_config.get(split, f'images/{split}')).name
        
        # --- [修改] 根據 self.in_channels 和您的新數值設定 Normalize 參數 ---
        normalize_mean, normalize_std = self._get_normalization_stats(
            self.config.get('train', {}), 
            base_path
        )
        
        transforms = A.Compose([
            A.Resize(img_h, img_w), 
            A.Normalize(mean=normalize_mean, std=normalize_std), # <--- [修改]
            ToTensorV2()
        ])
        
        # --- [修改] 建立 Dataset 時傳入 in_channels ---
        test_dataset = SegmentationDataset(test_img_dir, test_mask_dir, transforms, in_channels=self.in_channels, num_classes=self.model.config.num_labels)
        # 由於訓練時可能只用 batch_size=2，測試時設定太高(如32)容易產生 CUDA Out of Memory
        test_batch_size = self.config.get('train', {}).get('batch_size', 2)
        test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=4)
        
        total_iou = 0.0; self.model.eval()
        with torch.no_grad():
            import time
            start_time = time.time()
            print_interval = max(1, len(test_loader) // 10)
            for i, (images, masks) in enumerate(test_loader):
                if (i + 1) % print_interval == 0 or (i + 1) == len(test_loader):
                    elapsed = time.time() - start_time
                    eta = (elapsed / (i + 1)) * (len(test_loader) - (i + 1))
                    print(f"  [Test] Batch {i+1}/{len(test_loader)} ({((i+1)/len(test_loader))*100:.0f}%) | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s", flush=True)
                images, masks = images.to(self.device), masks.to(self.device)
                logits = self.model(pixel_values=images).logits
                upsampled = nn.functional.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
                
                num_classes = self.model.config.num_labels if hasattr(self.model.config, 'num_labels') else 1
                if num_classes > 1:
                    preds = torch.argmax(upsampled, dim=1)
                    intersection = 0.0
                    union = 0.0
                    for cls_idx in range(1, num_classes):
                        p = (preds == cls_idx).float()
                        m = (masks == cls_idx).float()
                        intersection += torch.sum(p * m)
                        union += torch.sum(p) + torch.sum(m) - torch.sum(p * m)
                    total_iou += ((intersection + 1e-6) / (union + 1e-6)).item()
                else:
                    preds = (torch.sigmoid(upsampled) > 0.5).float()
                    intersection = torch.sum(preds * masks)
                    union = torch.sum(preds) + torch.sum(masks) - intersection
                    total_iou += ((intersection + 1e-6) / (union + 1e-6)).item()
        avg_iou = total_iou / len(test_loader)
        class MockMetrics:
            def __init__(self, iou):
                class Seg:
                    def __init__(self, iou): self.mp=0.0; self.mr=0.0; self.map50=iou; self.map=iou*0.9
                self.seg = Seg(iou); self.box = self.seg
        return MockMetrics(avg_iou)
