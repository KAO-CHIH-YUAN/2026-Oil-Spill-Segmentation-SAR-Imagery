"""
[File] osdmamba_official_adapter.py
[Purpose] Adapter wrapping the OFFICIAL OSDMamba (SwinUMambaD) architecture.
[Architecture] VSSMEncoder (VMamba-Tiny backbone) + UNetResDecoder with deep supervision.
[Source] https://github.com/Multimodal-Intelligence-Lab-MIL/OSDMamba

Key differences vs. the self-implemented osdmamba_adapter:
  - Uses the official SS2D / VSSMEncoder / UNetResDecoder code verbatim.
  - forward() returns {'main', 'aux_4', 'aux_8', 'aux_16'} dict for deep supervision.
  - Supports loading VMamba-Tiny pretrained encoder weights (vmamba_pretrain_path).
  - Requires yuan_mamba_env (mamba_ssm must be installed); set workers: 0.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
import sys
import time
import copy
import segmentation_models_pytorch as smp

from ..models.osdmamba_official.SwinUMambaD import OfficialOSDMamba, load_pretrained_ckpt
from ..training_module import register_model
from ..calculate_stats import calculate_dataset_stats


# ---------------------------------------------------------------------------
# SAR speckle noise augmentation
# ---------------------------------------------------------------------------

class SARSpeckleNoise(A.ImageOnlyTransform):
    def __init__(self, variance=0.04, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.variance = variance

    def apply(self, img, **params):
        img_f = img.astype(np.float32)
        noise = np.random.gamma(1.0 / self.variance, self.variance,
                                img.shape).astype(np.float32)
        return np.clip(img_f * noise, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("variance",)


# ---------------------------------------------------------------------------
# EMA (identical to segformer_adapter_aux.py standard implementation)
# ---------------------------------------------------------------------------

class ModelEma:
    def __init__(self, model, decay=0.9999, device=None):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay  = decay
        self.device = device
        if device:
            self.module.to(device=device)

    def update(self, model):
        with torch.no_grad():
            for ema_p, m_p in zip(self.module.parameters(),
                                   model.parameters()):
                ema_p.data.mul_(self.decay).add_(m_p.data,
                                                  alpha=1.0 - self.decay)
            for ema_b, m_b in zip(self.module.buffers(), model.buffers()):
                if ema_b.dtype.is_floating_point:
                    ema_b.data.mul_(self.decay).add_(m_b.data,
                                                      alpha=1.0 - self.decay)
                else:
                    ema_b.copy_(m_b)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None,
                 in_channels=1, num_classes=1):
        self.image_dir  = Path(image_dir)
        self.mask_dir   = Path(mask_dir)
        self.image_files = sorted(
            list(self.image_dir.glob('*.png')) +
            list(self.image_dir.glob('*.jpg')))
        self.transforms  = transforms
        self.in_channels = in_channels
        self.num_classes = num_classes

    def get_labels(self):
        labels = []
        print("[Info] Scanning dataset for WeightedRandomSampler...")
        for img_path in tqdm(self.image_files):
            txt = self.mask_dir / (img_path.stem + '.txt')
            if txt.exists():
                labels.append(1 if txt.stat().st_size > 0 else 0)
                continue
            mask_path = self._find_mask(img_path)
            mask = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8),
                                cv2.IMREAD_GRAYSCALE)
            labels.append(1 if (mask is not None and np.any(mask > 0)) else 0)
        return labels

    def _find_mask(self, img_path):
        for ext in ['.png', '.jpg', img_path.suffix]:
            p = self.mask_dir / (img_path.stem + ext)
            if p.exists():
                return p
        return self.mask_dir / img_path.name

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path  = self.image_files[idx]
        mask_path = self._find_mask(img_path)

        image_bgr = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8),
                                  cv2.IMREAD_COLOR)
        if image_bgr is None:
            return (torch.zeros(self.in_channels, 512, 512),
                    torch.zeros(1, 512, 512))

        if self.in_channels == 1:
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        elif self.in_channels == 2:
            image = image_bgr[:, :, [2, 1]]
        else:
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mask = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8),
                             cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)

        if self.num_classes == 1:
            mask = (mask > 0).astype(np.float32)
        # multiclass: keep original integer class ids

        if self.transforms:
            out   = self.transforms(image=image, mask=mask)
            image = out['image']
            mask  = out['mask']

        # to tensor
        if isinstance(image, torch.Tensor):
            image_tensor = image.float()
        else:
            image = (image[np.newaxis] if image.ndim == 2
                     else image.transpose(2, 0, 1))
            image_tensor = torch.from_numpy(image).float()

        if isinstance(mask, torch.Tensor):
            mask_tensor = (mask.long() if self.num_classes > 1
                           else mask.float().unsqueeze(0)
                           if mask.ndim == 2 else mask.float())
        else:
            if self.num_classes > 1:
                mask_tensor = torch.from_numpy(mask).long()
            else:
                mask = (mask[np.newaxis] if mask.ndim == 2
                        else mask.transpose(2, 0, 1))
                mask_tensor = torch.from_numpy(mask).float()

        return image_tensor, mask_tensor


# ---------------------------------------------------------------------------
# Result / metrics objects
# ---------------------------------------------------------------------------

class OSDMambaOfficialResult:
    """Minimal result object satisfying evaluation_module / reconstruction_module."""
    def __init__(self, original_image, pred_mask_binary, pred_mask_prob):
        self.original_image      = original_image
        self.pred_mask_np        = pred_mask_binary
        self.pred_mask_binary_np = pred_mask_binary
        self.pred_mask_prob_np   = pred_mask_prob
        self.masks = (torch.from_numpy(pred_mask_binary).unsqueeze(0)
                      if pred_mask_binary.sum() > 0 else None)
        self.boxes = None
        self.names = {0: 'oil'}

    def plot(self, base_image=None):
        img = (base_image if base_image is not None
               else self.original_image.copy())
        if self.pred_mask_np.sum() == 0:
            return img
        mask = cv2.resize(self.pred_mask_np,
                          (img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST)
        if mask.max() > 1:
            colors = {1: (0, 255, 255), 2: (0, 0, 255),
                      3: (0, 76, 153),  4: (0, 153, 0)}
            overlay = np.zeros_like(img)
            for c, col in colors.items():
                overlay[mask == c] = col
            mask_bool = np.any(overlay > 0, axis=-1)
            img[mask_bool] = cv2.addWeighted(
                img[mask_bool], 0.4,
                overlay[mask_bool], 0.6, 0).reshape(-1, 3)
        else:
            overlay = np.zeros_like(img)
            overlay[mask == 1] = (0, 255, 255)
            img = cv2.addWeighted(img, 1.0, overlay, 0.5, 0)
        return img


class MockMetrics:
    def __init__(self, iou):
        class Seg:
            def __init__(self, v):
                self.map50 = v; self.map = v * 0.9
                self.mp = 0.0;  self.mr = 0.0
        self.seg = Seg(iou); self.box = self.seg


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

@register_model('osdmamba_official')
class OSDMambaOfficialAdapter(nn.Module):
    def __init__(self, exp_config):
        super().__init__()
        print("--- [Info] Initializing OSDMamba Official Adapter ---")
        self.config = exp_config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        arch_cfg         = self.config.get('architecture_cfg', {})
        self.in_channels = arch_cfg.get('in_channels', 1)
        self.num_classes = self.config.get('dataset', {}).get('nc', 1)
        depths           = arch_cfg.get('depths', [2, 2, 9, 2])
        dims             = arch_cfg.get('dims', 96)
        drop_path_rate   = arch_cfg.get('drop_path_rate', 0.2)
        d_state          = arch_cfg.get('d_state', 16)
        deep_supervision = arch_cfg.get('deep_supervision', True)

        print(f"  - in_channels={self.in_channels}, num_classes={self.num_classes}")
        print(f"  - depths={depths}, dims={dims}, deep_supervision={deep_supervision}")

        self.model = OfficialOSDMamba(
            in_chans=self.in_channels,
            num_classes=self.num_classes,
            depths=depths, dims=dims,
            drop_path_rate=drop_path_rate,
            d_state=d_state,
            deep_supervision=deep_supervision,
        ).to(self.device)

        self._load_weights(arch_cfg)
        self._norm_stats_cache = None
        self.val_ds = None
        self._setup_val_transform()   # expose val_ds for reconstruction_module

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _load_weights(self, arch_cfg):
        base_model_path   = self.config.get('base_model', '')
        vmamba_pretrain   = arch_cfg.get('vmamba_pretrain_path', '')

        if base_model_path and Path(base_model_path).exists():
            print(f"  - Loading checkpoint: {base_model_path}")
            ckpt = torch.load(base_model_path, map_location=self.device,
                              weights_only=True)
            state_dict = ckpt.get('model', ckpt)
            state_dict = {k.replace('module.', ''): v
                          for k, v in state_dict.items()}
            model_sd = self.model.state_dict()
            matched  = {k: v for k, v in state_dict.items()
                        if k in model_sd and v.shape == model_sd[k].shape}
            model_sd.update(matched)
            self.model.load_state_dict(model_sd)
            print(f"  - Loaded {len(matched)}/{len(model_sd)} keys from checkpoint.")

        elif vmamba_pretrain and Path(vmamba_pretrain).exists():
            load_pretrained_ckpt(self.model, self.in_channels, vmamba_pretrain)

        else:
            print("  - No pretrained weights. Using random initialisation.")

    # ------------------------------------------------------------------
    # val_ds transform – exposed so reconstruction_module can normalise correctly
    # ------------------------------------------------------------------

    def _setup_val_transform(self, mean=None, std=None):
        """Build self.val_ds with the correct normalisation transform.

        Called at init (uses dataset_stats.yaml found via base_model path, or
        hardcoded defaults) and again after training (with actual computed stats).
        """
        imgsz = self.config.get('imgsz', [512, 512])
        h = imgsz[0] if isinstance(imgsz, (list, tuple)) else imgsz
        w = imgsz[1] if isinstance(imgsz, (list, tuple)) else imgsz

        if mean is None or std is None:
            mean, std = self._get_normalization_stats(
                self.config.get('train', {}),
                self.config.get('dataset', {}).get('path'),
                verbose=False,
            )

        val_tf = A.Compose([
            A.Resize(h, w),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

        class _ValDs:
            def __init__(self, tf):
                self.transform = tf

        self.val_ds = _ValDs(val_tf)

    # ------------------------------------------------------------------
    # Normalisation stats (4-layer fallback, same as other adapters)
    # ------------------------------------------------------------------

    def _get_normalization_stats(self, train_params=None,
                                  dataset_path=None, verbose=True):
        if self._norm_stats_cache is not None:
            return self._norm_stats_cache

        # 1. experiments.yaml explicit
        if train_params and 'normalization' in train_params:
            cfg = train_params['normalization']
            if 'mean' in cfg and 'std' in cfg:
                if verbose:
                    print("  - [Norm] Using normalization from experiments.yaml")
                self._norm_stats_cache = tuple(cfg['mean']), tuple(cfg['std'])
                return self._norm_stats_cache

        # 2. dataset_stats.yaml next to base_model (test mode)
        base_model_path = self.config.get('base_model', '')
        if base_model_path:
            stats_file = Path(base_model_path).parent.parent / 'dataset_stats.yaml'
            if stats_file.exists():
                try:
                    with open(stats_file) as f:
                        s = yaml.safe_load(f)
                    if 'mean' in s and 'std' in s:
                        if verbose:
                            print(f"  - [Norm] Using {stats_file}")
                        self._norm_stats_cache = tuple(s['mean']), tuple(s['std'])
                        return self._norm_stats_cache
                except Exception:
                    pass

        # 3. dataset_stats.yaml in dataset root
        if dataset_path:
            stats_file = Path(dataset_path) / 'dataset_stats.yaml'
            if stats_file.exists():
                try:
                    with open(stats_file) as f:
                        s = yaml.safe_load(f)
                    if 'mean' in s and 'std' in s:
                        if verbose:
                            print(f"  - [Norm] Using {stats_file}")
                        self._norm_stats_cache = tuple(s['mean']), tuple(s['std'])
                        return self._norm_stats_cache
                except Exception:
                    pass

        # 4. hardcoded default
        if verbose:
            print("  - [Norm] Using hardcoded defaults")
        defaults = {1: ((0.417,), (0.267,)),
                    2: ((0.417, 0.417), (0.267, 0.267)),
                    3: ((0.417, 0.417, 0.417), (0.267, 0.267, 0.267))}
        self._norm_stats_cache = defaults.get(self.in_channels,
                                               defaults[3])
        return self._norm_stats_cache

    # ------------------------------------------------------------------
    # Loss builder (shared binary / multiclass, same as osdmamba_adapter)
    # ------------------------------------------------------------------

    def _build_loss_fn(self, loss_name, train_params):
        nc = self.num_classes
        ls = train_params.get('label_smoothing', 0.1)

        # ---- multiclass ----
        if nc > 1:
            if loss_name == 'focal_iou':
                class _L(nn.Module):
                    def forward(s, x, y):
                        return (smp.losses.FocalLoss('multiclass')(x, y) +
                                smp.losses.JaccardLoss('multiclass', from_logits=True)(x, y))
                return _L()
            if loss_name == 'focal_iou_ls':
                class _L(nn.Module):
                    def forward(s, x, y):
                        return (smp.losses.FocalLoss('multiclass')(x, y) +
                                smp.losses.JaccardLoss('multiclass', from_logits=True)(x, y) +
                                nn.CrossEntropyLoss(label_smoothing=ls)(x, y))
                return _L()
            if loss_name == 'focal_lovasz':
                class _L(nn.Module):
                    def forward(s, x, y):
                        return (smp.losses.FocalLoss('multiclass')(x, y) +
                                smp.losses.LovaszLoss('multiclass', per_image=False)(x, y))
                return _L()
            if loss_name == 'focal_lovasz_ls':
                class _L(nn.Module):
                    def forward(s, x, y):
                        return (smp.losses.FocalLoss('multiclass')(x, y) +
                                smp.losses.LovaszLoss('multiclass', per_image=False)(x, y) +
                                nn.CrossEntropyLoss(label_smoothing=ls)(x, y))
                return _L()
            if loss_name == 'unified_focal_iou':
                class _L(nn.Module):
                    def forward(s, x, y):
                        ufl = (0.5 * smp.losses.FocalLoss('multiclass')(x, y) +
                               0.5 * smp.losses.TverskyLoss('multiclass', alpha=0.3, beta=0.7, gamma=0.75, from_logits=True)(x, y))
                        return ufl + smp.losses.JaccardLoss('multiclass', from_logits=True)(x, y)
                return _L()
            if loss_name == 'ce_dice':
                class _L(nn.Module):
                    def forward(s, x, y):
                        return (0.5 * nn.CrossEntropyLoss()(x, y) +
                                0.5 * smp.losses.DiceLoss('multiclass', from_logits=True)(x, y))
                return _L()
            if loss_name == 'dice':
                return smp.losses.DiceLoss('multiclass', from_logits=True)
            if loss_name == 'focal':
                return smp.losses.FocalLoss('multiclass')
            if loss_name == 'lovasz':
                return smp.losses.LovaszLoss('multiclass', per_image=False)
            print(f"  [Warning] Unknown multiclass loss '{loss_name}'. Using CrossEntropyLoss.")
            return nn.CrossEntropyLoss()

        # ---- binary ----
        if loss_name == 'focal_iou':
            class _L(nn.Module):
                def forward(s, x, y):
                    return (smp.losses.FocalLoss('binary')(x, y) +
                            smp.losses.JaccardLoss('binary', from_logits=True)(x, y))
            return _L()
        if loss_name == 'focal_iou_ls':
            class _L(nn.Module):
                def forward(s, x, y):
                    y_s = y * (1 - ls) + 0.5 * ls
                    return (smp.losses.FocalLoss('binary')(x, y) +
                            smp.losses.JaccardLoss('binary', from_logits=True)(x, y) +
                            nn.BCEWithLogitsLoss()(x, y_s))
            return _L()
        if loss_name == 'focal_lovasz':
            class _L(nn.Module):
                def forward(s, x, y):
                    return (smp.losses.FocalLoss('binary')(x, y) +
                            smp.losses.LovaszLoss('binary', per_image=False)(x, y))
            return _L()
        if loss_name == 'focal_lovasz_ls':
            class _L(nn.Module):
                def forward(s, x, y):
                    y_s = y * (1 - ls) + 0.5 * ls
                    return (smp.losses.FocalLoss('binary')(x, y) +
                            smp.losses.LovaszLoss('binary', per_image=False)(x, y) +
                            nn.BCEWithLogitsLoss()(x, y_s))
            return _L()
        if loss_name == 'unified_focal_iou':
            class _L(nn.Module):
                def forward(s, x, y):
                    ufl = (0.5 * smp.losses.FocalLoss('binary')(x, y) +
                           0.5 * smp.losses.TverskyLoss('binary', alpha=0.3, beta=0.7, gamma=0.75, from_logits=True)(x, y))
                    return ufl + smp.losses.JaccardLoss('binary', from_logits=True)(x, y)
            return _L()
        if loss_name == 'bce_dice':
            class _L(nn.Module):
                def forward(s, x, y):
                    return (0.5 * nn.BCEWithLogitsLoss()(x, y) +
                            0.5 * smp.losses.DiceLoss('binary', from_logits=True)(x, y))
            return _L()
        if loss_name == 'bce_iou':
            class _L(nn.Module):
                def forward(s, x, y):
                    return (0.5 * nn.BCEWithLogitsLoss()(x, y) +
                            0.5 * smp.losses.JaccardLoss('binary', from_logits=True)(x, y))
            return _L()
        if loss_name == 'bce':
            return nn.BCEWithLogitsLoss()
        if loss_name == 'focal':
            return smp.losses.FocalLoss('binary')
        if loss_name == 'dice':
            return smp.losses.DiceLoss('binary', from_logits=True)
        if loss_name == 'lovasz':
            return smp.losses.LovaszLoss('binary', per_image=False)
        print(f"  [Warning] Unknown binary loss '{loss_name}'. Using focal_iou.")
        class _L(nn.Module):
            def forward(s, x, y):
                return (smp.losses.FocalLoss('binary')(x, y) +
                        smp.losses.JaccardLoss('binary', from_logits=True)(x, y))
        return _L()

    # ------------------------------------------------------------------
    # Plot helper
    # ------------------------------------------------------------------

    def _save_epoch_plot(self, history, results_path, epoch, loss_name='loss'):
        try:
            df = pd.DataFrame(history)
            if df.empty:
                return
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(df['epoch'], df['train_loss'], label=f'Train ({loss_name})')
            plt.plot(df['epoch'], df['val_loss'],   label=f'Val ({loss_name})')
            plt.title('Loss'); plt.xlabel('Epoch'); plt.legend(); plt.grid(True)

            plt.subplot(1, 2, 2)
            k_train = 'train_mIoU' if 'train_mIoU' in df else 'train_iou'
            k_val   = 'val_mIoU'   if 'val_mIoU'   in df else 'val_iou'
            plt.plot(df['epoch'], df[k_train], label='Train mIoU')
            plt.plot(df['epoch'], df[k_val],   label='Val mIoU')
            if 'val_iou_ema' in df and (df['val_iou_ema'] > 0).any():
                plt.plot(df['epoch'], df['val_iou_ema'], label='EMA mIoU',
                         linestyle='--')
            plt.title('mIoU'); plt.xlabel('Epoch'); plt.legend()
            plt.grid(True); plt.ylim(0, 1)

            plt.tight_layout()
            plt.savefig(results_path / f'training_curves_epoch_{epoch}.png')
            plt.close()
        except Exception as e:
            print(f"  [Warning] Plot failed: {e}")

    # ------------------------------------------------------------------
    # train()
    # ------------------------------------------------------------------

    def train(self, data, results_path, **train_params):
        print("--- [Info] OSDMamba Official: Training ---")
        results_path = Path(results_path)

        with open(data) as f:
            data_config = yaml.safe_load(f)
        base_path = Path(data_config['path'])

        class_names = data_config.get('names',
                                       [f'Class {i}' for i in range(self.num_classes)])
        if isinstance(class_names, dict):
            class_names = [class_names[i] for i in sorted(class_names)]

        # ---- augmentation ----
        imgsz   = train_params.get('imgsz', 512)
        h = imgsz[0] if isinstance(imgsz, (list, tuple)) else imgsz
        w = imgsz[1] if isinstance(imgsz, (list, tuple)) else imgsz

        aug_list = []
        if train_params.get('fliplr', 0):
            aug_list.append(A.HorizontalFlip(p=train_params['fliplr']))
        if train_params.get('flipud', 0):
            aug_list.append(A.VerticalFlip(p=train_params['flipud']))
        deg = train_params.get('degrees', 0)
        trn = train_params.get('translate', 0)
        scl = train_params.get('scale', 0)
        if deg or trn or scl:
            aug_list.append(A.ShiftScaleRotate(
                rotate_limit=deg, shift_limit=trn, scale_limit=scl,
                p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0))
        if train_params.get('random_brightness_contrast', 0):
            aug_list.append(A.RandomBrightnessContrast(
                p=train_params['random_brightness_contrast']))
        if train_params.get('gauss_noise', 0):
            aug_list.append(A.GaussNoise(p=train_params['gauss_noise']))
        if train_params.get('coarse_dropout', 0):
            aug_list.append(A.CoarseDropout(
                max_holes=8,
                max_height=int(h * 0.1), max_width=int(w * 0.1),
                p=train_params['coarse_dropout']))
        if train_params.get('elastic_transform', 0):
            aug_list.append(A.ElasticTransform(
                p=train_params['elastic_transform'],
                border_mode=cv2.BORDER_CONSTANT, value=0))
        if train_params.get('speckle_noise', 0):
            aug_list.append(SARSpeckleNoise(
                variance=train_params.get('speckle_variance', 0.04),
                p=train_params['speckle_noise']))

        # ---- dataset stats ----
        print("--- Calculating Dataset Statistics ---")
        mean, std = calculate_dataset_stats(
            base_path, save_yaml=True,
            output_path=results_path / 'dataset_stats.yaml')
        if mean is None:
            mean, std = self._get_normalization_stats(train_params, base_path)
        else:
            mean, std = tuple(mean), tuple(std)
            print(f"  - mean={mean}, std={std}")
        self._norm_stats_cache = mean, std
        self._setup_val_transform(mean, std)   # keep val_ds in sync with training stats

        train_tf = A.Compose(aug_list + [
            A.Resize(h, w), A.Normalize(mean=mean, std=std), ToTensorV2()])
        val_tf = A.Compose([
            A.Resize(h, w), A.Normalize(mean=mean, std=std), ToTensorV2()])

        # ---- datasets ----
        train_img_dir = base_path / data_config['train']
        val_img_dir   = base_path / data_config['val']
        train_lbl_dir = base_path / 'labels' / Path(data_config['train']).name
        val_lbl_dir   = base_path / 'labels' / Path(data_config['val']).name

        train_ds = SegmentationDataset(train_img_dir, train_lbl_dir,
                                        train_tf, self.in_channels, self.num_classes)
        val_ds   = SegmentationDataset(val_img_dir,   val_lbl_dir,
                                        val_tf, self.in_channels, self.num_classes)

        batch_size = train_params.get('batch_size', 4)
        num_workers = train_params.get('workers', 0)

        if train_params.get('use_weighted_sampler', False):
            try:
                labels    = train_ds.get_labels()
                num_pos   = labels.count(1)
                num_neg   = labels.count(0)
                if num_pos == 0:
                    raise ValueError("No positive samples")
                w_pos     = 1.0 / num_pos
                w_neg     = 1.0 / num_neg
                sw        = torch.tensor([w_pos if l else w_neg for l in labels]).double()
                sampler   = WeightedRandomSampler(sw, len(sw))
                train_loader = DataLoader(train_ds, batch_size=batch_size,
                                          sampler=sampler, num_workers=num_workers,
                                          pin_memory=True)
            except Exception as e:
                print(f"  [Warning] WeightedSampler failed: {e}. Using shuffle.")
                train_loader = DataLoader(train_ds, batch_size=batch_size,
                                          shuffle=True, num_workers=num_workers,
                                          pin_memory=True)
        else:
            train_loader = DataLoader(train_ds, batch_size=batch_size,
                                      shuffle=True, num_workers=num_workers,
                                      pin_memory=True)

        val_loader = DataLoader(val_ds, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)

        # ---- loss ----
        loss_name = train_params.get('loss_function', 'focal_iou')
        criterion = self._build_loss_fn(loss_name, train_params)
        print(f"  - Loss: {loss_name}")

        # ---- optimiser ----
        lr0          = train_params.get('lr0', 1e-4)
        weight_decay = train_params.get('weight_decay', 1e-4)
        freeze_epochs = train_params.get('freeze_epochs', 0)

        if freeze_epochs > 0:
            print(f"  - Freezing encoder for {freeze_epochs} epochs")
            self.model.freeze_encoder()

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr0, weight_decay=weight_decay)

        use_amp   = train_params.get('amp', False)
        scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)
        accum_steps = train_params.get('gradient_accumulation_steps', 1)
        grad_clip   = train_params.get('gradient_clipping', 0.0)

        # ---- EMA ----
        use_ema   = train_params.get('use_ema', True)
        ema_decay = train_params.get('ema_decay', 0.999)
        model_ema = None
        if use_ema:
            try:
                model_ema = ModelEma(self.model, decay=ema_decay,
                                      device=self.device)
                print(f"  - EMA enabled (decay={ema_decay})")
            except Exception as e:
                print(f"  [Warning] EMA init failed: {e}")

        # Lovasz warmup
        lovasz_warmup = train_params.get('lovasz_warmup_epochs', 0)

        best_iou, epochs_no_improve = 0.0, 0
        patience = train_params.get('patience', 0)
        history  = []
        epochs   = train_params.get('epochs', 100)

        for epoch in range(epochs):
            # unfreeze encoder
            if epoch == freeze_epochs and freeze_epochs > 0:
                print(f"\n  >>> Epoch {epoch+1}: Unfreezing encoder <<<")
                self.model.unfreeze_encoder()
                optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=lr0 * 0.1, weight_decay=weight_decay)

            # Lovasz warmup gate
            if hasattr(criterion, 'lovasz_weight'):
                criterion.lovasz_weight = (0.0 if epoch < lovasz_warmup else 1.0)

            # ---- train epoch ----
            self.model.train()
            run_loss = 0.0
            cm_dim   = self.num_classes if self.num_classes > 1 else 2
            total_cm = np.zeros((cm_dim, cm_dim), dtype=np.int64)

            use_tqdm = sys.stdout.isatty()
            iterator = (tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]")
                        if use_tqdm else train_loader)
            if not use_tqdm:
                print(f"Epoch {epoch+1} [Train] started...")
            log_interval = max(1, len(train_loader) // 5)
            t0 = time.time()

            for i, (images, masks) in enumerate(iterator):
                images = images.to(self.device)
                masks  = masks.to(self.device)

                with torch.amp.autocast('cuda', enabled=use_amp):
                    outputs = self.model(images)   # dict

                    # deep supervision loss
                    loss_main = criterion(outputs['main'], masks)
                    loss_aux  = 0.0
                    for k in ('aux_4', 'aux_8', 'aux_16'):
                        if k in outputs:
                            aux_up = F.interpolate(
                                outputs[k], size=masks.shape[-2:],
                                mode='bilinear', align_corners=False)
                            loss_aux = loss_aux + criterion(aux_up, masks)
                    loss = (loss_main + 0.4 * loss_aux) / accum_steps

                scaler.scale(loss).backward()

                if (i + 1) % accum_steps == 0:
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if model_ema is not None:
                        model_ema.update(self.model)

                run_loss += loss.item() * accum_steps

                # batch metrics
                main_out = outputs['main']
                if self.num_classes > 1:
                    preds = torch.argmax(main_out, 1).cpu().numpy().flatten().astype(np.int64)
                    gt    = masks.cpu().numpy().flatten().astype(np.int64)
                    total_cm += confusion_matrix(gt, preds, labels=list(range(self.num_classes)))
                else:
                    preds = (torch.sigmoid(main_out) > 0.5).cpu().numpy().flatten().astype(np.uint8)
                    gt    = masks.cpu().numpy().flatten().astype(np.uint8)
                    total_cm += confusion_matrix(gt, preds, labels=[0, 1])

                if use_tqdm:
                    iterator.set_postfix(loss=run_loss / (i + 1))
                elif (i + 1) % log_interval == 0:
                    elapsed = time.time() - t0
                    eta = (len(train_loader) - i - 1) * elapsed / (i + 1)
                    print(f"  Epoch {epoch+1} Batch {i+1}/{len(train_loader)} "
                          f"Loss={run_loss/(i+1):.4f} ETA={eta:.0f}s")

            avg_train_loss = run_loss / len(train_loader)
            train_miou = self._compute_miou(total_cm)

            # ---- val epoch ----
            eval_model = model_ema.module if model_ema is not None else self.model
            eval_model.eval()
            run_val_loss = 0.0
            total_cm_v   = np.zeros((cm_dim, cm_dim), dtype=np.int64)

            with torch.no_grad():
                val_iter = (tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
                            if use_tqdm else val_loader)
                if not use_tqdm:
                    print(f"Epoch {epoch+1} [Val] started...")
                for images, masks in val_iter:
                    images = images.to(self.device)
                    masks  = masks.to(self.device)
                    out    = eval_model(images)
                    main_o = out['main'] if isinstance(out, dict) else out
                    run_val_loss += criterion(main_o, masks).item()
                    if self.num_classes > 1:
                        preds = torch.argmax(main_o, 1).cpu().numpy().flatten().astype(np.int64)
                        gt    = masks.cpu().numpy().flatten().astype(np.int64)
                        total_cm_v += confusion_matrix(gt, preds, labels=list(range(self.num_classes)))
                    else:
                        preds = (torch.sigmoid(main_o) > 0.5).cpu().numpy().flatten().astype(np.uint8)
                        gt    = masks.cpu().numpy().flatten().astype(np.uint8)
                        total_cm_v += confusion_matrix(gt, preds, labels=[0, 1])

            avg_val_loss = run_val_loss / len(val_loader)
            val_miou     = self._compute_miou(total_cm_v)

            print(f"  Epoch {epoch+1}: train_loss={avg_train_loss:.4f} "
                  f"train_mIoU={train_miou:.4f} | "
                  f"val_loss={avg_val_loss:.4f} val_mIoU={val_miou:.4f}")

            log = {
                'epoch': epoch + 1,
                'train_loss': avg_train_loss, 'val_loss': avg_val_loss,
                'train_mIoU': train_miou, 'val_mIoU': val_miou,
                'train_iou': train_miou, 'val_iou': val_miou,
            }
            history.append(log)
            pd.DataFrame(history).to_csv(results_path / 'training_log.csv',
                                          index=False)

            # ---- checkpoint ----
            if val_miou > best_iou:
                best_iou = val_miou
                epochs_no_improve = 0
                (results_path / 'weights').mkdir(parents=True, exist_ok=True)
                save_model = (model_ema.module if model_ema is not None
                              else self.model)
                torch.save(save_model.state_dict(),
                           results_path / 'weights' / 'best.pt')
                self._best_metrics = {
                    'best_val_mIoU': float(best_iou),
                    'best_val_loss': float(avg_val_loss),
                    'best_epoch':    epoch + 1,
                }
            else:
                epochs_no_improve += 1

            if (epoch + 1) % 5 == 0:
                self._save_epoch_plot(history, results_path, epoch + 1,
                                       loss_name)
            if patience > 0 and epochs_no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        result = {'best_model_path': str(results_path / 'weights' / 'best.pt')}
        if hasattr(self, '_best_metrics'):
            result.update(self._best_metrics)
        return result

    def _compute_miou(self, cm: np.ndarray) -> float:
        if self.num_classes > 1:
            inter  = np.diag(cm)
            union  = cm.sum(1) + cm.sum(0) - inter
            return float(np.nanmean(inter / (union + 1e-9)))
        else:
            TN, FP, FN, TP = cm.ravel()
            return float(TP / (TP + FP + FN + 1e-9))

    # ------------------------------------------------------------------
    # predict()
    # ------------------------------------------------------------------

    def predict(self, source, imgsz, **kwargs):
        original_image = cv2.imdecode(
            np.fromfile(str(source), dtype=np.uint8), cv2.IMREAD_COLOR)
        if original_image is None:
            return [None]

        if self.in_channels == 1:
            img = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        elif self.in_channels == 2:
            img = original_image[:, :, [2, 1]]
        else:
            img = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        dataset_path = self.config.get('dataset', {}).get('path')
        mean, std = self._get_normalization_stats(
            self.config.get('train', {}), dataset_path, verbose=False)

        h = imgsz[0] if isinstance(imgsz, (list, tuple)) else imgsz
        w = imgsz[1] if isinstance(imgsz, (list, tuple)) else imgsz
        tf = A.Compose([A.Resize(h, w), A.Normalize(mean, std), ToTensorV2()])

        t_img = tf(image=img)['image'].unsqueeze(0).to(self.device)

        with torch.no_grad():
            self.model.eval()
            out      = self.model(t_img)
            main_out = out['main'] if isinstance(out, dict) else out

            oh, ow = original_image.shape[:2]
            main_out = F.interpolate(main_out, size=(oh, ow),
                                      mode='bilinear', align_corners=False)

            if self.num_classes > 1:
                prob      = torch.softmax(main_out, dim=1)
                prob_np   = prob[0, 1].cpu().numpy()          # oil class prob
                mask_np   = torch.argmax(prob, 1)[0].cpu().numpy().astype(np.uint8)
            else:
                prob_np   = torch.sigmoid(main_out)[0, 0].cpu().numpy()
                mask_np   = (prob_np > 0.5).astype(np.uint8)

        return [OSDMambaOfficialResult(original_image, mask_np, prob_np)]

    # ------------------------------------------------------------------
    # val()
    # ------------------------------------------------------------------

    def val(self, data, split='test', **kwargs):
        imgsz = kwargs.get('imgsz', 512)
        with open(data) as f:
            cfg = yaml.safe_load(f)
        base = Path(cfg['path'])
        mean, std = self._get_normalization_stats({}, base)
        h = imgsz[0] if isinstance(imgsz, (list, tuple)) else imgsz
        w = imgsz[1] if isinstance(imgsz, (list, tuple)) else imgsz
        tf = A.Compose([A.Resize(h, w), A.Normalize(mean, std), ToTensorV2()])

        split_key = cfg.get(split, f'images/{split}')
        img_dir   = base / split_key
        lbl_dir   = base / 'labels' / Path(split_key).name
        ds        = SegmentationDataset(img_dir, lbl_dir, tf,
                                         self.in_channels, self.num_classes)
        loader    = DataLoader(ds, batch_size=1, shuffle=False)

        cm_dim   = self.num_classes if self.num_classes > 1 else 2
        total_cm = np.zeros((cm_dim, cm_dim), dtype=np.int64)

        self.model.eval()
        with torch.no_grad():
            for img, mask in tqdm(loader, desc="Val"):
                img  = img.to(self.device)
                mask = mask.to(self.device)
                out  = self.model(img)
                main_o = out['main'] if isinstance(out, dict) else out
                if self.num_classes > 1:
                    preds = torch.argmax(main_o, 1).cpu().numpy().flatten().astype(np.int64)
                    gt    = mask.cpu().numpy().flatten().astype(np.int64)
                    total_cm += confusion_matrix(gt, preds,
                                                  labels=list(range(self.num_classes)))
                else:
                    preds = (torch.sigmoid(main_o) > 0.5).cpu().numpy().flatten().astype(np.uint8)
                    gt    = mask.cpu().numpy().flatten().astype(np.uint8)
                    total_cm += confusion_matrix(gt, preds, labels=[0, 1])

        miou = self._compute_miou(total_cm)
        return MockMetrics(miou)
