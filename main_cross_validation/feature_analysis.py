"""
[File] feature_analysis.py
[Purpose] Advanced Feature Space Analysis.
[Description]
    - Implements t-SNE and PCA visualization for model features.
    - Analyzes separation between classes in the latent space.
    - Generates plots for distribution analysis of training vs validation data.
[Author] Assistant (GitHub Copilot)
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
from pathlib import Path
import cv2
import pandas as pd
import seaborn as sns

class FeatureExtractor:
    """
    Helper class to extract features from a specific layer of the model.
    """
    def __init__(self, model, target_layer_name=None):
        self.model = model
        self.features = None
        self.hook = None
        self.target_layer_name = target_layer_name
        
        self._register_hook()

    def _register_hook(self):
        # If no layer specified, try to find a reasonable default (last decoder block usually)
        target_module = None
        
        if self.target_layer_name:
            for name, module in self.model.named_modules():
                if name == self.target_layer_name:
                    target_module = module
                    break
            if target_module is None:
                print(f"[Warning] Could not find layer: {self.target_layer_name}. Feature extraction might fail.")
        else:
            # Heuristics for common architectures
            # 1. SMP Unet / Unet++ / DeepLab
            if hasattr(self.model, 'decoder'):
                # Usually the last block of the decoder before segmentation head
                # smp.Unet: decoder.blocks[-1]
                if hasattr(self.model.decoder, 'blocks'):
                    target_module = self.model.decoder.blocks[-1]
                # smp.DeepLabV3+: decoder.last_conv
                elif hasattr(self.model.decoder, 'last_conv'):
                    target_module = self.model.decoder.last_conv
            
            # 2. Segformer (HuggingFace)
            elif hasattr(self.model, 'decode_head'):
                # linear_fuse is often the last step before classifier
                if hasattr(self.model.decode_head, 'linear_fuse'):
                    target_module = self.model.decode_head.linear_fuse
                elif hasattr(self.model.decode_head, 'batch_norm'):
                    target_module = self.model.decode_head.batch_norm
                # [New] Fallback for Segformer: use the last encoder block if decode_head is tricky
                elif hasattr(self.model, 'segformer') and hasattr(self.model.segformer, 'encoder'):
                     target_module = self.model.segformer.encoder.block[-1]

            # 3. OSDMamba
            elif 'OSDMamba' in type(self.model).__name__:
                if hasattr(self.model, 'dec_layer3'):
                    # Use the last block of the last decoder layer (High Res)
                    target_module = self.model.dec_layer3[-1]
                elif hasattr(self.model, 'final_head'):
                    # Fallback to final_head input (might need to hook input of final_head)
                    # But hooking output of dec_layer3 is safer
                    pass

            # 4. Generic fallback: Find the last Conv2d that is NOT the classification head
            if target_module is None:
                convs = [m for m in self.model.modules() if isinstance(m, nn.Conv2d)]
                if len(convs) > 1:
                    target_module = convs[-2] # Second to last conv
                elif len(convs) > 0:
                    target_module = convs[-1]

        if target_module:
            self.hook = target_module.register_forward_hook(self._hook_fn)
            print(f"[Info] Feature Extractor hooked to: {target_module}")
        else:
            print("[Error] Failed to register hook for feature extraction.")

    def _hook_fn(self, module, input, output):
        # output might be a Tensor or a tuple/dict
        if isinstance(output, torch.Tensor):
            self.features = output
        elif isinstance(output, (tuple, list)):
            # Check if the first element is a tensor
            if isinstance(output[0], torch.Tensor):
                self.features = output[0]
            else:
                # Try to find a tensor in the tuple
                for item in output:
                    if isinstance(item, torch.Tensor):
                        self.features = item
                        break
        elif hasattr(output, 'last_hidden_state'): # Transformers
            self.features = output.last_hidden_state
        elif hasattr(output, 'logits'): # Some HF outputs
             self.features = output.logits
        else:
            self.features = output

    def remove(self):
        if self.hook:
            self.hook.remove()

def collect_feature_samples(model_adapter, dataset_loader, max_samples=3000, samples_per_batch=50):
    """
    Collects feature samples categorized by TP, TN, FP, FN.
    Returns X (features) and y (labels).
    """
    extractor = FeatureExtractor(model_adapter.model)
    
    # Collections for different categories
    collected_data = {
        'TP': [], 'TN': [], 'FP': [], 'FN': []
    }
    
    model_adapter.model.eval()
    
    with torch.no_grad():
        for images, masks in tqdm(dataset_loader, desc="Collecting features"):
            images = images.to(model_adapter.device)
            masks = masks.to(model_adapter.device)
            
            # Forward pass
            if 'pixel_values' in model_adapter.model.forward.__code__.co_varnames:
                 output = model_adapter.model(pixel_values=images)
            else:
                 output = model_adapter.model(images)
            
            # Extract Logits/Predictions
            if isinstance(output, dict) and 'logits' in output:
                logits = output['logits']
            elif isinstance(output, (list, tuple)):
                logits = output[0]
            else:
                logits = output
                
            features = extractor.features # (B, C, H_feat, W_feat)
            if features is None: continue
            
            B, C_feat, H_feat, W_feat = features.shape
            
            # Resize masks to feature size
            if masks.dim() == 3: 
                masks = masks.unsqueeze(1)
            
            # Interpolate masks (Nearest for labels)
            masks_resized = torch.nn.functional.interpolate(masks.float(), size=(H_feat, W_feat), mode='nearest')
            
            # Resize logits and get preds (Bilinear for logits)
            logits_resized = torch.nn.functional.interpolate(logits, size=(H_feat, W_feat), mode='bilinear', align_corners=False)
            
            if logits_resized.shape[1] == 1:
                preds_resized = (torch.sigmoid(logits_resized) > 0.5).float()
            else:
                preds_resized = torch.argmax(logits_resized, dim=1, keepdim=True).float()
            
            # Flatten everything
            features_flat = features.permute(0, 2, 3, 1).reshape(-1, C_feat).cpu().numpy()
            masks_flat = masks_resized.reshape(-1).cpu().numpy()
            preds_flat = preds_resized.reshape(-1).cpu().numpy()
            
            # Identify categories
            is_pos = (masks_flat == 1)
            is_neg = (masks_flat == 0)
            is_pred_pos = (preds_flat == 1)
            is_pred_neg = (preds_flat == 0)
            
            indices = {
                'TP': np.where(is_pos & is_pred_pos)[0],
                'TN': np.where(is_neg & is_pred_neg)[0],
                'FP': np.where(is_neg & is_pred_pos)[0],
                'FN': np.where(is_pos & is_pred_neg)[0]
            }
            
            # Sample from this batch
            for cat, idxs in indices.items():
                if len(idxs) > 0:
                    n_take = min(len(idxs), samples_per_batch)
                    chosen = np.random.choice(idxs, n_take, replace=False)
                    collected_data[cat].append(features_flat[chosen])

    extractor.remove()
    
    # Combine and Downsample
    final_features = []
    final_labels = []
    
    # Flatten the lists
    for cat in collected_data:
        if collected_data[cat]:
            collected_data[cat] = np.concatenate(collected_data[cat], axis=0)
        else:
            collected_data[cat] = np.empty((0, C_feat))
            
    total_collected = sum(len(v) for v in collected_data.values())
    if total_collected == 0:
        print("[Warning] No features collected.")
        return None, None

    print(f"  - Collected samples: TP={len(collected_data['TP'])}, TN={len(collected_data['TN'])}, FP={len(collected_data['FP'])}, FN={len(collected_data['FN'])}")
    
    # Budget allocation
    remaining_budget = max_samples
    
    # Helper to add samples
    def add_samples(category, data, budget):
        n = len(data)
        if n == 0: return 0
        if n > budget:
            indices = np.random.choice(n, budget, replace=False)
            data = data[indices]
            n = budget
        final_features.append(data)
        final_labels.extend([category] * n)
        return n

    # 1. Prioritize Errors (FP, FN) - take all if possible
    n_fp = add_samples('FP', collected_data['FP'], remaining_budget)
    remaining_budget -= n_fp
    
    n_fn = add_samples('FN', collected_data['FN'], remaining_budget)
    remaining_budget -= n_fn
    
    # 2. Fill rest with Correct predictions (TP, TN)
    if remaining_budget > 0:
        budget_tp = remaining_budget // 2
        budget_tn = remaining_budget - budget_tp
        
        # Adjust if one class doesn't have enough
        if len(collected_data['TP']) < budget_tp:
            budget_tn += (budget_tp - len(collected_data['TP']))
        elif len(collected_data['TN']) < budget_tn:
            budget_tp += (budget_tn - len(collected_data['TN']))
            
        add_samples('TP', collected_data['TP'], budget_tp)
        add_samples('TN', collected_data['TN'], budget_tn)

    if not final_features:
        return None, None

    X = np.concatenate(final_features, axis=0)
    y = np.array(final_labels)
    return X, y

from matplotlib import animation

def plot_embedding_3d(X_embedded, y, save_path, title, palette):
    """
    Plots a 3D scatter plot of the embeddings and saves a rotating GIF.
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a DataFrame
    df_plot = pd.DataFrame({'x': X_embedded[:, 0], 'y': X_embedded[:, 1], 'z': X_embedded[:, 2], 'Label': y})
    
    # Plot each category
    scatters = []
    for label, color in palette.items():
        subset = df_plot[df_plot['Label'] == label]
        if not subset.empty:
            scatters.append(ax.scatter(subset['x'], subset['y'], subset['z'], c=color, label=label, alpha=0.6, s=30))
    
    ax.set_title(title)
    ax.legend()
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    
    # Save static view
    plt.savefig(save_path)
    print(f"  - 3D Plot saved to: {save_path}")
    
    # Create rotation animation
    def rotate(angle):
        ax.view_init(elev=30, azim=angle)
        return scatters

    # Save as GIF
    gif_path = save_path.with_suffix('.gif')
    print(f"  - Generating 3D rotation animation (this may take a moment)...")
    try:
        # [Modified] Slower and smoother rotation: 
        # frames step 2 (was 5) -> more frames
        # fps 15 -> smooth playback
        # Total duration = 180 frames / 15 fps = 12 seconds per rotation (was ~7s)
        anim = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2), interval=100, blit=False)
        anim.save(gif_path, writer='pillow', fps=15)
        print(f"  - 3D Animation saved to: {gif_path}")
    except Exception as e:
        print(f"  [Warning] Failed to save 3D animation: {e}")
        
    plt.close()

def run_tsne_analysis(model_adapter, dataset_loader, save_dir, max_samples=3000):
    """
    Runs t-SNE analysis on the features extracted from the model.
    """
    print("\n--- [Analysis] Starting t-SNE Analysis (3D) ---")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    X, y = collect_feature_samples(model_adapter, dataset_loader, max_samples=max_samples)
    if X is None: return

    print(f"  - Running t-SNE on {X.shape[0]} samples with {X.shape[1]} dimensions...")
    
    if X.shape[1] <= 1:
        print("  [Warning] Feature dimension is 1. t-SNE requires > 1 dimension. Skipping t-SNE.")
        return

    # Use init='pca' for better initialization and stability
    n_components = 3
    if X.shape[1] < n_components:
        print(f"  [Warning] Feature dimension ({X.shape[1]}) < n_components ({n_components}). Reducing n_components.")
        n_components = X.shape[1]
        
    tsne = TSNE(n_components=n_components, random_state=42, init='pca', learning_rate='auto')
    try:
        X_embedded = tsne.fit_transform(X)
    except Exception as e:
        print(f"  [Error] t-SNE failed: {e}")
        return
    
    palette = {'TP': 'green', 'TN': 'blue', 'FP': 'red', 'FN': 'orange'}
    plot_embedding_3d(X_embedded, y, save_dir / 'tsne_3d.png', 't-SNE 3D Visualization (TP/TN/FP/FN)', palette)

    # [New] Run 2: Focus on Object/Errors (Exclude TN)
    # This helps to see the separation between TP, FP, FN without the dominant TN cluster
    print("  - Running t-SNE on focus group (TP, FP, FN) to see details without Background...")
    mask = (y != 'TN')
    if np.sum(mask) > 50: # Ensure we have enough samples
        X_focus = X[mask]
        y_focus = y[mask]
        
        print(f"    -> Focusing on {len(X_focus)} samples (TP/FP/FN)...")
        tsne_focus = TSNE(n_components=3, random_state=42, init='pca', learning_rate='auto')
        X_embedded_focus = tsne_focus.fit_transform(X_focus)
        
        plot_embedding_3d(X_embedded_focus, y_focus, save_dir / 'tsne_3d_no_TN.png', 't-SNE Focus (TP/FP/FN - No Background)', palette)
    else:
        print("    -> Not enough non-TN samples for focus analysis.")

def run_pca_analysis(model_adapter, dataset_loader, save_dir, max_samples=3000):
    """
    Runs PCA analysis on the features extracted from the model.
    """
    print("\n--- [Analysis] Starting PCA Analysis (3D) ---")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    X, y = collect_feature_samples(model_adapter, dataset_loader, max_samples=max_samples)
    if X is None: return

    print(f"  - Running PCA on {X.shape[0]} samples...")
    pca = PCA(n_components=3)
    X_embedded = pca.fit_transform(X)
    
    palette = {'TP': 'green', 'TN': 'blue', 'FP': 'red', 'FN': 'orange'}
    plot_embedding_3d(X_embedded, y, save_dir / 'pca_3d.png', 'PCA 3D Visualization (TP/TN/FP/FN)', palette)
    
    print(f"  - PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")

def run_channel_importance_analysis(model_adapter, dataset_loader, save_dir, num_batches=50):
    """
    Analyzes input channel importance using Permutation Importance.
    Measures mIoU drop when each channel is shuffled.
    """
    print("\n--- [Analysis] Starting Channel Importance Analysis (Permutation) ---")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model_adapter.model.eval()
    
    # Helper to compute mIoU on a batch
    def compute_batch_iou(preds, masks):
        # preds: (B, H, W), masks: (B, H, W)
        intersection = ((preds == 1) & (masks == 1)).sum().item()
        union = ((preds == 1) | (masks == 1)).sum().item()
        return intersection, union

    # Collect a subset of data and STACK them into a single large batch
    # This is crucial for permutation to work if batch_size=1
    subset_images_list = []
    subset_masks_list = []
    
    print(f"  - Collecting subset of {num_batches} batches for analysis...")
    for i, (images, masks) in enumerate(dataset_loader):
        if i >= num_batches: break
        subset_images_list.append(images)
        subset_masks_list.append(masks)
        
    if not subset_images_list:
        print("[Warning] No data for channel importance analysis.")
        return

    # Stack into (N, C, H, W)
    # Note: This assumes all images have same size, which they should in this loader
    try:
        all_images = torch.cat(subset_images_list, dim=0)
        all_masks = torch.cat(subset_masks_list, dim=0)
    except Exception as e:
        print(f"[Error] Failed to stack images for importance analysis: {e}")
        return

    N, C, H, W = all_images.shape
    print(f"  - Analyzing {N} images with {C} input channels.")
    
    results = {}
    
    # 1. Baseline Performance
    total_inter = 0
    total_union = 0
    
    # Process in chunks to avoid OOM if N is large
    chunk_size = 10 
    
    with torch.no_grad():
        for i in range(0, N, chunk_size):
            batch_imgs = all_images[i:i+chunk_size].to(model_adapter.device)
            batch_masks = all_masks[i:i+chunk_size].to(model_adapter.device)
            
            if 'pixel_values' in model_adapter.model.forward.__code__.co_varnames:
                 output = model_adapter.model(pixel_values=batch_imgs)
            else:
                 output = model_adapter.model(batch_imgs)
            
            if isinstance(output, dict) and 'logits' in output: logits = output['logits']
            elif isinstance(output, (list, tuple)): logits = output[0]
            else: logits = output
            
            logits = torch.nn.functional.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
            if logits.shape[1] == 1: preds = (torch.sigmoid(logits) > 0.5).float()
            else: preds = torch.argmax(logits, dim=1, keepdim=True).float()
            
            if batch_masks.dim() == 3: batch_masks = batch_masks.unsqueeze(1)
            
            inter, union = compute_batch_iou(preds, batch_masks)
            total_inter += inter
            total_union += union
            
    baseline_iou = total_inter / (total_union + 1e-8)
    print(f"  - Baseline mIoU (Oil): {baseline_iou:.4f}")
    results['Baseline'] = baseline_iou
    
    # 2. Permutation Importance
    for c in range(C):
        print(f"  - Testing Channel {c} importance...")
        total_inter = 0
        total_union = 0
        
        # Create a shuffled version of the WHOLE dataset for this channel
        # We shuffle indices [0..N-1]
        permuted_indices = torch.randperm(N)
        
        # We can't modify all_images in place easily without affecting next loop, 
        # so we construct batches dynamically
        
        with torch.no_grad():
            for i in range(0, N, chunk_size):
                # Get original batch structure
                batch_imgs = all_images[i:i+chunk_size].clone() # (B, C, H, W)
                batch_masks = all_masks[i:i+chunk_size].to(model_adapter.device)
                
                # Apply permutation for channel c
                # The indices for this batch in the global permutation
                batch_perm_indices = permuted_indices[i:i+chunk_size]
                
                # Replace channel c with data from the permuted indices
                # We need to fetch the source images for these indices
                source_imgs = all_images[batch_perm_indices] # (B, C, H, W)
                batch_imgs[:, c, :, :] = source_imgs[:, c, :, :]
                
                batch_imgs = batch_imgs.to(model_adapter.device)
                
                if 'pixel_values' in model_adapter.model.forward.__code__.co_varnames:
                     output = model_adapter.model(pixel_values=batch_imgs)
                else:
                     output = model_adapter.model(batch_imgs)
                
                if isinstance(output, dict) and 'logits' in output: logits = output['logits']
                elif isinstance(output, (list, tuple)): logits = output[0]
                else: logits = output
                
                logits = torch.nn.functional.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)
                if logits.shape[1] == 1: preds = (torch.sigmoid(logits) > 0.5).float()
                else: preds = torch.argmax(logits, dim=1, keepdim=True).float()
                
                if batch_masks.dim() == 3: batch_masks = batch_masks.unsqueeze(1)
                
                inter, union = compute_batch_iou(preds, batch_masks)
                total_inter += inter
                total_union += union
        
        permuted_iou = total_inter / (total_union + 1e-8)
        drop = baseline_iou - permuted_iou
        # Ensure drop is non-negative (sometimes noise makes it slightly better)
        drop = max(0, drop)
        print(f"    -> Channel {c} shuffled mIoU: {permuted_iou:.4f} (Drop: {drop:.4f})")
        results[f'Channel {c}'] = drop

    # Plotting
    plt.figure(figsize=(10, 6))
    channels = [k for k in results.keys() if k != 'Baseline']
    drops = [results[k] for k in channels]
    
    # Normalize drops to relative importance %
    total_drop = sum(drops) + 1e-8
    importance_pct = [d / total_drop * 100 for d in drops]
    
    sns.barplot(x=channels, y=importance_pct, palette='viridis')
    plt.title(f'Input Channel Importance (Permutation)\n{model_adapter.config.get("architecture", "Model")}')
    plt.ylabel('Relative Importance (%)')
    plt.xlabel('Input Channel')
    plt.ylim(0, 100)
    
    for i, v in enumerate(importance_pct):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
        
    out_path = save_dir / 'channel_importance.png'
    plt.savefig(out_path)
    plt.close()
    print(f"  - Channel Importance plot saved to: {out_path}")

def analyze_rgb_error_distribution(model_adapter, dataset_loader, save_dir, max_pixels=1000000):
    """
    分析 RGB 通道在不同錯誤類型 (TP, FP, FN, TN) 下的分佈。
    這有助於找出特定顏色是否容易導致誤判 (例如：亮藍色容易 FP)。
    
    Args:
        max_pixels: 為了避免記憶體爆炸，每種類型最多採樣多少個像素。
    """
    print("\n--- [Analysis] Starting RGB Error Distribution Analysis (Histogram) ---")
    save_dir = Path(save_dir) / 'rgb_error_analysis'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model_adapter.model.eval()
    
    # 儲存像素值： {'TP': {'R': [], 'G': [], 'B': []}, 'FP': ...}
    pixel_data = {
        'TP': {'R': [], 'G': [], 'B': []},
        'FP': {'R': [], 'G': [], 'B': []},
        'FN': {'R': [], 'G': [], 'B': []},
        'TN': {'R': [], 'G': [], 'B': []}
    }
    
    # 計數器，用於控制採樣數量
    counts = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}
    
    # 嘗試獲取 Normalization 參數以進行反正規化 (還原回 0-255)
    norm_mean = np.array([0.485, 0.456, 0.406]) # Default ImageNet
    norm_std = np.array([0.229, 0.224, 0.225])   # Default ImageNet
    has_norm_params = False
    
    try:
        # 嘗試從 dataset transform 中獲取
        if hasattr(dataset_loader.dataset, 'transform') and dataset_loader.dataset.transform is not None:
            import albumentations as A
            transforms = dataset_loader.dataset.transform
            # 如果是 Compose，遍歷尋找 Normalize
            if isinstance(transforms, A.Compose):
                for t in transforms.transforms:
                    if isinstance(t, A.Normalize):
                        norm_mean = np.array(t.mean)
                        norm_std = np.array(t.std)
                        has_norm_params = True
                        print(f"  - [Info] Found normalization params: mean={norm_mean}, std={norm_std}")
                        break
            elif isinstance(transforms, A.Normalize):
                norm_mean = np.array(transforms.mean)
                norm_std = np.array(transforms.std)
                has_norm_params = True
                print(f"  - [Info] Found normalization params: mean={norm_mean}, std={norm_std}")
    except Exception as e:
        print(f"  - [Warning] Could not extract normalization params: {e}. Using defaults.")

    with torch.no_grad():
        for images, masks in tqdm(dataset_loader, desc="Collecting RGB Error Pixels"):
            # 如果所有類型都收集夠了，就提早結束
            if all(c >= max_pixels for c in counts.values()):
                break
                
            images = images.to(model_adapter.device)
            masks = masks.to(model_adapter.device)
            
            # 取得預測
            if 'pixel_values' in model_adapter.model.forward.__code__.co_varnames:
                outputs = model_adapter.model(pixel_values=images)
            else:
                outputs = model_adapter.model(images)
                
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs
                
            # Resize logits to match mask
            if logits.shape[-2:] != masks.shape[-2:]:
                logits = nn.functional.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            
            preds = (torch.sigmoid(logits) > 0.5).float()
            
            # 轉換為 numpy 以便處理
            # images 需要反正規化嗎？通常 dataset_loader 出來的是已經 normalize 過的
            # 這裡假設我們想看的是「正規化後」的特徵分佈，或者我們可以嘗試還原
            # 為了簡單起見，我們先直接分析輸入模型的數值 (這代表模型看到的顏色)
            
            imgs_np = images.cpu().numpy() # (B, 3, H, W)
            masks_np = masks.cpu().numpy() # (B, 1, H, W)
            preds_np = preds.cpu().numpy() # (B, 1, H, W)
            
            # 定義錯誤遮罩
            tp_mask = (preds_np == 1) & (masks_np == 1)
            fp_mask = (preds_np == 1) & (masks_np == 0)
            fn_mask = (preds_np == 0) & (masks_np == 1)
            tn_mask = (preds_np == 0) & (masks_np == 0)
            
            masks_dict = {'TP': tp_mask, 'FP': fp_mask, 'FN': fn_mask, 'TN': tn_mask}
            
            for cat, mask in masks_dict.items():
                if counts[cat] >= max_pixels:
                    continue
                
                # mask shape: (B, 1, H, W) -> (B, H, W)
                mask_sq = mask.squeeze(1)
                
                # 找出該類別的所有像素索引
                # np.where 回傳 (batch_idx, h, w)
                indices = np.where(mask_sq)
                
                if len(indices[0]) == 0:
                    continue
                
                # 隨機採樣以避免過多數據來自同一張圖 (Optional, 這裡先全取直到滿)
                num_pixels = len(indices[0])
                remaining = max_pixels - counts[cat]
                
                if num_pixels > remaining:
                    # 隨機選取 remaining 個
                    perm = np.random.choice(num_pixels, remaining, replace=False)
                    selected_b = indices[0][perm]
                    selected_h = indices[1][perm]
                    selected_w = indices[2][perm]
                else:
                    selected_b = indices[0]
                    selected_h = indices[1]
                    selected_w = indices[2]
                
                # 提取 RGB 值
                # imgs_np: (B, 3, H, W)
                # 我們需要 (N, 3)
                vals = imgs_np[selected_b, :, selected_h, selected_w] # (N, 3)
                
                # [Feature] 反正規化: (x * std + mean) * 255
                # vals is (N, 3), norm_std is (3,), norm_mean is (3,)
                # 確保維度匹配
                vals = vals * norm_std + norm_mean
                vals = vals * 255.0
                vals = np.clip(vals, 0, 255)
                
                pixel_data[cat]['R'].extend(vals[:, 0].tolist())
                pixel_data[cat]['G'].extend(vals[:, 1].tolist())
                pixel_data[cat]['B'].extend(vals[:, 2].tolist())
                
                counts[cat] += len(vals)

    print(f"  - Collected pixels: {counts}")
    
    # 開始繪圖
    # 針對 R, G, B 各畫一張圖
    channels = ['Red (Ch0)', 'Green (Ch1)', 'Blue (Ch2)']
    channel_keys = ['R', 'G', 'B']
    
    # 轉換為 DataFrame 方便 Seaborn 繪圖
    plot_data = []
    for cat in ['TP', 'FP', 'FN', 'TN']:
        # 為了圖表清晰，TN 可以少畫一點，或者畫在背景
        # 這裡全部都畫
        for i, ch_key in enumerate(channel_keys):
            vals = pixel_data[cat][ch_key]
            if not vals: continue
            
            # 隨機採樣繪圖 (如果數據太多，繪圖會很慢)
            if len(vals) > 10000:
                vals = np.random.choice(vals, 10000, replace=False)
            
            for v in vals:
                plot_data.append({
                    'Value': v,
                    'Type': cat,
                    'Channel': channels[i]
                })
    
    if not plot_data:
        print("[Warning] No pixel data collected for plotting.")
        return

    df_plot = pd.DataFrame(plot_data)
    
    # 畫圖：3個子圖 (R, G, B)
    print("  - Plotting histograms...")
    g = sns.FacetGrid(df_plot, col="Channel", hue="Type", sharex=True, sharey=False, height=5, aspect=1.2,
                      hue_order=['TP', 'FP', 'FN', 'TN'], 
                      palette={'TP': 'green', 'FP': 'red', 'FN': 'orange', 'TN': 'blue'})
    g.map(sns.kdeplot, "Value", fill=True, alpha=0.3, warn_singular=False)
    g.set_axis_labels("Pixel Value (0-255)", "Density")
    g.add_legend()
    
    out_path = save_dir / 'rgb_error_distribution.png'
    g.savefig(out_path)
    plt.close()
    print(f"  - RGB Error Distribution plot saved to: {out_path}")
    
    # 額外畫一張：只看 FP vs TP (誤報 vs 正確)
    # 這能幫我們看 FP 到底跟 TP 差在哪
    df_fp_tp = df_plot[df_plot['Type'].isin(['TP', 'FP'])]
    if not df_fp_tp.empty:
        g2 = sns.FacetGrid(df_fp_tp, col="Channel", hue="Type", sharex=True, sharey=False, height=5, aspect=1.2,
                          hue_order=['TP', 'FP'], palette={'TP': 'green', 'FP': 'red'})
        g2.map(sns.kdeplot, "Value", fill=True, alpha=0.3, warn_singular=False)
        g2.set_axis_labels("Pixel Value (0-255)", "Density")
        g2.add_legend()
        g2.savefig(save_dir / 'rgb_fp_vs_tp.png')
        plt.close()

    # 額外畫一張：只看 FN vs TP (漏報 vs 正確)
    df_fn_tp = df_plot[df_plot['Type'].isin(['TP', 'FN'])]
    if not df_fn_tp.empty:
        g3 = sns.FacetGrid(df_fn_tp, col="Channel", hue="Type", sharex=True, sharey=False, height=5, aspect=1.2,
                          hue_order=['TP', 'FN'], palette={'TP': 'green', 'FN': 'orange'})
        g3.map(sns.kdeplot, "Value", fill=True, alpha=0.3, warn_singular=False)
        g3.add_legend()
        g3.savefig(save_dir / 'rgb_fn_vs_tp.png')
        plt.close()

def visualize_feature_maps(model_adapter, dataset_loader, save_dir, num_images=5):
    """
    Visualizes feature maps (Attention Maps) for a few sample images.
    Uses PCA to project high-dimensional features to RGB.
    """
    print("\n--- [Analysis] Starting Feature Map Visualization (Attention Map) ---")
    save_dir = Path(save_dir) / 'feature_maps'
    save_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = FeatureExtractor(model_adapter.model)
    model_adapter.model.eval()
    
    count = 0
    with torch.no_grad():
        for images, masks in dataset_loader:
            if count >= num_images: break
            
            images = images.to(model_adapter.device)
            
            # Forward pass
            if 'pixel_values' in model_adapter.model.forward.__code__.co_varnames:
                 _ = model_adapter.model(pixel_values=images)
            else:
                 _ = model_adapter.model(images)
            
            features = extractor.features # (B, C, H, W)
            if features is None: continue
            
            # Process each image in batch
            for i in range(features.shape[0]):
                if count >= num_images: break
                
                feat = features[i] # (C, H, W)
                C, H, W = feat.shape
                
                # 1. Mean Activation Map (Heatmap)
                mean_activation = torch.mean(feat, dim=0).cpu().numpy()
                mean_activation = (mean_activation - mean_activation.min()) / (mean_activation.max() - mean_activation.min() + 1e-8)
                mean_activation = np.uint8(255 * mean_activation)
                mean_activation_color = cv2.applyColorMap(mean_activation, cv2.COLORMAP_JET)
                
                # 2. PCA Projection to RGB
                feat_flat = feat.permute(1, 2, 0).reshape(-1, C).cpu().numpy() # (H*W, C)
                
                # Standardize features before PCA for better visualization
                feat_mean = feat_flat.mean(axis=0)
                feat_std = feat_flat.std(axis=0) + 1e-8
                feat_norm = (feat_flat - feat_mean) / feat_std
                
                pca = PCA(n_components=3)
                feat_pca = pca.fit_transform(feat_norm)
                
                # Normalize to 0-255 for RGB
                feat_pca_norm = (feat_pca - feat_pca.min(axis=0)) / (feat_pca.max(axis=0) - feat_pca.min(axis=0) + 1e-8)
                feat_pca_rgb = (feat_pca_norm * 255).astype(np.uint8)
                feat_pca_img = feat_pca_rgb.reshape(H, W, 3)
                
                # Save images
                cv2.imwrite(str(save_dir / f'sample_{count}_mean_activation.png'), mean_activation_color)
                cv2.imwrite(str(save_dir / f'sample_{count}_pca_features.png'), cv2.cvtColor(feat_pca_img, cv2.COLOR_RGB2BGR))
                
                print(f"  - Saved feature maps for sample {count}")
                count += 1
                
    extractor.remove()
