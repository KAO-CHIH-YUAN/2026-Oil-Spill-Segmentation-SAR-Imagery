"""
[File] evaluation_module.py
[Purpose] Model Evaluation and Visualization Module.
[Description]
    - Computes pixel-level metrics (IoU, F1, Accuracy).
    - Generates visualization overlays (Prediction vs GT).
    - Integrates advanced feature analysis (t-SNE, PCA) hooks.
    - Handles post-training evaluation on Test sets.
[Author] Assistant (GitHub Copilot)
"""
from pathlib import Path
import yaml
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torch.nn.functional as F
import shutil
from concurrent.futures import ThreadPoolExecutor

from .reconstruction_module import run_reconstruction_evaluation, calculate_iou, compute_hd95
from .training_module import get_model_adapter
# 確保在檔案頂部引入
import albumentations as A
from albumentations.pytorch import ToTensorV2
from .gradcam_utils import GradCAM, apply_colormap_on_image, compute_input_saliency
# [New] 導入 t-SNE, PCA, Feature Map, Channel Importance 分析工具
try:
    from .feature_analysis import run_tsne_analysis, run_pca_analysis, visualize_feature_maps, run_channel_importance_analysis, analyze_rgb_error_distribution
except ImportError as e:
    print(f"[Error] Failed to import feature_analysis: {e}")
    run_tsne_analysis = None
    run_pca_analysis = None
    visualize_feature_maps = None
    run_channel_importance_analysis = None
    analyze_rgb_error_distribution = None

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
from pathlib import Path
import torch



def _get_gt_mask(label_path_base, h, w, architecture):
    """
    根據模型架構獲取真實標籤遮罩 (Ground Truth Mask)。
    - 對於 'yolo'，讀取 .txt 檔案並繪製多邊形。
    - 對於其他模型，讀取 .png 影像遮罩。
    """
    gt_mask = np.zeros((h, w), dtype=np.uint8)
    has_gt_object = False

    if architecture == 'yolo':
        # YOLO 的標籤檔案是 .txt
        label_path = label_path_base.with_suffix('.txt')
        if label_path.exists() and label_path.stat().st_size > 0:
            polygons = []
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 1:
                        has_gt_object = True
                        # 將歸一化座標轉換為絕對座標
                        poly = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
                        poly[:, 0] *= w
                        poly[:, 1] *= h
                        polygons.append(poly.astype(np.int32))
            if polygons:
                # 在空白遮罩上繪製所有多邊形
                cv2.fillPoly(gt_mask, polygons, 1)
    else:
        # 其他模型 (如 DeepLabV3+) 的標籤是 .png 影像
        label_path = label_path_base.with_suffix('.png')
        if label_path.exists():
            mask_img = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
            if mask_img is not None and mask_img.sum() > 0:
                gt_mask = (mask_img > 128).astype(np.uint8)
                has_gt_object = True
    
    return gt_mask, has_gt_object, label_path



# M4D Multiclass Color Palette (BGR for OpenCV)
CLASS_COLORS = {
    0: (0, 0, 0),       # Sea Surface (Black)
    1: (255, 255, 0),   # Oil Spill (Cyan) -> OpenCV BGR is (255, 255, 0)
    2: (0, 0, 255),     # Look-alike (Red) -> OpenCV BGR is (0, 0, 255)
    3: (0, 76, 153),    # Ship (Brown) -> (153, 76, 0) RGB -> (0, 76, 153) BGR
    4: (0, 153, 0)      # Land (Green) -> (0, 153, 0) RGB -> (0, 153, 0) BGR
}

def encode_mask_to_rgb(mask):
    """
    Convert 2D label mask (H, W) with values 0-4 to RGB image (H, W, 3)
    """
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_idx, color in CLASS_COLORS.items():
        rgb[mask == cls_idx] = color
    return rgb


def save_confusion_matrix_figure(cm, class_names, out_path,
                                 title="Confusion Matrix (pixel-level)"):
    """繪製並儲存 pixel-level confusion matrix 圖。

    cm          : 2D numpy array，rows = Ground Truth、cols = Predicted。
    class_names : 各列/欄對應的類別名稱。
    out_path    : 輸出 .png 路徑。
    每個 cell 標註「像素數 + 該列(GT)百分比」；底色為列正規化(row-normalized)。
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    cm = np.asarray(cm, dtype=np.float64)
    n = cm.shape[0]
    with np.errstate(all='ignore'):
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum != 0)

    fig, ax = plt.subplots(figsize=(max(4.0, n * 1.6), max(3.6, n * 1.4)))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0.0, vmax=1.0)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground Truth')
    ax.set_title(title)

    for i in range(n):
        for j in range(n):
            cnt = int(cm[i, j])
            pct = cm_norm[i, j] * 100.0
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, f"{cnt:,}\n{pct:.2f}%", ha='center', va='center',
                    color=color, fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    try:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=150)
        print(f"  - Confusion matrix 圖已儲存: {out_path}")
    finally:
        plt.close(fig)

# 檔案: main/evaluation_module.py
# (請替換這一個函式)

def generate_categorized_predictions(model_adapter, exp_config, results_path):
    """[Wrapper] Delegates to evaluate_patches_unified (categorized only, no metrics).

    保留此 API 給可能的外部呼叫；evaluate() 主流程已改用 evaluate_patches_unified 直接呼叫。
    """
    evaluate_patches_unified(model_adapter, exp_config, results_path,
                              do_metrics=False, do_categorized=True)


def calculate_pixel_level_metrics(model_adapter, exp_config):
    """[Wrapper] Delegates to evaluate_patches_unified (metrics only, no categorized).

    保留此 API 給可能的外部呼叫；evaluate() 主流程已改用 evaluate_patches_unified 直接呼叫。
    """
    return evaluate_patches_unified(model_adapter, exp_config, results_path=None,
                                     do_metrics=True, do_categorized=False)


def evaluate_patches_unified(model_adapter, exp_config, results_path=None,
                              do_metrics=True, do_categorized=None):
    """
    [Unified] 一次 batched inference 同時做兩件事：
      1. (do_metrics=True)     累積 pixel-level stats → 回傳 metrics dict (同 calculate_pixel_level_metrics 格式)
      2. (do_categorized=True) 生 categorized JPG 到 results_path/categorized_predictions/

    取代舊版兩個獨立 function 各跑一遍推論（重複 GPU 工作 ~2x）。

    do_categorized 預設規則：results_path 有給就開、沒給就關。
    若兩者都關，函式相當於 no-op。
    """
    if do_categorized is None:
        do_categorized = (results_path is not None)
    if do_categorized and results_path is None:
        print(f"  [警告] do_categorized=True 但 results_path 未提供，自動關閉 categorized 輸出")
        do_categorized = False

    label = []
    if do_metrics:     label.append("metrics")
    if do_categorized: label.append("categorized")
    label_str = " + ".join(label) if label else "no-op"
    print(f"\n--- [Evaluation] Start: Unified evaluation ({label_str}) ---")

    try:
        architecture = exp_config.get('architecture', 'unknown')
        dataset_cfg = exp_config.get('dataset', {})
        if not dataset_cfg: return {}

        # Mode (binary vs multiclass)
        num_classes = dataset_cfg.get('nc', 1)
        names = dataset_cfg.get('names', [])
        if len(names) > 2: num_classes = len(names)
        is_multiclass = num_classes > 1 and num_classes != 2
        if 'Binary' in exp_config.get('experiment_name', ''): is_multiclass = False

        base_path = Path(dataset_cfg['path'])
        test_img_dir = base_path / dataset_cfg.get('test', 'images/test')

        eval_on_orig_cfg = exp_config.get('evaluation_on_original', {})
        use_original_gt = eval_on_orig_cfg.get('enabled', False)
        original_data_root = eval_on_orig_cfg.get('original_data_root')
        gt_base_path = base_path
        original_img_base_path = base_path
        if use_original_gt and original_data_root:
            gt_base_path = Path(original_data_root)
            original_img_base_path = Path(original_data_root)

        test_label_dir = gt_base_path / 'labels' / test_img_dir.name.replace('images/', '')
        original_test_img_dir = original_img_base_path / 'images' / test_img_dir.name.replace('images/', '')

        if not test_img_dir.is_dir() or not test_label_dir.is_dir():
            print(f"  [警告] 找不到測試資料夾。")
            return {}

        image_files = list(test_img_dir.rglob('*.png')) + list(test_img_dir.rglob('*.jpg'))
        if not image_files: return {}

        # === Setup: categorized output dirs ===
        tp_dir = fp_dir = fn_dir = tn_dir = all_dir = None
        if do_categorized:
            output_base_dir = results_path / "categorized_predictions"
            if output_base_dir.exists(): shutil.rmtree(output_base_dir)
            if is_multiclass:
                all_dir = output_base_dir / "All_Predictions_Colored"
                all_dir.mkdir(parents=True)
            else:
                tp_dir = output_base_dir / "1_True_Positive"
                fp_dir = output_base_dir / "2_False_Positive"
                fn_dir = output_base_dir / "3_False_Negative"
                tn_dir = output_base_dir / "4_True_Negative"
                for d in [tp_dir, fp_dir, fn_dir, tn_dir]: d.mkdir(parents=True)

        # === Setup: metric accumulators ===
        total_cm = None
        total_tp = total_tn = total_fp = total_fn = 0
        if do_metrics:
            if is_multiclass:
                from sklearn.metrics import confusion_matrix
                total_cm = np.zeros((num_classes, num_classes), dtype=np.int64)

        # === Speed config ===
        cat_cfg = exp_config.get('categorized_predictions', {})
        metrics_cfg = exp_config.get('pixel_metrics', {})
        batch_size = int(cat_cfg.get('batch_size',
                          metrics_cfg.get('batch_size',
                           exp_config.get('train', {}).get('batch_size', 4))))
        save_tn = bool(cat_cfg.get('save_tn', True))
        async_writers = int(cat_cfg.get('async_writers', 4))
        val_tf = model_adapter.val_ds.transform if hasattr(model_adapter, 'val_ds') else None
        print(f"  - [Speed] batch_size={batch_size}, "
              f"async_writers={async_writers if do_categorized else 'N/A'}, "
              f"save_tn={save_tn if do_categorized else 'N/A'}, "
              f"val_ds={'ON' if val_tf else 'fallback'}")

        # === Overlay colors (binary categorized) ===
        color_tp = (0, 255, 255)
        color_fp = (0, 0, 255)
        color_fn = (255, 0, 0)
        alpha = exp_config.get('eval_alpha', 0.4)
        beta = 1.0 - alpha

        # === Async write pool ===
        write_pool = ThreadPoolExecutor(max_workers=async_writers) if do_categorized else None
        def _async_write(path, img, quality):
            if write_pool is not None:
                write_pool.submit(cv2.imwrite, str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])

        import time
        start_time = time.time()
        print_interval = max(1, len(image_files) // 10)

        try:
            for batch_start in range(0, len(image_files), batch_size):
                batch_paths = image_files[batch_start:batch_start+batch_size]

                # 1. Load batch + GT
                batch_inputs = []
                batch_meta = []
                for img_path in batch_paths:
                    low_res_img = cv2.imread(str(img_path))
                    if low_res_img is None: continue
                    target_img_to_draw = low_res_img.copy()
                    target_h, target_w = target_img_to_draw.shape[:2]

                    rel_parent = img_path.parent.relative_to(test_img_dir)
                    label_path_base = test_label_dir / rel_parent / img_path.stem

                    if use_original_gt and original_data_root:
                        _orig_cands = list((original_test_img_dir / rel_parent).glob(f"{img_path.stem}.*"))
                        original_img_path = _orig_cands[0] if _orig_cands else None
                        if original_img_path and original_img_path.exists():
                            target_img_to_draw = cv2.imread(str(original_img_path))
                            target_h, target_w = target_img_to_draw.shape[:2]

                    if architecture == 'yolo':
                        gt_mask_binary, has_gt_object, _ = _get_gt_mask(label_path_base, target_h, target_w, architecture)
                        gt_mask_raw = gt_mask_binary
                    else:
                        label_img_path = label_path_base.with_suffix('.png')
                        if label_img_path.exists():
                            gt_mask_raw = cv2.imread(str(label_img_path), cv2.IMREAD_GRAYSCALE)
                            if gt_mask_raw is None:
                                gt_mask_raw = np.zeros((target_h, target_w), dtype=np.uint8)
                        else:
                            gt_mask_raw = np.zeros((target_h, target_w), dtype=np.uint8)
                        has_gt_object = (gt_mask_raw > 128).any()
                        gt_mask_binary = (gt_mask_raw > 128).astype(np.uint8)
                        if gt_mask_raw.shape[:2] != (target_h, target_w):
                            gt_mask_raw = cv2.resize(gt_mask_raw, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                            gt_mask_binary = cv2.resize(gt_mask_binary, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

                    img_rgb = cv2.cvtColor(low_res_img, cv2.COLOR_BGR2RGB)
                    if val_tf is not None:
                        aug = val_tf(image=img_rgb, mask=np.zeros(img_rgb.shape[:2]))
                        inp = aug['image']
                    else:
                        inp = torch.from_numpy(img_rgb.transpose(2, 0, 1)).float() / 255.0
                    batch_inputs.append(inp)
                    batch_meta.append({
                        'img_path': img_path,
                        'target_img': target_img_to_draw,
                        'target_h': target_h, 'target_w': target_w,
                        'gt_mask_raw': gt_mask_raw,
                        'gt_mask_binary': gt_mask_binary,
                        'has_gt_object': has_gt_object,
                    })

                if not batch_inputs: continue

                # 2. Batch forward (single GPU call)
                batch_tensor = torch.stack(batch_inputs).to(model_adapter.device)
                model_adapter.model.eval()
                with torch.no_grad():
                    out = model_adapter.model(batch_tensor)
                    if isinstance(out, dict):
                        # SegFormer 變體常用 'main'；torchvision DeepLab/FCN 用 'out'
                        if 'main' in out:    out = out['main']
                        elif 'out' in out:   out = out['out']
                        elif 'logits' in out: out = out['logits']
                        else: out = next(iter(out.values()))
                    elif hasattr(out, 'logits'):
                        out = out.logits
                    if num_classes > 1:
                        batch_prob = torch.softmax(out, dim=1)
                    else:
                        batch_prob = torch.sigmoid(out)
                    target_h0, target_w0 = batch_meta[0]['target_h'], batch_meta[0]['target_w']
                    if batch_prob.shape[-1] != target_w0 or batch_prob.shape[-2] != target_h0:
                        batch_prob = F.interpolate(batch_prob, size=(target_h0, target_w0),
                                                   mode='bilinear', align_corners=False)
                    batch_prob_np = batch_prob.cpu().numpy()

                # 3. Per-image: accumulate metrics + write categorized JPG
                for b_idx, meta in enumerate(batch_meta):
                    img_path = meta['img_path']
                    target_img_to_draw = meta['target_img']
                    gt_mask_raw = meta['gt_mask_raw']
                    gt_mask_binary = meta['gt_mask_binary']
                    has_gt_object = meta['has_gt_object']
                    target_h, target_w = meta['target_h'], meta['target_w']

                    if target_h != target_h0 or target_w != target_w0:
                        pred_t = F.interpolate(batch_prob[b_idx:b_idx+1],
                                               size=(target_h, target_w),
                                               mode='bilinear', align_corners=False)[0].cpu().numpy()
                    else:
                        pred_t = batch_prob_np[b_idx]

                    if is_multiclass:
                        pred_map = np.argmax(pred_t, axis=0).astype(np.uint8)
                    else:
                        pred_map = (pred_t[0] > 0.5).astype(np.uint8)

                    # === Accumulate metrics ===
                    if do_metrics:
                        if is_multiclass:
                            gt_for_cm = gt_mask_raw
                            if gt_for_cm.shape != pred_map.shape:
                                gt_for_cm = cv2.resize(gt_for_cm, (pred_map.shape[1], pred_map.shape[0]),
                                                       interpolation=cv2.INTER_NEAREST)
                            total_cm += confusion_matrix(
                                gt_for_cm.flatten(), pred_map.flatten(),
                                labels=list(range(num_classes))
                            )
                        else:
                            # [FIX] > 128 確保 land(90) 不被誤判為正 GT
                            total_tp += int(np.sum(np.logical_and(pred_map, gt_mask_binary)))
                            total_tn += int(np.sum(np.logical_and(np.logical_not(pred_map), np.logical_not(gt_mask_binary))))
                            total_fp += int(np.sum(np.logical_and(pred_map, np.logical_not(gt_mask_binary))))
                            total_fn += int(np.sum(np.logical_and(np.logical_not(pred_map), gt_mask_binary)))

                    # === Write categorized JPG ===
                    if do_categorized:
                        if is_multiclass:
                            gt_vis = encode_mask_to_rgb(gt_mask_raw)
                            pred_vis = encode_mask_to_rgb(pred_map)
                            combined = np.hstack([target_img_to_draw, gt_vis, pred_vis])
                            _async_write(all_dir / f"{img_path.stem}_compare.jpg", combined, 80)
                        else:
                            prediction_made = pred_map.sum() > 0
                            temp_overlay = np.zeros_like(target_img_to_draw)
                            temp_overlay[np.logical_and(pred_map, gt_mask_binary)]                = color_tp
                            temp_overlay[np.logical_and(pred_map, np.logical_not(gt_mask_binary))] = color_fp
                            temp_overlay[np.logical_and(np.logical_not(pred_map), gt_mask_binary)] = color_fn
                            if temp_overlay.sum() > 0:
                                overlay_image = cv2.addWeighted(temp_overlay, alpha, target_img_to_draw, beta, 0)
                            else:
                                overlay_image = target_img_to_draw

                            per_image_iou = calculate_iou(pred_map, gt_mask_binary)
                            output_filename = f"{img_path.stem}_iou_{per_image_iou:.4f}.jpg"

                            if prediction_made and has_gt_object:
                                _async_write(tp_dir / output_filename, overlay_image, 80)
                            elif prediction_made and not has_gt_object:
                                _async_write(fp_dir / output_filename, overlay_image, 80)
                            elif not prediction_made and has_gt_object:
                                _async_write(fn_dir / output_filename, overlay_image, 80)
                            else:
                                if save_tn:
                                    _async_write(tn_dir / output_filename, target_img_to_draw, 40)

                # Progress
                done = min(batch_start + batch_size, len(image_files))
                if done >= len(image_files) or (batch_start // batch_size + 1) % print_interval == 0:
                    elapsed = time.time() - start_time
                    eta = (elapsed / done) * (len(image_files) - done) if done else 0
                    print(f"  [Evaluation] Unified: {done}/{len(image_files)} "
                          f"({done*100/len(image_files):.0f}%) | "
                          f"Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s", flush=True)
        finally:
            if write_pool is not None:
                print(f"  [Cleanup] Waiting for background JPEG writes to flush...", flush=True)
                write_pool.shutdown(wait=True)

        if do_categorized:
            print(f"  - Categorized predictions saved.")

        # === Compute final metrics dict ===
        if not do_metrics:
            print(f"--- [Evaluation] End: Unified evaluation (categorized only) ---")
            return {}

        epsilon = 1e-9
        if is_multiclass:
            print("\n--- Multiclass Metrics Report ---")
            intersection = np.diag(total_cm)
            union = total_cm.sum(axis=1) + total_cm.sum(axis=0) - intersection
            ious = intersection.astype(np.float64) / (union.astype(np.float64) + epsilon)
            ious[union == 0] = np.nan
            miou = np.nanmean(ious)
            pixel_accuracy = np.trace(total_cm) / (np.sum(total_cm) + epsilon)
            col_sum = total_cm.sum(axis=0)
            precisions = intersection.astype(np.float64) / (col_sum.astype(np.float64) + epsilon)
            precisions[col_sum == 0] = np.nan
            macro_precision = np.nanmean(precisions)
            row_sum = total_cm.sum(axis=1)
            recalls = intersection.astype(np.float64) / (row_sum.astype(np.float64) + epsilon)
            recalls[row_sum == 0] = np.nan
            macro_recall = np.nanmean(recalls)
            f1s = 2 * (precisions * recalls) / (precisions + recalls + epsilon)
            macro_f1 = np.nanmean(f1s)

            print(f"  - mIoU: {miou:.4f}")
            print(f"  - Pixel Accuracy (Global): {pixel_accuracy:.4f}")
            print(f"  - Macro Precision: {macro_precision:.4f}")
            print(f"  - Macro Recall: {macro_recall:.4f}")
            print(f"  - Macro F1: {macro_f1:.4f}")

            results_dict = {
                'mIoU(pixel)': f"{miou:.4f}",
                'Accuracy(pixel)': f"{pixel_accuracy:.4f}",
                'Precision(pixel)': f"{macro_precision:.4f}",
                'Recall(pixel)': f"{macro_recall:.4f}",
                'F1-score(pixel)': f"{macro_f1:.4f}",
                'IoU(pixel)': "N/A (Multiclass)",
                'IoU_Bg(pixel)': "N/A (Multiclass)"
            }
            for i, iou_v in enumerate(ious):
                class_name = names[i] if i < len(names) else f"Class_{i}"
                val_str = f"{iou_v:.4f}" if not np.isnan(iou_v) else "NaN"
                print(f"    - IoU ({class_name}): {val_str}")
                clean_name = class_name.replace(" ", "_")
                results_dict[f'IoU_{clean_name}(pixel)'] = val_str

            # === confusion matrix：以「Oil 類」做 one-vs-rest 匯出 TP/FP/FN/TN ===
            cm_class_names = [names[i] if i < len(names) else f"Class_{i}"
                              for i in range(num_classes)]
            oil_idx = next((i for i, nm in enumerate(cm_class_names)
                            if 'oil' in str(nm).lower()), 1 if num_classes > 1 else 0)
            tp_c = int(total_cm[oil_idx, oil_idx])
            fp_c = int(total_cm[:, oil_idx].sum() - tp_c)
            fn_c = int(total_cm[oil_idx, :].sum() - tp_c)
            tn_c = int(total_cm.sum() - tp_c - fp_c - fn_c)
            print(f"  - Confusion Matrix (pixel, one-vs-rest on '{cm_class_names[oil_idx]}') | "
                  f"TP={tp_c:,} FP={fp_c:,} FN={fn_c:,} TN={tn_c:,}")
            results_dict.update({
                'CM_positive_class':  cm_class_names[oil_idx],
                'CM_TP(pixel)':       str(tp_c),
                'CM_FP(pixel)':       str(fp_c),
                'CM_FN(pixel)':       str(fn_c),
                'CM_TN(pixel)':       str(tn_c),
            })

            # 完整 NxN confusion matrix 圖 (pixel 紀錄)
            if results_path is not None:
                save_confusion_matrix_figure(
                    total_cm, cm_class_names,
                    Path(results_path) / "confusion_matrix.png",
                    title="Confusion Matrix (pixel-level, Multiclass)")

            print(f"--- [Evaluation] End: Unified ({label_str}) ---")
            return results_dict
        else:
            print("\n--- Binary Metrics Report ---")
            pixel_accuracy   = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + epsilon)
            pixel_iou        = total_tp / (total_tp + total_fp + total_fn + epsilon)
            pixel_iou_bg     = total_tn / (total_tn + total_fn + total_fp + epsilon)
            pixel_mean_iou   = (pixel_iou + pixel_iou_bg) / 2
            pixel_precision  = total_tp / (total_tp + total_fp + epsilon)
            pixel_recall     = total_tp / (total_tp + total_fn + epsilon)
            pixel_f1_score   = 2 * (pixel_precision * pixel_recall) / (pixel_precision + pixel_recall + epsilon)
            print(f"  - Pixel-level Accuracy: {pixel_accuracy:.4f}")
            print(f"  - Pixel-level IoU (Fg): {pixel_iou:.4f}")
            print(f"  - Pixel-level IoU (Bg): {pixel_iou_bg:.4f}")
            print(f"  - Pixel-level mIoU:     {pixel_mean_iou:.4f}")
            print(f"  - Pixel-level Precision: {pixel_precision:.4f}")
            print(f"  - Pixel-level Recall: {pixel_recall:.4f}")
            print(f"  - Pixel-level F1-score: {pixel_f1_score:.4f}")
            print(f"  - Confusion Matrix (pixel) | TP={total_tp:,} FP={total_fp:,} "
                  f"FN={total_fn:,} TN={total_tn:,}")

            # === 產生 confusion matrix 圖 (pixel 紀錄) ===
            if results_path is not None:
                # cm[列=GT, 欄=Pred] = [[TN, FP], [FN, TP]]
                cm_bin = np.array([[total_tn, total_fp],
                                   [total_fn, total_tp]], dtype=np.int64)
                save_confusion_matrix_figure(
                    cm_bin, ['Background', 'Oil'],
                    Path(results_path) / "confusion_matrix.png",
                    title="Confusion Matrix (pixel-level, Binary)")

            print(f"--- [Evaluation] End: Unified ({label_str}) ---")
            return {
                'Accuracy(pixel)':  f"{pixel_accuracy:.4f}",
                'IoU(pixel)':       f"{pixel_iou:.4f}",
                'IoU_Bg(pixel)':    f"{pixel_iou_bg:.4f}",
                'mIoU(pixel)':      f"{pixel_mean_iou:.4f}",
                'Precision(pixel)': f"{pixel_precision:.4f}",
                'Recall(pixel)':    f"{pixel_recall:.4f}",
                'F1-score(pixel)':  f"{pixel_f1_score:.4f}",
                # confusion matrix 原始像素數 (寫入 Excel 最下方)
                'CM_TP(pixel)':     str(total_tp),
                'CM_FP(pixel)':     str(total_fp),
                'CM_FN(pixel)':     str(total_fn),
                'CM_TN(pixel)':     str(total_tn),
            }

    except Exception as e:
        print(f"  [錯誤] Unified evaluation 失敗: {e}")
        import traceback; traceback.print_exc()
        return {}


def evaluate_and_visualize(exp_config, data_yaml_path, model_path, results_path):
    exp_name = exp_config.get('test_name') or exp_config['experiment_name']
    print(f"\n--- [Evaluation] Start: Full evaluation for experiment '{exp_name}' ---")
    try:
        adapter_config = {
            'architecture': exp_config.get('architecture'),
            'base_model': model_path,
            'architecture_cfg': exp_config.get('architecture_cfg', {}),
            'dataset': exp_config.get('dataset', {})
        }
        model_adapter = get_model_adapter(adapter_config)
            
        eval_params = {
            'data': str(data_yaml_path),
            'split': 'test',
            'imgsz': exp_config.get('imgsz', 640),
            'conf': exp_config.get('eval_conf', 0.25),
            'iou': exp_config.get('eval_iou', 0.6)
        }
        # 為了與 YOLOv8 的 .val() 函式相容，需傳入 project 和 name
        eval_params['project'] = str(results_path)
        eval_params['name'] = "standard_evaluation_charts"
        eval_params['exist_ok'] = True
        
        # [Update] Handle custom val() returns (e.g. OSDMamba Multiclass)
        metrics = model_adapter.val(**eval_params)
        
        eval_results = {}
        
        # 1. Check for Multiclass Dictionary
        if hasattr(metrics, 'results_dict'):
            print("\n  - [Info] Merging OSDMamba Multiclass Metrics into Final Results...")
            eval_results.update(metrics.results_dict)
            
        # 2. Check for Standard YOLO box/seg
        if hasattr(metrics, 'box') and metrics.box is not None and getattr(metrics.box, 'map', None) is not None:
            p = getattr(metrics.box, 'mp', 0.0)
            r = getattr(metrics.box, 'mr', 0.0)
            m50 = getattr(metrics.box, 'map50', 0.0)
            m = getattr(metrics.box, 'map', 0.0)
            eval_results.update({'Precision(B)':f"{p:.4f}", 'Recall(B)':f"{r:.4f}", 'mAP50(B)':f"{m50:.4f}", 'mAP50-95(B)':f"{m:.4f}", 'F1-score(B)':f"{2*p*r/(p+r+1e-9):.4f}"})
        
        if hasattr(metrics, 'seg') and metrics.seg is not None and getattr(metrics.seg, 'map', None) is not None:
             p_seg = getattr(metrics.seg, 'mp', 0.0)
             r_seg = getattr(metrics.seg, 'mr', 0.0)
             m50_seg = getattr(metrics.seg, 'map50', 0.0)
             m_seg = getattr(metrics.seg, 'map', 0.0)
             eval_results.update({'Precision(M)':f"{p_seg:.4f}", 'Recall(M)':f"{r_seg:.4f}", 'mAP50(M)':f"{m50_seg:.4f}", 'mAP50-95(M)':f"{m_seg:.4f}", 'F1-score(M)':f"{2*p_seg*r_seg/(p_seg*r_seg+1e-9):.4f}"})


        # 3. Add Pixel Metrics + Categorized Predictions (合併為單一 batched inference，省 ~50% 時間)
        pixel_metrics = evaluate_patches_unified(model_adapter, exp_config, results_path,
                                                  do_metrics=True, do_categorized=True)
        eval_results.update(pixel_metrics)

        recon_config = exp_config.get('reconstruction')
        if recon_config and recon_config.get('enabled'):
            # --- [修改] 呼叫重建 (Reconstruction) 功能 ---
            print(f"\n--- [Evaluation] Start: Reconstruction evaluation for '{exp_name}' ---")
            
            # 獲取 patch (裁切圖) 的測試路徑
            dataset_cfg = exp_config.get('dataset', {});
            base_path = Path(dataset_cfg['path']); 
            test_img_dir = base_path / dataset_cfg.get('test', 'images/test')
            
            # 獲取原始大圖的路徑
            original_data_root = Path(recon_config.get('original_data_root'))
            
            train_cfg = exp_config.get('train', {})
            imgsz_to_use = exp_config.get('imgsz', train_cfg.get('imgsz', 640))

            # 準備視覺化參數 (從 exp_config 讀取，提供預設值)
            vis_params = {
                'min_conf': exp_config.get('eval_conf', 0.25),
                'nms_iou': exp_config.get('eval_iou', 0.6),
                'alpha': recon_config.get('alpha', 0.5), # 可在 yaml 的 reconstruction 下設定 alpha
                
                # --- [FIX 2a] ---
                # 新增：將 'original_patch_size' 從 reconstruction_config 傳遞到 vis_params
                # 如果未在 yaml 中提供，預設為 'imgsz' (保持舊有行為)
                'original_patch_size': recon_config.get('original_patch_size', imgsz_to_use),
                # --- [FIX 2a END] ---
                # [HD95] yaml 開關: reconstruction 下設 compute_hd95: false 可關閉全圖 HD95 (預設開)
                'compute_hd95': recon_config.get('compute_hd95', True),
                # [接線] 以下三項原本沒從 yaml 傳進 run_reconstruction_evaluation → 一律吃預設;
                #         現補上讓 reconstruction block 的設定真正生效。預設值與 reconstruction_module
                #         內部預設完全一致，故「未在 yaml 設定」者行為不變 (不影響舊 DV5 實驗)。
                'gpu_stitch': recon_config.get('gpu_stitch', False),
                'vis_include_original': recon_config.get('vis_include_original', True),
                'vis_max_dim': recon_config.get('vis_max_dim', 4096),
                # [TTA] yaml 開關: reconstruction 下設 tta: 'flip'/'d4' 開啟推論期增強 (預設 'none' = 行為不變)
                'tta': recon_config.get('tta', 'none'),
            }

            recon_metrics = run_reconstruction_evaluation(
                model_adapter=model_adapter,
                test_image_dir=test_img_dir,
                original_data_root=original_data_root,
                results_path=results_path,
                imgsz=imgsz_to_use,
                vis_params=vis_params
            )
            eval_results.update(recon_metrics)
            print(f"--- [Evaluation] End: Reconstruction evaluation for '{exp_name}' ---")
            # --- [修改] 結束 ---
        
        print(f"--- [Evaluation] End: Full evaluation for '{exp_name}' ---")
        return eval_results
    except Exception as e:
        print(f"[Error] An error occurred during evaluation: {e}"); import traceback; traceback.print_exc()
        return {"error": str(e)}
    



def calculate_single_image_advanced_metrics(pred_mask, gt_mask):
    """
    計算單張影像的 HD95、漏報數、誤報數
    """
    # 確保轉為 0/1 的 uint8
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    # --- 1. 計算 HD95 (真 95 百分位, 邊界KDTree; 取代舊的 max(directed_hausdorff)) ---
    # 舊版用全前景像素 + max → 其實是 HD(最大值)、對離群雜點敏感; 改用共用 compute_hd95。
    # 邊界情況由 compute_hd95 統一處理 (兩空→0；一空→該圖對角線, 512patch≈724px)。
    hd95 = compute_hd95(pred_bin, gt_bin)

    # --- 2. 物件級統計 (Object-Level) ---
    # 連通物件分析 (8-connectivity)
    n_gt, labels_gt = cv2.connectedComponents(gt_bin, connectivity=8)
    n_pred, labels_pred = cv2.connectedComponents(pred_bin, connectivity=8)
    
    # 扣掉背景 0
    gt_obj_count = n_gt - 1
    pred_obj_count = n_pred - 1
    
    missed = 0
    false_alarm = 0
    
    # 計算漏報 (Missed): GT 物件沒被 Pred 覆蓋
    for i in range(1, n_gt):
        gt_obj_mask = (labels_gt == i)
        if np.sum(np.logical_and(gt_obj_mask, pred_bin)) == 0:
            missed += 1
            
    # 計算誤報 (False Alarm): Pred 物件沒碰到 GT
    for i in range(1, n_pred):
        pred_obj_mask = (labels_pred == i)
        if np.sum(np.logical_and(pred_obj_mask, gt_bin)) == 0:
            false_alarm += 1
            
    return {
        'HD95': hd95,
        'GT_Count': gt_obj_count,
        'Pred_Count': pred_obj_count,
        'Missed': missed,
        'False_Alarm': false_alarm
    }

def run_advanced_evaluation(model_adapter, dataset_config, save_dir, imgsz=512, gradcam_config=None):
    """
    [修正版] 支援 Patch-level Grad-CAM 與 Input Saliency 分析
    """
    print(f"\n[Advanced Eval] 正在進行 Patch-level 進階評估 | imgsz={imgsz}...")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Grad-CAM 設定 ---
    vis_dir = None
    gradcam_obj = None
    # [Fix] Initialize variables to avoid UnboundLocalError
    enable_saliency = False
    enable_tsne = False
    enable_pca = False
    enable_feature_map = False
    enable_channel_importance = False
    enable_rgb_error_analysis = False
    
    # 預設為 False，除非 config 開啟
    if gradcam_config and gradcam_config.get('enabled', False):
        vis_dir = save_dir / "vis_patch_analysis"
        vis_dir.mkdir(parents=True, exist_ok=True)
        print(f"  [Visual Analysis] Enabled. Saving maps to {vis_dir}")
        
        gradcam_obj = GradCAM(model_adapter.model)
        enable_saliency = gradcam_config.get('saliency_map', False)
        enable_tsne = gradcam_config.get('tsne_analysis', False) # [New] t-SNE 開關
        enable_pca = gradcam_config.get('pca_analysis', False) # [New] PCA 開關
        enable_feature_map = gradcam_config.get('feature_map_analysis', False) # [New] Feature Map 開關
        enable_channel_importance = gradcam_config.get('channel_importance', False) # [New] Channel Importance 開關
        enable_rgb_error_analysis = gradcam_config.get('rgb_error_analysis', False) # [New] RGB Error Analysis 開關

    # --- 資料夾設定 ---
    dataset_root = Path(dataset_config['path'])
    img_dir = dataset_root / 'images' / 'test'
    lbl_dir = dataset_root / 'labels' / 'test'
    
    if not img_dir.exists():
        img_dir = dataset_root / 'images' / 'val'
        lbl_dir = dataset_root / 'labels' / 'val'
    if not img_dir.exists(): return

    image_files = sorted([f for f in img_dir.rglob('*') if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
    results = []
    
    # Transform (給 GradCAM 用)
    # [FIX] 動態獲取 in_channels 和 normalization stats
    in_channels = getattr(model_adapter, 'in_channels', 3)
    
    # 嘗試讀取 dataset_stats.yaml
    stats_file = dataset_root / 'dataset_stats.yaml'
    if stats_file.exists():
        try:
            with open(stats_file, 'r') as f: stats = yaml.safe_load(f)
            mean, std = tuple(stats['mean']), tuple(stats['std'])
            print(f"  [Info] Loaded normalization stats from {stats_file}")
        except:
            mean, std = (0.417, 0.417, 0.417), (0.267, 0.267, 0.267)
    else:
        # Fallback defaults based on channels
        if in_channels == 1: mean, std = (0.417,), (0.267,)
        elif in_channels == 2: mean, std = (0.417, 0.417), (0.267, 0.267)
        else: mean, std = (0.417, 0.417, 0.417), (0.267, 0.267, 0.267)

    # [Fix] correct unpacking of imgsz if list
    if isinstance(imgsz, (list, tuple)): h, w = imgsz[0], imgsz[1]
    else: h, w = imgsz, imgsz

    val_transform = A.Compose([
        A.Resize(h, w),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    # [優化] 改用 DataLoader 進行 Patch-level 的批次推論
    from torch.utils.data import DataLoader
    from .adapters.segformer_adapter_aux import SegmentationDataset
    
    num_classes = getattr(model_adapter.model.config, 'num_labels', 1) if hasattr(model_adapter.model, 'config') else 1

    patched_ds = SegmentationDataset(
        img_dir, lbl_dir, val_transform, 
        in_channels=in_channels, 
        num_classes=num_classes
    )
    eval_batch_size = dataset_config.get('batch_size', 2) # [修改]
    if hasattr(model_adapter, 'config') and model_adapter.config:
        eval_batch_size = model_adapter.config.get('train', {}).get('batch_size', eval_batch_size)
    patched_loader = DataLoader(patched_ds, batch_size=eval_batch_size, shuffle=False, num_workers=4)

    # 取得檔案名稱對應
    img_filenames = [p.name for p in patched_ds.image_files]
    
    model_adapter.model.eval()
    batch_idx = 0
    total_batches = len(patched_loader)
    log_interval = max(1, total_batches // 10)
    
    with torch.no_grad():
        for images, masks in patched_loader:
            images = images.to(model_adapter.device)
            # 在 GPU 上批次推論
            try:
                out = model_adapter.model(pixel_values=images)
            except TypeError:
                out = model_adapter.model(images)
                
            if isinstance(out, dict):
                # SegFormer 變體常用 'main'；torchvision DeepLab/FCN 用 'out'；多種 fallback
                if 'main' in out:    logits = out['main']
                elif 'out' in out:   logits = out['out']
                elif 'logits' in out: logits = out['logits']
                else: logits = next(iter(out.values()))   # 拿第一個 tensor
            elif hasattr(out, 'logits'):
                logits = out.logits
            else:
                logits = out
            
            upsampled = torch.nn.functional.interpolate(logits, size=masks.shape[-2:], mode='bilinear', align_corners=False)
            
            if num_classes > 1:
                preds = torch.argmax(torch.softmax(upsampled, dim=1), dim=1).cpu().numpy() # (B, H, W)
                masks_np = masks.cpu().numpy()
                
                for b in range(preds.shape[0]):
                    # Check Class 1 (Oil)
                    curr_pred = (preds[b] == 1).astype(np.uint8) 
                    curr_mask = (masks_np[b] == 1).astype(np.uint8)
                    filename = img_filenames[batch_idx * eval_batch_size + b]
                    
                    try:
                        metrics = calculate_single_image_advanced_metrics(curr_pred, curr_mask)
                        results.append({'filename': filename, **metrics})
                    except: pass
            else:
                preds = (torch.sigmoid(upsampled) > 0.5).cpu().numpy().astype(np.uint8)
                masks_np = masks.cpu().numpy().astype(np.uint8)
                for b in range(preds.shape[0]):
                    filename = img_filenames[batch_idx * eval_batch_size + b]
                    try:
                        # squeeze is needed for binary output masks of size (1,H,W)
                        p_mask = preds[b][0] if preds.ndim == 4 else preds[b]
                        t_mask = masks_np[b][0] if masks_np.ndim == 4 else masks_np[b]
                        metrics = calculate_single_image_advanced_metrics(p_mask, t_mask)
                        results.append({'filename': filename, **metrics})
                    except Exception as e: pass
                    
            batch_idx += 1
            if batch_idx % log_interval == 0 or batch_idx == total_batches:
                print(f"  Patch-level Eval: {batch_idx}/{total_batches} ({batch_idx * 100 / total_batches:.0f}%)")

    # --- [New] Feature Analysis (t-SNE, PCA, Attention Map, Channel Importance) ---
    if (enable_tsne or enable_pca or enable_feature_map or enable_channel_importance or enable_rgb_error_analysis) and run_tsne_analysis is not None:
        # 建立一個臨時的 DataLoader 供特徵分析使用
        # 為了簡單起見，我們重用 SegmentationDataset 的邏輯，但這裡我們手動構建一個簡單的 loader
        
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, img_files, lbl_dir, transform, in_channels):
                self.img_files = img_files
                self.lbl_dir = lbl_dir
                self.transform = transform
                self.in_channels = in_channels
            def __len__(self): return len(self.img_files)
            def __getitem__(self, idx):
                img_path = self.img_files[idx]
                mask_path = self.lbl_dir / f"{img_path.stem}.png"
                
                img_bgr = cv2.imread(str(img_path))
                if img_bgr is None: return torch.zeros(3, 512, 512), torch.zeros(1, 512, 512)
                
                if self.in_channels == 1: img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                elif self.in_channels == 2: img = img_bgr[:, :, [2, 1]]
                else: img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if mask is None: mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
                mask = (mask > 0).astype(np.float32)
                mask = np.expand_dims(mask, axis=-1) # (H, W, 1)
                
                if self.transform:
                    transformed = self.transform(image=img, mask=mask)
                    img_t = transformed['image']
                    mask_t = transformed['mask'].permute(2, 0, 1) # (1, H, W)
                else:
                    img_t = torch.from_numpy(img).float()
                    mask_t = torch.from_numpy(mask).float()
                return img_t, mask_t

        tsne_dataset = SimpleDataset(image_files, lbl_dir, val_transform, in_channels)
        tsne_loader = torch.utils.data.DataLoader(tsne_dataset, batch_size=1, shuffle=True) # Shuffle to get random samples
        
        # [Config] Analysis Sample Size
        analysis_samples = 10000 # Increased from 3000 to 10000 for better coverage

        if enable_tsne:
            run_tsne_analysis(model_adapter, tsne_loader, save_dir, max_samples=analysis_samples)
        if enable_pca:
            run_pca_analysis(model_adapter, tsne_loader, save_dir, max_samples=analysis_samples)
        if enable_feature_map:
            visualize_feature_maps(model_adapter, tsne_loader, save_dir)
        if enable_channel_importance:
            run_channel_importance_analysis(model_adapter, tsne_loader, save_dir)
        if enable_rgb_error_analysis:
            analyze_rgb_error_distribution(model_adapter, tsne_loader, save_dir)

    if results:
        df = pd.DataFrame(results)
        summary = {}
        for col in df.columns:
            if col == 'filename': summary[col] = 'TOTAL / AVERAGE'
            elif col in ['HD95', 'IoU']: summary[col] = df[col].mean()
            elif col in ['GT_Count', 'Pred_Count', 'Missed', 'False_Alarm']: summary[col] = df[col].sum()
            elif pd.api.types.is_numeric_dtype(df[col]): summary[col] = df[col].mean()
        df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
        df.to_excel(save_dir / "Advanced_Analysis.xlsx", index=False)
        print(f"[Advanced Eval] 完成。")