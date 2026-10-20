"""
[File] calculate_stats.py
[Purpose] Dataset Statistics Calculation.
[Description]
    - Calculates global mean and standard deviation for datasets.
    - Useful for normalizing input data during training.
    - Iterates through all images in a dataset to compute aggregate stats.
[Author] Assistant (GitHub Copilot)
"""
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob
import os
import sys
import yaml
import argparse
from pathlib import Path

def calculate_dataset_stats(dataset_path, save_yaml=True, output_path=None):
    """
    計算資料集的 Mean 和 Std，並儲存為 dataset_stats.yaml
    """
    dataset_path = Path(dataset_path)
    # 假設訓練圖片在 images/train 或 images
    image_dir = dataset_path / 'images' / 'train'
    if not image_dir.exists():
        image_dir = dataset_path / 'images'
        if not image_dir.exists():
            print(f"[Error] Cannot find images directory in {dataset_path}")
            return None, None

    print(f"Scanning images in: {image_dir}")
    # [FIX] 改用 rglob 遞迴掃描：nested HNM 的結構是 images/train/<event>/<file>.png
    # (多一層 event 子資料夾)，舊版非遞迴 glob("*.png") 只掃第一層 → 掃空 → fallback 預設
    # mean/std (0.417/0.268)，等於從未用真實資料統計做正規化。rglob 同時相容無 event 子層的舊結構。
    # follow symlink：nested HNM 影像是 symlink，需 resolve 後讀。
    image_files = sorted(set(list(image_dir.rglob("*.png")) + list(image_dir.rglob("*.jpg"))))

    if not image_files:
        print("[Error] No images found.")
        return None, None
    print(f"  - [Info] Found {len(image_files)} images (recursive) for stats calculation.")

    # 初始化累加變數
    # 為了通用性，我們先讀取第一張圖片來確定通道數
    first_img = np.array(Image.open(image_files[0]))
    if first_img.ndim == 2:
        channels = 1
    else:
        channels = first_img.shape[2]
    
    print(f"Detected {channels} channels.")
    
    channel_sum = np.zeros(channels)
    channel_sum_sq = np.zeros(channels)
    pixel_count = 0

    print("Calculating Mean and Std...")
    if sys.stdout.isatty():
        iterator = tqdm(image_files)
    else:
        iterator = image_files
    
    log_interval = max(1, int(len(image_files) * 0.2))

    for i, img_path in enumerate(iterator):
        if not sys.stdout.isatty() and (i + 1) % log_interval == 0:
            print(f"  Processing image {i+1}/{len(image_files)} ({((i+1)/len(image_files))*100:.0f}%)")
        try:
            img = Image.open(img_path)
            if channels == 1:
                img = img.convert('L')
            elif channels == 4:
                img = img.convert('RGBA')  # [RGBA] 保留第4通道(texture alpha)
            else:
                img = img.convert('RGB') # 假設多通道是 RGB
                
            img_np = np.array(img) / 255.0 # Normalize to [0, 1]
            
            if channels == 1:
                # (H, W) -> (H, W, 1) for consistent processing
                img_np = np.expand_dims(img_np, axis=-1)
            
            # 確保維度正確
            if img_np.shape[2] != channels:
                print(f"[Warning] Skipping {img_path}: Expected {channels} channels, got {img_np.shape[2]}")
                continue

            h, w, c = img_np.shape
            pixel_count += h * w
            
            # Sum over height and width
            channel_sum += np.sum(img_np, axis=(0, 1))
            channel_sum_sq += np.sum(np.square(img_np), axis=(0, 1))
            
        except Exception as e:
            print(f"[Warning] Error processing {img_path}: {e}")

    if pixel_count == 0:
        print("[Error] No valid pixels processed.")
        return None, None

    # Calculate Mean and Std
    mean = channel_sum / pixel_count
    std = np.sqrt(channel_sum_sq / pixel_count - np.square(mean))

    print("\n" + "="*40)
    print("Results:")
    print(f"Mean: {mean}")
    print(f"Std:  {std}")
    print("="*40)

    if save_yaml:
        stats = {
            'mean': mean.tolist(),
            'std': std.tolist()
        }
        if output_path is None:
            output_path = dataset_path / 'dataset_stats.yaml'
        else:
            output_path = Path(output_path)
            
        with open(output_path, 'w') as f:
            yaml.dump(stats, f)
        print(f"\nStats saved to: {output_path}")
        print("You can now use this dataset for training, and the adapters will automatically load these values.")
    
    return mean.tolist(), std.tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate dataset Mean and Std")
    parser.add_argument('--dataset_path', type=str, required=True, help="Path to the dataset root (containing images/train)")
    args = parser.parse_args()
    
    calculate_dataset_stats(args.dataset_path)
