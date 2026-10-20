"""
[File] calcuate_dataset_mean_std.py
[Purpose] Calculate Mean and Std for Dataset Normalization.
[Description]
    - Computes global RGB mean and standard deviation from a directory of images.
    - Used to determine normalization parameters for data augmentation.
[Author] Assistant (GitHub Copilot)
"""
import numpy as np
from PIL import Image
from tqdm import tqdm
import glob
import os

# 指向您訓練集圖片的資料夾
image_dir = "/home/yuan/Oil_Project_10-8/dataset/datasetv4/Dv4_SAR_Patch/Patched_P2048_O0_BG100p_Resize512/Dv4_SAR/images/train" 

# 找到所有圖片
image_files = glob.glob(os.path.join(image_dir, "*.png")) # 或其他格式

# 用於累加的變數
# 我們需要計算所有像素的總和，以及所有像素平方的總和
channel_sum = np.zeros(3)
channel_sum_sq = np.zeros(3)
pixel_count = 0

for img_path in tqdm(image_files):
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img) / 255.0 # 將像素值縮放到 [0, 1]
    
    # 確保影像是 HxWx3
    if img_np.ndim == 3:
        h, w, c = img_np.shape
        pixel_count += h * w
        channel_sum += np.sum(img_np, axis=(0, 1))
        channel_sum_sq += np.sum(np.square(img_np), axis=(0, 1))

# 計算最終的 mean 和 std
mean = channel_sum / pixel_count
std = np.sqrt(channel_sum_sq / pixel_count - np.square(mean))

print(f"Calculated Mean: {mean}")
print(f"Calculated Std: {std}")

# 範例輸出 (您的數值會不同):
# Calculated Mean: [0.312, 0.289, 0.312]
# Calculated Std: [0.156, 0.142, 0.156]

"""
Calculated Mean: [0.47704014 0.46236495 0.47704014]
Calculated Std: [0.18484827 0.17782841 0.18484827]
"""