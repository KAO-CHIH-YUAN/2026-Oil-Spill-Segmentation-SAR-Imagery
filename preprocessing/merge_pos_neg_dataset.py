"""
[File] merge_pos_neg_dataset.py
[Purpose] Merge Positive and Negative Datasets.
[Description]
    - Merges separate folders of positive (oil spill) and negative (clean sea) samples.
    - Standardizes the directory structure for training.
[Author] Assistant (GitHub Copilot)
"""
import os
import shutil
from tqdm import tqdm

# --- 使用者設定區塊 ---

# 1. 設定您的輸入資料夾路徑 (即包含 'images' 和 'labels' 的上一層目錄)
#    例如圖片中的: .../Patched_P2048_O512_BG100p_Separated/DV4_SAR...
INPUT_DIR = r"/home/yuan/Yuan/OIL_Project_12_7/dataset/DV5_reorder/DV5_SAR_All_Patch/P2048_TrainO512_TestO1024_BG100_Split"

# 2. 設定您的輸出資料夾路徑 (合併後的檔案將存放於此)
OUTPUT_DIR = r"/home/yuan/Yuan/OIL_Project_12_7/dataset/DV5_reorder/DV5_SAR_All_Patch/P2048_TrainO512_TestO1024"

# ----------------------

def merge_pos_neg_folders(input_base, output_base):
    """
    將 dataset 結構中的 *_pos 和 *_neg 資料夾合併為單一資料夾。
    例如: images/train_pos + images/train_neg -> images/train
    """
    
    print(f"--- 開始合併資料集 ---")
    print(f"來源: {os.path.abspath(input_base)}")
    print(f"輸出: {os.path.abspath(output_base)}")
    
    # 定義要處理的子目錄類型
    sub_types = ['images', 'labels']
    
    # 定義要合併的目標名稱 (splits)
    target_splits = ['train', 'val', 'test']
    
    # 定義來源的後綴 (pos 和 neg)
    suffixes = ['_pos', '_neg']

    # 1. 檢查輸入路徑是否有效
    if not os.path.isdir(input_base):
        print(f"錯誤：輸入資料夾不存在: {input_base}")
        return

    # 2. 遍歷 images 和 labels
    for sub_type in sub_types:
        # e.g., input/images
        input_sub_dir = os.path.join(input_base, sub_type)
        
        if not os.path.isdir(input_sub_dir):
            print(f"警告：找不到 '{sub_type}' 資料夾，跳過。")
            continue

        # 3. 遍歷 train, val, test
        for split in target_splits:
            # 建立目標資料夾: e.g., output/images/train
            dest_dir = os.path.join(output_base, sub_type, split)
            os.makedirs(dest_dir, exist_ok=True)
            
            print(f"\n正在處理: {sub_type}/{split} ...")
            
            # 4. 遍歷 _pos 和 _neg 來源資料夾
            for suffix in suffixes:
                # 來源資料夾名稱: e.g., train_pos
                source_folder_name = f"{split}{suffix}"
                source_dir = os.path.join(input_sub_dir, source_folder_name)

                # 檢查來源資料夾是否存在
                if not os.path.isdir(source_dir):
                    print(f"  - 找不到來源資料夾 '{source_folder_name}'，略過。")
                    continue

                # 5. 遞迴收集所有檔案（支援事件子資料夾）
                all_entries = []
                for dirpath, _, filenames in os.walk(source_dir):
                    for filename in filenames:
                        all_entries.append((dirpath, filename))

                if not all_entries:
                    print(f"  - 資料夾 '{source_folder_name}' 是空的。")
                    continue

                # 6. 執行複製，保留事件子資料夾結構
                for dirpath, filename in tqdm(all_entries, desc=f"  複製 {source_folder_name}", unit="file"):
                    rel_dir = os.path.relpath(dirpath, source_dir)
                    # rel_dir == '.' 代表平坦結構，否則為事件子資料夾名稱
                    if rel_dir == '.':
                        target_dir = dest_dir
                    else:
                        target_dir = os.path.join(dest_dir, rel_dir)
                    os.makedirs(target_dir, exist_ok=True)

                    src_file = os.path.join(dirpath, filename)
                    dst_file = os.path.join(target_dir, filename)
                    try:
                        shutil.copy2(src_file, dst_file)
                    except Exception as e:
                        print(f"  [錯誤] 複製檔案 '{filename}' 失敗: {e}")

    print("\n" + "="*40)
    print("--- 合併任務已完成！ ---")
    print(f"新資料集已建立於: {output_base}")
    print("="*40)

if __name__ == "__main__":
    merge_pos_neg_folders(INPUT_DIR, OUTPUT_DIR)