"""
[File] utils.py
[Purpose] General Utility Functions.
[Description]
    - Provides helper functions for YAML handling and path management.
    - Contains miscellaneous tools used across the project modules.
[Author] Assistant (GitHub Copilot)
"""
import yaml
from pathlib import Path

def create_temp_data_yaml(dataset_config, target_dir):
    """在目標資料夾中建立一個暫時的 data.yaml 檔案。"""
    temp_yaml_path = target_dir / 'data.yaml'
    with open(temp_yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(dataset_config, f, allow_unicode=True, sort_keys=False)
    return temp_yaml_path

def get_image_counts(dataset_config, yaml_parent_dir):
    """
    根據 dataset 字典計算 train/val/test 圖片數量。
    """
    try:
        # 獲取資料集路徑
        dataset_path_str = dataset_config.get('path', '')
        if not dataset_path_str:
            return {'train_count': 'Path not specified', 'val_count': 'Path not specified', 'test_count': 'Path not specified'}

        base_path = Path(dataset_path_str)

        # 判斷路徑是絕對還是相對
        if not base_path.is_absolute():
            # 如果是相對路徑，以專案根目錄為基準
            project_root = Path.cwd() 
            base_path = project_root / base_path

        counts = {}
        for split in ['train', 'val', 'test']:
            if split in dataset_config:
                # dataset_config[split] 的值可能是 'images/train'
                image_dir = base_path / dataset_config[split]
                if image_dir.is_dir():
                    # 計算 .jpg 和 .png 檔案的總數
                    num_images = len(list(image_dir.rglob('*.jpg'))) + len(list(image_dir.rglob('*.png')))
                    counts[f'{split}_count'] = num_images
                else:
                    counts[f'{split}_count'] = f'Path not found: {image_dir}'
            else:
                counts[f'{split}_count'] = 'Not specified'
        return counts
    except Exception as e:
        print(f" 計算圖片數量時出錯: {e}")
        return {'train_count': -1, 'val_count': -1, 'test_count': -1}

