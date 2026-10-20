"""
[File] training_module.py
[Purpose] Core Training Loop and Model Factory.
[Description]
    - Manages the training and validation loops per epoch.
    - Handles optimizer stepping, mixed precision (AMP), and gradient accumulation.
    - Implements the 'get_model_adapter' factory to load models dynamically.
    - Saves checkpoints and training histories.
[Author] Assistant (GitHub Copilot)
"""
import time, torch, psutil, importlib, pkgutil
import sys
print(f" 目前使用的 Python 路徑: {sys.executable}")
from pathlib import Path
from .utils import create_temp_data_yaml

# 嘗試導入 debug_utils，若檔案不存在則捕獲 ImportError
try:
    from .debug_utils import debug_verify_dataset
except ImportError:
    # 如果檔案不存在，定義一個假的函式以避免崩潰
    def debug_verify_dataset(*args, **kwargs):
        print("[Warning] debug_utils.py not found. Skipping dataset verification.")
        return True

# --- 模型註冊表與工廠函式 ---

MODEL_REGISTRY = {}

def register_model(name):
    """一個裝飾器，用於將模型適配器註冊到 MODEL_REGISTRY。"""
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def _dynamic_import_adapters():
    """動態導入 adapters 目錄下的所有模組以觸發註冊。"""
    print("--- Dynamically importing adapters ---")
    adapters_dir = Path(__file__).parent / 'adapters'
    for filepath in adapters_dir.glob('*.py'):
        if filepath.name.startswith('_'):
            continue
        
        module_name = f'.adapters.{filepath.stem}'
        try:
            importlib.import_module(module_name, package=__package__)
            print(f"  - Successfully imported: {filepath.stem}")
        except ImportError as e:
            print(f"  - Failed to import '{filepath.stem}': {e}")
        except Exception as e:
            print(f"載入 adapter '{filepath.stem}' 時發生未知錯誤: {e}")


def get_model_adapter(config):
    """
    工廠函式：根據設定檔中的 'architecture' 動態取得模型適配器實例。
    """
    # --- 在首次呼叫時，動態載入所有 adapters ---
    if not MODEL_REGISTRY:
        _dynamic_import_adapters()
        print(f"--- Registered Model Adapters: {list(MODEL_REGISTRY.keys())} ---")

    architecture = config.get('architecture')
    if not architecture:
        raise ValueError("設定檔中未指定 'architecture'。")

    adapter_class = MODEL_REGISTRY.get(architecture)
    
    if adapter_class is None:
        raise ValueError(f"找不到名為 '{architecture}' 的模型適配器。請檢查 'architecture' 名稱是否正確，以及是否已在 MODEL_REGISTRY 中註冊。")

    # 傳遞整個 config 字典給適配器的建構函式
    print(f"--- Getting adapter for architecture: '{architecture}' ---")
    return adapter_class(config)

# --- 主訓練函式 ---

def train_model(config, results_path):
    exp_name = config['experiment_name']
    print(f"\n{'='*20} 開始訓練: {exp_name} {'='*20}")
    print(f"結果將儲存於: {results_path}")

    try:
        dataset_config = config.get('dataset', {})
        if not dataset_config:
            print("  [錯誤] 訓練任務未定義 'dataset'。")
            return None
        
        # # --- [DEBUG] 在訓練前驗證資料集 ---
        # if config.get('architecture') == 'yolo':
        #     print("\n--- Running Pre-flight Dataset Verification for YOLO ---")
        #     dataset_root = dataset_config.get('path')
        #     train_images = dataset_config.get('train', 'images/train')
        #     val_images = dataset_config.get('val', 'images/val')
            
        #     # 假設 labels 的子目錄與 images 的子目錄名稱相同
        #     train_labels = Path(train_images).name
        #     val_labels = Path(val_images).name

        #     is_train_ok = debug_verify_dataset(dataset_root, train_images, f"labels/{train_labels}")
        #     is_val_ok = debug_verify_dataset(dataset_root, val_images, f"labels/{val_labels}")

        #     if not is_train_ok or not is_val_ok:
        #         print("[FATAL] Dataset verification failed. Aborting training.")
        #         return None
        #     print("--- Dataset Verification Passed ---\n")
        # # --- [DEBUG] 驗證結束 ---

        dataset_config.setdefault('train', 'images/train')
        dataset_config.setdefault('val', 'images/val')
        dataset_config.setdefault('test', 'images/test')
        dataset_config.setdefault('nc', 1)
        dataset_config.setdefault('names', ['oil'])
        
        temp_yaml_path = create_temp_data_yaml(dataset_config, results_path)

        # --- 使用工廠函式取得模型適配器 ---
        model_adapter = get_model_adapter(config)
        
        start_time = time.time()
        process = psutil.Process()
        initial_ram = process.memory_info().rss
        
        # --- 呼叫適配器的 train 方法 ---
        # 訓練參數現在由 config['train'] 控制
        train_params = config.get('train', {})
        training_output = model_adapter.train(
            data=str(temp_yaml_path),
            results_path=results_path,
            **train_params
        )

        training_time = time.time() - start_time
        final_ram = process.memory_info().rss
        
        # 從適配器回傳的結果中取得模型路徑
        best_model_path_str = training_output.get('best_model_path')
        if not best_model_path_str:
            print("  [錯誤] 訓練結束，但適配器未回傳 'best_model_path'。")
            return None

        best_model_path = Path(best_model_path_str)
        if not best_model_path.exists():
            print(f"  [錯誤] 訓練結束，但找不到最佳模型檔案於: {best_model_path}")
            return None
            
        training_results = {
            'training_time_minutes': f"{training_time / 60:.2f}",
            'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            # --- [FIX 1] ---
            # 修正 RAM 紀錄方式：紀錄當前程序的總 RAM，而不是增量。
            'Ram_GB': f"{final_ram / (1024 ** 3):.2f}",
            # --- [FIX 1 END] ---
            'best_model_path': str(best_model_path)
        }

        # [UPDATE] 將 adapter 回傳的任何額外指標 (如 validation mIoU, class IoU) 也加入 training_results
        # 這樣就能一路被帶到 log_to_excel
        for k, v in training_output.items():
            if k != 'best_model_path': # path 已經被加入過了
                 training_results[k] = v

        print(f"訓練完成！耗時: {training_results['training_time_minutes']} 分鐘")
        return training_results
        
    except Exception as e:
        print(f"訓練過程中發生嚴重錯誤: {e}")
        import traceback; traceback.print_exc()
        return None