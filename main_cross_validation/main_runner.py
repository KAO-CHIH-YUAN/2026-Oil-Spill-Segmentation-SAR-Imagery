# 專案主進入點
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# [FIX] 放寬 OpenCV 讀圖像素上限 (預設 ~2^30)。必須在任何 import cv2 之前設定才生效。
# 否則 fold_4 等含超大原始 SAR 影像的折，在 reconstruction 讀整張原圖時會觸發
# CV_IO_MAX_IMAGE_PIXELS assertion，導致整折評估失敗、指標寫成 NaN。
# 注意：evaluation_module 在子模組 import 鏈中比 reconstruction_module 更早 import cv2，
# 故此設定必須放在 main_runner 最源頭，不能只放 reconstruction_module。
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(pow(2, 40)))

import yaml
from pathlib import Path
import datetime
import copy
import torch
import gc

# 從本地模組導入
from .training_module import train_model
from .evaluation_module import evaluate_and_visualize, run_advanced_evaluation
from .tracking_module import log_to_excel
from .reconstruction_module import run_reconstruction_pipeline
from .training_module import get_model_adapter
from .utils import (
    get_image_counts, create_temp_data_yaml,
)
# 導入 CV 工具
from .cross_validation_utils import prepare_kfold_datasets, materialize_train_val_split

# 取得此腳本所在的目錄
script_dir = Path(__file__).parent


def run_evaluation_job(exp_config, model_path, results_path, excel_path, desired_order, run_timestamp, training_metrics=None):
    """
    執行單個評估任務，包括模型預測、視覺化和結果記錄。
    """
    test_name = exp_config.get('test_name', exp_config.get('experiment_name'))
    print(f"\n--- 開始評估任務: {test_name} ---")
    print(f"--- 使用模型: {model_path} ---")

    # 準備評估用的資料集設定
    eval_dataset_config = exp_config.get('dataset', {})
    if not eval_dataset_config:
        print("   [錯誤] 評估任務未定義 'dataset'。")
        return

    # 為資料集設定提供預設值
    for split in ['train', 'val', 'test']:
        eval_dataset_config.setdefault(split, f'images/{split}')
    eval_dataset_config.setdefault('nc', 1)
    eval_dataset_config.setdefault('names', ['oil'])
    
    # 建立評估結果目錄和臨時資料設定檔
    eval_results_path = results_path
    eval_results_path.mkdir(exist_ok=True, parents=True)
    temp_yaml_path = create_temp_data_yaml(eval_dataset_config, eval_results_path)

    # 準備要記錄到 Excel 的日誌資料
    log_data = {k: v for k, v in exp_config.items() if not isinstance(v, (dict, list))}
    log_data['run_timestamp'] = run_timestamp
    log_data['Experiment_name'] = log_data.pop('experiment_name', 'N/A')
    if 'test_name' in log_data:
        log_data['Test name'] = log_data.pop('test_name')

    # 格式化日誌欄位名稱
    train_params = exp_config.get('train', {})
    log_data['Epochs'] = train_params.get('epochs')
    log_data['Batch size'] = train_params.get('batch_size')
    log_data['Image size'] = train_params.get('imgsz')
    log_data.pop('imgsz', None)  # 確保不與 Image size 重複
    log_data['Patience'] = train_params.get('patience')

    # 載入額外的超參數
    log_data['mosaic'] = train_params.get('mosaic')
    log_data['degrees'] = train_params.get('degrees')
    log_data['translate'] = train_params.get('translate')
    log_data['scale'] = train_params.get('scale')
    log_data['dropout'] = train_params.get('dropout')
    
    # 新增的額外參數
    log_data['Loss_function'] = train_params.get('loss_function')
    log_data['lr0'] = train_params.get('lr0')
    log_data['weight_decay'] = train_params.get('weight_decay')
    log_data['mixup'] = train_params.get('mixup')

    # 合併訓練階段的指標
    if training_metrics:
        log_data.update(training_metrics)

    # 獲取圖片數量並更新日誌
    log_data.update(get_image_counts(eval_dataset_config, eval_results_path))
    log_data['best_model_path'] = str(model_path)
    log_data['Results_folder'] = str(eval_results_path)

    # 執行評估與視覺化
    eval_metrics = evaluate_and_visualize(exp_config, temp_yaml_path, model_path, eval_results_path)
    if eval_metrics:
        log_data.update(eval_metrics)

    # ========================================================
    # 準備模型 Adapter (供後續進階分析與重建使用)
    # ========================================================
    adapter_for_eval = None
    try:
        # 建立 Config，強制指定 'base_model' 為當前評估的模型路徑
        config_for_eval = exp_config.copy()
        config_for_eval['base_model'] = str(model_path) 
        
        # 初始化 Adapter
        adapter_for_eval = get_model_adapter(config_for_eval)
        
        # 取得 imgsz (預設 512)
        imgsz = exp_config.get('train', {}).get('imgsz', 512)

        gradcam_config = exp_config.get('grad_cam', {'enabled': False})
        
    except Exception as e:
        print(f"[Warning] 無法載入模型 Adapter，後續進階分析將跳過: {e}")
        import traceback; traceback.print_exc()

    if adapter_for_eval:
        # --- [2] 呼叫 Patch-level 進階評估 (HD95, Missed, FP) ---
        print("\n>>> 步驟 2/3: 執行 Patch-level 進階分析 (HD95, Missed)...")
        try:
            run_advanced_evaluation(
                model_adapter=adapter_for_eval, 
                dataset_config=exp_config['dataset'], 
                save_dir=results_path,
                imgsz=imgsz,
                gradcam_config=gradcam_config
            )
        except Exception as e:
            print(f"[Warning] Patch-level 進階評估執行失敗: {e}")
            import traceback; traceback.print_exc()

        # --- [3] 呼叫 Reconstruction 進階評估 (若有開啟) ---
        # 檢查 config 中是否有開啟 reconstruction (通常在 post_tests 裡設定)
        if exp_config.get('reconstruction', False):
            print("\n>>> 步驟 3/3: 執行大圖重建與進階分析...")
            recon_cfg = exp_config.get('reconstruction')
            if isinstance(recon_cfg, bool): recon_cfg = {}
            explicit_root = None
            if 'original_data_root' in recon_cfg:
                explicit_root = Path(recon_cfg['original_data_root'])
            elif 'dataset' in exp_config and 'path' in exp_config['dataset']:
                 # 備用：有時候 original root 跟 dataset root 有關
                 pass 

            try:
                run_reconstruction_pipeline(
                    model_adapter=adapter_for_eval,
                    dataset_config=exp_config['dataset'],
                    save_dir=results_path,
                    gradcam_config=gradcam_config,
                    reconstruction_config=recon_cfg,
                    explicit_original_root=explicit_root  
                )
            except Exception as e: print(f"[Err] Recon: {e}")
        else:
            print("\n>>> 步驟 3/3: 跳過大圖重建 (reconstruction: False)")
    # ========================================================
    current_desired_order = []
    
    # 判斷是否為多類別
    try:
        nc = exp_config.get('dataset', {}).get('nc', 1)
        names = exp_config.get('dataset', {}).get('names', [])
        # NC > 2 視為多類別，或者名字列表大於2
        is_multiclass = (nc > 2) or (len(names) > 2)
    except:
        is_multiclass = False

    # 1. 全部欄位排列
    full_order = [
        'Experiment_name', 'training_time_minutes', 'mode', 'Test name', 'Image size',
        'train_count', 'val_count', 'test_count',
        'Precision(B)', 'Recall(B)', 'mAP50(B)', 'mAP50-95(B)', 'F1-score(B)',
        'Precision(M)', 'Recall(M)', 'mAP50(M)', 'mAP50-95(M)', 'F1-score(M)',
        'Precision(pixel)', 'Recall(pixel)', 'F1-score(pixel)', 'Accuracy(pixel)', 
        'IoU(pixel)', 'IoU_Bg(pixel)', 'mIoU(pixel)'
    ]
    
    current_desired_order.extend(full_order)

    # 2. 如果是多類別，在 mIoU(pixel) 後面插入各類別的 IoU
    if is_multiclass:
        dataset_names = exp_config.get('dataset', {}).get('names', [])
        for name in dataset_names:
            clean_name = name.replace(" ", "_")
            key_name = f"IoU_{clean_name}(pixel)"
            if key_name not in current_desired_order:
                current_desired_order.append(key_name)

    # 3. 接續剩餘欄位
    remaining_fields = [
        'reconstruction_f1_score', 'reconstruction_accuracy', 'reconstruction_iou_oil',
        'reconstruction_iou_bg', 'reconstruction_mean_iou', 'reconstruction_hd95', 'architecture', 'run_timestamp',
        'gpu_name', 'Ram_GB', 'original_dataset_path_for_cv', 'Results_folder', 'best_model_path',
        'base_model', 'Epochs', 'Batch size', 'Patience', 'Loss_function', 'degrees', 'translate',
        'scale', 'dropout', 'mosaic', 'mixup', 'lr0', 'weight_decay', 'draw_bounding_boxes'
    ]
    current_desired_order.extend(remaining_fields)

    # 4. Confusion matrix (pixel-level) 原始像素數 —— 固定排在 Excel 最下方
    cm_tail = ['CM_positive_class', 'CM_TP(pixel)', 'CM_FP(pixel)', 'CM_FN(pixel)', 'CM_TN(pixel)']
    current_desired_order.extend([k for k in cm_tail if k not in current_desired_order])

    # 根據預設順序排序日誌並寫入 Excel
    # 使用本次動態產生的 current_desired_order
    ordered_log_data = {key: log_data.get(key) for key in current_desired_order}
    
    # 將剩餘未在列表中的 key 補在後面 (以防遺漏額外資訊)
    ordered_log_data.update({k: v for k, v in log_data.items() if k not in current_desired_order})
    
    log_to_excel(excel_path, ordered_log_data, current_desired_order)


def run_experiment_core(current_exp_config, results_path, excel_path, desired_order, timestamp, completed_experiments_paths):
    """
    封裝核心的實驗流程：訓練 -> 儲存 -> 後處理測試。
    這樣做是為了支援 CV 迴圈可以重複呼叫此邏輯。
    """
    # Note: desired_order 来自外部传入，但在 evaluation job 中会被覆盖成动态列表，所以这里传入旧列表也无妨
    model_to_evaluate = None

    training_metrics = {}

    # --- 訓練或測試模式 ---
    if current_exp_config['mode'] == 'train':
        training_results = train_model(current_exp_config, results_path)
        if training_results and 'best_model_path' in training_results:
            model_to_evaluate = training_results['best_model_path']
            training_metrics = training_results
        else:
            print(f"[錯誤] 實驗 '{current_exp_config['experiment_name']}' 訓練失敗，跳過後續步驟。")
            completed_experiments_paths[current_exp_config['experiment_name']] = None
            return # 訓練失敗直接返回
    
    elif current_exp_config['mode'] == 'test':
        model_to_evaluate = current_exp_config['base_model']
        training_metrics = {} 
        if not model_to_evaluate or not Path(model_to_evaluate).exists():
            print(f"[錯誤] 測試模式下找不到模型: {model_to_evaluate}")
            return
    
    # --- 後處理測試 ---
    if model_to_evaluate:
        post_tests = current_exp_config.get('post_tests', [])
        if not post_tests:
            # 如果沒有定義 post_tests，則對主實驗進行一次評估
            print(f"\n沒有 post_tests, 對主任務 '{current_exp_config['experiment_name']}' 進行評估。")
            run_evaluation_job(current_exp_config, model_to_evaluate, results_path, excel_path, desired_order, timestamp, training_metrics)
        else:
            print(f"\n偵測到 post_tests, 開始執行後續測試任務...")
            for test_job in post_tests:
                if not test_job.get('run', True):
                    continue

                test_timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
                test_results_path = results_path / f"post_test_{test_timestamp}_{test_job['test_name']}"

                # 組合設定：測試任務的設定會覆寫主實驗的設定
                test_run_config = {**current_exp_config, **test_job}
                
                # 自動更新 CV 模式下的資料集路徑 (如果 test 使用的是原訓練集)
                # 這是為了確保 fold 的測試是在該 fold 的 split 上進行
                original_dataset_path = current_exp_config.get('original_dataset_path_for_cv', None)
                if original_dataset_path and test_run_config.get('dataset', {}).get('path') == original_dataset_path:
                     test_run_config['dataset']['path'] = current_exp_config['dataset']['path']
                     print(f"  [CV] Post-test '{test_job['test_name']}' 自動指向 Fold 資料集")

                run_evaluation_job(test_run_config, model_to_evaluate, test_results_path, excel_path, desired_order, test_timestamp, training_metrics)

    # 記錄成功訓練的實驗結果路徑
    if current_exp_config['mode'] == 'train':
        completed_experiments_paths[current_exp_config['experiment_name']] = model_to_evaluate


"""
[File] main_runner.py
[Purpose] Main Entry Point for Cross-Validation Experiments.
[Description]
    - Orchestrates the entire experiment lifecycle: Train -> Test -> Eval.
    - Handles Cross-Validation loops (K-Fold or Random Split).
    - Manages configuration parsing and result logging.
    - Supports dynamic model adapter loading.
[Author] Assistant (GitHub Copilot)
"""
import sys
import argparse

def main():
    """
    專案主進入點函式，負責讀取設定、管理實驗流程。
    """
    parser = argparse.ArgumentParser(description="Run Oil Project Experiments")
    parser.add_argument('--config', type=str, default='experiments.yaml', help='Path to experiment config yaml')
    args = parser.parse_args()

    try:
        if os.path.isabs(args.config):
            config_path = Path(args.config)
        else:
            config_path = script_dir / args.config
            
        print(f"Loading config from: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            master_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"錯誤：找不到設定檔 '{config_path}'！請確認檔案是否存在。")
        return

    results_base_dir = Path(master_config['results_base_dir'])
    excel_path = master_config['excel_log_path']
    completed_experiments_paths = {}

    # 定義 Excel 欄位的理想順序
    desired_order = [
        'Experiment_name', 'training_time_minutes', 'mode', 'Test name', 'Image size',
        'train_count', 'val_count', 'test_count',
        'Precision(B)', 'Recall(B)', 'mAP50(B)', 'mAP50-95(B)', 'F1-score(B)',
        'Precision(M)', 'Recall(M)', 'mAP50(M)', 'mAP50-95(M)', 'F1-score(M)',
        'Precision(pixel)', 'Recall(pixel)', 'F1-score(pixel)', 'Accuracy(pixel)', 
        'IoU(pixel)', 'IoU_Bg(pixel)', 'mIoU(pixel)', 
        'IoU_Sea_Surface(pixel)', 'IoU_Oil_Spill(pixel)', 'IoU_Look-alike(pixel)', 'IoU_Ship(pixel)', 'IoU_Land(pixel)',
        'reconstruction_f1_score', 'reconstruction_accuracy', 'reconstruction_iou_oil',
        'reconstruction_iou_bg', 'reconstruction_mean_iou', 'reconstruction_hd95', 'architecture', 'run_timestamp',
        'gpu_name', 'Ram_GB', 'original_dataset_path_for_cv', 'Results_folder', 'best_model_path',
        'base_model', 'Epochs', 'Batch size', 'Patience', 'Loss_function', 'degrees', 'translate',
        'scale', 'dropout', 'mosaic', 'mixup', 'lr0', 'weight_decay', 'draw_bounding_boxes'
    ]

    for exp_config in master_config.get('experiments', []):
        
        try:
            if not exp_config.get('run', True):
                print(f"\n--- 跳過實驗: {exp_config.get('experiment_name')} (run 設為 False) ---")
                continue

            timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
            
            # 建立單一的主實驗資料夾
            # 例如: .../20231219-100000_ExpName/
            base_results_path = results_base_dir / f"{timestamp}_{exp_config['experiment_name']}"
            base_results_path.mkdir(exist_ok=True, parents=True)

            print("\n" + "="*80)
            print(f"==> 開始處理實驗: {exp_config.get('experiment_name')}")
            print(f"==> 結果路徑: {base_results_path}")

            # 檢查是否啟用 Cross Validation
            cv_options = exp_config.get('cross_validation', {})
            # [Modification] Support CV in both 'train' and 'test' modes
            cv_enabled = cv_options.get('enabled', False)
            
            if cv_enabled:
                cv_mode = cv_options.get('mode', 'kfold_split')
                if cv_mode == 'nested_folds':
                    # 使用 HNM script ENABLE_5FOLD_NESTED_SPLIT 預生成的 fold 目錄
                    # yaml 範例:
                    #   cross_validation:
                    #     enabled: true
                    #     mode: nested_folds
                    #     nested_folds_base_dir: /path/to/HNM_root      # 含 fold_0/ fold_1/ ...
                    #     folds_subpath: Global_Dataset_Ratio_1_1.5     # fold_X/ 底下實際 dataset 子路徑
                    nested_base = Path(cv_options['nested_folds_base_dir'])
                    folds_sub   = cv_options.get('folds_subpath', '')
                    fold_dirs   = sorted([d for d in nested_base.iterdir()
                                          if d.is_dir() and d.name.startswith('fold_')])
                    fold_paths  = [str(d / folds_sub) if folds_sub else str(d) for d in fold_dirs]
                    k_folds     = len(fold_paths)
                    if k_folds == 0:
                        raise RuntimeError(f"[CV nested_folds] 在 {nested_base} 沒找到任何 fold_* 子資料夾")
                    for p in fold_paths:
                        if not Path(p).exists():
                            raise RuntimeError(f"[CV nested_folds] fold dataset 不存在: {p}")
                    print(f"==> [CV模式] 使用 HNM 預生成的 nested_folds (K={k_folds})，base={nested_base}")
                    for i, p in enumerate(fold_paths):
                        print(f"  fold_{i+1}: {p}")
                else:
                    print(f"==> [CV模式] 啟用 Cross Validation (K={cv_options.get('k_folds', 3)})...")
                    k_folds = cv_options.get('k_folds', 3)
                    seed = cv_options.get('random_state', 42)

                    # 準備資料集 (現切現拆)
                    fold_paths = prepare_kfold_datasets(
                        original_dataset_path=exp_config['dataset']['path'],
                        results_base_dir=results_base_dir,
                        k_folds=k_folds,
                        seed=seed,
                        cv_settings=cv_options
                    )
                
                # [CV Test Mode Logic]
                # If testing, 'base_model' in config should point to the PARENT experiment folder
                # e.g., result/model/.../20231219-ExperimentName
                parent_model_dir = None
                if exp_config.get('mode') == 'test':
                    parent_model_dir = Path(exp_config.get('base_model', ''))
                    if not parent_model_dir.exists():
                        print(f"[Warning] CV Test Mode: Base model dir not found: {parent_model_dir}")

                # 迴圈執行每一折
                # 兩種方式控制只跑某一 fold (任一觸發即生效，無需兩個都設)：
                #   1. single_fold_only=True            → 跑 fold_1 (向後相容預設)
                #   2. single_fold_index=N (N>=1)       → 跑 fold_N (任意指定，自動隱含 single 模式)
                #   single_fold_index=0 or 未設 + single_fold_only=False → 跑全部
                single_fold_only  = cv_options.get('single_fold_only', False)
                # 注意：single_fold_index 預設 None (不是 1) 才能區分「user 顯式設 1」vs「沒設」
                _raw_idx = cv_options.get('single_fold_index', None)
                single_fold_index = int(_raw_idx) if _raw_idx is not None else None

                # 判定要跑的 target_fold (1-indexed)：
                # 1. index 顯式設 (>=1) → 用 index 指定的 fold (不管 single_fold_only)
                # 2. 否則 single_fold_only=True → fold_1
                # 3. 都沒設 → 跑全部 (target_fold = None)
                if single_fold_index is not None and single_fold_index >= 1:
                    target_fold = single_fold_index
                elif single_fold_only:
                    target_fold = 1
                else:
                    target_fold = None

                if target_fold is not None:
                    print(f"  >>> [CV模式] 只跑 fold_{target_fold}（其他 fold 會 skip）")

                for i, fold_dataset_path in enumerate(fold_paths):
                    fold_idx_1based = i + 1
                    if target_fold is not None and fold_idx_1based != target_fold:
                        print(f"  >>> [CV模式] skip fold_{fold_idx_1based} (target=fold_{target_fold})")
                        continue

                    fold_name = f"fold_{fold_idx_1based}"
                    print(f"\n  >>> 執行 {fold_name} / {k_folds} <<<")
                    
                    # 建立子資料夾: .../ExpName/fold_1/
                    fold_results_dir = base_results_path / fold_name
                    fold_results_dir.mkdir(exist_ok=True)
                    
                    # 準備 Fold 的設定
                    fold_config = copy.deepcopy(exp_config)
                    # 修改名稱以利 Excel 區分 (ExpName_fold_1)
                    fold_config['experiment_name'] = f"{exp_config['experiment_name']}_{fold_name}"
                    # 記錄原始路徑以便 Post-test 比對
                    fold_config['original_dataset_path_for_cv'] = exp_config['dataset']['path']
                    # 指向 Fold 資料集
                    fold_config['dataset']['path'] = fold_dataset_path
                    # 關閉子實驗的 CV 設定
                    fold_config['cross_validation']['enabled'] = False

                    # Nested folds 模式：HNM 不切 val，這裡用 scene_level_events grouping 切
                    # train → train + val (symlink 旁邊 sibling tree)。post_tests 跟 evaluation_on_original
                    # 也指向同一個新 path (含 val 的版本)。
                    if cv_mode == 'nested_folds':
                        scene_lvl = cv_options.get('scene_level_events', [])
                        val_ratio = float(cv_options.get('val_ratio', 0.1))
                        val_seed  = int(cv_options.get('val_seed',
                                                       cv_options.get('random_state', 42)))
                        manifest_path = fold_results_dir / 'train_val_split_manifest.xlsx'
                        new_path = materialize_train_val_split(
                            Path(fold_dataset_path),
                            scene_level_events=scene_lvl,
                            val_ratio=val_ratio,
                            seed=val_seed,
                            manifest_xlsx=manifest_path,
                        )
                        fold_dataset_path = str(new_path)
                        fold_config['dataset']['path'] = fold_dataset_path
                        for pt in fold_config.get('post_tests', []):
                            if 'dataset' in pt:
                                pt['dataset']['path'] = fold_dataset_path
                            if 'evaluation_on_original' in pt:
                                pt['evaluation_on_original']['original_data_root'] = fold_dataset_path

                    # [CV Test Mode Auto-Resolve]
                    if exp_config.get('mode') == 'test' and parent_model_dir:
                        # Attempt to find weights in: parent_dir / fold_X / weights / best.pt
                        # Or potentially just parent_dir / fold_X / best.pt depending on structure
                        # Standard structure: .../ExpName/fold_1/weights/best.pt
                        candidate_weights = parent_model_dir / fold_name / 'weights' / 'best.pt'
                        if not candidate_weights.exists():
                            # Fallback check
                            candidate_weights = parent_model_dir / fold_name / 'best.pt'
                        
                        if candidate_weights.exists():
                            print(f"  - [CV Test] Auto-detected model weights: {candidate_weights}")
                            fold_config['base_model'] = str(candidate_weights)
                        else:
                            print(f"  - [CV Test Warning] Could not find weights for {fold_name} in {parent_model_dir}")

                    config_save_path = fold_results_dir / 'exp_config.yaml'
                    print(f"[Info] 備份實驗設定至: {config_save_path}")
                    with open(config_save_path, 'w', encoding='utf-8') as f:
                        yaml.dump(fold_config, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
                    
                    # 執行核心流程 (訓練 -> 測試 -> 記錄)
                    run_experiment_core(
                        fold_config, 
                        fold_results_dir, 
                        excel_path, 
                        desired_order, 
                        timestamp, 
                        completed_experiments_paths
                    )
            
            else:
                config_save_path = base_results_path / 'exp_config.yaml'
                print(f"[Info] 備份實驗設定至: {config_save_path}")
                with open(config_save_path, 'w', encoding='utf-8') as f:
                    yaml.dump(exp_config, f, sort_keys=False, allow_unicode=True, default_flow_style=False)

                # 一般模式 (非 CV)
                run_experiment_core(
                    copy.deepcopy(exp_config), 
                    base_results_path, 
                    excel_path, 
                    desired_order, 
                    timestamp, 
                    completed_experiments_paths
                )

        except Exception as e:
            print(f"[嚴重錯誤] 實驗 '{exp_config.get('experiment_name')}' 執行期間發生未預期的錯誤: {e}")
            import traceback
            traceback.print_exc()

        finally:
            print(f"\n--- 完成實驗區塊，正在進行記憶體清理... ---")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n所有已設定的實驗執行完畢！")


if __name__ == '__main__':
    # 為了讓動態導入 `adapters` 正常工作，需要從專案的根目錄來執行
    # 例如: `export PYTHONPATH=$PYTHONPATH:/path/to/your/OIL_PROJECT`
    # 然後執行: `python -m OIL_PROJECT.code_10_7.main.main_runner`
    # export PYTHONPATH=$PYTHONPATH:/home/yuan && nohup /home/yuan/.conda/envs/yuan_oil_env_py3.10/bin/python -u -m Yuan.OIL_Project_12_7.main_cross_validation.main_runner > /home/yuan/Yuan/OIL_Project_12_7/runner.log 2>&1 &
    # export PYTHONPATH=$PYTHONPATH:/home/yuan && nohup /home/yuan/.conda/envs/yuan_mamba_env/bin/python -u -m Yuan.OIL_Project_12_7.main_cross_validation.main_runner > /home/yuan/Yuan/OIL_Project_12_7/runner.log 2>&1 &
    # export PYTHONPATH=$PYTHONPATH:/home/yuan && nohup /home/yuan/.conda/envs/yuan_mamba_env/bin/python -u -m Yuan.OIL_Project_12_7.main_cross_validation.main_runner --config "/home/yuan/Yuan/OIL_Project_12_7/main_cross_validation/experiments_m4d_multiclass.yaml" > /home/yuan/Yuan/OIL_Project_12_7/runner.log 2>&1 &
    # export PYTHONPATH=$PYTHONPATH:/home/yuan && nohup /home/yuan/.conda/envs/yuan_mamba_env/bin/python -u -m Yuan.OIL_Project_12_7.main_cross_validation.main_runner --config "/home/yuan/Yuan/OIL_Project_12_7/main_cross_validation/experiments_m4d_binary.yaml" > /home/yuan/Yuan/OIL_Project_12_7/runner.log 2>&1 &
    # export PYTHONPATH=$PYTHONPATH:/home/yuan && nohup /home/yuan/.conda/envs/yuan_mamba_env/bin/python -u -m Yuan.OIL_Project_12_7.main_cross_validation.main_runner --config "/home/yuan/Yuan/OIL_Project_12_7/main_cross_validation/experiments.yaml" > /home/yuan/Yuan/OIL_Project_12_7/runner.log 2>&1 &
    main()