# 專案說明手冊 (Project Manual)

本文件說明此深度學習框架的結構、設定與使用方法，適用於 SAR 影像油汙偵測任務。內容涵蓋資料準備、模型訓練、評估與結果記錄的完整流程。

---

## 目錄

0. [環境安裝與驗證](#0-環境安裝與驗證)
1. [專案概觀](#1-專案概觀)
2. [資料準備](#2-資料準備)
   - 2.1 資料集目錄結構
   - 2.2 遮罩格式說明
   - 2.3 雙資料集策略
   - 2.4 裁切工具 patch_method.py（通用）
   - 2.5 困難樣本探勘（HNM）
   - 2.6 SAR 影像 RGB 轉換工具
   - 2.7 困難樣本探勘腳本總覽
3. [支援的模型架構](#3-支援的模型架構)
4. [experiments.yaml 設定參考](#4-experimentsyaml-設定參考)
5. [進階功能](#5-進階功能)
6. [評估與視覺化](#6-評估與視覺化)
7. [典型使用場景](#7-典型使用場景)

---

## 0. 環境安裝與驗證

本專案在 **Python 3.10 + CUDA 12.8（RTX 5090）** 上驗證通過。套件版本鎖定於交接根目錄的 `requirements.txt`。

### 0.1 建立 conda 環境
```bash
# 1) 建立並啟用 Python 3.10 環境
conda create -n yuan_handoff_env_py3.10 python=3.10 -y
conda activate yuan_handoff_env_py3.10
```

### 0.2 安裝 PyTorch

```bash
# 本機 CUDA 12.8；換卡請改對應 cuXXX，或用 CPU 版（下方）
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

# （純 CPU 環境，僅供測試，訓練會很慢）
# pip install torch==2.8.0 torchvision==0.23.0
```

### 0.3 安裝其餘套件
```bash
cd C:\\ 根目錄 # 在交接根目錄（含 requirements.txt 的那層）執行
pip install -r requirements.txt
```


### 0.4 驗證安裝（一鍵自我檢查）
在交接根目錄執行：
```bash
python check_env.py
```
此腳本會依序檢查：
1. **核心套件**是否齊全（並比對驗證版本；版本不同標 `OK*`，非錯誤）。
2. **CUDA** 是否可用（`torch.cuda.is_available()` 與 GPU 名稱）。
3. **模型 adapter** 是否能正確動態註冊。

預期結尾輸出：
```
== 模型 adapter 註冊 ==
  已註冊：['segformer', 'segformer_deepsup']
  [OK ] 主力 adapter segformer / segformer_deepsup 已就緒
  [note] osdmamba_official 未註冊屬正常（需額外安裝 mamba_ssm，非主力流程不需要）
...
結果：環境就緒 ✅ (含 CUDA)
```


---

## 1. 專案概觀

### 1.1 核心模組

| 模組 | 職責 |
|------|------|
| `main_runner.py` | 總控制器：讀取 `experiments.yaml`，依序執行訓練 → 評估 → 記錄 |
| `experiments.yaml` | 實驗設定中心：所有超參數、資料集路徑、模型架構皆在此定義 |
| `training_module.py` | 工廠函式 `get_model_adapter()`：依 `architecture` 名稱動態載入對應 Adapter |
| `adapters/` | 模型包裝器目錄：統一 `train()` / `predict()` / `val()` 介面 |
| `evaluation_module.py` | 像素級指標計算（IoU, F1）與分類視覺化（TP/FP/FN 疊加圖）|
| `reconstruction_module.py` | Patch 預測拼回原始大圖，計算全圖指標 |
| `cross_validation_utils.py` | K-Fold 資料集切分（Symlink 策略，不複製實體檔案）|
| `tracking_module.py` | 將指標寫入 Excel，欄位依 Binary/Multiclass 模式動態調整 |
| `gradcam_utils.py` | Grad-CAM、Saliency Map 視覺化 |
| `feature_analysis.py` | t-SNE、PCA、特徵圖分析 |

### 1.2 執行流程

```
main_runner.py 讀取 experiments.yaml
        │
        ├─[啟用 CV]─→ cross_validation_utils.py 建立 K 份 Symlink 資料集
        │              └─→ 對每一折執行以下流程
        │
        ├─[mode=train]─→ training_module.get_model_adapter()
        │                └─→ adapters/*.py 執行訓練迴圈
        │                     └─→ 儲存 best.pt（EMA 權重）
        │
        └─[post_tests]─→ evaluation_module  (Patch 級指標 + 視覺化)
                      ─→ reconstruction_module (拼回大圖，全圖指標)
                      ─→ gradcam_utils / feature_analysis (進階分析)
                      ─→ tracking_module (寫入 Excel)
```

---

## 2. 資料準備

### 2.1 資料集目錄結構

框架預期的標準目錄結構如下（`dataset.path` 指向根目錄）：

```
dataset/
├── images/
│   ├── train/   ← 訓練圖片（.png 或 .jpg）
│   ├── val/     ← 驗證圖片
│   └── test/    ← 測試圖片
└── labels/
    ├── train/   ← 遮罩（.png）或 YOLO 標籤（.txt）
    ├── val/
    └── test/
```

訓練結束後，`calculate_dataset_stats()` 會自動計算並儲存 `dataset_stats.yaml` 到訓練結果目錄，供後續 Test 模式使用正確的正規化參數。

### 2.2 遮罩格式說明

- **格式**：灰階或 RGB 的 `.png` 檔。
- **Binary 任務**：非零像素 = 油汙（前景）；0 = 背景。
- **Multiclass 任務**：像素值直接對應類別 ID（0=Sea Surface, 1=Oil, 2=Lookalike, 3=Ship, 4=Land）。
- 資料集可另含 YOLO 格式的 `.txt` 標籤檔（純背景圖塊為空白 `.txt`）；框架用它加速 `use_weighted_sampler` 與 HNM 的正負樣本掃描（無則改讀 PNG mask）。

### 2.3 雙資料集策略（推薦）

由於 SAR 原始影像尺寸過大（如 10000×10000），訓練時需先裁切成小圖塊，但評估時應盡量在原始解析度上計算指標。建議維護兩份資料集：

| 資料集 | 用途 | 說明 |
|--------|------|------|
| **訓練用 Patch** | `dataset.path` | 裁切後再縮放，如 512×512 |
| **原始 Patch** | `original_data_root` | 裁切但未縮放，如 1024×1024，供重建與高解析度評估使用 |

### 2.4 裁切工具 `preprocessing/patch_method.py`

將高解析度原始大圖裁切為固定尺寸圖塊。

**主要功能：**

1. **雙軌裁切策略**：
   - **Train/Val**：自適應步長。油汙區域用 `OVERLAP_TRAIN_VAL` 密集採樣（增加正樣本）。
   - **Test**：標準滑動窗口，強制使用 `OVERLAP_TEST` 均勻裁切，確保重建品質。
2. **左上對齊**：所有圖塊統一左上對齊，解決重建時的位移問題。
3. **邊界補白**：邊緣不足尺寸的圖塊自動補白，確保所有圖塊尺寸一致。
4. **正負分離**：設定 `SEPARATE_OUTPUT = True` 可將含目標的圖塊（Positive）與純背景（Negative）分開儲存為 `train_pos/` 和 `train_neg/`，用於後續 HNM。
5. **多進程加速**：使用 `concurrent.futures` 平行裁切。需要按照硬體配置來修正。
6. **座標檔名**：生成檔名包含 `_x{X}_y{Y}`，供 `reconstruction_module` 定位使用。

### 2.5 困難樣本探勘 (HNM)

**腳本位置**：`preprocessing/Hard_Negative_Mining/run_auto_hnm_MaskSplit_EL2N_Submodular_CV.py`

**目標**：SAR 影像中「Lookalike（相似物）」（如海浪、低風速區）常導致嚴重誤報。HNM 篩選出鑑別價值最高的負樣本，建立高品質資料集。本腳本**一次同時完成 5-fold 巢狀切分（nested CV）＋負樣本選樣**，直接輸出各 fold 的訓練資料集。

**以 env 參數化**（不改檔即可切換來源/輸出/配額）：
- `HNM_SOURCE`：close patch 來源（如 `{TAG}_RGB_LandNeutral_close_Split`）；`HNM_OUTPUT`：輸出根目錄。
- `HNM_HARD_ALLOC` / `HNM_COAST_ALLOC` / `HNM_LAND_ALLOC`：負樣本配額（ocean = 1 − 其餘），預設 `0.50 / 0.25 / 0`。

**流程（每個 outer fold，各 phase 皆 idempotent，中斷可重跑續行）：**
- **Phase 0**：依 mask 將負樣本 pre-split 為 land / coast / ocean（`hnm_0_global_staging`，屬暫存，訓練用不到，可於完成後刪除）。
- **Phase 1b**：用全資料訓練一個 EL2N 分類器（`hnm_classifier_full.pt`）。
- **Phase 2**：算 OOF（out-of-fold）EL2N 分數與嵌入，輸出 `global_oof_mined`（含 OOF 視覺化與分數，供稽核，建議保留）。
- **Phase 3**：組裝。負樣本總量 = n_pos × 1.5，分四桶：**hard**（EL2N 最高分，不分類別）先取 `HARD_ALLOC`，其餘依 land/coast/ocean 各跑 **submodular（facility location）** 取代表性樣本，ocean 取餘數。

**輸出**：各 fold 的 `Global_Dataset_Ratio_1_1.5/`。5 個 fold 都產出此資料夾才算成功。


### 2.6 SAR 影像 RGB 轉換工具

**路徑**：`preprocessing/RGB_Transformation/`

針對非 M4D 的原始 SAR 單通道影像（如 DV5 資料集），提供轉換為 3 通道 RGB 格式的腳本。

| 腳本 | 轉換方式 | 說明 |
|------|----------|------|
| `conver_origin_img_to_RGB_LandMasked.py` | pseudo-RGB + 陸地遮罩 | 單通道 SAR → windowed pseudo-RGB，並以 label 遮蔽陸地區域（`RGB_LandNeutral_close` 的基底）|
| `conver_origin_img_to_RGB_LandNeutral_close.py` ★ | pseudo-RGB + 陸地中和(close) | **本研究 Ours 編碼**：整幅一次做 windowed RGB，並將陸地/偽nodata/河流(morph close)→ 海值中和。裁切消融即用此輸出再裁成 2048/1024/512 |


---

### 2.7 困難樣本探勘 (HNM) — 腳本總覽

所有 HNM 腳本集中於 `preprocessing/Hard_Negative_Mining/`：

| 腳本 | 說明 |
|------|------|
| `run_auto_hnm_MaskSplit_EL2N_Submodular_CV.py` | mask land/coast/ocean 切分 + EL2N + Submodular，nested 5-fold，輸出 `Global_Dataset_Ratio_*` |
| `run_auto_hnm_M4D_ourMethod_NoPhase0.py` | **M4D 二元**專用 HNM（EL2N + Submodular，因為固定資料集分布故無 Phase 0）|
| `_hnm_m4d_neutral.py` | M4D HNM 輔助模組（被上者載入）|

---

## 3. 支援的模型架構

所有 Adapter 都遵循統一的 `train()` / `predict()` / `val()` 介面，詳見 `adapters/adapter.md`。

### 3.1 架構總覽


| `architecture` 名稱 | 說明 | 特殊依賴 |
|---------------------|------|----------|
| `segformer_deepsup`  | SegFormer-B5 + 深監督輔助頭 | transformers |
| `segformer` | 原始 SegFormer（無輔助頭） | transformers |
| `osdmamba_official` | OSDMamba 官方實作（VMamba/SSM） | mamba-ssm（缺少時自動略過）|

> **註**：`training_module.py` 以 `_dynamic_import_adapters()`動態載入，清單以 `adapters/*.py` 內 `@register_model('...')` 實際註冊名稱為準；新增模型只需放入對應 `.py`。

### 3.2 SegFormer 系列（主力架構 `segformer_deepsup`）

- 預設強制讀取本地離線模型：`/pretrain/local_model/segformer-b5`（資料夾內需有 `config.json`）。
- 支援 Binary 和 Multiclass 兩種模式（依 `nc` 自動切換）。
- **`segformer_deepsup`** 提供：輔助頭（deep-supervision）、EMA、Freeze Epochs、WeightedSampler 等進階訓練技巧；`segformer`（原版）為無輔助頭的對照。
- 詳細訓練參數見 [4.3 架構專用設定]。

### 3.3 OSDMamba

**環境需求：**
- CUDA 12.4 + PyTorch 2.6.0。
- 安裝：`mamba-ssm`, `causal-conv1d`, `einops`, `mmcv`。缺少時 `_dynamic_import_adapters()` 。
- DataLoader 建議設 `workers: 0` 避免 Pin Memory 錯誤。

---

## 4. `experiments.yaml` 設定參考

以下為一份**完整的範例**（ `segformer_deepsup` + nested 5-fold）。一個檔可放多個 `experiments`，逐一執行。

```yaml
results_base_dir: '/abs/path/result/model/MyExp'   # 所有實驗結果的根目錄
excel_log_path:   '/abs/path/result/excel_log/MyExp.xlsx'  # 指標寫入的 Excel

experiments:
  - experiment_name: 'Segformer_deepsup_Demo'  # 唯一名稱 → 結果子目錄 {timestamp}_{name}
    run: true                     # false 可跳過此實驗（其他仍會跑）
    mode: 'train'                 # 'train' 訓練+測試 / 'test' 只測（需給 base_model）
    architecture: 'segformer_deepsup'   # 見第 3 節（segformer / segformer_deepsup / osdmamba_official）
    base_model: ''                # ''=SegFormer 自動載本地 B5 Encoder；給 .pt 路徑=載該權重(test 必給)
    imgsz: 512                    # 模型輸入尺寸（也需放頂層，供部分流程讀取）

    architecture_cfg:
      in_channels: 3              # 1=SAR灰階 / 2=VV,VH / 3=RGB
      hf_model_name: '/abs/path/pretrain/local_model/segformer-b5'  # 本地 B5(含 config.json)
      drop_path_rate: 0.1         # MiT Encoder Stochastic Depth 正則
      classifier_dropout_prob: 0.0
      # ↓ 深監督輔助頭(segformer_deepsup 專用；原版 segformer 不需要)
      aux_loss_weight: 0.4        # total = loss_main + aux_weight × loss_aux；設 0.0 等同無 aux
      aux_in_channels: 160        # 對應 B5 hidden_states[-2]
      aux_delay_epochs: 10        # 前 N epoch aux_weight=0，讓主幹先學
      aux_warmup_epochs: 30       # delay 後 aux_weight 由 0 線性升到目標

    dataset:
      path: '/unused/placeholder/will-be-overridden-per-fold'  # nested_folds 會逐 fold 覆寫
      nc: 1                       # 類別數：1=Binary(油/背景)，5=Multiclass
      names: ['oil']             # 類別名稱（Multiclass 列全部 5 類）

    train:
      epochs: 70
      imgsz: 512                  # 載入時 resize 到此(1024/512 crop 皆→512)
      batch_size: 12
      lr0: 0.0001                 # AdamW 初始學習率
      weight_decay: 0.02          # AdamW L2
      patience: 20                # Early Stopping 耐心(0=關閉)
      workers: 4                  # DataLoader 執行緒(遇 CUDA/pin 錯誤設 0)
      loss_function: 'focal_iou_ls'   # 完整清單見 adapters/adapter.md 5(6)
      label_smoothing: 0.1        # *_ls 系列生效
      lovasz_warmup_epochs: 10    # 前 N epoch Lovasz 權重=0(用到 lovasz 時)
      # ── 訓練技巧 ──
      amp: true                   # 混合精度
      gradient_accumulation_steps: 4  # 等效放大 batch
      freeze_epochs: 10           # 前 N epoch 凍結 Encoder，第 N+1 解凍
      use_ema: true               # 存 EMA 權重(泛化更穩)
      ema_decay: 0.999
      use_weighted_sampler: true  # 依正負樣本數自動平衡採樣
      # ── 資料擴增(Albumentations，影像/遮罩同步) ──
      degrees: 45                 # 旋轉角度
      translate: 0.3              # 位移比例
      scale: 0.5                  # 縮放 1±scale
      fliplr: 0.5
      flipud: 0.5
      coarse_dropout: 0.1         # 隨機遮擋
      random_brightness: 0.2      # 亮度
      random_contrast: 0.2        # 對比
      speckle_noise: 0.0          # SAR 專用 Gamma 乘性雜訊機率
      speckle_variance: 0.04
      cutmix: 0
      copy_paste: 0.1
      # normalization:            # 省略時 4 層 fallback：train.normalization →
      #   mean: [0.417,0.417,0.417]   # base_model 旁 dataset_stats.yaml → 資料集根 dataset_stats.yaml
      #   std:  [0.268,0.268,0.268]   # → 內建預設(灰階 0.417/0.268)。訓練時會自動算並存 dataset_stats.yaml

    cross_validation:
      enabled: true
      mode: nested_folds          # 用 HNM 預生成的 fold（DV5 主線）；M4D 可改 k_folds
      nested_folds_base_dir: '/abs/path/dataset/.../HNM_..._nested'  # 內含 fold_0/ fold_1/...
      folds_subpath: 'Global_Dataset_Ratio_1_1.5'  # 每個 fold_X 底下實際資料集子路徑
      val_ratio: 0.2              # 從每 fold 的 train 再切出 val 比例
      val_seed: 43               # 控制 train/val 切分(本研究用 42 與 43)
      scene_level_events: ["沉船 1487","疑似油污","艙底水排放","似乎源自石油平台","Taylor 平台","破損管道","天然滲漏點"]  # 同場景不跨 train/val
      # single_fold_index: 3     # 只跑指定 fold(續跑/除錯用)；不設則跑全部 5 fold

    post_tests:                   # 訓練後依序執行的測試任務(可多個)
      - test_name: 'MyTest'
        dataset: { path: '/unused/placeholder/will-be-overridden-per-fold' }
        reconstruction:           # patch 預測拼回整圖，算整圖指標
          enabled: true
          original_data_root: '/unused/placeholder/will-be-overridden-per-fold'
          original_patch_size: 2048   # 原始裁切尺寸(告訴重建放大倍率；2048/1024/512)
          gpu_stitch: true
          compute_hd95: true
        evaluation_on_original: { enabled: true, original_data_root: '/unused/placeholder/will-be-overridden-per-fold' }
        draw_bounding_boxes: false
        grad_cam:                 # 選用：進階特徵分析
          enabled: false
          saliency_map: true
          tsne_analysis: true
          pca_analysis: true
```

**幾個常用變化：**
- **只測不練**：`mode: 'test'` + `base_model: '/.../fold_1/weights/best.pt'`（CV 模式會自動對每個 fold 找 `fold_X/weights/best.pt`）。
- **M4D / 一般 k-fold**：`cross_validation.mode` 省略或用 `k_folds`，加 `k_folds: 5`、`random_state: 42`；設 `val_ratio` 會切換為 Repeated Random Split。
- **Multiclass**：`dataset.nc: 5`、`names` 列 5 類、`architecture_cfg` 加 `ignore_index: 255`（padded 邊緣）。

## 5. 進階功能

### 5.1 交叉驗證 (Cross Validation)

在實驗設定中加入 `cross_validation` 欄位以啟用。

```yaml
cross_validation:
  enabled: true
  k_folds: 5
  random_state: 42
  val_ratio: 0.05    # 選填；設定後切換為 Repeated Random Split 模式
```

**兩種切分策略：**

| 策略 | 觸發條件 | 說明 |
|------|----------|------|
| **Stratified Group K-Fold** | 未設定 `val_ratio` | 同一原始大圖的 Patch 留在同一折（防 Data Leakage）；每折正負比例均衡 |
| **Repeated Group Shuffle Split** | 設定 `val_ratio` | 每次隨機切出固定比例 Val，重複 `k_folds` 次；用於重現固定比例切分的論文實驗(OSDMamba論文) |

- 使用 Symlink，不複製實體檔案，節省磁碟空間。

### 5.2 進階特徵分析（Grad-CAM、t-SNE 等）

在 `post_tests` 的 `grad_cam` 欄位啟用：

```yaml
grad_cam:
  enabled: true
  saliency_map: true       # 各輸入通道（VV, VH）的顯著性圖
  tsne_analysis: true      # 高維特徵 t-SNE 降維視覺化
  pca_analysis: true       # PCA 降維視覺化
  feature_map_analysis: true  # 中間層特徵圖視覺化
  channel_importance: true    # 各輸入通道重要性分析
  rgb_error_analysis: false   # RGB 誤差分析（RGB 影像適用）
```

### 5.3 Excel 結果記錄規則

`tracking_module.py` 依 `nc` 自動切換記錄欄位：

**Binary 模式（nc ≤ 2）：**
- 記錄：`IoU(pixel)`（前景）、`IoU_Bg(pixel)`（背景）、`mIoU(pixel)`、`Precision`、`Recall`、`F1`
- 若啟用重建：`reconstruction_accuracy`、`reconstruction_mean_iou` 等

**Multiclass 模式（nc > 2）：**
- 記錄：`Accuracy(pixel)`、`mIoU(pixel)`、`Macro Precision/Recall/F1`
- Per-Class IoU 自動展開：如 `IoU_Sea_Surface(pixel)`、`IoU_Oil_Spill(pixel)` 等
- 隱藏 Binary 專用欄位（`IoU_Bg`、重建指標）

---

## 6. 評估與視覺化

### 6.1 指標體系

`evaluation_module` 與 `reconstruction_module` 計算以下像素級指標：

| 指標 | 說明 |
|------|------|
| **IoU（前景）** | 僅關注目標類別（最嚴格的單一指標）|
| **IoU（背景）** | 背景類別準確度 |
| **mIoU** | `(IoU_fg + IoU_bg) / 2` |
| **Precision** | `TP / (TP + FP)` |
| **Recall** | `TP / (TP + FN)` |
| **F1-score** | Precision 與 Recall 的調和平均 |
| **HD95** | Hausdorff Distance（95th percentile），邊界誤差 |

### 6.2 視覺化輸出

**標準疊加圖（所有模型）：**

| 顏色 | 含義 |
|------|------|
| 青色（Cyan）| TP：正確偵測 |
| 紅色（Red）| FP：誤報 |
| 藍色（Blue）| FN：漏偵 |

**SegFormer（`segformer_deepsup`）進階視覺化：**
每張預測圖額外生成左右拼接圖：
- **左圖**：原圖 + TP/FP/FN 疊加
- **右圖**：RGB 像素分布 KDE 圖，比較「預測油汙」vs「GT 油汙」vs「全局背景」的特徵分布，輔助分析誤報/漏偵原因

### 6.3 大圖重建流程

```
測試 Patch（512×512）
    ↓ 模型預測 → 機率圖（0.0~1.0）
    ↓ 放大回原始 Patch 尺寸（如 512→1024）
    ↓ stitch_masks()：依座標定位 + Gaussian 加權疊加
    ↓ argmax 取最終類別
完整重建大圖 → 與原始大圖 GT 計算指標
```

---

## 7. 典型使用場景

### 場景 A：SegFormer deepsup 完整訓練

```yaml
- experiment_name: 'SegFormer_DeepSup_Full'
  mode: 'train'
  architecture: 'segformer_deepsup'
  base_model: ''    # 自動從本地 segformer-b5 載入 Encoder 預訓練權重
  architecture_cfg:
    in_channels: 1
    aux_loss_weight: 0.4
    aux_delay_epochs: 5
    aux_warmup_epochs: 10
    drop_path_rate: 0.1
    classifier_dropout_prob: 0.1
  dataset:
    path: '/data/oil_patches_512'
    nc: 1
    names: ['oil']
  train:
    epochs: 100
    imgsz: 512
    batch_size: 8
    lr0: 0.0001
    weight_decay: 0.01
    patience: 20
    loss_function: 'focal_iou'
    amp: true
    gradient_accumulation_steps: 2
    use_ema: true
    ema_decay: 0.999
    freeze_epochs: 5
    use_weighted_sampler: true
    fliplr: 0.5
    flipud: 0.3
    speckle_noise: 0.3
    speckle_variance: 0.04
    coarse_dropout: 0.2
  post_tests:
    - test_name: 'Test_Patches'
      dataset:
        path: '/data/oil_patches_test_512'
      reconstruction:
        enabled: true
        original_data_root: '/data/oil_patches_original_1024'
        original_patch_size: 1024
      evaluation_on_original:
        enabled: true
        original_data_root: '/data/oil_patches_original_1024'
```

### 場景 B：已訓練模型直接測試

```yaml
- experiment_name: 'SegFormer_Test_Only'
  mode: 'test'
  architecture: 'segformer_deepsup'
  base_model: '/results/SegFormer_DeepSup_Full/fold_0/weights/best.pt'
  architecture_cfg:
    in_channels: 1
  dataset:
    path: '/data/oil_patches_512'
    nc: 1
  post_tests:
    - test_name: 'New_Dataset'
      dataset:
        path: '/data/new_test_patches'
      evaluation_on_original:
        enabled: true
        original_data_root: '/data/new_test_original'
```

### 場景 C：5-Fold 交叉驗證

```yaml
- experiment_name: 'SegFormer_5Fold_CV'
  mode: 'train'
  architecture: 'segformer_deepsup'
  base_model: ''
  architecture_cfg:
    in_channels: 1
    aux_loss_weight: 0.4
  dataset:
    path: '/data/oil_patches_512'
    nc: 1
  train:
    epochs: 100
    imgsz: 512
    batch_size: 8
    lr0: 0.0001
    loss_function: 'focal_iou'
    use_ema: true
  cross_validation:
    enabled: true
    k_folds: 5
    random_state: 42
  post_tests:
    - test_name: 'CV_Test'
      dataset:
        path: '/data/oil_patches_test_512'
      reconstruction:
        enabled: true
        original_data_root: '/data/oil_patches_original_1024'
        original_patch_size: 1024
```

### 場景 D：含進階特徵分析

```yaml
- experiment_name: 'SegFormer_With_Analysis'
  mode: 'test'
  architecture: 'segformer_deepsup'
  base_model: '/results/best.pt'
  architecture_cfg:
    in_channels: 1
  dataset:
    path: '/data/oil_patches_512'
    nc: 1
  post_tests:
    - test_name: 'Analysis'
      dataset:
        path: '/data/oil_patches_test_512'
      grad_cam:
        enabled: true
        saliency_map: true
        tsne_analysis: true
        pca_analysis: true
        feature_map_analysis: true
        channel_importance: true
```

---
