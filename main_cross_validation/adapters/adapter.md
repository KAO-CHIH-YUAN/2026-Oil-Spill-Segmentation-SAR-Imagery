# Adapter 說明手冊 (Adapter Manual)

## 1. 核心概念：適配器模式 (Adapter Pattern)

本框架的核心設計是「適配器模式」。`main_runner.py` (總控制器) 並不知道如何「具體」訓練一個 SegFormer 或 OSDMamba。

`main_runner.py` 只會對 `training_module.py` 下達一個統一的指令：「根據 `experiments.yaml` 中的 `architecture: 'segformer_deepsup'` 設定，給我一個模型物件，並呼叫它的 `train()` 方法。」

`training_module.py` 中的 `get_model_adapter` 工廠函式，會去 `adapters/` 目錄中尋找**被 `@register_model('segformer_deepsup')` 裝飾器 註冊**的 `SegformerDeepSupAdapter` 類別，並將其回傳。

`Adapter` (適配器) 的職責，就是將**不同模型**（SegFormer, OSDMamba...）的**不同實作細節**，全部**封裝 (Wrap)** 起來，對外提供一組**完全統一的標準介面**。

## 2. 現有適配器列表 (Available Adapters)

本交接包已**精簡至 3 個適配器**，於 `experiments.yaml` 的 `architecture` 欄位指定。
（清單以 `@register_model('...')` 實際註冊的名稱為準。）

### 主力模型
*   **`segformer_deepsup`** ★：SegFormer-B5 + 深監督輔助頭（deep-supervision aux）。**DV5、M4D 二元 / 多類別實驗採用的主力架構**。支援 Binary 與 Multiclass、aux loss 動態排程、EMA、Freeze Epochs、WeightedSampler、進階視覺化等完整訓練技巧（見第 5、6 節）。檔案：`segformer_adapter_deepsup.py`。

### 比較對象（baseline）
*   **`segformer`**：原始 SegFormer-B5（無輔助頭）。作為主力的消融對照。檔案：`segformer_adapter.py`。
*   **`osdmamba_official`**：OSDMamba 官方實作（VMamba / SSM，需 `mamba_ssm` 依賴；缺少時該 adapter 會自動被略過）。檔案：`osdmamba_official_adapter.py`。

> **註 1**：`training_module.py` 以 `_dynamic_import_adapters()`（`glob('adapters/*.py')`）動態載入，因此新增／移除 adapter 只需增刪對應 `.py` 檔，無須改動其他程式。
> **註 2**：本交接包已移除其餘比較用適配器（`deeplabv3+`、`osdmamba`(非官方)、`segformer_aux`、`yolo`、`unet`/`unet++`/`unet_aux`、`paper_unet`、`attention_unet`、`famobileunet`、`rs3mamba`）以精簡體積。若 `experiments.yaml` 指定到這些已移除的名稱會找不到對應模型；需要時請自行補回對應 `.py`。
> **註 3**：RUE 雙分支（`segformer_rue`）亦已停用，連同 `models/RUE/` 與相關腳本移至專案根目錄的 `_legacy_rue/`。

## 3. 標準介面 (Standard Interface)

所有 `adapters/` 目錄下的適配器類別，都必須實作以下標準函式：

* **`__init__(self, config)`**:
    * **職責：** 建構模型。
    * **動作：** 讀取 `config` 字典（來自 `experiments.yaml`），特別是 `architecture_cfg` 和 `base_model`。
    * **來源：** 它會決定是從 `Hugging Face transformers`（SegFormer）載入模型，或是像 `osdmamba_official_adapter.py` 一樣整合外部模型原始碼手動建構。

* **`train(self, data, results_path, **train_params)`**:
    * **職責：** 執行完整的模型訓練。
    * **動作：** 實作完整的 PyTorch 訓練迴圈，包含資料增強、`DataLoader`、`AMP`、`Early Stopping`、儲存權重，並在結束後繪製 Loss/IoU 圖表。
    * **回傳：** 一個包含 `best_model_path` 的字典。

* **`predict(self, source, imgsz, **kwargs)`**:
    * **職責：** 對單一張圖片執行預測。
    * **動作：** 載入圖片、套用 `Normalize` 和 `Resize`、執行 `model.eval()` 和 `model(image)`、將預測結果放大回原始尺寸。
    * **回傳：** 一個**自訂結果物件** (例如 `SegformerPredictionResult`)，該物件必須包含 `.masks`, `.boxes`, `.plot()` 等屬性，以供 `evaluation_module.py` 統一呼叫。
    * **[關鍵] 回傳值規範：** 必須回傳一個自訂結果物件，該物件必須包含：
        *   `pred_mask_binary_np`: 二值化 (0/1) 的遮罩 (供 `evaluation_module` 計算指標使用)。
        *   `pred_mask_prob_np`: 原始機率圖 (0.0~1.0) (供 `reconstruction_module` 進行加權拼貼，這對於消除拼接縫隙至關重要)。
    * **參數 boxes=False:** 適配器應支援此參數以關閉 Bounding Box 生成（沿用早期 YOLO 介面的相容參數；分割模型維持接受但忽略即可）。

* **`val(self, data, split='test', **kwargs)`**:
    * **職責：** 在 `val` 或 `test` 資料集上計算標準指標 (通常是 Pixel IoU)。
    * **動作：** 遍歷 `DataLoader`，累計 IoU。
    * **回傳：** 一個**模仿 YOLO 驗證結果的 `MockMetrics` 物件**，使其具有 `.seg.map50` 屬性，以供 `main_runner.py` 統一記錄。

## 4. 介面相容性設計背景

標準介面（`predict` 回傳含 `.masks`/`.boxes`/`.plot()` 的結果物件、`val` 回傳含 `.seg.map50` 的 `MockMetrics`）沿用早期 YOLO/Ultralytics 的結果格式，讓 `main_runner.py` 與 `evaluation_module.py` 能以統一方式呼叫所有模型。本交接包已移除 YOLO 包裝器（`yolo_adapter.py`），但此相容格式仍保留在現存的 SegFormer / OSDMamba 適配器中，勿更動其回傳結構。

## 5. 進階功能與實作細節

### (1) 多通道輸入 (`in_channels`)
現存 Adapter (`segformer`, `segformer_deepsup`, `osdmamba_official`) 皆支援此功能。
- 讀取 `architecture_cfg.in_channels` 參數（例如 1, 2, 3）。
- `SegmentationDataset` 會根據此參數動態讀取影像：
    *   `1`: 讀取為灰階 (Gray)。
    *   `2`: 讀取為 R/G 雙通道 (通常對應 VV/VH)。
    *   `3`: 讀取為 RGB。
- `__init__` 會將 `in_channels` 參數傳遞給模型建構函式，自動調整第一層卷積的輸入通道數。

### (2) Binary 與 Multiclass 模式 (`nc`)
`segformer_deepsup`（主力）根據 `dataset.nc` 自動切換模式：
- **Binary (nc=1)**：輸出 sigmoid 機率，Loss 使用 Binary 系列，視覺化使用 TP/FP/FN 色彩。
- **Multiclass (nc>1)**：輸出 softmax 機率，Loss 使用 Multiclass 系列，視覺化使用各類別獨立顏色（Oil=Cyan, Lookalike=Red, Ship=Brown, Land=Green）。

Excel 記錄欄位也會依據模式自動切換（Binary 記錄 IoU(Fg)/IoU(Bg)；Multiclass 記錄 mIoU 和 per-class IoU）。

### (3) 輔助損失 (`aux_loss_weight`, `aux_delay_epochs`, `aux_warmup_epochs`)
`segformer_adapter_deepsup.py`（主力）支援此功能（原始 `segformer` 無輔助頭，不適用）。
- **原理**: 在 Encoder 的倒數第二個隱藏層（`hidden_states[-2]`）接一個 `SimpleSegmentationHead`，訓練時計算額外的 Loss，有助於深層網路的梯度傳遞，加速收斂。
- **總 Loss**: `total_loss = loss_main + (current_aux_weight * loss_aux)`
- **動態權重排程（Delay + Warmup）**：
    *   `aux_delay_epochs`（預設 0）：前 N epoch 內 aux weight = 0，讓主幹先學習基礎特徵。
    *   `aux_warmup_epochs`（預設 0）：Delay 結束後，aux weight 從 0 線性爬升到 `aux_loss_weight`，避免突然加入輔助 Loss 造成梯度震盪。

### (4) EMA（Exponential Moving Average）(`use_ema`, `ema_decay`)
`segformer_adapter_deepsup.py`（主力）實作了 EMA 機制（自訂 `ModelEma` class）。
- **原理**：維護一份 model 權重的指數移動平均副本，每個 batch 後更新：`ema_weight = ema_weight * decay + current_weight * (1 - decay)`。
- **訓練期間**：用 EMA 模型進行驗證，以獲得更穩定的 Val IoU 指標。
- **儲存最佳模型**：儲存 EMA 權重（而非當前 epoch 的原始權重），使最終模型更具泛化能力。
- 參數：`use_ema: true`（預設啟用）、`ema_decay: 0.999`。

### (5) Encoder Freeze Epochs (`freeze_epochs`)
`segformer_adapter_deepsup.py`（主力）支援訓練初期凍結 Encoder 的策略。
- `freeze_epochs` 設定凍結的 epoch 數，前 N epoch 凍結 `segformer.parameters()`（Encoder），只訓練 Decoder 和 Aux Head。
- 第 N+1 epoch 自動解凍，開始 Full Fine-tuning。
- 適用於預訓練權重品質高、需要避免早期過擬合的場景。

### (6) 損失函式 (`loss_function`)
Binary 和 Multiclass 模式各支援一套完整的損失函式，透過 `train.loss_function` 參數指定。

**Binary 模式可用損失函式：**
| 名稱 | 組成 |
|------|------|
| `dice` | DiceLoss（預設） |
| `bce` | BCEWithLogitsLoss |
| `focal` | FocalLoss |
| `lovasz` | LovaszLoss |
| `bce_dice` | 0.5×BCE + 0.5×Dice |
| `bce_iou` | 0.5×BCE + 0.5×IoU |
| `bce_iou_ls` | 0.5×BCE(label_smoothing) + 0.5×IoU |
| `focal_iou` | Focal + IoU |
| `focal_lovasz` | Focal + Lovasz |
| `focal_iou_ls` | Focal + IoU + BCE(label_smoothing) |
| `focal_lovasz_ls` | Focal + Lovasz + BCE(label_smoothing) |
| `unified_focal_iou` | 0.5×Focal + 0.5×Tversky(α=0.3,β=0.7) + IoU |
| `unified_focal_lovasz` | 0.5×Focal + 0.5×Tversky + Lovasz |
| `unified_focal_iou_ls` | unified_focal_iou + BCE(label_smoothing) |

**Multiclass 模式可用損失函式：**
| 名稱 | 組成 |
|------|------|
| `dice` | DiceLoss（預設） |
| `ce` | CrossEntropyLoss |
| `focal` | FocalLoss |
| `lovasz` | LovaszLoss |
| `ce_dice` | 0.5×CE + 0.5×Dice |
| `ce_iou` | 0.5×CE + 0.5×IoU |
| `focal_iou` | Focal + IoU |
| `focal_lovasz` | Focal + Lovasz |
| `focal_iou_ls` | Focal + IoU + CE(label_smoothing) |
| `focal_lovasz_ls` | Focal + Lovasz + CE(label_smoothing) |
| `unified_focal_iou` | 0.5×Focal + 0.5×Tversky + IoU |
| `unified_focal_lovasz` | 0.5×Focal + 0.5×Tversky + Lovasz |
| `unified_focal_iou_ls` | unified_focal_iou + CE(label_smoothing) |

**Label Smoothing 相關參數：**
- `label_smoothing`（預設 0.1）：套用 Label Smoothing 的損失函式（`*_ls` 系列）時生效。

**Lovasz 動態 Warmup：**
- `lovasz_warmup_epochs`（預設 10）：前 N epoch Lovasz Loss 權重 = 0，避免訓練初期 loss 不穩定（Lovasz 對早期隨機預測不友善），第 N+1 epoch 起恢復為 1.0。

### (7) 資料擴增 (Data Augmentation)
透過 `train` 區塊的參數動態啟用，基於 `Albumentations` 實作，**影像和遮罩同步變換**。

| 參數 | 說明 |
|------|------|
| `fliplr` | 水平翻轉機率 |
| `flipud` | 垂直翻轉機率 |
| `degrees` | 旋轉角度範圍 |
| `translate` | 位移比例 |
| `scale` | 縮放範圍（`1±scale`） |
| `elastic_transform` | 彈性形變機率 (`border_mode=CONSTANT`) |
| `coarse_dropout` | 隨機遮擋機率（每次 1~8 個方塊，大小約 5%~10% 影像尺寸） |
| `random_brightness_contrast` | 隨機亮度/對比調整機率 |
| `gauss_noise` | 高斯雜訊機率 |
| `speckle_noise` | SAR Speckle Noise 機率（Gamma 分布乘性雜訊，符合 SAR 物理特性） |
| `speckle_variance` | Speckle Noise 強度（預設 0.04） |

### (8) WeightedRandomSampler (`use_weighted_sampler`)
針對正負樣本嚴重不均衡的資料集。
- `use_weighted_sampler: true` 時，先掃描訓練集統計 Positive/Negative 數量（優先讀取 YOLO `.txt` 標註檔加速掃描，若無則讀 PNG Mask）。
- 計算每個樣本的採樣權重（`weight = 1 / class_count`），使每個 epoch 中正負樣本出現機率趨於均等。

### (9) 正規化參數自動偵測（4 層 Fallback）
`_get_normalization_stats()` 依以下優先順序取得 mean/std：
1. `experiments.yaml` 的 `train.normalization` 欄位（最高優先）
2. 訓練結果目錄（`base_model` 的上兩層）中的 `dataset_stats.yaml`（Test Mode 自動沿用訓練時的統計）
3. 資料集根目錄的 `dataset_stats.yaml`
4. Hardcoded 預設值（灰階 mean=0.417, std=0.268）

訓練時會自動呼叫 `calculate_dataset_stats()` 計算並儲存到 `results_path/dataset_stats.yaml`，供後續 Test 時使用。

### (10) 訓練曲線快照
訓練期間每 20 epoch 自動呼叫 `_save_epoch_plot()` 儲存當前 Loss/IoU 曲線快照（`training_curves_epoch_N.png`），方便在訓練中途監控學習狀況，不需等到訓練結束。

### (11) DropPath 與 Classifier Dropout 正則化
- `architecture_cfg.drop_path_rate`（預設 0.1）：MiT Encoder FFN 的 Stochastic Depth 比率，隱式訓練模型集成，提升泛化。
- `architecture_cfg.classifier_dropout_prob`（預設 0.0）：Decoder MLP Head 的 Dropout 機率。

### (12) OSDMamba 支援 (`osdmamba_official_adapter.py`)
- **環境**: 需 `mamba_ssm` / `causal-conv1d` 依賴；缺少時 `_dynamic_import_adapters()` 會捕捉 ImportError 並自動略過此 adapter，不影響其他模型。
- **架構**: 整合 OSDMamba 官方原始碼，封裝為標準分割模型介面。

---

## 6. 進階視覺化與相容性 (Advanced Visualization & Backward Compatibility)

此進階視覺化功能由**主力 `segformer_adapter_deepsup.py`** 實作；`evaluation_module.py` 亦配合支援不同層級的視覺化輸出。（下文以 `SegformerPredictionResult` 泛指此類支援進階繪圖的結果物件。）

### (1) 進階視覺化 (`SegformerPredictionResult`)

*   **功能**: 不僅僅是將 Mask 覆蓋在原圖上 (Overlay)，還會生成一個拼接圖，包含：
    *   **左圖**: 原始影像 + 分割結果 (Overlay)。
    *   **右圖**: RGB 分佈分析圖 (KDE Plot)，用於比較「預測出的油污區域 (Pred Oil)」、「實際油污區域 (GT Oil)」與「全局背景 (Global)」的 RGB 特徵差異。
*   **觸發條件**: 當 `prediction_made` 為 True (有預測) 且使用支援進階繪圖的 SegFormer 適配器（`segformer_deepsup`）時自動觸發。
*   **False Negative (FN) 支援**: 即使模型預測全黑 (漏判)，系統也會嘗試建立一個 `Dummy Result` 物件，利用 Ground Truth 生成右側的 RGB 分析圖，以便開發者分析為何模型漏判 (例如：RGB 分佈是否與背景過於相似)。
*   **Global RGB Stats Cache**: `predict()` 時會自動嘗試載入或計算訓練集的 Oil vs Background RGB 統計（快取於資料集根目錄的 `global_rgb_stats_v1.pkl`），供分析圖的 Global 參考線使用。

### (2) 評估模組的相容性邏輯 (`evaluation_module.py`)

評估模組已更新 (Fix v1.5)，具備自動偵測與向下相容的能力：

1.  **自動偵測 (Auto-detection)**:
    *   檢查 `results[0]` 是否具有 `plot()` 方法、`ground_truth` 屬性或類別名稱包含 `SegformerPredictionResult`。

2.  **分流處理**:
    *   **進階模式 (Advanced Mode)**: 若偵測到支援進階繪圖的 Adapter (如 `segformer_deepsup` 的 `SegformerPredictionResult`)，則直接呼叫 `result.plot(base_image=...)`。這會產出帶有詳細資訊的拼接大圖。
    *   **傳統模式 (Legacy Mode)**: 若使用不支援進階繪圖的 Adapter（如 `osdmamba_official`），其結果物件無 `.plot()`，系統會退回至**手動 OpenCv 繪圖**。這會產出標準的 Mask 疊加圖 (TP=Cyan, FP=Red, FN=Blue)。

### (3) 開發建議

*   若您開發了新的 Adapter 並希望支援進階視覺化，請在回傳的 Result 物件中實作 `.plot(base_image)` 方法，並確保其行為符合預期。
*   若不實作 `.plot()`，系統將自動使用預設的圖層疊加邏輯 (Overlay only)，確保基本功能不受影響。
