# 前處理腳本索引與操作流程 (Preprocessing Index & Operation Order)

部分路徑硬編碼。

## 0. 上游前處理 `DV5_SAR_Land_upstream_Preprocess/`
原始 TIF+JSON → 整幅 Step4_Dataset。
| 腳本 | 用途 |
|---|---|
| ★ `run_upstream_preprocess.py` | 單一入口，依 `_pp_config.py` 連續跑 Step1→4 |
| `pipeline_step1_land_json.py` / `pipeline_step1_overlay.py` | TIF → OSM 陸地多邊形 JSON + overlay |
| `pipeline_step2_merge_json.py` | 合併 land JSON + 標註 JSON |
| `pipeline_step3_png_mask.py` | TIF+JSON → 整幅 PNG / Mask / YOLO TXT |
| `pipeline_step4_build_dataset_onlyDoneOne.py` | 組成 Step4_Dataset（含中途檔名校正）|


## A. DV5 主流程（Ours = 整幅 RGB 陸地中和 close）
| 腳本 | 用途 |
|---|---|
| ★ `RGB_Transformation/conver_origin_img_to_RGB_LandNeutral_close.py` | **Ours 編碼**：整幅一次 windowed pseudo-RGB + 陸地/偽nodata/河流(close)→ 海值中和 |
| `RGB_Transformation/conver_origin_img_to_RGB_LandMasked.py` | 上者的基底（windowed RGB + 陸地遮罩；被 close 版 import）|
| `RGB_Transformation/conver_origin_img_to_RGB.py` | 最基礎的 VV→pseudo-RGB（Kussul 風格）；M4D 的 `m4d_ourmethod_*_fullres.py` 動態載入它 |
| ★ `patch_method.py` | 整幅 close → 依 `PATCH_SIZE`/`PATCH_UNIFORM_OVERLAP`/`PATCH_TAG` 切 patch（2048/1024/512）|
| ★ `Hard_Negative_Mining/run_auto_hnm_MaskSplit_EL2N_Submodular_CV.py` | **DV5 最終 HNM**：mask land/coast/ocean + EL2N + Submodular，nested 5-fold，輸出 `Global_Dataset_Ratio_*`（訓練直接讀這個）|
| `calcuate_dataset_mean_std.py` | 計算 mean/std（正規化）|

> DV5 資料鏈：上游 Step1-4 → 整幅 close 編碼（一次）→ `patch_method.py` → `Hard_Negative_Mining/run_auto_hnm_MaskSplit_EL2N_Submodular_CV.py`。**完整指令見下方 [C 節](#c-dv5-操作流程整幅-close--裁切--hnm--訓練)。**

## B. M4D 流程
| 腳本 | 用途 |
|---|---|
| ★ `m4d_mklab_5class_reorg.py` | 原始 M4D(MKLab) 重組成 pipeline 格式（images / labels_1D）|
| ★ `m4d_ourmethod_rgb_neutral_fullres.py` | M4D 二元 Ours（RGB 陸地中和）原尺寸 |
| ★ `m4d_ourmethod_mc_rgb_fullres.py` | M4D 5 類 Ours（RGBours）原尺寸 |
| `patch_method_m4d_binary.py` | M4D 二元切 patch（分 pos/neg）|
| `patch_method_m4d_multiclass.py` | M4D 5 類 native patch512 |
| `merge_pos_neg_dataset.py` | 合併二元 pos/neg 為標準 train/val/test |
| `Hard_Negative_Mining/run_auto_hnm_M4D_ourMethod_NoPhase0.py` | M4D 二元 HNM（EL2N + Submodular）|
| `Hard_Negative_Mining/_hnm_m4d_neutral.py` | M4D HNM 輔助 |

> 對應 yaml：`experiments_m4d_5class_deepsup_patch.yaml`（多類別）、`experiments_m4d_ourmethod_deepsup.yaml`（二元，含 allData / HNM 兩實驗）。** [D 節](#d-m4d-操作流程)。**

---

## C. DV5 操作流程（整幅 close → 裁切 → HNM → 訓練）


**整體順序**：⓪ 設定 `preprocess_config.yaml` → 跑 `run_upstream_preprocess.py` → ① 跑 RGB close 編碼 → ② 設定 `patch_method.py` 參數後跑 → ③ 設定 HNM 腳本參數後跑 → ④ 跑 `main_runner`。每一步都是「先在檔案裡設定，再執行」。

### ⓪ 上游前處理（原始 TIF → Step4_Dataset）
```bash
cd /home/yuan/Yuan/OIL_Project_12_7/OIL_Project_Handoff/preprocessing/DV5_SAR_Land_upstream_Preprocess
python _pp_config.py                 # (可選) 檢視解析後路徑、確認輸入存在
python run_upstream_preprocess.py --dry   # (可選) 印計畫不實跑
python run_upstream_preprocess.py    # Step1→Step4；只跑子集用 --from N --to M
```
- 所有路徑集中在 `preprocess_config.yaml`（換機器只改 `handoff_root` 一行）。
- 輸出：`$H/dataset/DV5_SAR_Preprocess_Output/Step4_Dataset`（175 整幅）。

### ① 整幅 RGB LandNeutral close 編碼（一次，三裁切共用）
```bash
cd $H/preprocessing/RGB_Transformation
python conver_origin_img_to_RGB_LandNeutral_close.py

在檔案裡設定參數 → 執行。
```

### ② 切 patch — 設定 `patch_method.py` 後執行
打開 `$H/preprocessing/patch_method.py`，在檔案上方設定區（約 48–71 行）確認/修改：

| 變數 | 2048 | 1024 | 512 | 說明 |
|---|---|---|---|---|
| `CATEGORIES` | `["Step4_Dataset_RGB_LandNeutral_close"]` | 同左 | 同左 | 要切的整幅資料夾（＝①的輸出）|
| `PATCH_SIZE` | `2048` | `1024` | `512` | 裁切邊長 |
| `UNIFORM_OVERLAP` | `512` | `256` | `128` | 重疊（stride = SIZE − OVERLAP）|
| `SOURCE_BASE_DIR` / `OUTPUT_BASE_DIR` | 交接包預設 | 同 | 同 | 換機器才需改 |


```bash
cd $H/preprocessing
python patch_method.py
```
- 輸出資料夾自動命名 `P{SIZE}_OUni{OVERLAP}_BG100_RGB_LandNeutral_close_Split`（2048 → `P2048_OUni512_BG100_RGB_LandNeutral_close_Split`），存在 `OUTPUT_BASE_DIR`。

### ③ HNM 篩選 — 設定 `Hard_Negative_Mining/run_auto_hnm_MaskSplit_EL2N_Submodular_CV.py` 後執行
打開該檔，在上方 USER CONFIG 區（約 53–119 行）設定：

| 變數 | 值（以 2048 為例）| 說明 |
|---|---|---|
| `_PATCH_TAG` | `P2048_OUni512_BG100` | tag 名稱（1024/512 改成 `P1024_OUni256_BG100` / `P512_OUni128_BG100`）|
| `SOURCE_PATCH_DIR` | `{_PATCH_BASE}/{_PATCH_TAG}_RGB_LandNeutral_close_Split` | input dataset（＝②的輸出）；**檔案預設是 VV 版 `_Split`，close 流程一定要改成 `_RGB_LandNeutral_close_Split`** |
| `HNM_PIPELINE_BASE_DIR` | `{_PATCH_BASE}/HNM_{_PATCH_TAG}_RGB_LandNeutral_close_nested` | 輸出根（＝訓練 yaml 的 `nested_folds_base_dir`）|
| `HARD_ALLOC` / `COAST_ALLOC` / `LAND_ALLOC` | 見下方配額表 | 負樣本配額（ocean = 1 − 其餘）|



**負樣本抽樣配額（三裁切統一）**：Phase 3 負樣本總量 = n_pos × 1.5，先取 **hard**（EL2N 最高分，不分類別），其餘依 mask land/coast/ocean 各跑 submodular，ocean 取餘數。三裁切一律用同一組配額：

| `HARD_ALLOC` | `COAST_ALLOC` | `LAND_ALLOC` | OCEAN(餘) |
|---|---|---|---|
| 0.50 | 0.25 | 0.00 | 0.25 |

> `LAND_ALLOC=0`：純陸地 patch 不再抽（油汙不會出現在整張皆陸地的圖）；hard 仍不分類別，故 EL2N 很高的純陸地 patch 仍可能入選。腳本內建預設為 `0.50 / 0.15 / 0.15`，請在 USER CONFIG 區把 `COAST_ALLOC / LAND_ALLOC` 改為 `0.25 / 0.00`（或用 env `HNM_COAST_ALLOC=0.25 HNM_LAND_ALLOC=0.00`）。

```bash
cd $H/preprocessing
python Hard_Negative_Mining/run_auto_hnm_MaskSplit_EL2N_Submodular_CV.py
```
- 輸出各 fold 的 `Global_Dataset_Ratio_1_1.5/`。5 個 fold 都產出才算成功。

### ④ 訓練 + 測試
`experiments_crop2048_close.yaml` 設定訓練超參數與測試：
```bash
cd $H
python -m main_cross_validation.main_runner --config experiments_crop2048_close.yaml
```

- 輸出：`$H/result/model/DV5_cropablation/`、`$H/result/excel_log/DV5_cropablation.xlsx`。

> **懶人法（不想逐檔改）**：以上 ②③ 的參數都可用環境變數覆寫，一鍵腳本 `run_cropsize_ablation.sh` 已把三裁切全部串好（見下方）。

### 清理與一鍵腳本
- 訓練確認完成後刪 `HNM_{TAG}_..._nested/fold_*/hnm_0_global_staging`（Phase 0 暫存，~3G/fold；訓練只讀 `Global_Dataset_Ratio`）。**保留** `global_oof_mined`（OOF 稽核）。
- 一鍵：`bash run_cropsize_ablation.sh`（整幅 close 一次 → 依序三裁切 patch→HNM→train→清 staging）。背景執行：
  `setsid bash run_cropsize_ablation.sh > MASTER.log 2>&1 < /dev/null & disown`
  > 若要三裁切統一用上表配額（`0.50/0.25/0.00/0.25`），請確認腳本頂部三個 crop 都 `export HNM_COAST_ALLOC=0.25 HNM_LAND_ALLOC=0.00`。
- HNM 每個 phase idempotent（輸出存在即跳過），中斷可直接重跑續行；成功判定＝每 crop 5 個 fold 都產出 `Global_Dataset_Ratio_1_1.5`。


---

## D. M4D 操作流程

M4D = 公開資料集 MKLab / Krestenitis 2019；採用與 DV5 相同主力 `segformer_deepsup`（影像編碼 Ours = VV→RGB + 陸地中和）。路徑相對 `$H/dataset/M4Ds/`。

### ① 重組原始 M4D
```bash
cd $H
python preprocessing/m4d_mklab_5class_reorg.py
#   → dataset/M4Ds/M4D/{train,test}/images/ + labels_1D/ (5-class ID mask 0–4)
```

### ② 影像編碼（Ours = VV→RGB + 陸地中和），原尺寸中間產物
```bash
# 二元：RGB 陸地中和 原尺寸 + binary label
python preprocessing/m4d_ourmethod_rgb_neutral_fullres.py

# 5 類別：RGBours 原尺寸(1250×650) + 5-class label（不改陸地=class4）
python preprocessing/m4d_ourmethod_mc_rgb_fullres.py
```

### ③ 切 patch
```bash
# 二元（切 patch，分 pos/neg）
python preprocessing/patch_method_m4d_binary.py
python preprocessing/merge_pos_neg_dataset.py      # 合併 pos/neg → 標準 train/val/test

# 5 類別（native patch512：512 patch, oil overlap256）
python preprocessing/patch_method_m4d_multiclass.py
```
> ★ 5 類別重點：padded 邊緣區 label = **255 (ignore_index)**；`segformer_deepsup` 的 multiclass label loader 會偵測原生 class ID（0–4）並套用 ignore_index，不可用 DV5 的灰階 remap。
