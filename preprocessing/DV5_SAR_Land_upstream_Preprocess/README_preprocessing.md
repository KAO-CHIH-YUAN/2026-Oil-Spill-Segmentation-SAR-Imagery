# SAR 油污前處理 Pipeline 筆記

## 概覽

三支腳本依序執行，將 SAR TIF 轉換成訓練用的 PNG、Mask、YOLO TXT。

```
DV5_All_TIF/
  └── <事件名稱>/
        └── <日期_衛星>/ # 多層資料夾結構
              └── *.tif
              └── *.json   ← 標註 JSON（使用者放置）
```

```
Step 1 ─→ Step 2 ─→ Step 3
TIF        Land        PNG
           JSON    ─→  Mask PNG
           + 標註       YOLO TXT
           JSON
         = 合併 JSON
```

---

## 目錄結構

```
DV5_SAR_Preprocess/
├── DV5_All_TIF/                  ← 原始 TIF（含使用者標註 JSON）
├── osm_cache/                    ← OSM 海岸線快取（約 916 MB，自動下載）
│   └── land-polygons-complete-4326/
│       └── land_polygons.shp
├── Preprocess_Output/
│   ├── Step1_LandJSON/           ← Step 1 輸出：land 多邊形 JSON
│   ├── Step1_Overlay/            ← Step 1 輸出：目視檢查用 JPG
│   ├── Step2_MergedJSON/         ← Step 2 輸出：合併後完整 JSON
│   ├── Step3_PNG/                ← Step 3 輸出：RGB PNG + convert_log.csv
│   └── Step3_Mask/               ← Step 3 輸出：彩色 Mask PNG + YOLO TXT
└── Preprocess/
    ├── pipeline_step1_land_json.py
    ├── pipeline_step2_merge_json.py
    ├── pipeline_step3_png_mask.py
    └── pipeline_step1_overlay.py  ← 獨立 overlay 補跑腳本
```

---

## Step 1 — TIF → Land JSON + Overlay JPG

**腳本**：`pipeline_step1_land_json.py`

### 功能
- 對每個 TIF，從 OSM 海岸線資料查詢對應範圍的陸地多邊形
- 輸出 labelme 格式 JSON（`label = "land"`）
- 同時輸出目視檢查用 JPG（陸地藍色半透明覆蓋）

### 輸入
| 項目 | 路徑 |
|------|------|
| TIF 資料夾 | `DV5_All_TIF/` |
| OSM 海岸線 | 自動下載並快取至 `osm_cache/` |

### 輸出
| 項目 | 路徑 | 格式 |
|------|------|------|
| Land JSON | `Step1_LandJSON/<相對路徑>/<stem>.json` | labelme JSON |
| Overlay JPG | `Step1_Overlay/<相對路徑>/<多層_stem>.jpg` | JPEG quality 50 |

### 關鍵設定
```python
MIN_LAND_PIXELS = 20       # 小於此面積（像素）的陸地多邊形捨棄
OVERLAY_SCALE   = 0.5      # 縮放比例（50%）
OVERLAY_MAXDIM  = 4096     # 最長邊上限（防止超出 JPEG 大小限制）
JPEG_QUALITY    = 50
LAND_COLOR      = (0, 0, 255, 51)  # 藍色 20% 透明度
```

### Land JSON 格式
```json
{
  "version": "5.3.1",
  "flags": {},
  "shapes": [
    {
      "label": "land",
      "points": [[col, row], ...],
      "shape_type": "polygon",
      "imageData": null
    }
  ],
  "imagePath": "file.tif",
  "imageHeight": H,
  "imageWidth": W
}
```

### 執行
```bash
python pipeline_step1_land_json.py
```

---

## Step 2 — 合併 Land JSON + 標註 JSON

**腳本**：`pipeline_step2_merge_json.py`

### 功能
- 以**資料夾**為單位比對（不是檔名），相同相對路徑下的 JSON 視為一對
- 輸出檔名以**標註 JSON 的原始檔名**為主

### 輸入
| 項目 | 路徑說明 |
|------|----------|
| Land JSON | `Step1_LandJSON/`（Step 1 輸出） |
| 標註 JSON | `DV5_All_TIF/`（使用者放置，與 TIF 同目錄） |

### 合併邏輯
| 情況 | 處理方式 |
|------|----------|
| 兩者都有 | 以標註 JSON 為底，移除其中舊 land，加入新 land shapes |
| 只有 land JSON | 直接複製 |
| 只有標註 JSON | 直接複製 |

### 輸出
```
Step2_MergedJSON/<相對路徑>/<標註JSON檔名>.json
```

### 執行
```bash
python pipeline_step2_merge_json.py
```

---

## Step 3 — TIF + JSON → PNG + Mask + YOLO TXT

**腳本**：`pipeline_step3_png_mask.py`

### 功能
- TIF → RGB PNG（全解析度）
- 合併 JSON → 彩色 Mask PNG
- 合併 JSON → YOLO polygon `.txt`（與 Mask 同資料夾）

### 輸出檔名規則

相對路徑所有層級用 `_` 串接：

```
DV5_All_TIF/事件名稱/日期_衛星/file_8b.tif
→ 事件名稱_日期_衛星_file_8b.png / .txt
```

### 輸出
| 項目 | 路徑 |
|------|------|
| PNG | `Step3_PNG/<相對路徑>/<stem>.png` |
| Mask | `Step3_Mask/<相對路徑>/<stem>.png` |
| YOLO TXT | `Step3_Mask/<相對路徑>/<stem>.txt` |
| Log | `Step3_PNG/convert_log.csv` |

### 顏色對應（Mask）
| Label | 顏色 | RGB |
|-------|------|-----|
| `"0"` (oil spill) | 白色 | (255, 255, 255) |
| `"land"` | 綠色 | (0, 153, 0) |
| 背景 | 黑色 | (0, 0, 0) |

### Class ID 對應（YOLO TXT）
| Label | Class ID |
|-------|----------|
| `"0"` (oil spill) | 0 |
| `"land"` | 1 |

### YOLO TXT 格式
```
<class_id> x1/W y1/H x2/W y2/H ... xn/W yn/H
```
座標已正規化至 [0, 1]，polygon 格式（非 bounding box）。

### 開關設定
```python
GENERATE_PNG  = True   # TIF → PNG
GENERATE_MASK = True   # JSON → Mask PNG
GENERATE_TXT  = True   # JSON → YOLO TXT
```
只需補生成某一項時，將其他兩個設為 `False` 即可快速重跑。

### 執行
```bash
python pipeline_step3_png_mask.py
```

---

## 完整執行順序

```bash
# 1. 從 OSM 生成 land JSON + overlay（首次執行會下載 OSM 約需數分鐘）
python pipeline_step1_land_json.py

# 2. 合併 land JSON + 使用者標註 JSON
python pipeline_step2_merge_json.py

# 3. 生成 PNG / Mask / YOLO TXT
python pipeline_step3_png_mask.py
```

---

## 常見情況處理

### Step 3 找不到 JSON（NO_JSON）
- 確認 Step 2 已執行完成
- 確認 `Step2_MergedJSON/` 中對應目錄下有 `.json` 檔
- Step 3 是以**目錄**找 JSON（取第一個），不要求檔名與 TIF 一致

### OSM 快取已存在
- `osm_cache/land-polygons-complete-4326/land_polygons.shp` 存在時自動跳過下載
