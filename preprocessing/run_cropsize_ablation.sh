#!/bin/bash
# ============================================================================
# 裁切大小消融 一鍵重現 (DV5 Ours = 整幅 RGB LandNeutral-close，再裁切)
#   整幅 close 編碼(一次) → 各 crop: patch → HNM → segformer_deepsup 訓練+測試 → 清 staging
#
# 三個 crop 皆 resize 512 訓練，唯一變數 = 裁切大小。
# 負樣本配額(見 docs/OPERATION_ORDER_CropSize.md)：
#   2048  = 舊配額 hard0.50 / coast0.15 / land0.15 / ocean0.20 (HNM 預設值)
#   1024  = 新配額 hard0.50 / coast0.25 / land0.00 / ocean0.25
#   512   = 新配額 hard0.50 / coast0.25 / land0.00 / ocean0.25
# (2048 與 1024/512 配額不同：使用者刻意選擇，論文須註明；如要三者一致，把下方
#  crop 2048 那行前面的 export 兩個 ALLOC 打開即可。)
#
# ⚠️ 交接精簡：僅保留 experiments_crop2048_close.yaml；1024/512 yaml 已移除。
#    要完整跑三裁切，請先從 2048 版複製出 experiments_crop{1024,512}_close.yaml
#    （改 cross_validation.nested_folds_base_dir 與 reconstruction.original_patch_size），
#    否則下方 1024/512 的訓練階段會因找不到 yaml 而失敗。
#
# 用法：  cd <handoff_root>/preprocessing && bash run_cropsize_ablation.sh
#   要背景不隨終端關閉：  setsid bash run_cropsize_ablation.sh > MASTER.log 2>&1 < /dev/null & disown
# ============================================================================
set -u
# 交接根：本腳本位於 <root>/preprocessing/ → 上一層即 root
H="$(cd "$(dirname "$0")/.." && pwd)"
PPO=$H/dataset/DV5_SAR_Preprocess_Output
PB=$H/dataset/DV5_SAR_Land_Patch
PP=$H/preprocessing
LOGDIR=$PPO/train_logs
mkdir -p "$LOGDIR"
# conda：本機在 /home/sallylab/miniconda3；換機器請改這行或改用 `conda run`
CONDA="conda run -n yuan_oil_env_py3.10 python -u"
[ -x /home/sallylab/miniconda3/bin/conda ] && CONDA="/home/sallylab/miniconda3/bin/conda run -n yuan_oil_env_py3.10 python -u"
KOUTER=5
FOLDSUB=Global_Dataset_Ratio_1_1.5
log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# ── A) 整幅 RGB LandNeutral close 編碼 (一次，三 crop 共用；已存在會自行跳過) ──
log "== A) 整幅 RGB LandNeutral close 編碼 =="
cd "$PP/RGB_Transformation"
$CONDA conver_origin_img_to_RGB_LandNeutral_close.py > "$PPO/wholeclose.log" 2>&1 || { log "FAIL wholeclose (見 wholeclose.log)"; exit 1; }
NC=$(find "$PPO/Step4_Dataset_RGB_LandNeutral_close/images" -name '*.png' 2>/dev/null | wc -l)
log "整幅 close images=$NC"
[ "$NC" -lt 175 ] && { log "FAIL close <175 ($NC)"; exit 1; }

crop(){  # $1=TAG $2=SIZE $3=OV $4=yaml
  local TAG=$1 SIZE=$2 OV=$3 YAML=$4
  export PATCH_BASE=$PB PATCH_TAG=$TAG PATCH_SIZE=$SIZE PATCH_UNIFORM_OVERLAP=$OV
  local CLOSE=$PB/${TAG}_RGB_LandNeutral_close_Split
  local HNMOUT=$PB/HNM_${TAG}_RGB_LandNeutral_close_nested
  cd "$PP"

  # ① patch (已存在則跳過)
  local NP=$(find "$CLOSE/images" -name '*.png' 2>/dev/null | wc -l)
  if [ "$NP" -lt 1 ]; then
    log "[$TAG] 裁切 close (size=$SIZE ov=$OV)"
    PATCH_CATEGORIES=Step4_Dataset_RGB_LandNeutral_close $CONDA patch_method.py > "$LOGDIR/../patch_${TAG}.log" 2>&1 || { log "FAIL patch $TAG"; return 1; }
    NP=$(find "$CLOSE/images" -name '*.png' 2>/dev/null | wc -l)
  else
    log "[$TAG] 裁切 close 已存在 ($NP)，跳過 patch"
  fi
  log "[$TAG] close patches=$NP"
  [ "$NP" -lt 1 ] && { log "FAIL patch empty $TAG"; return 1; }

  # ② HNM (定 5-fold + EL2N + Submodular；per-phase idempotent，可續跑)
  log "[$TAG] HNM (5-fold)"
  HNM_SOURCE=$CLOSE HNM_OUTPUT=$HNMOUT \
    $CONDA Hard_Negative_Mining/run_auto_hnm_MaskSplit_EL2N_Submodular_CV.py > "$LOGDIR/../hnm_${TAG}.log" 2>&1
  # HNM 會吞例外並 exit 0 → 以「5 個 fold 都有 Global_Dataset_Ratio」判定真正成功
  local NG=$(find "$HNMOUT" -maxdepth 2 -type d -name "$FOLDSUB" 2>/dev/null | wc -l)
  if [ "$NG" -lt "$KOUTER" ]; then log "FAIL HNM $TAG (Global_Dataset_Ratio dirs=$NG < $KOUTER，見 hnm_${TAG}.log)"; return 1; fi
  log "[$TAG] HNM done (folds=$NG)"

  # ③ 訓練 + 測試
  log "[$TAG] TRAIN ($YAML)"
  cd "$H"
  $CONDA -m main_cross_validation.main_runner --config "$YAML" > "$LOGDIR/${YAML}.log" 2>&1
  local RC=$?
  log "[$TAG] TRAIN rc=$RC done"
  [ "$RC" -ne 0 ] && { log "FAIL train $TAG (rc=$RC)"; return 1; }

  # ④ 清理：刪 Phase 0 暫存 (訓練只讀 Global_Dataset_Ratio；保留 OOF 視覺化)
  local FREED=$(du -sch "$HNMOUT"/fold_*/hnm_0_global_staging 2>/dev/null | tail -1 | awk '{print $1}')
  rm -rf "$HNMOUT"/fold_*/hnm_0_global_staging
  log "[$TAG] cleanup: 刪 hnm_0_global_staging (釋放 ~$FREED，保留 OOF)"
}

# ── 2048：舊配額 (不設 ALLOC env → 用 HNM 預設 0.50/0.15/0.15/0.20) ──
( crop P2048_OUni512_BG100 2048 512 experiments_crop2048_close.yaml ) || { log "STOP at 2048"; exit 1; }

# ── 1024 / 512：新配額 (land=0，coast=ocean=0.25) ──
export HNM_HARD_ALLOC=0.50 HNM_COAST_ALLOC=0.25 HNM_LAND_ALLOC=0.00
crop P1024_OUni256_BG100 1024 256 experiments_crop1024_close.yaml || { log "STOP at 1024"; exit 1; }
crop P512_OUni128_BG100  512  128 experiments_crop512_close.yaml  || { log "STOP at 512"; exit 1; }

log "== ALL DONE (2048 + 1024 + 512 建置 + 訓練 + 測試 結束) =="
