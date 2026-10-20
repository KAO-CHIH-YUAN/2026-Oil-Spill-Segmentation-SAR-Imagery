#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
環境自我檢查：確認套件、CUDA、以及模型 adapter 能否正確載入。
用法（在交接根目錄）：  python check_env.py
全部顯示 [OK] 即代表環境安裝正確，可以開始訓練。
"""
import importlib
import sys

# import 名稱 -> 期望版本（僅供對照，版本不同不視為失敗）
PKGS = [
    ("torch", "2.8.0"), ("torchvision", "0.23.0"), ("transformers", "4.57.1"),
    ("timm", "1.0.20"), ("segmentation_models_pytorch", "0.5.0"),
    ("numpy", "2.2.6"), ("pandas", "2.3.3"), ("scipy", "1.15.3"),
    ("sklearn", "1.7.2"), ("skimage", "0.25.2"), ("cv2", "4.12.0"),
    ("PIL", "11.3.0"), ("albumentations", "2.0.8"), ("matplotlib", "3.10.6"),
    ("seaborn", "0.13.2"), ("openpyxl", "3.1.5"), ("yaml", "6.0.2"),
    ("tqdm", "4.67.1"), ("psutil", "7.0.0"), ("requests", "2.32.5"),
]
# 僅上游前處理 (Step1-4 產陸地遮罩) 需要；缺少不影響訓練
GEO_PKGS = [("rasterio", "1.4.4"), ("shapely", "2.1.2"), ("fiona", "1.10.1")]

def check(pkgs, hard=True):
    ok = True
    for name, want in pkgs:
        try:
            m = importlib.import_module(name)
            got = getattr(m, "__version__", "?")
            tag = "OK " if got == want else "OK*"  # * = 版本與驗證版不同
            print(f"  [{tag}] {name:28s} {got}  (驗證版 {want})")
        except Exception as e:
            print(f"  [{'MISS' if hard else 'skip'}] {name:28s} — {type(e).__name__}: {e}")
            if hard:
                ok = False
    return ok

def main():
    print(f"Python {sys.version.split()[0]}\n")
    print("== 核心套件 ==")
    core_ok = check(PKGS, hard=True)

    print("\n== 地理套件（僅上游前處理需要，缺少不影響訓練）==")
    check(GEO_PKGS, hard=False)

    print("\n== CUDA ==")
    cuda_ok = False
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        print(f"  [{'OK ' if cuda_ok else 'WARN'}] torch.cuda.is_available() = {cuda_ok} "
              f"| build cuda={torch.version.cuda} "
              f"| device={torch.cuda.get_device_name(0) if cuda_ok else 'CPU only'}")
    except Exception as e:
        print(f"  [WARN] 無法檢查 CUDA：{e}")

    print("\n== 模型 adapter 註冊 ==")
    adapters_ok = False
    try:
        from main_cross_validation.training_module import _dynamic_import_adapters, MODEL_REGISTRY
        _dynamic_import_adapters()
        keys = sorted(MODEL_REGISTRY.keys())
        print(f"  已註冊：{keys}")
        need = {"segformer", "segformer_deepsup"}
        if need.issubset(MODEL_REGISTRY):
            adapters_ok = True
            print("  [OK ] 主力 adapter segformer / segformer_deepsup 已就緒")
        else:
            print(f"  [MISS] 缺少 {need - set(MODEL_REGISTRY)}")
        if "osdmamba_official" not in MODEL_REGISTRY:
            print("  [note] osdmamba_official 未註冊屬正常（需額外安裝 mamba_ssm，非主力流程不需要）")
    except Exception as e:
        print(f"  [MISS] adapter 載入失敗：{type(e).__name__}: {e}")

    print("\n" + "=" * 60)
    if core_ok and adapters_ok:
        print("結果：環境就緒 ✅ " + ("(含 CUDA)" if cuda_ok else "(⚠ 僅 CPU，訓練會很慢)"))
        return 0
    print("結果：環境不完整 ❌ 請對照上方 [MISS] 項目安裝。")
    return 1

if __name__ == "__main__":
    sys.exit(main())
