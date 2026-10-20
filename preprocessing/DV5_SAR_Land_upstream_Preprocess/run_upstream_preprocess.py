#!/usr/bin/env python3
"""
上游前處理 orchestrator —— 一次連續跑 Step1 → Step4。

所有路徑/超參統一在 preprocess_config.yaml 設定一次即可（見 _pp_config.py）。

用法：
    python run_upstream_preprocess.py                 # 連續跑 Step1→Step4
    python run_upstream_preprocess.py --from 3        # 從 Step3 開始跑到 Step4
    python run_upstream_preprocess.py --from 1 --to 3 # 只跑 Step1~Step3
    python run_upstream_preprocess.py --config /path/to/other.yaml
    python run_upstream_preprocess.py --dry           # 只印將執行的步驟與解析後路徑，不實跑

每一步是獨立子行程；任一步失敗即中止（回傳非 0）。
依賴（在 config 內指定）：TIF、標註 JSON、OSM 快取(Step1)、source_dataset+Excel(Step4)。
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

STEPS = [
    (1, "陸地JSON + overlay",      "pipeline_step1_land_json.py"),
    (2, "合併 land+標註 JSON",     "pipeline_step2_merge_json.py"),
    (3, "PNG / Mask / YOLO TXT",   "pipeline_step3_png_mask.py"),
    (4, "組裝 Step4_Dataset",       "pipeline_step4_build_dataset_onlyDoneOne.py"),
]


def main():
    ap = argparse.ArgumentParser(description="DV5 上游前處理 Step1-4 連續執行")
    ap.add_argument("--from", dest="start", type=int, default=None, help="起始步驟 (1-4)；預設讀 config 的 run.from")
    ap.add_argument("--to", dest="end", type=int, default=None, help="結束步驟 (1-4)；預設讀 config 的 run.to")
    ap.add_argument("--config", type=str, default=None, help="preprocess_config.yaml 路徑 (預設同資料夾)")
    ap.add_argument("--dry", action="store_true", help="只印計畫與解析後路徑，不實跑")
    args = ap.parse_args()

    env = dict(os.environ)
    if args.config:
        env["PREPROCESS_CONFIG"] = str(Path(args.config).resolve())
        os.environ["PREPROCESS_CONFIG"] = env["PREPROCESS_CONFIG"]

    # 未指定 --from/--to 時，改用 config 的 run 預設值
    from _pp_config import CFG
    if args.start is None:
        args.start = int(CFG.run.get("from", 1))
    if args.end is None:
        args.end = int(CFG.run.get("to", 4))

    # 印出解析後的設定（同時驗證 config 可載入）
    print("=" * 64)
    print("上游前處理 orchestrator")
    print("=" * 64)
    r = subprocess.run([sys.executable, str(HERE / "_pp_config.py")], env=env)
    if r.returncode != 0:
        print("[中止] 設定檔載入失敗。")
        return 1

    todo = [s for s in STEPS if args.start <= s[0] <= args.end]
    print("\n將執行：")
    for n, desc, script in todo:
        print(f"  Step{n}: {desc}  ({script})")

    if args.dry:
        print("\n(--dry) 未實跑。")
        return 0

    for n, desc, script in todo:
        print("\n" + "─" * 64)
        print(f"▶ Step{n}: {desc}")
        print("─" * 64)
        r = subprocess.run([sys.executable, "-u", str(HERE / script)], env=env, cwd=str(HERE))
        if r.returncode != 0:
            print(f"\n[中止] Step{n} ({script}) 失敗，回傳碼 {r.returncode}。後續步驟不執行。")
            return r.returncode
        print(f"✔ Step{n} 完成")

    print("\n" + "=" * 64)
    print(f"全部完成 (Step{todo[0][0]}→Step{todo[-1][0]})。輸出見 config 的 output_base。")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
