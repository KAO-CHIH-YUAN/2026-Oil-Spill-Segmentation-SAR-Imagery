"""
上游前處理設定載入器。
讀 preprocess_config.yaml，代換 {handoff_root}/{output_base}，計算各步驟輸出路徑。
用法：
    from _pp_config import CFG
    CFG.inputs['tif_root'], CFG.out['step1_landjson'], CFG.step1['min_land_pixels'] ...
設定檔位置可用環境變數 PREPROCESS_CONFIG 覆寫。
"""
import os
import yaml
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_CFG_PATH = Path(os.environ.get("PREPROCESS_CONFIG", _HERE / "preprocess_config.yaml"))


def _subst(val, mapping):
    if isinstance(val, str):
        for k, v in mapping.items():
            val = val.replace("{" + k + "}", str(v))
        return val
    if isinstance(val, dict):
        return {k: _subst(v, mapping) for k, v in val.items()}
    if isinstance(val, list):
        return [_subst(v, mapping) for v in val]
    return val


class _Config:
    def __init__(self, path: Path):
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        handoff_root = raw["handoff_root"]
        mapping = {"handoff_root": handoff_root}
        output_base = _subst(raw["output_base"], mapping)
        mapping["output_base"] = output_base

        raw = _subst(raw, mapping)

        self.path = path
        self.handoff_root = Path(handoff_root)
        self.output_base = Path(output_base)
        self.inputs = {k: Path(v) for k, v in raw["inputs"].items()}
        self.step1 = raw["step1"]
        self.step3 = raw["step3"]
        self.step4 = raw["step4"]
        self.run = raw.get("run", {"from": 1, "to": 4})

        # 各步驟輸出的絕對路徑
        o = raw["outputs"]
        self.out = {k: self.output_base / v for k, v in o.items()}

    def __repr__(self):
        return f"<PreprocessConfig {self.path}>"


CFG = _Config(_CFG_PATH)


if __name__ == "__main__":
    # 快速檢視解析後的路徑
    print("設定檔:", CFG.path)
    print("handoff_root:", CFG.handoff_root)
    print("output_base :", CFG.output_base)
    print("\n[inputs]")
    for k, v in CFG.inputs.items():
        print(f"  {k:22s} {v}   {'✓' if v.exists() else '✗ 不存在'}")
    print("\n[outputs]")
    for k, v in CFG.out.items():
        print(f"  {k:16s} {v}")
    print("\n[run]  ", CFG.run)
    print("[step1]", CFG.step1)
    print("[step3]", CFG.step3)
    print("[step4]", CFG.step4)
