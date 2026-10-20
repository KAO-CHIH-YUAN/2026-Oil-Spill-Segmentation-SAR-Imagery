#!/usr/bin/env bash
# =====================================================================
# 路徑可攜化 (retarget_paths)
# 把交接包內所有硬編碼的專案根路徑，換成你機器上的新位置。
#
# 背景：程式內多處寫死絕對路徑 /home/yuan/Yuan/OIL_Project_12_7/...
#       （指向 dataset / pretrain / result）。同機沿用母專案可直接跑；
#       換機 / 換路徑時，先跑這支腳本一次即可。
#
# 用法：
#   ./retarget_paths.sh /new/abs/path/to/project            # 預覽 (dry-run，不改檔)
#   ./retarget_paths.sh /new/abs/path/to/project --apply     # 實際修改 (先備份 *.bak)
#
# 還原：把各 *.bak 覆蓋回原檔即可。
# =====================================================================
set -euo pipefail

OLD="/home/yuan/Yuan/OIL_Project_12_7"
NEW="${1:-}"
MODE="${2:-dry}"

if [ -z "$NEW" ]; then
  echo "用法: $0 <新專案根絕對路徑> [--apply]"
  exit 1
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
SELF="$HERE/$(basename "$0")"

echo "OLD 前綴 = $OLD"
echo "NEW 前綴 = $NEW"
echo "掃描目錄 = $HERE  (*.py *.yaml *.md)"
echo

# 找出含 OLD 前綴的檔（排除腳本自己與既有 .bak）
mapfile -t FILES < <(grep -rl "$OLD" "$HERE" --include=*.py --include=*.yaml --include=*.md 2>/dev/null | grep -v "$SELF" || true)

echo "受影響檔案數: ${#FILES[@]}"
TOTAL=$(grep -rho "$OLD" "$HERE" --include=*.py --include=*.yaml --include=*.md 2>/dev/null | grep -v "$SELF" | wc -l || true)
echo "總出現次數:   $TOTAL"
echo

if [ "$MODE" == "--apply" ]; then
  for f in "${FILES[@]}"; do
    [ -n "$f" ] || continue
    cp "$f" "$f.bak"
    sed -i "s#${OLD}#${NEW}#g" "$f"
  done
  echo "✅ 已替換完成，原檔備份為對應的 *.bak。"
  echo "   還原方式：for f in \$(find $HERE -name '*.bak'); do mv \"\$f\" \"\${f%.bak}\"; done"
else
  echo "(dry-run) 未修改任何檔。加上 --apply 才會實際替換。"
  echo "範例（前 3 檔的受影響行）："
  for f in "${FILES[@]:0:3}"; do
    [ -n "$f" ] || continue
    echo "  # $f"
    grep -n "$OLD" "$f" | head -2 | sed 's/^/    /'
  done
fi
