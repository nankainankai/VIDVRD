#!/usr/bin/env bash
# mock 检测/追踪 + 全链路真实 VL
# export DASHSCOPE_API_KEY="sk-你的key"
# ./scripts/run_with_vl.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

API_KEY="${1:-${DASHSCOPE_API_KEY:-}}"
VIDEO="${VIDEO:-data/validation_dummy.mp4}"
RUN_DIR="${RUN_DIR:-runs/live_vl}"

if [[ -z "$API_KEY" ]]; then
  echo "ERROR: 设置 DASHSCOPE_API_KEY 或: ./scripts/run_with_vl.sh sk-xxx"
  exit 1
fi

[[ -f "$VIDEO" ]] || python scripts/make_validation_dummy.py

python scripts/run_vidvrd_auto.py \
  --video "$VIDEO" \
  --run_dir "$RUN_DIR" \
  --config configs/run_with_vl.json \
  --resume \
  --skip_eval \
  --api_key "$API_KEY"

python scripts/generate_run_report.py --run_dir "$RUN_DIR"
echo ""
echo "完成: $RUN_DIR (检查各节点 *.json 中 used_images)"
