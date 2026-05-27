#!/usr/bin/env bash
# 全自动跑通（mock 检测/追踪 + 真实 DashScope 关系 LLM）
# export DASHSCOPE_API_KEY="sk-你的key"
# ./scripts/run_with_api.sh

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

API_KEY="${1:-${DASHSCOPE_API_KEY:-}}"
VIDEO="${VIDEO:-data/validation_dummy.mp4}"
RUN_DIR="${RUN_DIR:-runs/live_api}"

if [[ -z "$API_KEY" ]]; then
  echo "ERROR: 设置 DASHSCOPE_API_KEY 或: ./scripts/run_with_api.sh sk-xxx"
  exit 1
fi

[[ -f "$VIDEO" ]] || python scripts/make_validation_dummy.py

python scripts/run_vidvrd_auto.py \
  --video "$VIDEO" \
  --run_dir "$RUN_DIR" \
  --config configs/run_with_api.json \
  --resume \
  --skip_eval \
  --api_key "$API_KEY"

echo ""
echo "完成: $RUN_DIR/run_manifest.json"
echo "      $RUN_DIR/pred/relations_pred.json"
