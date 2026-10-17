#!/bin/bash
# SafeGuard-Agent runner (server). Usage: run_gate.sh <task> <gate_mode> <try_id> <prob> <gpu> <camera>
set -e
TASK="${1:-cook_tofu__with_bottle__of__oil}"
GATE_MODE="${2:-off}"
TRY_ID="${3:-smoke_off_01}"
PROB="${4:-0.8}"
GPU="${5:-}"
CAMERA="${6:-0}"

WS=/home/sdb/ruochen/SafeMem-workspace
ENV_PY=$WS/envs/isbench/bin/python
REPO=$WS/repos/SafeMem

# keys from server-local files (NOT committed to git)
OPENAI_KEY=$(python3 -c "import json; print(json.load(open('$HOME/.codex/auth.json'))['OPENAI_API_KEY'])")
if [ -f "$HOME/.config/safemem/gemini.env" ]; then
  source "$HOME/.config/safemem/gemini.env"
fi

# api.wpironman.top is directly reachable; do NOT route through a proxy
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy 2>/dev/null || true

export OPENAI_API_KEY="$OPENAI_KEY"
export OPENAI_API_BASE='https://api.wpironman.top/v1'

# Scene graph VLM: gemini-3-flash (gemini key) by default.
# Set SCENE_GRAPH_MODEL=gpt-6-sol to use gpt-6-sol (codex key) when Gemini is down.
if [ "$SCENE_GRAPH_MODEL" = "gpt-6-sol" ]; then
  export GEMINI_API_KEY="$OPENAI_KEY"
  export GEMINI_MODEL='gpt-6-sol'
else
  export GEMINI_API_KEY="${GEMINI_API_KEY:-}"
  export GEMINI_MODEL="${GEMINI_MODEL:-gemini-3-flash}"
fi
export GEMINI_BASE_URL='https://api.wpironman.top/v1'

export SCENE_GRAPH_CHECKPOINT="$WS/checkpoints/sam2.1_hiera_large.pt"
export OMNIGIBSON_DATASET_PATH="$WS/data/omnigibson"
export OMNIGIBSON_ASSET_PATH="$WS/assets/omnigibson"
export OMNIGIBSON_HEADLESS=1
if [ -n "$GPU" ]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi

cd "$REPO"
echo "=== run: task=$TASK gate=$GATE_MODE try_id=$TRY_ID prob=$PROB model=gpt-6-sol ==="
"$ENV_PY" -m og_ego_prim.cli.online_benchmark_once \
  --task "$TASK" --try_id "$TRY_ID" --work_dir ./results/smoke \
  --use_scene_graph --scene_graph_camera_index "$CAMERA" \
  --use_risk_predictor --risk_check_probability "$PROB" --risk_gate_mode "$GATE_MODE" \
  --model gpt-6-sol --risk_predictor_model gpt-6-sol \
  --risk_max_replans 10 --prompt_setting v1 > "$WS/run_${TRY_ID}.log" 2>&1
