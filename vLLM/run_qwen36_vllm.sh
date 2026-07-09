#!/usr/bin/env bash
set -euo pipefail

VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest-x86_64-cu129-ubuntu2404}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.6-27B-FP8}"

HF_ENV=()

if [[ -n "${HF_TOKEN:-}" ]]; then
  if ! python3 - <<'PY'
import os, sys
t = os.environ.get("HF_TOKEN", "")
sys.exit(0 if t.isascii() else 1)
PY
  then
    echo "ERROR: HF_TOKEN contains non-ASCII characters. Remove Chinese placeholder text from HF_TOKEN." >&2
    exit 1
  fi

  if [[ "${HF_TOKEN}" != hf_* ]]; then
    echo "WARNING: HF_TOKEN is set but does not start with hf_ ." >&2
  fi

  HF_ENV=(-e "HF_TOKEN=${HF_TOKEN}")
fi

docker rm -f qwen36-vllm 2>/dev/null || true

docker run -d \
  --name qwen36-vllm \
  --runtime nvidia \
  --gpus all \
  --ipc=host \
  --network host \
  -v qwen-hf-cache:/root/.cache/huggingface \
  -v /data/videos:/videos:ro \
  "${HF_ENV[@]}" \
  -e VLLM_VIDEO_FETCH_TIMEOUT=3600 \
  "$VLLM_IMAGE" \
  "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.88 \
  --reasoning-parser qwen3 \
  --allowed-local-media-path /videos \
  --media-io-kwargs '{"video": {"num_frames": -1, "frame_recovery": true}}' \
  --limit-mm-per-prompt.video 1
