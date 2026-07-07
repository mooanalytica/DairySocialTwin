#!/usr/bin/env bash
#SBATCH --job-name=dcsna_gx040008_a100
#SBATCH --account=rrg-sureshra
#SBATCH --time=16:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/home/your_account/scratch/DCSNA/run_logs/narval/%x_%j.out
#SBATCH --error=/home/your_account/scratch/DCSNA/run_logs/narval/%x_%j.err

set -euo pipefail

module load apptainer/1.3.5

PROJECT_HOST="/home/your_account/scratch/DCSNA"
SCRATCH_HOST="/home/your_account/scratch"
IMAGE_HOST="/home/your_account/scratch/dcsna_a100_raw_4090.sif"
IMAGE_LOCAL="${SLURM_TMPDIR}/dcsna_a100_raw_4090.sif"

SOURCE_VIDEO_HOST="/home/your_account/scratch/V0531/Dairy Farm Videos/May 7 2025 Dairy Farm 3 Videos/Video/Gopro2/GX040008.MP4"
INPUT_HOST="${PROJECT_HOST}/input_a100/gx040008"
INPUT_LINK_HOST="${INPUT_HOST}/GX040008.MP4"
INPUT_CONTAINER="/work/input_a100/gx040008"
OUTPUT_CONTAINER="/work/output_a100_gx040008"

LOG_CHECK_SECONDS="20"
LOG_CHECK_INTERVAL="10"
GPU_MONITOR_INTERVAL="60"

mkdir -p "${PROJECT_HOST}/run_logs/narval" "${PROJECT_HOST}/.ultralytics" "${PROJECT_HOST}/.cache/torch" "${INPUT_HOST}"

test -d "${PROJECT_HOST}"
test -d "${SCRATCH_HOST}"
test -f "${IMAGE_HOST}"
test -f "${SOURCE_VIDEO_HOST}"

if [[ -e "${INPUT_LINK_HOST}" && ! -L "${INPUT_LINK_HOST}" ]]; then
  echo "[fatal] refusing to replace non-symlink input path: ${INPUT_LINK_HOST}"
  exit 2
fi
ln -sfn "${SOURCE_VIDEO_HOST}" "${INPUT_LINK_HOST}"

cp "${IMAGE_HOST}" "${IMAGE_LOCAL}"

export APPTAINERENV_PYTHONNOUSERSITE=1
export APPTAINERENV_YOLO_CONFIG_DIR=/work/.ultralytics
export APPTAINERENV_TORCH_HOME=/work/.cache/torch
export APPTAINERENV_MPLCONFIGDIR=/work/.cache/matplotlib

echo "[host] job=${SLURM_JOB_ID} node=${SLURMD_NODENAME:-unknown}"
echo "[host] source_video=${SOURCE_VIDEO_HOST}"
echo "[host] input_link=${INPUT_LINK_HOST}"
nvidia-smi

(
  while true; do
    echo "[gpu-monitor] $(date -Is)"
    nvidia-smi --query-gpu=name,index,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
    sleep "${GPU_MONITOR_INTERVAL}"
  done
) &
GPU_MONITOR_PID="$!"
trap 'kill "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true' EXIT

APPTAINER_ARGS=(
  --nv
  --bind "${PROJECT_HOST}:/work"
  --bind "${SCRATCH_HOST}:/scratch"
  --bind "${SCRATCH_HOST}:${SCRATCH_HOST}"
  --pwd /work
)

apptainer exec "${APPTAINER_ARGS[@]}" "${IMAGE_LOCAL}" \
  python -u /work/cluster/narval/preflight_a100_single_video.py \
    --project-root /work \
    --input-dir "${INPUT_CONTAINER}" \
    --expected-video GX040008.MP4 \
    --require-device-substring A100 \
    --log-check-seconds "${LOG_CHECK_SECONDS}" \
    --log-check-interval "${LOG_CHECK_INTERVAL}"

apptainer exec "${APPTAINER_ARGS[@]}" "${IMAGE_LOCAL}" \
  python -u /work/run_full_interaction_pipeline.py \
    --input-dir "${INPUT_CONTAINER}" \
    --output-dir "${OUTPUT_CONTAINER}" \
    --det-device 0 \
    --pose-device cuda:0 \
    --visualize-every 1

echo "[done] output directory: ${PROJECT_HOST}/output_a100_gx040008"
