#!/usr/bin/env bash
set -Eeuo pipefail

# Runs one bounded CourtVision job on an EC2 worker. The instance is stopped
# after the job when SHUTDOWN_AFTER_JOB=1 and EC2's shutdown behavior is Stop.

ROOT="${COURTVISION_ROOT:-/opt/courtvision}"
IMAGE="${COURTVISION_IMAGE:-courtvision:latest}"
INPUT_NAME="${1:-}"
JOB_ID="${JOB_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
TARGET_FPS="${TARGET_FPS:-15}"
MAX_WIDTH="${MAX_WIDTH:-960}"
MAX_DURATION_SECONDS="${MAX_DURATION_SECONDS:-30}"
MAX_JOB_MINUTES="${MAX_JOB_MINUTES:-90}"
SHUTDOWN_AFTER_JOB="${SHUTDOWN_AFTER_JOB:-0}"

stop_instance() {
  if [[ "$SHUTDOWN_AFTER_JOB" == "1" ]]; then
    echo "Stopping the EC2 instance to end compute charges."
    if [[ "${EUID:-$(id -u)}" == "0" ]]; then
      shutdown -h now
    else
      sudo shutdown -h now
    fi
  fi
}
trap stop_instance EXIT

if [[ -z "$INPUT_NAME" || "$INPUT_NAME" == */* || "$INPUT_NAME" == *\\* ]]; then
  echo "Usage: $0 <filename from $ROOT/input>" >&2
  exit 64
fi

INPUT_PATH="$ROOT/input/$INPUT_NAME"
OUTPUT_NAME="${INPUT_NAME%.*}_${JOB_ID}.mp4"
OUTPUT_PATH="$ROOT/output/$OUTPUT_NAME"
CACHE_PATH="$ROOT/cache/$JOB_ID"

for model in player_detector.pt ball_detector_model.pt court_keypoint_detector.pt; do
  if [[ ! -f "$ROOT/models/$model" ]]; then
    echo "Missing model: $ROOT/models/$model" >&2
    exit 66
  fi
done

if [[ ! -f "$INPUT_PATH" ]]; then
  echo "Input does not exist: $INPUT_PATH" >&2
  exit 66
fi

mkdir -p "$ROOT/output" "$CACHE_PATH"

docker_args=(
  --rm
  --memory=14g
  --cpus=4
  --pids-limit=512
  --security-opt=no-new-privileges
  --read-only
  --tmpfs=/tmp:rw,noexec,nosuid,size=1g
  --env=HOME=/tmp
  --env=XDG_CACHE_HOME=/tmp/.cache
  --mount="type=bind,src=$ROOT/models,dst=/app/backend/models,readonly"
  --mount="type=bind,src=$ROOT/input,dst=/data/input,readonly"
  --mount="type=bind,src=$ROOT/output,dst=/data/output"
  --mount="type=bind,src=$CACHE_PATH,dst=/data/cache"
)

if command -v nvidia-smi >/dev/null 2>&1; then
  docker_args+=(--gpus=all)
fi

echo "Job: $JOB_ID"
echo "Input: $INPUT_PATH"
echo "Output: $OUTPUT_PATH"
echo "Limit: ${MAX_JOB_MINUTES} minutes"

timeout --signal=TERM --kill-after=2m "${MAX_JOB_MINUTES}m" \
  docker run "${docker_args[@]}" "$IMAGE" \
  "/data/input/$INPUT_NAME" \
  --duration-seconds "$MAX_DURATION_SECONDS" \
  --target-fps "$TARGET_FPS" \
  --max-width "$MAX_WIDTH" \
  --stub-path /data/cache \
  --output-video "/data/output/$OUTPUT_NAME"

echo "Completed: $OUTPUT_PATH"
