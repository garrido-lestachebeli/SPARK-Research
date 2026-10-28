#!/bin/bash
#SBATCH --output=data/slurm-%j.out
#SBATCH --error=data/slurm-%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

set -euo pipefail

cd "${SPARK_PROJECT_ROOT:?SPARK_PROJECT_ROOT is required}"
mkdir -p "${SPARK_RUNS_DIRECTORY:-data}"

if [[ -n "${SPARK_VENV:-}" && -f "${SPARK_VENV}/bin/activate" ]]; then
  source "${SPARK_VENV}/bin/activate"
elif [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
fi

args=(
  --environment "${SPARK_ENVIRONMENT:?}"
  --table-size "${SPARK_TABLE_SIZE:-2}"
  --run-id "${SPARK_RUN_ID:?}"
  --run-directory "${SPARK_RUNS_DIRECTORY:-data}"
  --seed "${SPARK_SEED:-0}"
  --checkpoint-interval "${SPARK_CHECKPOINT_INTERVAL:-100}"
  --eval-interval "${SPARK_EVAL_INTERVAL:-0}"
  --eval-episodes "${SPARK_EVAL_EPISODES:-20}"
  --league-reroll-interval "${SPARK_LEAGUE_REROLL_INTERVAL:-0}"
  --agent-memory-size "${SPARK_AGENT_MEMORY_SIZE:-100}"
  --starting-stack "${SPARK_STARTING_STACK:-50}"
  --small-blind "${SPARK_SMALL_BLIND:-1}"
  --big-blind "${SPARK_BIG_BLIND:-2}"
  --episode-mode "${SPARK_EPISODE_MODE:-session}"
  --max-raises "${SPARK_MAX_RAISES:-0}"
)
if [[ -n "${SPARK_SEAT_CONFIG_B64:-}" ]]; then
  args+=(--seat-config-b64 "${SPARK_SEAT_CONFIG_B64}")
fi
if [[ -n "${SPARK_EPISODES:-}" && "${SPARK_EPISODES}" != "none" ]]; then
  args+=(--episodes "${SPARK_EPISODES}")
fi
python3 -m spark.training.cli "${args[@]}" &
worker_pid=$!

checkpoint_and_requeue() {
  echo "Slurm wall-time warning received; checkpointing and requeueing job ${SLURM_JOB_ID}."
  kill -TERM "${worker_pid}" 2>/dev/null || true
  wait "${worker_pid}" || true
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}

stop_worker() {
  kill -TERM "${worker_pid}" 2>/dev/null || true
  wait "${worker_pid}" || true
  exit 0
}

trap checkpoint_and_requeue USR1
trap stop_worker TERM INT
wait "${worker_pid}"
