#!/bin/bash
#SBATCH --output=runs/slurm-%j.out
#SBATCH --error=runs/slurm-%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

set -euo pipefail

cd "${SPARK_PROJECT_ROOT:?SPARK_PROJECT_ROOT is required}"
mkdir -p "${SPARK_RUNS_DIRECTORY:-runs}"

if [[ -n "${SPARK_VENV:-}" && -f "${SPARK_VENV}/bin/activate" ]]; then
  source "${SPARK_VENV}/bin/activate"
elif [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
fi

args=(
  --environment "${SPARK_ENVIRONMENT:?}"
  --run-id "${SPARK_RUN_ID:?}"
  --run-directory "${SPARK_RUNS_DIRECTORY:-runs}"
  --seed "${SPARK_SEED:-0}"
  --checkpoint-interval "${SPARK_CHECKPOINT_INTERVAL:-100}"
  --agent-memory-size "${SPARK_AGENT_MEMORY_SIZE:-100}"
)
if [[ -n "${SPARK_EPISODES:-}" && "${SPARK_EPISODES}" != "none" ]]; then
  args+=(--episodes "${SPARK_EPISODES}")
fi
if [[ "${SPARK_SEPARATE_SEAT_AGENTS:-0}" == "1" ]]; then
  args+=(--separate-seat-agents)
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
