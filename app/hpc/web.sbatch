#!/bin/bash
#SBATCH --output=data/spark-web-%j.out
#SBATCH --error=data/spark-web-%j.err
#SBATCH --requeue
#SBATCH --signal=B:USR1@120

set -euo pipefail

cd "${SPARK_PROJECT_ROOT:?SPARK_PROJECT_ROOT is required}"
mkdir -p data

if [[ -n "${SPARK_VENV:-}" && -f "${SPARK_VENV}/bin/activate" ]]; then
  source "${SPARK_VENV}/bin/activate"
elif [[ -f ".venv/bin/activate" ]]; then
  source ".venv/bin/activate"
fi

export SPARK_RUNS_DIRECTORY="${SPARK_RUNS_DIRECTORY:-${SPARK_PROJECT_ROOT}/data}"
node_name="$(hostname -s)"
port="${SPARK_WEB_PORT:-8000}"
# 0.0.0.0 is required for Open OnDemand /rnode and /node proxies.
bind_host="${SPARK_BIND_HOST:-0.0.0.0}"
ondemand_base="${SPARK_ONDEMAND_URL:-https://dh-ood.hpc.msoe.edu}"

echo "SPARK dashboard node: ${node_name}"
echo "SPARK dashboard port: ${port}"
echo "SPARK bind host: ${bind_host}"
echo "SPARK SSH user: ${USER}"
echo "Token: ${SPARK_ACCESS_TOKEN}"
echo ""
echo "Preferred access on Rosie (Open OnDemand):"
echo "  ${ondemand_base}/rnode/${node_name}/${port}/?token=${SPARK_ACCESS_TOKEN}"
echo "  ${ondemand_base}/node/${node_name}/${port}/?token=${SPARK_ACCESS_TOKEN}"
echo ""
echo "SSH tunnels only work if Windows can resolve a login host."
echo "Compute nodes like ${node_name} are usually not in campus DNS."
echo ""

python3 -m uvicorn app.app:app \
  --host "${bind_host}" \
  --port "${port}" \
  --no-access-log &
server_pid=$!

requeue_web() {
  kill -TERM "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" || true
  scontrol requeue "${SLURM_JOB_ID}"
  exit 0
}
stop_web() {
  kill -TERM "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" || true
  exit 0
}
trap requeue_web USR1
trap stop_web TERM INT
wait "${server_pid}"
