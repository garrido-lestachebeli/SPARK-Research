#!/bin/bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"
mkdir -p runs

port="${SPARK_WEB_PORT:-8000}"
token="${SPARK_ACCESS_TOKEN:-$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')}"
partition_args=()
if [[ -n "${SPARK_WEB_PARTITION:-}" ]]; then
  partition_args+=("--partition=${SPARK_WEB_PARTITION}")
fi

job_id="$(
  sbatch --parsable \
    --job-name=spark-web \
    --time="${SPARK_WEB_WALL_TIME:-08:00:00}" \
    --export="ALL,SPARK_PROJECT_ROOT=${project_root},SPARK_WEB_PORT=${port},SPARK_ACCESS_TOKEN=${token},SPARK_VENV=${project_root}/.venv" \
    "${partition_args[@]}" \
    hpc/web.sbatch
)"
job_id="${job_id%%;*}"

echo "Submitted SPARK dashboard as Slurm job ${job_id}."
echo "Token: ${token}"
echo "Wait for the job to start, then inspect:"
echo "  cat runs/spark-web-${job_id}.out"
echo "That file contains the compute node, SSH tunnel command, and dashboard URL."
