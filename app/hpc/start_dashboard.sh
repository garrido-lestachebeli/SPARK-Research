#!/bin/bash
# Submit the SPARK dashboard as a Slurm job from a Rosie login node, then exit.
# Closing this SSH session does not stop the dashboard or training jobs.

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${project_root}"
mkdir -p data

remember_job() {
  printf '%s\n' "$1" > data/dashboard.job
}

existing_line=""
if command -v squeue >/dev/null 2>&1; then
  existing_line="$(
    squeue -u "${USER}" -n spark-web -h -o "%i %T" |
      awk '$2 == "PENDING" || $2 == "RUNNING" { print; exit }'
  )" || true
fi

if [[ -n "${existing_line}" ]]; then
  job_id="${existing_line%% *}"
  state="${existing_line##* }"
  remember_job "${job_id}"
  echo "SPARK dashboard is already Slurm job ${job_id} (${state})."
  echo "A second dashboard was not submitted."
  echo "Check the address with:"
  echo "  bash app/hpc/dashboard_status.sh"
  echo "You can close this SSH session. The dashboard and any training jobs keep running."
  exit 0
fi

rm -f data/dashboard.url

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
    --mem=2G \
    --cpus-per-task=1 \
    --export="ALL,SPARK_PROJECT_ROOT=${project_root},SPARK_WEB_PORT=${port},SPARK_ACCESS_TOKEN=${token},SPARK_VENV=${project_root}/.venv" \
    "${partition_args[@]}" \
    app/hpc/web.sbatch
)"
job_id="${job_id%%;*}"
remember_job "${job_id}"

echo "Submitted SPARK dashboard as Slurm job ${job_id}."
echo "Check the address with:"
echo "  bash app/hpc/dashboard_status.sh"
echo "You can close this SSH session. The dashboard and any training jobs keep running."
