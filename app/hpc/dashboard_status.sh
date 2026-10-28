#!/bin/bash
# Print the live dashboard Slurm job and the current OnDemand URL.
# Run this from a later SSH session. It does not start or stop anything.

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${project_root}"

job_file="data/dashboard.job"
url_file="data/dashboard.url"

if [[ ! -f "${job_file}" ]]; then
  echo "No dashboard job recorded. Start one with:"
  echo "  bash app/hpc/start_dashboard.sh"
  exit 1
fi

job_id="$(tr -d '[:space:]' < "${job_file}")"
echo "Dashboard Slurm job: ${job_id}"

if ! command -v squeue >/dev/null 2>&1; then
  echo "squeue is not available on this machine."
elif ! squeue -j "${job_id}"; then
  echo "Job ${job_id} is not in the queue."
fi

if [[ -f "${url_file}" ]]; then
  echo "Dashboard URL:"
  cat "${url_file}"
else
  echo "The job has not written a URL yet. Run this again after it starts:"
  echo "  bash app/hpc/dashboard_status.sh"
fi
