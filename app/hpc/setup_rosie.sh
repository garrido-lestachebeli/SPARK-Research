#!/bin/bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${project_root}"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
mkdir -p data

echo "ROSIE environment ready at ${project_root}/.venv"
echo "From an SSH session on a Rosie login node:"
echo "  bash app/hpc/start_dashboard.sh"
echo "Close SSH after it prints the job id. From a later session:"
echo "  bash app/hpc/dashboard_status.sh"
