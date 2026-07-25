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
echo "Start the dashboard with: bash app/hpc/start_dashboard.sh"
