#!/bin/bash
# Run the SPARK dashboard in the current OnDemand/interactive shell.
# Use this when you are already on a Rosie compute node (prompt shows dh-node*).

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${project_root}"
mkdir -p data

if [[ -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source ".venv/bin/activate"
fi

port="${SPARK_WEB_PORT:-8000}"
token="${SPARK_ACCESS_TOKEN:-$(python3 -c 'import secrets; print(secrets.token_urlsafe(24))')}"
node_name="$(hostname -s)"
export SPARK_ACCESS_TOKEN="${token}"
export SPARK_RUNS_DIRECTORY="${SPARK_RUNS_DIRECTORY:-${project_root}/data}"
export SPARK_LOCAL_JOBS="${SPARK_LOCAL_JOBS:-1}"

echo "Starting dashboard on ${node_name}:${port}"
echo "Token: ${token}"
echo ""
echo "In your browser, open ONE of:"
echo "  https://dh-ood.hpc.msoe.edu/rnode/${node_name}/${port}/?token=${token}"
echo "  https://dh-ood.hpc.msoe.edu/node/${node_name}/${port}/?token=${token}"
echo ""
echo "Press Ctrl+C to stop."
echo ""

exec python3 -m uvicorn app.app:app --host 0.0.0.0 --port "${port}"
