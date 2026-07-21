# SPARK Research

Infrastructure for reinforcement-learning research in stochastic, partially
observable poker environments. The current release supports Leduc, Limit Texas,
and No-Limit Texas Hold'em through RLCard. Blackjack is retained as an earlier
experiment but is not part of the training dashboard.

## What is included

- A shared environment interface for three poker variants
- A separate configurable agent class for each environment
- Versioned episodes that separate player-visible observations from hidden,
perfect information
- An agent `update()` hook that runs after every completed episode
- Continuous, checkpointed training workers
- Per-run SQLite databases for episodes, actions, metrics, and checkpoints
- Legacy JSON import and portable JSON export
- A FastAPI/HTML dashboard for Slurm jobs, statistics, and full-state replay
- Requeueable Slurm scripts for the MSOE ROSIE supercomputer

No learning algorithm is selected yet. Each environment agent chooses uniformly
from its legal actions. The classes provide memory, update, and checkpoint hooks
for adding Q-learning, SARSA, DQN, or another algorithm later.

## Repository layout

```text
spark/
  agents/             Environment-specific random agents and shared contract
  environments/       Shared RLCard adapters and environment registry
  storage/            SQLite persistence and JSON import/export
  training/           Continuous trainer and CLI
  slurm/              Safe sbatch/squeue/sacct/scancel wrapper
web/
  app.py               FastAPI application
  templates/           Dashboard HTML
  static/              Local CSS and JavaScript (no CDN required)
hpc/
  train.sbatch         Requeueable training worker
  web.sbatch           Dashboard server job
  setup_rosie.sh       ROSIE virtual-environment setup
  start_dashboard.sh   Submit the dashboard and generate an access token
tests/                 Environment, trainer, storage, Slurm, and API tests
runs/                  Generated per-run databases, checkpoints, and logs
```

The old `LeducPoker/`, `LimitHoldem/`, and `Blackjack/` command-line collectors
remain available. New training should use the shared `spark` package.

## Local setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements-dev.txt
```

Run the test suite:

```bash
python -m pytest
```



## Training from one command

Run a finite local Leduc experiment:

```bash
python -m spark.training.cli \
  --environment leduc-holdem \
  --run-id leduc-demo \
  --episodes 1000 \
  --seed 42
```

Run Limit Hold'em continuously until interrupted:

```bash
python -m spark.training.cli \
  --environment limit-holdem \
  --run-id limit-continuous \
  --checkpoint-interval 100 \
  --seed 42
```

Pressing Ctrl+C finishes the current action sequence, saves a checkpoint, and
marks the run stopped. Re-running the same command and run ID resumes from
`runs/<run-id>/checkpoint.json`.

Every episode follows this order:

1. The environment-specific agent receives only the acting player's observation and legal
  actions.
2. The environment records visible and hidden states separately for replay.
3. The completed episode is committed to SQLite.
4. `agent.update(episode)` records the episode and provides the future learning hook.
5. The agent version advances and its state is checkpointed.
6. The next episode starts with the updated agent.

This guarantees that a future learning agent can update between every episode.

## Extending an agent

Each game folder owns its agent next to `main.py`, `replay.py`, and `storage.py`
(for example `LeducPoker/agent.py`). Subclass the shared `Agent` contract from
`spark.agents`:

```python
# LeducPoker/agent.py
from spark.agents import Agent

class MyLeducAgent(Agent):
    name = "my-leduc-agent"
    environment_id = "leduc-holdem"

    def select_action(self, observation, legal_actions, player_id):
        # observation contains no opponent hand/perfect information
        return legal_actions[0]

    def update(self, episode):
        # Train from the completed trajectory and terminal payoffs.
        return {"loss": 0.0}

    def state_dict(self):
        return {"model": "..."}

    def load_state_dict(self, state):
        ...
```

Map it in `spark/agents/registry.py`. The trainer chooses the agent automatically
from `--environment`; there is no separate policy selector. Poker defaults to one
shared self-play agent, with separate seat agents available through
`--separate-seat-agents`.

## SQLite run data

Each worker is the only writer to its own database:

```text
runs/<run-id>/
  run.sqlite
  checkpoint.json
```

The database stores run metadata, episodes, every action, observable states,
hidden states, payoffs, outcomes, agent versions, checkpoints, metrics, and
worker events. The dashboard opens short-lived read connections and calculates
aggregate statistics in SQL rather than loading an entire JSON file.

### Import old episode files

```python
from spark.storage import RunDatabase
from spark.storage.import_export import import_legacy_json

database = RunDatabase("runs/imported-leduc/run.sqlite")
import_legacy_json(
    "/path/to/legacy/episodes.json",
    database,
    run_id="imported-leduc",
)
```



### Export a run

The dashboard exposes `GET /api/runs/<run-id>/export`, or use:

```python
from spark.storage import RunDatabase
from spark.storage.import_export import export_json

database = RunDatabase("runs/leduc-demo/run.sqlite", initialize=False)
export_json(database, "runs/leduc-demo/leduc-demo.json")
```



## Running the web dashboard

The dashboard can run on your PC for local testing, or on Rosie through Open
OnDemand at [https://dh-ood.hpc.msoe.edu](https://dh-ood.hpc.msoe.edu).

Dashboard capabilities:

- Start one campaign containing independent Leduc, Limit, and No-Limit jobs
- Configure seed, episode limit, CPUs, memory, wall time, partition, and
checkpoint interval
- Monitor and stop jobs and inspect worker logs
- Remove jobs from campaign controls without deleting their run data or replays
- Save named agent snapshots and warm-start new compatible campaigns from them
- View rolling payoff, outcome counts, action frequencies, episode lengths, and
throughput
- Browse episodes and move backward/forward/autoplay through their timelines
- Inspect the exact visible agent input and all hidden hands/perfect state at
every step
- Export a run as JSON



### Agent library

Each run writes its latest durable agent state to
`runs/<run-id>/checkpoint.json`. In the dashboard's **Agent library**, choose a
run and a name to copy that checkpoint into `runs/agent-snapshots/`. Saving uses
the latest checkpoint already on disk; if a running job has not reached its
first checkpoint interval, wait for one or stop it cleanly before saving.

When creating a campaign, each selected environment has its own **Starting
agent** selector. A saved snapshot can only be loaded into the environment and
shared/separate-seat topology it was created for. Loading warm-starts agent
state in a new run while resetting that run's episode and version counters.
Resuming an existing run continues to use its own checkpoint.

These operations have intentionally different scopes:

- **Remove job:** stops it if needed and hides it from campaign controls, while
retaining its database, statistics, replay, checkpoint, and campaign label.
- **Delete saved agent:** deletes only the reusable snapshot copy.
- **Stop job:** keeps both campaign controls and all run data available.



### On your PC

1. Create and activate a virtual environment, then install dependencies (see
  [Local setup](#local-setup)).
2. Start the dashboard with local job mode enabled so campaigns can launch
  worker subprocesses:

```bash
# Linux/macOS
SPARK_LOCAL_JOBS=1 python -m uvicorn web.app:app --reload --host 127.0.0.1 --port 8000

# Windows PowerShell
$env:SPARK_LOCAL_JOBS = "1"
py -m uvicorn web.app:app --reload --host 127.0.0.1 --port 8000
```

1. Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

Without `SPARK_LOCAL_JOBS=1` and without Slurm, the dashboard is read-only.
Leave the terminal running while you use the page; Ctrl+C stops the server.

### On Rosie (Open OnDemand)

Rosie compute hosts (`dh-node*`) are usually not in campus DNS, and
`rosie.msoe.edu` is not a valid SSH hostname. Use Open OnDemand instead of
Windows SSH tunnels to compute nodes.

#### 1. Connect and prepare

1. Open [https://dh-ood.hpc.msoe.edu](https://dh-ood.hpc.msoe.edu) and sign in.
2. Start **Shell Access** (or an interactive job that lands you on a `dh-node`*).
3. In that Rosie shell:

```bash
cd ~/SummerResearch   # or your clone path
bash hpc/setup_rosie.sh
source .venv/bin/activate
python -m pip install -r requirements.txt
```

If a Python module is required first:

```bash
module avail python
module load python
```



#### 2. Start the dashboard in the OnDemand shell

When your prompt already shows a compute node such as `dh-node2`:

```bash
bash hpc/run_dashboard_ondemand.sh
```

Leave that shell running. The script prints a token and the node name. In your
browser, open (replace `dh-node2` and `TOKEN` with the values from the script):

```text
https://dh-ood.hpc.msoe.edu/rnode/dh-node2/8000/?token=TOKEN
```

If `/rnode/` does not work, try:

```text
https://dh-ood.hpc.msoe.edu/node/dh-node2/8000/?token=TOKEN
```

Keep the `?token=...` query string. After updating code on Rosie, restart the
dashboard script and hard-refresh the browser (Ctrl+Shift+R).

#### 3. Optional: submit the dashboard as a Slurm job

```bash
bash hpc/start_dashboard.sh
cat runs/spark-web-JOB_ID.out
```

Use the OnDemand `/rnode/...` or `/node/...` URL and token printed in that log.

### Continuous Slurm behavior (training campaigns)

From the Rosie dashboard, starting a campaign submits one Slurm job per selected
environment with `--requeue`. Two minutes before the wall-time limit, Slurm
sends `USR1`; the batch script asks the Python worker to stop cleanly, writes
its SQLite transaction and agent checkpoint, then requeues the same job.
Training resumes from the checkpoint. Use the dashboard Stop button or
`scancel JOB_ID` to stop a campaign.

Cluster scheduling policies and partition names can change. Confirm resource limits in the
current [ROSIE User Guide](https://msoe.dev/#/) before submitting large jobs.

## Environment status

- **Leduc Hold'em:** enabled (`rlcard.make("leduc-holdem")`)
- **Limit Texas Hold'em:** enabled (`rlcard.make("limit-holdem")`)
- **No-Limit Texas Hold'em:** enabled (`rlcard.make("no-limit-holdem")`)
- **Blackjack:** legacy collector only; excluded from current campaigns

