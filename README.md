# SPARK Research

Infrastructure for reinforcement-learning research in stochastic, partially
observable poker environments. The project supports **Leduc**, **Limit Texas**,
and **No-Limit Texas Hold'em** through [RLCard](https://github.com/datamllab/rlcard),
with continuous training workers, a League opponent roster, and a FastAPI
dashboard for campaigns, statistics, and full-state replay on local machines or
the MSOE ROSIE cluster.

## Repository layout

The repo has three top-level folders with clear jobs:

```text
SPARK-Research/
  spark/     # Research code — environments, agents, training, league algorithms
  data/      # Generated artifacts — episodes, campaigns, checkpoints, league files
  app/       # Platform — dashboard UI/API, HPC scripts, tests
  README.md
  requirements.txt
  requirements-dev.txt
  pytest.ini
```

| Folder | What belongs here | When you open it |
| --- | --- | --- |
| [`spark/`](#spark-research-code) | Python you read and change for research | New agents, envs, learning rules, league algorithms |
| [`data/`](#data-storage) | Runtime outputs (gitignored) | Inspect runs, checkpoints, league member files |
| [`app/`](#app-platform) | Dashboard, Slurm helpers, pytest | Launch UI, submit jobs, run tests |

If you still have an older local `runs/` directory from before this layout,
rename it once:

```bash
mv runs data
```

---

## `spark/` — research code

```text
spark/
  agents/                 Agent contract, registry, and per-environment implementations
    base.py               Abstract Agent API
    registry.py           Discovers Agent subclasses under spark.agents.*
    rlcard_model.py       Evaluate-only wrappers for RLCard model-zoo bots
    leduc_holdem/         Random, Q-learning, CFR, rule bots
    limit_holdem/         Random, rule bots
    no_limit_holdem/      Random baseline
  environments/           Shared adapters + legacy collectors
    leduc.py / limit_holdem.py / no_limit_holdem.py
    rlcard_poker.py       Shared RLCard adapter
    registry.py
    LeducPoker/           Legacy CLI collector (archived)
    LimitHoldem/          Legacy CLI collector (archived)
    NoLimitHoldem/        Legacy CLI collector (archived)
  league/                 Research algorithms for the League (stubs today)
    elo.py                Win-rate pairwise ELO transfers after each N-episode window
    evolution.py          Planned Lamarckian evolution over trained members
  storage/                Persistence *code* (not the files themselves)
    database.py           Per-run SQLite schema and queries
    league.py             League member JSON store under data/league-members/
    agent_snapshots.py    Agent library snapshots under data/agent-snapshots/
    import_export.py      Legacy JSON import / portable JSON export
    stats.py              Aggregate statistics for the dashboard
  training/               Continuous trainer and CLI entry point
  slurm/                  Safe sbatch / squeue / scancel wrapper
  EXAMPLE/                Teaching notebooks (Battleship DQN / classical RL)
  models.py               Shared episode / step dataclasses
```

### Environments

Modern training uses the adapters in `spark/environments/` (`leduc-holdem`,
`limit-holdem`, `no-limit-holdem`). They expose a shared interface: reset, step
through decisions, and record both **player-visible** observations and **hidden**
perfect information for replay.

The folders `LeducPoker/`, `LimitHoldem/`, and `NoLimitHoldem/` under
`spark/environments/` are older standalone collectors kept for reference. New
work should use the shared adapters and `python -m spark.training.cli`.

### Agents

Each implementation lives under the matching package:

```text
spark/agents/<environment_package>/<name>_agent.py
```

The registry maps environment IDs to packages:

| Environment ID | Package |
| --- | --- |
| `leduc-holdem` | `spark.agents.leduc_holdem` |
| `limit-holdem` | `spark.agents.limit_holdem` |
| `no-limit-holdem` | `spark.agents.no_limit_holdem` |

Add a new agent by dropping a module into the right package:

```python
# spark/agents/leduc_holdem/my_agent.py
from spark.agents.base import Agent

class MyLeducAgent(Agent):
    agent_id = "my-leduc-agent"
    name = "My Leduc Agent"
    environment_id = "leduc-holdem"

    def select_action(self, observation, legal_actions, player_id):
        return legal_actions[0]

    def update(self, episode):
        return {"loss": 0.0}

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        ...
```

Agent IDs must be unique. Invalid or duplicate modules are reported by
`/api/config` without hiding healthy agents. Each table seat gets its own agent
instance and can train or run frozen in evaluation mode independently.

### Training loop

`spark/training/trainer.py` plays episodes, buffers learning updates, checkpoints
agent state, and optionally re-rolls League opponents. Entry point:

```bash
python -m spark.training.cli \
  --environment leduc-holdem \
  --run-id leduc-demo \
  --episodes 1000 \
  --seed 42
```

Omit `--episodes` to run until interrupted. Ctrl+C finishes the current action
sequence, saves a checkpoint, and marks the run stopped. Re-running the same
`--run-id` resumes from `data/<run-id>/checkpoint.json`.

Per episode (simplified):

1. The acting agent sees only its observation and legal actions.
2. Visible and hidden states are recorded for replay.
3. The episode is committed to SQLite under `data/<run-id>/`.
4. Training seats buffer the episode; every **N** episodes (`--update-interval`)
   the trainer calls `agent.update(episode)` for each buffered episode.
5. League-backed training seats persist `state_dict()` back to the League store.
6. Checkpoints advance so jobs can resume or requeue on Slurm.

### Storage code vs `data/`

`spark/storage/` is **code**: schemas, writers, and readers. The files those APIs
create live under [`data/`](#data-storage). Do not put large generated databases
inside `spark/`.

### League research (`spark/league/`)

Campaigns use a League **roster** (active vs bench members) persisted by
`spark.storage.league.LeagueMemberStore`.

**ELO (`spark/league/elo.py`)**

1. Every new member starts at **1000**.
2. After each training window of **N** episodes (`update_interval`), the trainer
   computes each League-backed seat's win rate over that window.
3. Every unordered pair of those members is compared once. Points move with
   `transfer_points(winrate_a, winrate_b, elo_a, elo_b)`:
   - win-rate ratio near 1:1 → small transfer
   - outperforming a lower-rated opponent → fewer points gained than an upset of
     a higher-rated one (standard ELO expectation term)
4. Transfers are zero-sum; updated ratings are written back to
   `data/league-members/`.

Fresh (non-League) seats do not participate in ELO. At least two League-backed
seats must be present for an update.

**Lamarckian evolution (`spark/league/evolution.py`)** remains a stub
(`propose_offspring` / `inherit_state` raise `NotImplementedError`).

Import surface:

```python
from spark.league import expected_score, transfer_points, update_ratings, LamarckianEvolution
```

### EXAMPLE notebooks

`spark/EXAMPLE/` holds teaching material (e.g. Battleship DQN). It is not used by
the trainer or dashboard.

---

## `data/` — storage

Generated at runtime (gitignored). Override the root with
`SPARK_RUNS_DIRECTORY` if needed; the default is `data/` at the project root.

```text
data/
  dashboard.sqlite              # Campaign and job index for the UI
  agent-snapshots/              # Optional Agent library warm-starts
    agent-*.json
  league-members/               # League roster (active + bench)
    league-*.json
  <run-id>/
    run.sqlite                  # Episodes, steps, metrics, events
    checkpoint.json             # Resumable seat agent states
    worker.log                  # Local campaign worker stdout
  slurm-%j.out / .err           # Slurm training logs
  spark-web-%j.out / .err       # Slurm dashboard logs
```

Each worker is the sole writer to its own `run.sqlite`. The dashboard opens
short-lived read connections and aggregates statistics in SQL.

### Import / export

```python
from spark.storage import RunDatabase
from spark.storage.import_export import import_legacy_json, export_json

database = RunDatabase("data/imported-leduc/run.sqlite")
import_legacy_json(
    "/path/to/legacy/episodes.json",
    database,
    run_id="imported-leduc",
)

database = RunDatabase("data/leduc-demo/run.sqlite", initialize=False)
export_json(database, "data/leduc-demo/leduc-demo.json")
```

The dashboard also exposes `GET /api/runs/<run-id>/export`.

---

## `app/` — platform

```text
app/
  app.py                 FastAPI application
  services/              Campaigns, run listing, dashboard SQLite index
  templates/             Dashboard HTML
  static/                Local CSS and JavaScript (no CDN)
  hpc/                   Rosie / Slurm scripts
    setup_rosie.sh
    start_dashboard.sh
    run_dashboard_ondemand.sh
    train.sbatch
    web.sbatch
  tests/                 Pytest suite
```

You normally do not need to change this for research experiments, except when
extending the dashboard API or cluster launch path.

### Local setup

Python 3.10+ recommended.

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

python -m pip install -r requirements-dev.txt
python -m pytest
```

### Training from the CLI

Finite Leduc experiment:

```bash
python -m spark.training.cli \
  --environment leduc-holdem \
  --run-id leduc-demo \
  --episodes 1000 \
  --seed 42
```

Continuous Limit Hold'em until interrupted:

```bash
python -m spark.training.cli \
  --environment limit-holdem \
  --run-id limit-continuous \
  --checkpoint-interval 100 \
  --seed 42
```

Useful flags: `--update-interval N`, `--league-reroll-interval K`,
`--table-size`, `--seat-config-b64`, `--run-directory` (default `data`).

### Web dashboard

Capabilities:

- Start a campaign with independent jobs per selected environment
- Choose table size for Limit / No-Limit (2–10 seats)
- Seat players from **Fresh implementation**, a specific **League member**, or
  **League (random)** from the active roster
- Buffer learning with **update interval N**; re-roll random League seats every
  **K** episodes (`K=0` keeps one opponent)
- Monitor, stop, and remove jobs; inspect worker logs
- Manage League (active vs Bench), seed built-in RLCard bots
- Optional Agent library snapshots for Fresh warm-starts
- Rolling payoff / outcomes / action frequencies / throughput
- Episode browser with step-by-step visible vs hidden state
- JSON export of a run

#### Campaign seat sources

| Source | Behavior |
| --- | --- |
| Fresh implementation | Pick an agent class + Train/Evaluate; optional library warm-start |
| League member | Seat a specific member; uses that member's train/snapshot mode |
| League (random) | Sample an **active** League member at submit time |

**Update interval (N):** buffer completed episodes; every N episodes call
`update()` on each buffered episode for training seats, then clear the buffer.
Stopping a run flushes any leftover buffer.

**League re-roll (K):** for `league_random` seats only. `K=0` keeps the resolved
member for the whole campaign. `K>0` re-samples every K episodes. Training seats
write state back to `data/league-members/` before a re-roll and on checkpoints.

#### League vs Agent library

- **League** (`data/league-members/`): primary opponent roster for campaigns.
  Active members are drawn by random seats; Bench is sidelined. ELO starts at
  1000 and updates from pairwise win-rate comparisons after each N-episode
  training window (see `spark/league/elo.py`).
- **Agent library** (`data/agent-snapshots/`): optional warm-starts for Fresh
  seats. Prefer importing snapshots into the League when you want them in the
  pool. Collapsed under **Advanced** in the Campaigns UI.

#### On your PC

```bash
# Linux/macOS
SPARK_LOCAL_JOBS=1 python -m uvicorn app.app:app --reload --host 127.0.0.1 --port 8000

# Windows PowerShell
$env:SPARK_LOCAL_JOBS = "1"
py -m uvicorn app.app:app --reload --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Without `SPARK_LOCAL_JOBS=1`
and without Slurm, the dashboard is read-only for launching campaigns.

#### On Rosie (Open OnDemand)

Rosie compute hosts (`dh-node*`) are usually not in campus DNS. Prefer Open
OnDemand over SSH tunnels to compute nodes:
[https://dh-ood.hpc.msoe.edu](https://dh-ood.hpc.msoe.edu).

Prepare the environment:

```bash
cd ~/SummerResearch   # or your clone path
bash app/hpc/setup_rosie.sh
source .venv/bin/activate
```

If needed: `module load python` (check `module avail python` first).

Start the dashboard in an OnDemand shell that is already on a compute node:

```bash
bash app/hpc/run_dashboard_ondemand.sh
```

Open the printed URL (keep `?token=...`):

```text
https://dh-ood.hpc.msoe.edu/rnode/dh-node2/8000/?token=TOKEN
```

If `/rnode/` fails, try `/node/...`. After code updates, restart the script and
hard-refresh the browser.

Optional: submit the dashboard as a Slurm job:

```bash
bash app/hpc/start_dashboard.sh
cat data/spark-web-JOB_ID.out
```

#### Continuous Slurm training

From the Rosie dashboard, a campaign submits one Slurm job per environment via
`app/hpc/train.sbatch` with `--requeue`. Two minutes before the wall-time limit,
Slurm sends `USR1`; the worker checkpoints and the job requeues. Use the
dashboard Stop button or `scancel JOB_ID` to halt a campaign.

Confirm current partitions and limits in the
[ROSIE User Guide](https://msoe.dev/#/).

---

## Environment status

| Environment | Status |
| --- | --- |
| Leduc Hold'em | Enabled (`rlcard.make("leduc-holdem")`) |
| Limit Texas Hold'em | Enabled, 2–10 players |
| No-Limit Texas Hold'em | Enabled, 2–10 players |

## Extending SPARK

| Goal | Where to work |
| --- | --- |
| New learning agent | `spark/agents/<env_package>/` |
| New or changed game rules / adapter | `spark/environments/` |
| Win-rate League ELO | `spark/league/elo.py` |
| Population / Lamarckian evolution | `spark/league/evolution.py` |
| Trainer cadence / checkpoints | `spark/training/` |
| Dashboard UI or API | `app/` |
| Cluster launch scripts | `app/hpc/` |
| Inspect episodes / league files | `data/` |
