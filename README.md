# SPARK Research

Infrastructure for reinforcement-learning research in imperfect-information poker.  
SPARK trains agents on **Leduc**, **Limit Texas**, and **No-Limit Texas Hold'em**  
([RLCard](https://github.com/datamllab/rlcard)), runs campaigns locally or on the  
MSOE ROSIE cluster, and ships a dashboard for League opponents, ELO tracking, and  
full episode replay (visible + hidden state).

**Core ideas**


| Piece            | Role                                                                             |
| ---------------- | -------------------------------------------------------------------------------- |
| **Environments** | Poker variants with player-visible observations vs hidden perfect info           |
| **Agents**       | Auto-discovered `Agent` subclasses; each seat trains or evaluates independently  |
| **Trainer**      | Online loop: play → log episode → `update()` → checkpoint; optional League eval  |
| **League**       | Opponent roster with ELO; seats can load members or sample active ones at random |
| **Campaigns**    | One or more env jobs launched from the dashboard (local or Slurm)                |
| **Runs**         | Per-job SQLite DB + checkpoint under `data/<run-id>/`                            |


Research work usually means editing `spark/agents/` and `spark/league/`. Use `data/`
to inspect results; change `app/` when extending the UI or cluster path.

## Repository layout


| Folder   | Contents                                                        |
| -------- | --------------------------------------------------------------- |
| `spark/` | Environments, agents, trainer, league, storage, Slurm helpers   |
| `app/`   | FastAPI dashboard, HPC scripts, tests                           |
| `data/`  | Runtime output (gitignored): run DBs, checkpoints, League, logs |


```text
spark/
  agents/         Per-environment agents (auto-discovered)
  environments/   RLCard adapters
  league/         ELO (+ evolution stub)
  storage/        SQLite, League JSON, snapshots
  training/       Online trainer + CLI
  slurm/          sbatch / squeue / scancel wrappers
app/
  app.py, static/, templates/   Dashboard
  hpc/                          Rosie setup and sbatch scripts
  tests/
data/
  <run-id>/run.sqlite, checkpoint.json
  league-members/, agent-snapshots/, dashboard.sqlite
```



## Setup

Python 3.10+ recommended.

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install -r requirements-dev.txt
pytest
```



## Quick start



### CLI training

```bash
python -m spark.training.cli \
  --environment leduc-holdem \
  --run-id leduc-demo \
  --episodes 1000 \
  --seed 42
```

Omit `--episodes` to run until Ctrl+C (finishes the current hand, checkpoints, stops).
Re-running the same `--run-id` resumes from `data/<run-id>/checkpoint.json`.


| Flag                         | Meaning                                                      |
| ---------------------------- | ------------------------------------------------------------ |
| `--eval-interval K`          | Every K episodes, freeze and test vs the League (`0` = off)  |
| `--eval-episodes N`          | Test hands per League member per eval round (default 20)     |
| `--league-reroll-interval K` | Re-sample random League seats every K episodes               |
| `--table-size`               | Players for Limit / No-Limit (2–10)                          |
| `--starting-stack`           | Chip buy-in per episode (default 50)                         |
| `--episode-mode`             | `session` (play until bust) or `hand` (one hand per episode) |
| `--run-directory`            | Override default `data/` root                                |




### Dashboard

```bash
# Linux/macOS
SPARK_LOCAL_JOBS=1 python -m uvicorn app.app:app --reload --host 127.0.0.1 --port 8000

# Windows PowerShell
$env:SPARK_LOCAL_JOBS = "1"
py -m uvicorn app.app:app --reload --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Without `SPARK_LOCAL_JOBS=1`
(and without Slurm), you can browse runs but not start new workers.

From the UI: launch campaigns, seat players, monitor/stop jobs, manage the League,
view stats, and replay episodes. **Delete data** on a job or campaign removes that
run’s files under `data/` (League members and Agent library snapshots are kept).

## Campaigns, League, and episodes

**Seats** can be a fresh agent, a specific League member, or a random active League
member. Training seats call `update()` after each episode. With `--eval-interval K`
(`K > 0`), the trainer periodically freezes the learner and plays test episodes
against every active League member (stored separately; not used for learning).

**Episode mode:** `session` plays until someone cannot post the blind (or a safety
hand limit); `hand` is one poker hand per episode.

**League** members are active or benched. ELO starts at 1000 and updates from
windowed pairwise win rates among League-backed seats (`spark/league/elo.py`).
The Agent library (`data/agent-snapshots/`) is for warm-starting fresh seats;
prefer the League as the live roster.

Lamarckian evolution hooks in `spark/league/evolution.py` are not implemented yet.

## Rosie (MSOE HPC)

Use Open OnDemand: [https://dh-ood.hpc.msoe.edu](https://dh-ood.hpc.msoe.edu)
(compute nodes are usually not reachable by SSH from a laptop).

```bash
cd ~/SummerResearch   # or your clone path
bash app/hpc/setup_rosie.sh
source .venv/bin/activate

# From Shell Access / an interactive job on dh-node*:
bash app/hpc/run_dashboard_ondemand.sh
```

Open the printed URL (keep `?token=...`). Campaigns submit one Slurm job per
environment via `app/hpc/train.sbatch`. Near wall-time the job checkpoints and
requeues. See the [ROSIE User Guide](https://msoe.dev/#/) for partitions and limits.

## Adding an agent

Drop a class under `spark/agents/leduc_holdem/`, `limit_holdem/`, or
`no_limit_holdem/`. Subclasses of `Agent` are discovered at startup:

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

Agent IDs must be unique; `environment_id` must match the package.


| Goal                        | Where to edit                 |
| --------------------------- | ----------------------------- |
| New agent                   | `spark/agents/<env_package>/` |
| Game / observation adapter  | `spark/environments/`         |
| ELO                         | `spark/league/elo.py`         |
| Evolution / population      | `spark/league/evolution.py`   |
| Train cadence / checkpoints | `spark/training/`             |
| Dashboard or API            | `app/`                        |
| Cluster scripts             | `app/hpc/`                    |


