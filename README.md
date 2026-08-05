# SPARK Research

SPARK is infrastructure for reinforcement-learning research in imperfect-information
poker. It supports **Leduc**, **Limit Texas**, and **No-Limit Texas Hold'em** via
[RLCard](https://github.com/datamllab/rlcard).

You can train agents from the command line or through a web dashboard that launches
campaigns (locally or on the MSOE ROSIE cluster), manages a **League** of opponents,
tracks **ELO**, and lets you replay full episodes with both visible and hidden state.

## Repository layout

Everything sits in three top-level folders:


| Folder   | What it is                                                                 |
| -------- | -------------------------------------------------------------------------- |
| `spark/` | Research code you edit — environments, agents, trainer, league algorithms  |
| `data/`  | Runtime output — episode DBs, checkpoints, league files, logs (gitignored) |
| `app/`   | Platform — FastAPI dashboard, Slurm helpers, tests                         |


```text
spark/
  agents/           Per-environment agent implementations (auto-discovered)
  environments/     RLCard adapters for the three poker variants
  league/           ELO updates; Lamarckian evolution stub
  storage/          Code that reads/writes data/ (SQLite, league JSON, snapshots)
  training/         Online trainer and CLI
  slurm/            Safe wrappers around sbatch / squeue / scancel
app/
  app.py, static/, templates/   Dashboard
  hpc/                          Rosie setup and sbatch scripts
  tests/                        Pytest suite
data/
  <run-id>/run.sqlite, checkpoint.json
  league-members/, agent-snapshots/, dashboard.sqlite
```

When doing research, spend most of your time in `spark/agents/` and `spark/league/`.
Use `data/` to inspect results; change `app/` only when extending the UI or cluster path.

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



## Training from the CLI

```bash
python -m spark.training.cli \
  --environment leduc-holdem \
  --run-id leduc-demo \
  --episodes 1000 \
  --seed 42
```

Omit `--episodes` to run until you press Ctrl+C. The worker finishes the current
hand, writes a checkpoint, and stops cleanly. Re-running the same `--run-id`
resumes from `data/<run-id>/checkpoint.json`.

Useful flags:


| Flag                         | Meaning                                                            |
| ---------------------------- | ------------------------------------------------------------------ |
| `--update-interval N`        | Buffer N episodes, then call `update()` on all of them (default 1) |
| `--league-reroll-interval K` | Re-roll random League seats every K episodes (`0` = never)         |
| `--table-size`               | Players for Limit / No-Limit (2–10)                                |
| `--starting-stack`           | Chip buy-in at the start of each session episode (default 50)      |
| `--run-directory`            | Override the default `data/` root                                  |


Each **episode** is a cash session: players start with a chip stack (default 50)
and keep playing hands until someone cannot post the big blind (bust), or a
safety hand limit is reached. Terminal payoffs are net chips won or lost over
the whole session. Observations include `stacks`, `my_stack`, and `hand_index`
so agents can condition on bankroll across hands.

Player-visible observations are stored separately from hidden perfect
information, so the dashboard can show exactly what the agent saw versus the
true cards. Training seats only learn from their own observations via
`agent.update()`.

## Dashboard

Local launch:

```bash
# Linux/macOS
SPARK_LOCAL_JOBS=1 python -m uvicorn app.app:app --reload --host 127.0.0.1 --port 8000

# Windows PowerShell
$env:SPARK_LOCAL_JOBS = "1"
py -m uvicorn app.app:app --reload --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Without `SPARK_LOCAL_JOBS=1`
(and without Slurm), you can browse runs but cannot start new campaign workers.

From the UI you can:

- Start a campaign across one or more environments
- Seat each player, set N / K, seed, episode limit, and Slurm resources
- Monitor jobs, stop them, and read worker logs
- Manage the League roster and optional Agent library snapshots
- View statistics and step through episode replays



### Campaign seats


| Source               | Behavior                                                                   |
| -------------------- | -------------------------------------------------------------------------- |
| Fresh implementation | New agent instance (Train or Evaluate); optional library warm-start        |
| League member        | Load a specific roster member; mode follows that member (train / snapshot) |
| League (random)      | Draw an **active** League member at submit time                            |


**Update interval (N):** episodes are buffered; every N completed episodes the
trainer runs `update()` on each buffered episode for training seats, then clears
the buffer. Stopping a run flushes any leftover buffer so learning is not lost.

**Starting stack:** chip buy-in for each seat at the start of a session episode
(default 50). Play continues until someone cannot post the big blind.

**League re-roll (K):** only for `League (random)` seats. `K = 0` keeps the same
opponent for the whole campaign. `K > 0` samples a new active member every K
episodes (preferring a different one when the pool has more than one). Training
seats write their weights back to the League before a re-roll.

### League and Agent library

The **League** (`data/league-members/`) is the main opponent roster. Members are
either **active** (eligible for random seats) or on the **Bench**. You can create
blank agents, import snapshots, seed built-in RLCard bots, and toggle train vs
snapshot mode.

**ELO** starts at **1000**. After each N-episode training window, every unordered
pair of League-backed seats at the table is compared using win rates over that
window. Points transfer with `spark/league/elo.py`: near 1:1 win rates move few
points, and beating a much weaker (lower-ELO) opponent yields less than an upset
of a stronger one. Transfers are zero-sum and written back to the member files.
Fresh (non-League) seats do not participate; at least two League seats are required.

The **Agent library** (`data/agent-snapshots/`) is a secondary store for warm-starting
Fresh seats. Prefer the League as the live roster; import library snapshots into
the League when you want them in the pool.

Lamarckian evolution hooks live in `spark/league/evolution.py` but are not
implemented yet.

## Rosie (MSOE HPC)

Rosie compute nodes are usually not reachable by SSH from a laptop. Use Open
OnDemand: [https://dh-ood.hpc.msoe.edu](https://dh-ood.hpc.msoe.edu).

```bash
cd ~/SummerResearch   # or your clone path
bash app/hpc/setup_rosie.sh
source .venv/bin/activate

# From a Shell Access / interactive job already on dh-node*:
bash app/hpc/run_dashboard_ondemand.sh
```

Open the printed URL (keep `?token=...`). If `/rnode/` fails, try `/node/...`.

Campaigns on Rosie submit one Slurm job per environment via `app/hpc/train.sbatch`.
Near the wall-time limit the job checkpoints and requeues so training can continue.
Stop a job from the dashboard or with `scancel`. See the
[ROSIE User Guide](https://msoe.dev/#/) for partitions and limits.

## Adding an agent

Put a new class under the matching package — `spark/agents/leduc_holdem/`,
`limit_holdem/`, or `no_limit_holdem/`. The registry discovers every `Agent`
subclass when the trainer or dashboard starts:

```python
# spark/agents/leduc_holdem/my_agent.py
from spark.agents.base import Agent

class MyLeducAgent(Agent):
    agent_id = "my-leduc-agent"
    name = "My Leduc Agent"
    environment_id = "leduc-holdem"

    def select_action(self, observation, legal_actions, player_id):
        # observation has no opponent hole cards / perfect info
        return legal_actions[0]

    def update(self, episode):
        # Learn from the completed trajectory and terminal payoffs
        return {"loss": 0.0}

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        ...
```

Agent IDs must be unique. `environment_id` must match the package. Each seat gets
its own instance and can train or evaluate independently.


| Goal                        | Where to edit                 |
| --------------------------- | ----------------------------- |
| New agent                   | `spark/agents/<env_package>/` |
| Game / observation adapter  | `spark/environments/`         |
| ELO formula                 | `spark/league/elo.py`         |
| Evolution / population      | `spark/league/evolution.py`   |
| Train cadence / checkpoints | `spark/training/`             |
| Dashboard or API            | `app/`                        |
| Cluster scripts             | `app/hpc/`                    |




## Bayesian hidden-state inference (work in progress)

Goal: when an opponent acts, update a belief over their hole card using their known policy and Bayes' rule

### `agent.action_probs(observation, legal_actions)`

Returns the likelihood ingredient: π(a | observation) over currently legal actions.

**Signature**

```python
probs = agent.action_probs(observation, legal_actions)  # -> dict[int, float]
```

**Return format**

- `dict[int, float]` mapping each **legal** action id → probability
- Values are ≥ 0 and sum to `1.0`
- Illegal actions are omitted (not listed as 0)

Example (Leduc legal set `[0, 1, 2]` = call / raise / fold):

```python
{0: 0.25, 1: 0.50, 2: 0.25}
```

For a Bayes update, use only the observed action:

```python
likelihood = agent.action_probs(obs_for_hypothesized_card, legal)[observed_action]
```

**Where it lives**


| Agent            | File                                   | Behavior                                                     |
| ---------------- | -------------------------------------- | ------------------------------------------------------------ |
| Default / random | `spark/agents/base.py`                 | Uniform over `legal_actions`                                 |
| Q-learning       | `spark/agents/leduc_holdem/q_agent.py` | ε-greedy mix while training; greedy one-hot while evaluating |
| CFR / rule bots  | `spark/agents/rlcard_model.py`         | CFR: mixed average policy; rules: one-hot on chosen action   |


CFR and rule agents inherit from `RLCardModelAgent`; they do not redefine the method.

**Observation format**

Same player-visible dict the environment already builds (`decision.observation` /
`step.visible_before`):

```python
{
    "raw": {...},           # e.g. hand, public_card, chips, ...
    "encoded": [...],       # float vector (Leduc length 36)
    "action_record": [...],
    "stacks": [...],
    "my_stack": float,
    "hand_index": int,
    "starting_stack": float,
    "big_blind": float,
}
```

`action_probs` does **not** loop over possible hole cards. Each call assumes one
info state already encoded in `observation`. The Bayes loop calls it once per
hypothesized card after constructing a matching `obs_c`.

### Opponent wiring and per-card likelihood scaffold

- `Agent.set_opponent(other)` stores a runtime reference to another seat (not saved
  in checkpoints). The trainer wires seats automatically after create, league
  re-roll, and checkpoint load (2-player tables point at each other).
- `LeducQAgent` calls `_query_opponent_action_probs` at the start of
  `select_action`, builds one hypothetical opponent observation per remaining
  hole card (5 preflop / 4 postflop), and stores:

```python
agent.opponent_action_probs_by_card  # dict[card_id, dict[action_id, float]]
```

**Still to implement**

- Prior / posterior belief over opponent cards (Bayes update using the scaffold)
- A Bayes agent that uses the belief to act

