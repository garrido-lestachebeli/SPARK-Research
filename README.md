# SPARK-Research

Implementing Bayes' rule in reinforcement learning for stochastic, partially observable environments.

This repo builds small card-game environments (Blackjack, Leduc Hold'em, and later Limit / No-Limit Texas Hold'em) that:

---

## Project layout

```
SPARK-Research/
├── Blackjack/          # Gymnasium Blackjack-v1 (single agent vs dealer)
│   ├── main.py         # run episodes, print reports, optional --save
│   ├── storage.py      # save / load / append episodes.json
│   ├── replay.py       # interactive ASCII replay
│   └── episodes.json   # saved episode data
├── LeducPoker/         # RLCard leduc-holdem (2 players)
│   ├── main.py
│   ├── storage.py
│   ├── replay.py
│   └── episodes.json
├── LimitHoldem/        # planned (RLCard limit-holdem)
├── NoLimitHoldem/      # planned (RLCard no-limit-holdem)
└── README.md
```

Each environment folder is self-contained: collect data with `main.py`, inspect it with `replay.py`, and load the same JSON later for training.

---

## Setup

**Windows (local):**

```bash
pip install gymnasium rlcard
```

**Linux / school supercomputer:**

```bash
python3 -m pip install --user gymnasium rlcard
```

Use `python` / `py` on Windows and `python3` on Linux. Do **not** use the Windows `py` launcher on the cluster.

---

## Shared workflow

For every game the pattern is the same:

| Step | Command idea | What it does |
|------|----------------|--------------|
| Collect | `.../main.py --count N --seed S --save` | Run `N` random episodes and write `episodes.json` |
| Inspect | `.../replay.py` | Step through saved episodes in the terminal |
| Train (later) | policy + training loop | Consume episode reports to update an agent |

### Replay controls (Blackjack and Leduc)

| Key | Action |
|-----|--------|
| `n` / `p` | Next / previous step in the current episode |
| `f` / `l` | Jump to first / last step |
| `>` / `<` | Next / previous episode in the file |
| `N` | Run a new random episode (auto-appended to the JSON file) |
| `g #` | Go to timeline step `#` |
| `r #` | Switch to report `#` |
| `a` | Auto-play through the timeline |
| `s` / `load` | Save or load an episodes file |
| `q` | Quit |

---

## Blackjack

**Library:** [Gymnasium](https://gymnasium.farama.org/) `Blackjack-v1`

**What it is:** Classic single-player blackjack against a fixed dealer. Observation is `(player_sum, dealer_showing, usable_ace)`. Actions: `0 = stick`, `1 = hit`. Reward is assigned only on the final step (`+1` win, `-1` loss, `0` draw).

**Run episodes:**

```bash
# Windows
py Blackjack/main.py --count 5 --seed 42 --save

# Linux / supercomputer
python3 Blackjack/main.py --count 5 --seed 42 --save
```

Episodes are written to `Blackjack/episodes.json`.

**Replay:**

```bash
py Blackjack/replay.py
py Blackjack/replay.py Blackjack/custom.json
```

**CLI flags for `main.py`:**

- `--count N` — number of episodes (default `1`)
- `--seed S` — first episode seed; later episodes use `S+1`, `S+2`, ...
- `--save [path]` — save to `Blackjack/episodes.json` or an optional custom path
- `--print` — print each report even when saving

**Example episode (simplified):**

```json
{
  "seed": 42,
  "initial_state": {
    "player_sum": 17,
    "dealer_showing": 10,
    "usable_ace": true
  },
  "steps": [
    {
      "step": 0,
      "action": 0,
      "action_name": "stick",
      "state_before": {
        "player_sum": 17,
        "dealer_showing": 10,
        "usable_ace": true
      },
      "state_after": {
        "player_sum": 17,
        "dealer_showing": 10,
        "usable_ace": true
      },
      "reward": -1.0,
      "terminated": true,
      "truncated": false,
      "info": {}
    }
  ],
  "total_reward": -1.0,
  "num_steps": 1,
  "terminated": true,
  "truncated": false,
  "outcome": "loss",
  "final_state": {
    "player_sum": 17,
    "dealer_showing": 10,
    "usable_ace": true
  }
}
```

**Next steps for Blackjack:**

- Pass a real `policy` into `run_episode` instead of random actions
- Add a training loop that updates the policy from episode reports

---

## Leduc Poker

**Library:** [RLCard](https://rlcard.org/) `leduc-holdem`

**What it is:** A tiny two-player poker game used a lot in imperfect-information RL research. Deck has six cards (Jack, Queen, King × two suits). Each player gets one private card; one public card is revealed before the second betting round. Both seats currently play randomly.

**Important differences from Blackjack:**

- Two competing agents; each step records which `player_id` acted
- No meaningful per-step reward — RLCard only returns **payoffs at the end** of the hand
- Seed is set when the environment is **created**, not on `reset()`
- Reports also store `perfect_info` (both hands, public card, chips) for replay / analysis

**Actions:** `0 = call`, `1 = raise`, `2 = fold`, `3 = check`

**Run episodes:**

```bash
python LeducPoker/main.py --count 5 --seed 42 --save
# or on the cluster:
python3 LeducPoker/main.py --count 5 --seed 42 --save
```

Episodes are written to `LeducPoker/episodes.json`.

**Replay:**

```bash
python LeducPoker/replay.py
python LeducPoker/replay.py LeducPoker/custom.json
```

In the ASCII viewer, Player 0's hand is shown, Player 1's hand stays hidden until the final step (showdown / fold), and the public card appears when it is dealt.

**Example episode (simplified):**

```json
{
  "seed": 42,
  "num_players": 2,
  "initial_state": {
    "hand": "HJ",
    "public_card": null,
    "my_chips": 1,
    "all_chips": [2, 1],
    "current_player": 1,
    "legal_actions": ["call", "raise", "fold"]
  },
  "steps": [
    {
      "step": 0,
      "player_id": 1,
      "action": 0,
      "action_name": "call",
      "state_before": { "...": "..." },
      "state_after": { "...": "..." },
      "legal_actions": ["call", "raise", "fold"],
      "perfect_info": {
        "hand_cards": ["SJ", "HJ"],
        "public_card": null,
        "chips": [2, 2]
      }
    }
  ],
  "payoffs": [1.0, -1.0],
  "num_steps": 2,
  "outcomes": ["win", "loss"],
  "winner": 0
}
```

**Dependency:** `pip install rlcard` (or `python3 -m pip install --user rlcard` on the supercomputer).

---

## Limit Texas Hold'em

Planned. Will use RLCard `limit-holdem` with the same `main.py` / `storage.py` / `replay.py` pattern as Leduc.

## No-Limit Texas Hold'em

Planned. Will use RLCard `no-limit-holdem` with the same pattern.
