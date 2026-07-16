from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

import rlcard

ACTION_NAMES = {0: "call", 1: "raise", 2: "fold", 3: "check"}

def make_env(seed: Optional[int] = None):
    config = {}
    if seed is not None:
        config["seed"] = seed
    return rlcard.make("limit-holdem", config=config)


def _jsonify(value: Any) -> Any:
    """Convert numpy scalars/arrays and nested values into JSON-friendly types."""
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except (ValueError, AttributeError):
            pass
    return value

def format_state(state: dict[str, Any] | None) -> dict[str, Any]:
    if state is None:
        return {}

    raw_obs = state.get("raw_obs", {}) or {}

    legal_actions = state.get("raw_legal_actions")
    if legal_actions is None:
        legal_actions = raw_obs.get("legal_actions", [])

    return {
        "hand": _jsonify(raw_obs.get("hand")),
        "public_cards": _jsonify(raw_obs.get("public_cards")),
        "all_chips": _jsonify(raw_obs.get("all_chips")),
        "my_chips": _jsonify(raw_obs.get("my_chips")),
        "pot": _jsonify(raw_obs.get("pot")),
        "stage": raw_obs.get("stage"),
        "current_player": _jsonify(raw_obs.get("current_player")),
        "legal_actions": list(legal_actions),
    }



#def format_perfect_info(env) -> dict[str, Any]:
    try:
        info = env.get_perfect_information()
    except Exception:
        return {}

    return {
        "hand_cards": _jsonify(info.get("hand_cards")),
        "public_cards": _jsonify(info.get("public_cards")),
        "chips": _jsonify(info.get("chips")),
        "dealer_button": _jsonify(info.get("dealer_button")),
        "current_player": _jsonify(info.get("current_player")),
        "legal_actions": _jsonify(info.get("legal_actions")),
        "round_counter": _jsonify(info.get("round_counter")),
    }

def format_perfect_info(env) -> dict[str, Any]:
    try:
        info = env.get_perfect_information()
    except Exception:
        info = {}

    public_cards = (
        info.get("public_cards")
        or info.get("public_card")
        or []
    )

    return {
        "hand_cards": _jsonify(info.get("hand_cards")),
        "public_cards": _jsonify(public_cards),
        "chips": _jsonify(info.get("chips")),
        "dealer_button": _jsonify(info.get("dealer_button")),
        "current_player": _jsonify(info.get("current_player")),
        "legal_actions": _jsonify(info.get("legal_actions")),
        "round_counter": _jsonify(info.get("round_counter")),
    }

@dataclass
class StepRecord:
    step: int
    player_id: int
    action: int
    action_name: str
    state_before: dict[str, Any]
    state_after: dict[str, Any]
    legal_actions: list[str] = field(default_factory=list)
    perfect_info: dict[str, Any] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepRecord:
        return cls(
            step=int(data["step"]),
            player_id=int(data["player_id"]),
            action=int(data["action"]),
            action_name=str(data["action_name"]),
            state_before=dict(data["state_before"]),
            state_after=dict(data.get("state_after", {})),
            legal_actions=list(data.get("legal_actions", [])),
            perfect_info=dict(data.get("perfect_info", {})),
            info=dict(data.get("info", {})),
        )

@dataclass
class EpisodeReport:
    seed: Optional[int]
    num_players: int = 2
    initial_state: dict[str, Any] = field(default_factory=dict)
    initial_perfect_info: dict[str, Any] = field(default_factory=dict)
    steps: list[StepRecord] = field(default_factory=list)
    payoffs: list[float] = field(default_factory=list)
    num_steps: int = 0
    outcomes: list[str] = field(default_factory=list)
    winner: Optional[int] = None
    final_perfect_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "num_players": self.num_players,
            "initial_state": self.initial_state,
            "initial_perfect_info": self.initial_perfect_info,
            "steps": [step.to_dict() for step in self.steps],
            "payoffs": self.payoffs,
            "num_steps": self.num_steps,
            "outcomes": self.outcomes,
            "winner": self.winner,
            "final_perfect_info": self.final_perfect_info,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodeReport:
        return cls(
            seed=data.get("seed"),
            num_players=int(data.get("num_players", 2)),
            initial_state=dict(data.get("initial_state", {})),
            initial_perfect_info=dict(data.get("initial_perfect_info", {})),
            steps=[
                StepRecord.from_dict(step)
                for step in data.get("steps", [])
            ],
            payoffs=[
                float(x)
                for x in data.get("payoffs", [])
            ],
            num_steps=int(data.get("num_steps", 0)),
            outcomes=list(data.get("outcomes", [])),
            winner=data.get("winner"),
            final_perfect_info=dict(
                data.get("final_perfect_info", {})
            ),
        )

def describe_outcomes(payoffs: list[float]) -> list[str]:
    outcomes: list[str] = []
    for payoff in payoffs:
        if payoff > 0:
            outcomes.append("win")
        elif payoff < 0:
            outcomes.append("loss")
        else:
            outcomes.append("draw")
    return outcomes


def choose_winner(payoffs: list[float]) -> Optional[int]:
    if not payoffs:
        return None
    best = max(payoffs)
    winners = [index for index, payoff in enumerate(payoffs) if payoff == best]
    if best <= 0 or len(winners) != 1:
        return None
    return winners[0]


def random_action_provider(env) -> Callable[[dict[str, Any]], int]:
    """Return a callable that samples uniformly from legal actions."""

    def choose_action(state: dict[str, Any]) -> int:
        legal = list(state.get("legal_actions", {}).keys())
        if not legal:
            raise RuntimeError("No legal actions available.")
        return int(random.choice(legal))

    return choose_action


def run_episode(
    env=None,
    *,
    seed: Optional[int] = None,
    action_fn: Optional[Callable[[dict[str, Any]], int]] = None,
) -> EpisodeReport:
    """Run one Leduc Hold'em episode and return a detailed report.

    Args:
        env: Optional pre-built RLCard environment. If omitted (or when a seed
            is provided), a fresh env is created with that seed.
        seed: Optional RNG seed used when creating the environment.
        action_fn: Callable(state) -> action id. Defaults to random legal action.
    """
    if env is None or seed is not None:
        env = make_env(seed=seed)

    if action_fn is None:
        action_fn = random_action_provider(env)

    state, player_id = env.reset()
    report = EpisodeReport(
        seed=seed,
        num_players=int(env.num_players),
        initial_state=format_state(state),
        initial_perfect_info=format_perfect_info(env),
    )

    step_index = 0
    while not env.is_over():
        state_before = format_state(state)
        legal_actions = list(state_before.get("legal_actions", []))
        action = int(action_fn(state))
        action_name = ACTION_NAMES.get(action, f"unknown({action})")

        next_state, next_player_id = env.step(action)

        print("game.public_cards:", env.game.public_cards)
        print("perfect info:", env.get_perfect_information()) #added two lines

        state_after = format_state(None if env.is_over() else next_state)

        report.steps.append(
            StepRecord(
                step=step_index,
                player_id=int(player_id),
                action=action,
                action_name=action_name,
                state_before=state_before,
                state_after=state_after,
                legal_actions=legal_actions,
                perfect_info=format_perfect_info(env),
                info={},
            )
        )
        step_index += 1
        state = next_state
        player_id = next_player_id

    payoffs = [_jsonify(value) for value in env.get_payoffs()]
    report.payoffs = [float(value) for value in payoffs]
    report.num_steps = step_index
    report.outcomes = describe_outcomes(report.payoffs)
    report.winner = choose_winner(report.payoffs)
    report.final_perfect_info = format_perfect_info(env)
    return report


def format_report(report: EpisodeReport) -> str:
    """Return a human-readable summary of an episode report."""
    lines = [
        "=== Limit Hold'em Episode Report ===",
        f"Seed: {report.seed}",
        f"Players: {report.num_players}",
        f"Initial state: {report.initial_state}",
        f"Initial perfect info: {report.initial_perfect_info}",
        f"Payoffs: {report.payoffs}",
        f"Outcomes: {report.outcomes}",
        f"Winner: {report.winner}",
        f"Steps taken: {report.num_steps}",
        "",
        "Step-by-step:",
    ]

    for step in report.steps:
        lines.append(
            f"  [{step.step}] player {step.player_id} {step.action_name} "
            f"| before: {step.state_before} "
            f"| after: {step.state_after} "
            f"| legal: {step.legal_actions}"
        )

    lines.extend(
        [
            "",
            f"Final perfect info: {report.final_perfect_info}",
        ]
    )
    return "\n".join(lines)

def main() -> None:
    import argparse

    from storage import DEFAULT_EPISODES_FILE, save_episodes

    parser = argparse.ArgumentParser(description="Run Leduc Hold'em episodes with RLCard.")
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of episodes to run (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for the first episode. Later episodes use seed+1, seed+2, ...",
    )
    parser.add_argument(
        "--save",
        nargs="?",
        const=str(DEFAULT_EPISODES_FILE),
        default=None,
        help="Save episodes to a JSON file (default: LeducPoker/episodes.json).",
    )
    parser.add_argument(
        "--print",
        dest="print_reports",
        action="store_true",
        help="Print each episode report to the terminal.",
    )
    args = parser.parse_args()

    reports: list[EpisodeReport] = []

    for index in range(args.count):
        episode_seed = None if args.seed is None else args.seed + index
        report = run_episode(seed=episode_seed)
        reports.append(report)

        if args.print_reports or args.save is None:
            print(format_report(report))
            print()

    if args.save is not None:
        saved_path = save_episodes(reports, args.save)
        print(f"Saved {len(reports)} episode(s) to {saved_path}")
    elif args.count == 1:
        print("Structured report:")
        print(reports[0].to_dict())


if __name__ == "__main__":
    main()