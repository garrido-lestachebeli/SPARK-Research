"""Blackjack episode runner using Gymnasium's Blackjack-v1 environment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Iterator, Optional

import gymnasium as gym
from gymnasium import Env

ACTION_NAMES = {0: "stick", 1: "hit"}


def format_state(state: tuple[int, int, int]) -> dict[str, Any]:
    player_sum, dealer_card, usable_ace = state
    return {
        "player_sum": player_sum,
        "dealer_showing": dealer_card,
        "usable_ace": bool(usable_ace),
    }


@dataclass
class StepRecord:
    step: int
    action: int
    action_name: str
    state_before: dict[str, Any]
    state_after: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepRecord:
        return cls(
            step=int(data["step"]),
            action=int(data["action"]),
            action_name=str(data["action_name"]),
            state_before=dict(data["state_before"]),
            state_after=dict(data["state_after"]),
            reward=float(data["reward"]),
            terminated=bool(data["terminated"]),
            truncated=bool(data["truncated"]),
            info=dict(data.get("info", {})),
        )


@dataclass
class EpisodeReport:
    seed: Optional[int]
    initial_state: dict[str, Any]
    steps: list[StepRecord] = field(default_factory=list)
    total_reward: float = 0.0
    num_steps: int = 0
    terminated: bool = False
    truncated: bool = False
    outcome: str = ""
    final_state: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "initial_state": self.initial_state,
            "steps": [step.to_dict() for step in self.steps],
            "total_reward": self.total_reward,
            "num_steps": self.num_steps,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "outcome": self.outcome,
            "final_state": self.final_state,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EpisodeReport:
        return cls(
            seed=data.get("seed"),
            initial_state=dict(data["initial_state"]),
            steps=[StepRecord.from_dict(step) for step in data.get("steps", [])],
            total_reward=float(data.get("total_reward", 0.0)),
            num_steps=int(data.get("num_steps", 0)),
            terminated=bool(data.get("terminated", False)),
            truncated=bool(data.get("truncated", False)),
            outcome=str(data.get("outcome", "")),
            final_state=dict(data.get("final_state", {})),
        )


def describe_outcome(total_reward: float) -> str:
    if total_reward > 0:
        return "win"
    if total_reward < 0:
        return "loss"
    return "draw"


def random_action_provider(env: Env) -> Callable[[], int]:
    def choose_action() -> int:
        return int(env.action_space.sample())

    return choose_action


def run_episode(
    env: Env,
    *,
    seed: Optional[int] = None,
    actions: Optional[Iterator[int]] = None,
    action_fn: Optional[Callable[[], int]] = None,
) -> EpisodeReport:
    """Run one Blackjack episode and return a detailed report.

  Args:
      env: A Gymnasium Blackjack-v1 environment.
      seed: Optional RNG seed passed to env.reset().
      actions: Optional iterable of actions (0=stick, 1=hit). Falls back to
          action_fn or random sampling when exhausted.
      action_fn: Callable returning the next action. Used when `actions` is
          not provided or is exhausted.
  """
    if actions is None and action_fn is None:
        action_fn = random_action_provider(env)

    action_iter = iter(actions) if actions is not None else None

    def next_action() -> int:
        if action_iter is not None:
            try:
                return int(next(action_iter))
            except StopIteration:
                pass
        if action_fn is not None:
            return int(action_fn())
        return int(env.action_space.sample())

    reset_kwargs: dict[str, Any] = {}
    if seed is not None:
        reset_kwargs["seed"] = seed

    state, _info = env.reset(**reset_kwargs)
    report = EpisodeReport(
        seed=seed,
        initial_state=format_state(state),
        final_state=format_state(state),
    )

    step_index = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action = next_action()
        state_before = state
        state, reward, terminated, truncated, info = env.step(action)

        report.steps.append(
            StepRecord(
                step=step_index,
                action=action,
                action_name=ACTION_NAMES.get(action, f"unknown({action})"),
                state_before=format_state(state_before),
                state_after=format_state(state),
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                info=dict(info),
            )
        )
        report.total_reward += float(reward)
        step_index += 1

    report.num_steps = step_index
    report.terminated = terminated
    report.truncated = truncated
    report.final_state = format_state(state)
    report.outcome = describe_outcome(report.total_reward)
    return report


def format_report(report: EpisodeReport) -> str:
    """Return a human-readable summary of an episode report."""
    lines = [
        "=== Blackjack Episode Report ===",
        f"Seed: {report.seed}",
        f"Initial state: {report.initial_state}",
        f"Outcome: {report.outcome} (total reward: {report.total_reward})",
        f"Steps taken: {report.num_steps}",
        "",
        "Step-by-step:",
    ]

    for step in report.steps:
        lines.append(
            f"  [{step.step}] {step.action_name} "
            f"| state: {step.state_before} -> {step.state_after} "
            f"| reward: {step.reward} "
            f"| done: {step.terminated or step.truncated}"
        )

    lines.extend(
        [
            "",
            f"Final state: {report.final_state}",
            f"Terminated: {report.terminated}, Truncated: {report.truncated}",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    import argparse

    from storage import DEFAULT_EPISODES_FILE, save_episodes

    parser = argparse.ArgumentParser(description="Run Blackjack episodes with Gymnasium.")
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
        help="Save episodes to a JSON file (default: Blackjack/episodes.json).",
    )
    parser.add_argument(
        "--print",
        dest="print_reports",
        action="store_true",
        help="Print each episode report to the terminal.",
    )
    args = parser.parse_args()

    env = gym.make("Blackjack-v1")
    reports: list[EpisodeReport] = []

    for index in range(args.count):
        episode_seed = None if args.seed is None else args.seed + index
        report = run_episode(env, seed=episode_seed)
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
