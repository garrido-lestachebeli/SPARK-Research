"""Single local/Slurm command for continuous training."""

from __future__ import annotations

import argparse
import base64
import json
import uuid

from spark.environments import list_environments

from .trainer import Trainer, TrainerConfig


def build_parser() -> argparse.ArgumentParser:
    enabled = [item.id for item in list_environments() if item.enabled]
    parser = argparse.ArgumentParser(description="Run a SPARK online training worker.")
    parser.add_argument("--environment", choices=enabled, required=True)
    parser.add_argument(
        "--table-size",
        type=int,
        default=2,
        help="Number of seats (Leduc requires 2; Hold'em supports 2-10).",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-directory", default="data")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=0,
        help=(
            "Freeze the agent every K episodes and test it against every active "
            "League member; 0 disables evaluation."
        ),
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        help="Test episodes played against each League member per evaluation round.",
    )
    parser.add_argument(
        "--league-reroll-interval",
        type=int,
        default=0,
        help="Re-roll league_random seats every K episodes; 0 keeps them fixed.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Stop after this many episodes; omit to run until stopped.",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--separate-seat-agents",
        action="store_true",
        help="Instantiate one agent per poker seat instead of sharing one.",
    )

    parser.add_argument(
        "--agents",
        nargs="+",
        default=None,
        help=(
            "Agents by seat order. Example: "
            "--agents leduc-random leduc-q"
        ),
    )
    parser.add_argument(
        "--seat-config",
        default=None,
        help='JSON list of {"agent_id", "mode"} objects in seat order.',
    )
    parser.add_argument(
        "--seat-config-b64",
        default=None,
        help="URL-safe base64 encoded seat-config JSON (used by Slurm).",
    )
    parser.add_argument(
        "--agent-memory-size",
        type=int,
        default=100,
        help="Maximum completed episodes retained in each agent checkpoint.",
    )
    parser.add_argument(
        "--starting-stack",
        type=float,
        default=50.0,
        help="Chip buy-in for each seat at the start of a cash-session episode.",
    )
    parser.add_argument(
        "--episode-mode",
        choices=["session", "hand"],
        default="session",
        help=(
            "session = play hands until bust (cash game); "
            "hand = one poker hand per episode."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    encoded_config = args.seat_config
    if args.seat_config_b64:
        encoded_config = base64.urlsafe_b64decode(
            args.seat_config_b64.encode("ascii")
        ).decode("utf-8")
    seat_configs = json.loads(encoded_config) if encoded_config else None
    if seat_configs is not None and not isinstance(seat_configs, list):
        raise ValueError("--seat-config must decode to a JSON list.")
    if seat_configs == []:
        seat_configs = None
    run_id = args.run_id or f"{args.environment}-{uuid.uuid4().hex[:10]}"
    trainer = Trainer(
        TrainerConfig(
            run_id=run_id,
            environment=args.environment,
            table_size=args.table_size,
            seat_configs=seat_configs,
            agents=args.agents,
            seed=args.seed,
            run_directory=args.run_directory,
            checkpoint_interval=max(1, args.checkpoint_interval),
            episode_limit=args.episodes,
            shared_agents=not args.separate_seat_agents,
            agent_memory_size=max(0, args.agent_memory_size),
            eval_interval=max(0, args.eval_interval),
            eval_episodes=max(1, args.eval_episodes),
            league_reroll_interval=max(0, args.league_reroll_interval),
            starting_stack=max(1.0, float(args.starting_stack)),
            episode_mode=args.episode_mode,
        )
    )
    completed = trainer.run(resume=not args.no_resume)
    # Fold the write-ahead log back into run.sqlite before exiting; a leftover
    # log makes every later dashboard connection pay to re-validate it.
    trainer.database.checkpoint_wal()
    print(f"Run {run_id} stopped after {completed} episode(s).")


if __name__ == "__main__":
    main()
