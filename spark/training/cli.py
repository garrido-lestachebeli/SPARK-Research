"""Single local/Slurm command for continuous training."""

from __future__ import annotations

import argparse
import uuid

from spark.environments import list_environments

from .trainer import Trainer, TrainerConfig


def build_parser() -> argparse.ArgumentParser:
    enabled = [item.id for item in list_environments() if item.enabled]
    parser = argparse.ArgumentParser(description="Run a SPARK online training worker.")
    parser.add_argument("--environment", choices=enabled, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-directory", default="runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
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
        "--agent-memory-size",
        type=int,
        default=100,
        help="Maximum completed episodes retained in each agent checkpoint.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_id = args.run_id or f"{args.environment}-{uuid.uuid4().hex[:10]}"
    trainer = Trainer(
        TrainerConfig(
            run_id=run_id,
            environment=args.environment,
            seed=args.seed,
            run_directory=args.run_directory,
            checkpoint_interval=max(1, args.checkpoint_interval),
            episode_limit=args.episodes,
            shared_agents=not args.separate_seat_agents,
            agent_memory_size=max(0, args.agent_memory_size),
        )
    )
    completed = trainer.run(resume=not args.no_resume)
    print(f"Run {run_id} stopped after {completed} episode(s).")


if __name__ == "__main__":
    main()
