"""Import legacy episode JSON and export run data as portable JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spark.models import Episode, EpisodeStep, outcomes_from_payoffs, winner_from_payoffs

from .database import RunDatabase


def import_legacy_json(
    source: str | Path,
    database: RunDatabase,
    *,
    run_id: str,
    agent_name: str = "legacy-random",
) -> int:
    payload = json.loads(Path(source).read_text(encoding="utf-8"))
    raw_episodes = payload if isinstance(payload, list) else payload.get("episodes", [])
    environment = (
        payload.get("game", "unknown") if isinstance(payload, dict) else "unknown"
    )
    first_seed = next(
        (int(item["seed"]) for item in raw_episodes if item.get("seed") is not None),
        0,
    )
    database.create_run(
        run_id,
        environment,
        agent_name,
        first_seed,
        {"imported_from": str(source)},
    )
    imported = 0
    for number, raw in enumerate(raw_episodes):
        episode = _legacy_episode(raw, run_id, number, environment)
        database.save_episode(episode)
        imported += 1
    database.update_run(status="imported", episode_count=imported)
    return imported


def export_json(database: RunDatabase, destination: str | Path) -> Path:
    run = database.get_run()
    if not run:
        raise ValueError("Database contains no run.")
    summaries = database.list_episodes(limit=2_147_483_647)
    episodes = []
    for summary in sorted(summaries, key=lambda item: item["episode_number"]):
        episode = database.get_episode(summary["id"])
        if episode:
            episodes.append(episode.to_dict())
    payload = {
        "version": 2,
        "game": run["environment"],
        "run_id": run["id"],
        "episode_count": len(episodes),
        "episodes": episodes,
    }
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _legacy_episode(
    raw: dict[str, Any],
    run_id: str,
    number: int,
    environment: str,
) -> Episode:
    payoffs = [float(value) for value in raw.get("payoffs", [])]
    if not payoffs and "total_reward" in raw:
        payoffs = [float(raw["total_reward"])]
    outcomes = [str(value) for value in raw.get("outcomes", [])]
    if not outcomes:
        outcomes = outcomes_from_payoffs(payoffs)
    hidden = dict(raw.get("initial_perfect_info", {}))
    steps: list[EpisodeStep] = []
    for index, old_step in enumerate(raw.get("steps", [])):
        hidden_after = dict(old_step.get("perfect_info", hidden))
        legal_names = list(old_step.get("legal_actions", []))
        action_ids = {
            "call": 0,
            "raise": 1,
            "fold": 2,
            "check": 3,
            "stick": 0,
            "hit": 1,
        }
        steps.append(
            EpisodeStep(
                index=int(old_step.get("step", index)),
                player_id=int(old_step.get("player_id", 0)),
                agent_version=0,
                action=int(old_step["action"]),
                action_name=str(old_step.get("action_name", old_step["action"])),
                legal_actions=[
                    action_ids[name] for name in legal_names if name in action_ids
                ],
                visible_before=dict(old_step.get("state_before", {})),
                visible_after=dict(old_step.get("state_after", {})),
                hidden_before=hidden,
                hidden_after=hidden_after,
            )
        )
        hidden = hidden_after
    return Episode(
        run_id=run_id,
        episode_number=number,
        environment=environment,
        seed=int(raw.get("seed") or number),
        agent_version=0,
        steps=steps,
        payoffs=payoffs,
        outcomes=outcomes,
        winner=raw.get("winner", winner_from_payoffs(payoffs)),
    )
