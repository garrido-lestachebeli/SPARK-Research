import json
from pathlib import Path

import pytest

from spark.storage import RunDatabase
from spark.training.trainer import Trainer, TrainerConfig


def config(directory: Path, run_id: str, episodes: int) -> TrainerConfig:
    return TrainerConfig(
        run_id=run_id,
        environment="leduc-holdem",
        seed=91,
        run_directory=str(directory),
        episode_limit=episodes,
        checkpoint_interval=1,
    )


def episode_payoffs(database: RunDatabase) -> list[list[float]]:
    summaries = sorted(database.list_episodes(limit=100), key=lambda item: item["episode_number"])
    return [list(item["payoffs"]) for item in summaries]


def test_agent_updates_after_every_episode_and_metadata_is_recorded(tmp_path):
    trainer = Trainer(config(tmp_path, "online", 3))
    trainer.run()

    database = RunDatabase(tmp_path / "online" / "run.sqlite", initialize=False)
    run = database.get_run()
    assert run["episode_count"] == 3
    assert run["agent_version"] == 3
    assert trainer.agents[0].episodes_seen == 3
    assert len(trainer.agents[0].episode_memory) == 3

    summary = database.list_episodes(limit=1)[0]
    assert summary["step_count"] > 0
    assert len(summary["payoffs"]) == 2
    episode = database.get_episode(summary["id"])
    # Observation logs are intentionally not persisted; replay has no frames.
    assert episode.steps == []
    stats = database.statistics()
    assert stats["episodes"] == 3
    assert stats["average_steps"] > 0
    assert stats["actions"]


def test_checkpoint_resume_matches_uninterrupted_random_agent(tmp_path):
    baseline = Trainer(config(tmp_path, "baseline", 5))
    baseline.run()

    first = Trainer(config(tmp_path, "resumed", 3))
    first.run()
    checkpoint_path = tmp_path / "resumed" / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["policy_version"] = checkpoint.pop("agent_version")
    checkpoint["policies"] = checkpoint.pop("agents")
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    resumed = Trainer(config(tmp_path, "resumed", 5))
    resumed.run()

    baseline_db = RunDatabase(tmp_path / "baseline" / "run.sqlite", initialize=False)
    resumed_db = RunDatabase(tmp_path / "resumed" / "run.sqlite", initialize=False)
    assert episode_payoffs(baseline_db) == episode_payoffs(resumed_db)


def test_separate_seat_agents_are_configurable(tmp_path):
    trainer = Trainer(
        TrainerConfig(
            run_id="separate",
            environment="limit-holdem",
            run_directory=str(tmp_path),
            episode_limit=0,
            seat_configs=[
                {"agent_id": "limit-random", "mode": "train"},
                {"agent_id": "limit-random", "mode": "evaluate"},
            ],
        )
    )
    assert trainer.agents[0] is not trainer.agents[1]
    assert trainer.agents[0].training
    assert not trainer.agents[1].training


def test_frozen_agent_is_not_updated(tmp_path):
    trainer = Trainer(
        TrainerConfig(
            run_id="frozen",
            environment="leduc-holdem",
            run_directory=str(tmp_path),
            episode_limit=3,
            seat_configs=[
                {"agent_id": "leduc-random", "mode": "train"},
                {"agent_id": "leduc-q", "mode": "evaluate"},
            ],
        )
    )
    trainer.run()
    assert trainer.agents[0].episodes_seen == 3
    assert trainer.agents[1].episodes_seen == 0


def test_q_agent_trains_and_round_trips_checkpoint(tmp_path):
    seat_configs = [
        {"agent_id": "leduc-q", "mode": "train"},
        {"agent_id": "leduc-random", "mode": "evaluate"},
    ]
    trainer = Trainer(
        TrainerConfig(
            run_id="q-learning",
            environment="leduc-holdem",
            run_directory=str(tmp_path),
            episode_limit=3,
            checkpoint_interval=1,
            seat_configs=seat_configs,
        )
    )
    trainer.run()
    assert trainer.agents[0].episodes_seen == 3
    assert len(trainer.agents[0].q) > 0
    restored = Trainer(
        TrainerConfig(
            run_id="q-learning",
            environment="leduc-holdem",
            run_directory=str(tmp_path),
            episode_limit=4,
            checkpoint_interval=1,
            seat_configs=seat_configs,
        )
    )
    restored.load_checkpoint()
    assert restored.agents[0].q == trainer.agents[0].q


@pytest.mark.parametrize(
    "environment_id",
    ["leduc-holdem", "limit-holdem", "no-limit-holdem"],
)
def test_each_environment_runs_with_its_matching_agent(tmp_path, environment_id):
    trainer = Trainer(
        TrainerConfig(
            run_id=f"smoke-{environment_id}",
            environment=environment_id,
            run_directory=str(tmp_path),
            episode_limit=1,
        )
    )
    assert trainer.agents[0].environment_id == environment_id
    assert trainer.run() == 1


def test_training_seat_learns_after_every_episode(tmp_path):
    trainer = Trainer(
        TrainerConfig(
            run_id="per-episode",
            environment="leduc-holdem",
            run_directory=str(tmp_path),
            episode_limit=3,
            seat_configs=[
                {"agent_id": "leduc-q", "mode": "train"},
                {"agent_id": "leduc-random", "mode": "evaluate"},
            ],
        )
    )
    trainer.run()
    assert trainer.agent_version == 3
    assert trainer.agents[0].episodes_seen == 3
    assert len(trainer.agents[0].q) > 0


def evaluation_config(directory: Path, run_id: str, **overrides) -> TrainerConfig:
    settings = {
        "run_id": run_id,
        "environment": "leduc-holdem",
        "run_directory": str(directory),
        "episode_limit": 2,
        "eval_interval": 2,
        "eval_episodes": 3,
        "seat_configs": [
            {"agent_id": "leduc-q", "mode": "train"},
            {"agent_id": "leduc-random", "mode": "evaluate"},
        ],
    }
    settings.update(overrides)
    return TrainerConfig(**settings)


def test_evaluation_round_tests_every_active_league_member(tmp_path):
    from spark.storage import LeagueMemberStore

    league = LeagueMemberStore(tmp_path)
    league.create_blank("A", "leduc-holdem", "leduc-q", status="league")
    league.create_blank("B", "leduc-holdem", "leduc-random", status="league")
    league.create_blank("Benched", "leduc-holdem", "leduc-random", status="bench")

    trainer = Trainer(evaluation_config(tmp_path, "sweep"))
    trainer.run()

    # Evaluation must not consume the episode budget or the training tables.
    assert trainer.next_episode_number == 2
    database = RunDatabase(tmp_path / "sweep" / "run.sqlite", initialize=False)
    assert database.get_run()["episode_count"] == 2
    assert len(database.list_episodes(limit=100)) == 2

    summary = database.evaluation_summary()
    assert summary["rounds"] == 1
    # Two active members times three test episodes; the benched member is excluded.
    assert summary["episodes"] == 6
    opponents = summary["seats"]["0"]["opponents"]
    assert [item["name"] for item in opponents] == ["A", "B"]
    boxes = summary["seats"]["0"]["rounds"][0]["opponents"]
    assert all(item["n"] == 3 for item in boxes)


def test_evaluation_leaves_the_live_training_agent_untouched(tmp_path):
    from spark.storage import LeagueMemberStore

    league = LeagueMemberStore(tmp_path)
    league.create_blank("A", "leduc-holdem", "leduc-q", status="league")

    trainer = Trainer(evaluation_config(tmp_path, "frozen-sweep", episode_limit=1, eval_interval=1))
    trainer.run_episode(0)
    before = json.dumps(trainer.agents[0].state_dict(), sort_keys=True)
    seen_before = trainer.agents[0].episodes_seen
    trainer.next_episode_number = 1
    trainer._run_evaluation_round()

    assert json.dumps(trainer.agents[0].state_dict(), sort_keys=True) == before
    assert trainer.agents[0].episodes_seen == seen_before


def test_evaluation_without_league_opponents_is_skipped(tmp_path):
    trainer = Trainer(evaluation_config(tmp_path, "no-opponents"))
    trainer.run()

    database = RunDatabase(tmp_path / "no-opponents" / "run.sqlite", initialize=False)
    assert database.evaluation_summary()["episodes"] == 0
    assert database.get_run()["status"] == "completed"


def test_evaluation_excludes_the_training_seats_own_league_entry(tmp_path):
    from spark.storage import LeagueMemberStore

    league = LeagueMemberStore(tmp_path)
    own = league.create_blank("Self", "leduc-holdem", "leduc-q", status="league")
    league.create_blank("Rival", "leduc-holdem", "leduc-random", status="league")

    trainer = Trainer(
        evaluation_config(
            tmp_path,
            "self-excluded",
            seat_configs=[
                {
                    "source": "league_member",
                    "agent_id": "leduc-q",
                    "mode": "train",
                    "league_member_id": own["id"],
                },
                {"agent_id": "leduc-random", "mode": "evaluate"},
            ],
        )
    )
    trainer.run()

    database = RunDatabase(tmp_path / "self-excluded" / "run.sqlite", initialize=False)
    opponents = database.evaluation_summary()["seats"]["0"]["opponents"]
    assert [item["name"] for item in opponents] == ["Rival"]
    assert [
        item["name"]
        for item in database.evaluation_summary()["seats"]["0"]["rounds"][0]["opponents"]
    ] == ["Rival"]


def test_league_random_reroll_persists_and_swaps(tmp_path):
    from spark.storage import LeagueMemberStore

    league = LeagueMemberStore(tmp_path)
    first = league.create_blank("A", "leduc-holdem", "leduc-q", status="league")
    second = league.create_blank("B", "leduc-holdem", "leduc-random", status="league")
    trainer = Trainer(
        TrainerConfig(
            run_id="reroll",
            environment="leduc-holdem",
            run_directory=str(tmp_path),
            episode_limit=2,
            league_reroll_interval=1,
            seat_configs=[
                {
                    "source": "fresh",
                    "agent_id": "leduc-random",
                    "mode": "evaluate",
                },
                {
                    "source": "league_random",
                    "agent_id": first["agent_id"],
                    "mode": "train",
                    "league_member_id": first["id"],
                },
            ],
        )
    )
    # Warm-start seat 1 from league member A.
    trainer.agents[1].load_state_dict(league.state_for(first["id"], environment="leduc-holdem"))
    trainer.agents[1].set_training(True)
    trainer.run()
    # After each episode (K=1), seat 1 should re-roll; final member is one of the pool.
    final_id = trainer.seat_configs[1]["league_member_id"]
    assert final_id in {first["id"], second["id"]}
    # Prior training state was written back at least once.
    saved = league.get(first["id"])
    assert "state" in saved
