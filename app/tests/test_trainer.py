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


def episode_actions(database: RunDatabase) -> list[list[int]]:
    summaries = sorted(database.list_episodes(limit=100), key=lambda item: item["episode_number"])
    return [
        [step.action for step in database.get_episode(item["id"]).steps]
        for item in summaries
    ]


def test_agent_updates_after_every_episode_and_hidden_state_is_recorded(tmp_path):
    trainer = Trainer(config(tmp_path, "online", 3))
    trainer.run()

    database = RunDatabase(tmp_path / "online" / "run.sqlite", initialize=False)
    run = database.get_run()
    assert run["episode_count"] == 3
    assert run["agent_version"] == 3
    assert trainer.agents[0].episodes_seen == 3
    assert len(trainer.agents[0].episode_memory) == 3

    episode = database.get_episode(database.list_episodes(limit=1)[0]["id"])
    assert episode.steps[0].hidden_before["hand_cards"]
    assert "hand_cards" not in episode.steps[0].visible_before


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
    assert episode_actions(baseline_db) == episode_actions(resumed_db)


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


def test_update_interval_buffers_episodes_then_trains_on_all(tmp_path):
    trainer = Trainer(
        TrainerConfig(
            run_id="buffered",
            environment="leduc-holdem",
            run_directory=str(tmp_path),
            episode_limit=3,
            update_interval=3,
            seat_configs=[
                {"agent_id": "leduc-q", "mode": "train"},
                {"agent_id": "leduc-random", "mode": "evaluate"},
            ],
        )
    )
    trainer.run()
    # One flush after 3 episodes → one agent_version bump; all 3 episodes learned.
    assert trainer.agent_version == 1
    assert trainer.agents[0].episodes_seen == 3
    assert len(trainer.agents[0].q) > 0


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
            update_interval=1,
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
