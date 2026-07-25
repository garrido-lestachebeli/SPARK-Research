import json

import pytest

from spark.storage import AgentSnapshotError, AgentSnapshotStore
from spark.training.trainer import Trainer, TrainerConfig


def run_config(tmp_path, run_id, environment="leduc-holdem", **values):
    return TrainerConfig(
        run_id=run_id,
        environment=environment,
        run_directory=str(tmp_path),
        episode_limit=values.pop("episode_limit", 3),
        checkpoint_interval=1,
        **values,
    )


def test_save_materialize_load_and_delete_snapshot(tmp_path):
    Trainer(run_config(tmp_path, "source")).run()
    source_checkpoint = tmp_path / "source" / "checkpoint.json"
    store = AgentSnapshotStore(tmp_path)

    saved = store.save_from_run("Leduc baseline", "source")
    assert saved["environment"] == "leduc-holdem"
    assert saved["source"]["agent_version"] == 3
    assert store.list()[0]["name"] == "Leduc baseline"

    result = store.materialize_run_checkpoint(
        saved["id"],
        run_id="warm-start",
        environment="leduc-holdem",
        seed=99,
        checkpoint_interval=1,
        episode_limit=1,
        agent_memory_size=100,
    )
    checkpoint = json.loads(
        (tmp_path / "warm-start" / "checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["next_episode_number"] == 0
    assert checkpoint["agent_version"] == 0
    assert checkpoint["agents"][0]["episodes_seen"] == 3
    assert not result["shared_agents"]

    trainer = Trainer(run_config(tmp_path, "warm-start", episode_limit=1))
    assert trainer.run() == 1
    assert trainer.agents[0].episodes_seen == 4

    store.delete(saved["id"])
    assert store.list() == []
    assert source_checkpoint.exists()
    assert (tmp_path / "source" / "run.sqlite").exists()


def test_snapshot_rejects_wrong_environment_and_invalid_ids(tmp_path):
    Trainer(run_config(tmp_path, "source")).run()
    store = AgentSnapshotStore(tmp_path)
    saved = store.save_from_run("Safe snapshot", "source")

    with pytest.raises(AgentSnapshotError, match="not compatible"):
        store.materialize_run_checkpoint(
            saved["id"],
            run_id="wrong-environment",
            environment="limit-holdem",
            seed=1,
            checkpoint_interval=1,
            episode_limit=1,
            agent_memory_size=10,
        )
    with pytest.raises(AgentSnapshotError):
        store.get("../checkpoint")
    with pytest.raises(AgentSnapshotError):
        store.save_from_run("Bad run", "../source")


def test_legacy_checkpoint_keys_are_normalized(tmp_path):
    Trainer(run_config(tmp_path, "legacy-source")).run()
    path = tmp_path / "legacy-source" / "checkpoint.json"
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    checkpoint["policies"] = checkpoint.pop("agents")
    checkpoint["policy_version"] = checkpoint.pop("agent_version")
    path.write_text(json.dumps(checkpoint), encoding="utf-8")

    saved = AgentSnapshotStore(tmp_path).save_from_run(
        "Legacy checkpoint", "legacy-source"
    )
    assert saved["source"]["agent_version"] == 3


def test_snapshot_saves_one_selected_seat(tmp_path):
    Trainer(
        run_config(
            tmp_path,
            "separate-source",
            episode_limit=1,
            seat_configs=[
                {"agent_id": "leduc-random", "mode": "train"},
                {"agent_id": "leduc-q", "mode": "evaluate"},
            ],
        )
    ).run()
    store = AgentSnapshotStore(tmp_path)
    saved = store.save_from_run("Q evaluator", "separate-source", seat=1)
    assert not saved["shared_agents"]
    assert saved["agent_count"] == 1
    assert saved["agent_id"] == "leduc-q"
    state = store.state_for(
        saved["id"],
        environment="leduc-holdem",
        agent_id="leduc-q",
    )
    assert state["agent_id"] == "leduc-q"
    with pytest.raises(AgentSnapshotError, match="not 'leduc-random'"):
        store.state_for(
            saved["id"],
            environment="leduc-holdem",
            agent_id="leduc-random",
        )
