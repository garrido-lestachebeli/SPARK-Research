import json

import pytest

from spark.agents import create_agent, list_agents
from spark.environments import create_environment
from spark.storage import (
    AgentSnapshotStore,
    LeagueError,
    LeagueMemberNotFound,
    LeagueMemberStore,
)
from spark.training.trainer import Trainer, TrainerConfig


def test_league_blank_import_copy_move_and_delete(tmp_path):
    Trainer(
        TrainerConfig(
            run_id="league-source",
            environment="leduc-holdem",
            run_directory=str(tmp_path),
            episode_limit=2,
            checkpoint_interval=1,
        )
    ).run()
    snapshots = AgentSnapshotStore(tmp_path)
    snapshot = snapshots.save_from_run("Baseline", "league-source")
    league = LeagueMemberStore(tmp_path)

    blank = league.create_blank("Fresh Q", "leduc-holdem", "leduc-q", status="league")
    assert blank["elo"] == 1000
    assert blank["status"] == "league"
    assert blank["source_kind"] == "blank"
    assert blank["stats"]["matches"] == 0
    # Trainable implementations default to training mode.
    assert blank["mode"] == "train"
    assert blank["supports_training"] is True

    imported = league.create_from_snapshot(
        "From snapshot",
        snapshot["id"],
        status="bench",
    )
    assert imported["status"] == "bench"
    assert imported["source_kind"] == "snapshot"
    assert imported["agent_id"] == snapshot["agent_id"]

    copied = league.copy_member(imported["id"], "Copied member")
    assert copied["elo"] == 1000
    assert copied["id"] != imported["id"]
    full = league.get(copied["id"])
    assert "q_table" in full["state"] or copied["agent_id"] == "leduc-random"

    moved = league.set_status(blank["id"], "bench")
    assert moved["status"] == "bench"
    renamed = league.rename(blank["id"], "Renamed Q")
    assert renamed["name"] == "Renamed Q"

    snapshotted = league.set_mode(blank["id"], "snapshot")
    assert snapshotted["mode"] == "snapshot"
    trained = league.set_mode(blank["id"], "train")
    assert trained["mode"] == "train"
    with pytest.raises(LeagueError):
        league.set_mode(blank["id"], "bogus")

    active = league.list(environment="leduc-holdem", status="league")
    bench = league.list(environment="leduc-holdem", status="bench")
    assert all(item["status"] == "league" for item in active)
    assert {item["id"] for item in bench} >= {blank["id"], imported["id"]}

    detail = league.detail(blank["id"])
    assert "elo_note" in detail
    assert "q_table" not in detail["state_preview"]

    league.delete(blank["id"])
    with pytest.raises(LeagueMemberNotFound):
        league.get(blank["id"])


def test_league_seed_builtins_and_reject_bad_ids(tmp_path):
    league = LeagueMemberStore(tmp_path)
    created = league.seed_builtins("leduc-holdem")
    assert {item["agent_id"] for item in created} == {
        "leduc-cfr",
        "leduc-rule-v1",
        "leduc-rule-v2",
    }
    again = league.seed_builtins("leduc-holdem")
    assert again == []

    limit_created = league.seed_builtins("limit-holdem")
    assert {item["agent_id"] for item in limit_created} == {"limit-rule-v1"}

    # RLCard opponents cannot train, so they are forced to snapshot mode.
    cfr = next(item for item in created if item["agent_id"] == "leduc-cfr")
    assert cfr["mode"] == "snapshot"
    assert cfr["supports_training"] is False
    forced = league.create_blank(
        "Frozen CFR",
        "leduc-holdem",
        "leduc-cfr",
        mode="train",
    )
    assert forced["mode"] == "snapshot"

    with pytest.raises(LeagueError, match="No agent"):
        league.create_blank("Bad", "leduc-holdem", "limit-random")
    with pytest.raises(LeagueError):
        league.get("../league-bad")


def test_rlcard_agents_register_and_act():
    ids = {item["id"] for item in list_agents()}
    assert {
        "leduc-cfr",
        "leduc-rule-v1",
        "leduc-rule-v2",
        "limit-rule-v1",
    } <= ids

    environment = create_environment("leduc-holdem", 2)
    decision = environment.reset(0)
    for agent_id in ("leduc-cfr", "leduc-rule-v1", "leduc-rule-v2"):
        agent = create_agent(agent_id, seed=1, config={"training": False, "seat_id": 0})
        assert agent.supports_training is False
        action = agent.select_action(
            decision.observation,
            list(decision.legal_actions),
            decision.player_id,
        )
        assert action in decision.legal_actions

    payload = create_agent("leduc-cfr").state_dict()
    restored = create_agent("leduc-cfr")
    restored.load_state_dict(payload)
    assert restored.training is False
