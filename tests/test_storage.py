import json
import sqlite3

from spark.storage import RunDatabase
from spark.storage.import_export import export_json, import_legacy_json


def test_legacy_import_statistics_and_export(tmp_path):
    source = tmp_path / "episodes.json"
    source.write_text(
        json.dumps(
            {
                "version": 1,
                "game": "leduc-holdem",
                "episodes": [
                    {
                        "seed": 7,
                        "initial_perfect_info": {
                            "hand_cards": ["SJ", "HQ"],
                            "chips": [2, 1],
                        },
                        "steps": [
                            {
                                "step": 0,
                                "player_id": 1,
                                "action": 2,
                                "action_name": "fold",
                                "legal_actions": ["call", "raise", "fold"],
                                "state_before": {"hand": "HQ"},
                                "state_after": {},
                                "perfect_info": {
                                    "hand_cards": ["SJ", "HQ"],
                                    "chips": [2, 1],
                                },
                            }
                        ],
                        "payoffs": [0.5, -0.5],
                        "outcomes": ["win", "loss"],
                        "winner": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    database = RunDatabase(tmp_path / "run.sqlite")
    assert import_legacy_json(source, database, run_id="imported") == 1
    stats = database.statistics()
    assert stats["episodes"] == 1
    assert stats["outcomes"]["0"]["win"] == 1
    assert stats["seat_payoffs"]["0"]["mean"] == 0.5
    assert stats["seat_outcomes"]["0"]["win_rate"]["rate"] == 1.0
    assert stats["actions"][0]["action"] == "fold"

    destination = export_json(database, tmp_path / "export.json")
    exported = json.loads(destination.read_text(encoding="utf-8"))
    assert exported["episode_count"] == 1
    assert exported["episodes"][0]["steps"][0]["hidden_before"]["hand_cards"]


def test_policy_era_database_columns_remain_readable_and_are_migrated(tmp_path):
    path = tmp_path / "legacy.sqlite"
    database = RunDatabase(path)
    database.create_run("legacy", "leduc-holdem", "random", 3, {})

    with sqlite3.connect(path) as connection:
        connection.execute("ALTER TABLE runs RENAME COLUMN agent_name TO policy_name")
        connection.execute(
            "ALTER TABLE runs RENAME COLUMN agent_version TO policy_version"
        )
        connection.execute(
            "ALTER TABLE episodes RENAME COLUMN agent_version TO policy_version"
        )
        connection.execute(
            "ALTER TABLE steps RENAME COLUMN agent_version TO policy_version"
        )
        connection.execute(
            "ALTER TABLE checkpoints RENAME COLUMN agent_version TO policy_version"
        )

    legacy_run = RunDatabase(path, initialize=False).get_run()
    assert legacy_run["agent_name"] == "random"
    assert legacy_run["agent_version"] == 0

    RunDatabase(path)
    with sqlite3.connect(path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(runs)").fetchall()
        }
    assert {"agent_name", "agent_version"} <= columns
