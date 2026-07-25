from app.services.campaigns import CampaignService
from app.services.dashboard_store import DashboardStore
from app.services.runs import RunsService
from spark.training.trainer import Trainer, TrainerConfig


class FakeSlurm:
    available = True

    def __init__(self):
        self.submitted = []
        self.cancelled = []

    def submit_training(self, config):
        self.submitted.append(config)
        return str(10_000 + len(self.submitted))

    def status(self, job_id):
        return {"job_id": job_id, "state": "RUNNING", "elapsed": "00:01", "node": "node1"}

    def cancel(self, job_id):
        self.cancelled.append(job_id)


def test_campaign_submits_one_job_per_environment_and_cancels(tmp_path):
    store = DashboardStore(tmp_path / "dashboard.sqlite")
    service = CampaignService(tmp_path, RunsService(tmp_path / "runs"), store)
    fake = FakeSlurm()
    service.slurm = fake

    result = service.create(
        {
            "name": "test",
            "environments": ["leduc-holdem", "limit-holdem"],
            "seed": 42,
            "checkpoint_interval": 10,
            "cpus": 1,
            "memory": "1G",
            "wall_time": "00:10:00",
        }
    )
    assert len(result["jobs"]) == 2
    assert [item["environment"] for item in fake.submitted] == [
        "leduc-holdem",
        "limit-holdem",
    ]
    assert fake.submitted[0]["table_size"] == 2
    assert len(fake.submitted[1]["seat_configs"]) == 2
    metadata = store.run_campaign_metadata()
    assert {
        value["campaign_name"] for value in metadata.values()
    } == {"test"}
    assert len(metadata) == 2

    service.stop_campaign(result["campaign_id"])
    assert fake.cancelled == ["10001", "10002"]


def test_remove_job_keeps_run_data_and_campaign_label(tmp_path):
    store = DashboardStore(tmp_path / "dashboard.sqlite")
    runs = RunsService(tmp_path / "runs")
    service = CampaignService(tmp_path, runs, store)
    fake = FakeSlurm()
    service.slurm = fake
    result = service.create(
        {
            "name": "keep-history",
            "environments": ["leduc-holdem"],
            "seed": 7,
        }
    )
    run_id = result["jobs"][0]["run_id"]
    Trainer(
        TrainerConfig(
            run_id=run_id,
            environment="leduc-holdem",
            run_directory=str(runs.runs_directory),
            episode_limit=1,
        )
    ).run()

    removed = service.remove(run_id)
    assert removed == {"run_id": run_id, "stopped": True}
    assert store.list_campaigns()[0]["jobs"] == []
    assert store.run_campaign_metadata()[run_id]["campaign_name"] == "keep-history"
    assert (runs.runs_directory / run_id / "run.sqlite").exists()
    assert runs.list_runs()[0]["id"] == run_id
    assert fake.cancelled == ["10001"]


def test_campaign_materializes_selected_agent_snapshot(tmp_path):
    store = DashboardStore(tmp_path / "dashboard.sqlite")
    runs = RunsService(tmp_path / "runs")
    Trainer(
        TrainerConfig(
            run_id="source",
            environment="leduc-holdem",
            run_directory=str(runs.runs_directory),
            episode_limit=2,
        )
    ).run()
    service = CampaignService(tmp_path, runs, store)
    snapshot = service.agent_snapshots.save_from_run("starter", "source")
    fake = FakeSlurm()
    service.slurm = fake

    result = service.create(
        {
            "name": "warm",
            "tables": {
                "leduc-holdem": {
                    "table_size": 2,
                    "seats": [
                        {
                            "source": "fresh",
                            "agent_id": "leduc-random",
                            "mode": "train",
                            "snapshot_id": snapshot["id"],
                        },
                        {
                            "source": "fresh",
                            "agent_id": "leduc-q",
                            "mode": "evaluate",
                        },
                    ],
                }
            },
            "seed": 9,
            "update_interval": 5,
            "league_reroll_interval": 0,
        }
    )
    run_id = result["jobs"][0]["run_id"]
    checkpoint = runs.runs_directory / run_id / "checkpoint.json"
    assert checkpoint.exists()
    assert fake.submitted[0]["seat_configs"][0]["snapshot_id"] == snapshot["id"]
    assert fake.submitted[0]["seat_configs"][1]["mode"] == "evaluate"
    assert fake.submitted[0]["update_interval"] == 5


def test_campaign_resolves_league_random_and_rejects_empty_pool(tmp_path):
    store = DashboardStore(tmp_path / "dashboard.sqlite")
    runs = RunsService(tmp_path / "runs")
    service = CampaignService(tmp_path, runs, store)
    fake = FakeSlurm()
    service.slurm = fake

    import pytest

    with pytest.raises(ValueError, match="No active League members"):
        service.create(
            {
                "name": "empty-league",
                "tables": {
                    "leduc-holdem": {
                        "table_size": 2,
                        "seats": [
                            {"source": "fresh", "agent_id": "leduc-q", "mode": "train"},
                            {"source": "league_random"},
                        ],
                    }
                },
            }
        )

    member = service.league.create_blank(
        "Opponent",
        "leduc-holdem",
        "leduc-cfr",
        status="league",
    )
    result = service.create(
        {
            "name": "with-league",
            "tables": {
                "leduc-holdem": {
                    "table_size": 2,
                    "seats": [
                        {"source": "fresh", "agent_id": "leduc-q", "mode": "train"},
                        {"source": "league_random"},
                    ],
                }
            },
            "seed": 3,
            "league_reroll_interval": 10,
        }
    )
    seat = fake.submitted[0]["seat_configs"][1]
    assert seat["source"] == "league_random"
    assert seat["league_member_id"] == member["id"]
    assert seat["agent_id"] == "leduc-cfr"
    assert fake.submitted[0]["league_reroll_interval"] == 10
    assert result["jobs"][0]["seats"][1]["league_member_id"] == member["id"]
