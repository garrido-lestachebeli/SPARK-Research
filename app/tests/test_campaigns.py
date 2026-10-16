from app.services.campaigns import CampaignService
from app.services.dashboard_store import DashboardStore
from app.services.runs import RunsService
from spark.storage import RunDatabase
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

    def statuses(self, job_ids):
        return {str(job_id): self.status(str(job_id)) for job_id in job_ids}

    def cancel(self, job_id):
        self.cancelled.append(job_id)


def test_refresh_tolerates_empty_run_database(tmp_path):
    import sqlite3

    store = DashboardStore(tmp_path / "dashboard.sqlite")
    runs = RunsService(tmp_path / "runs")
    service = CampaignService(tmp_path, runs, store)
    service.slurm = FakeSlurm()
    result = service.create(
        {
            "name": "broken-db",
            "environments": ["leduc-holdem"],
            "seed": 1,
        }
    )
    run_id = result["jobs"][0]["run_id"]
    db_path = runs.runs_directory / run_id / "run.sqlite"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    sqlite3.connect(db_path).close()

    assert RunDatabase(db_path, initialize=False).get_run() is None
    campaigns = service.refresh()
    assert len(campaigns) == 1
    assert campaigns[0]["jobs"][0]["run_id"] == run_id
    assert campaigns[0]["jobs"][0]["run"] is None


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


def test_remove_job_deletes_run_storage(tmp_path):
    store = DashboardStore(tmp_path / "dashboard.sqlite")
    runs = RunsService(tmp_path / "runs")
    service = CampaignService(tmp_path, runs, store)
    fake = FakeSlurm()
    service.slurm = fake
    result = service.create(
        {
            "name": "reclaim-space",
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
    # Slurm stdout/stderr live beside the run folders and should go too.
    (runs.runs_directory / "slurm-10001.out").write_text("log", encoding="utf-8")
    (runs.runs_directory / "slurm-10001.err").write_text("log", encoding="utf-8")

    removed = service.remove(run_id)
    assert removed["run_id"] == run_id
    assert removed["stopped"] is True
    assert removed["bytes_freed"] > 0
    assert not (runs.runs_directory / run_id).exists()
    assert not (runs.runs_directory / "slurm-10001.out").exists()
    assert not (runs.runs_directory / "slurm-10001.err").exists()
    assert store.list_campaigns()[0]["jobs"] == []
    assert run_id not in store.run_campaign_metadata()
    assert runs.list_runs() == []
    assert fake.cancelled == ["10001"]


def test_delete_campaign_removes_every_run_including_hidden_jobs(tmp_path):
    store = DashboardStore(tmp_path / "dashboard.sqlite")
    runs = RunsService(tmp_path / "runs")
    service = CampaignService(tmp_path, runs, store)
    service.slurm = FakeSlurm()
    result = service.create(
        {
            "name": "big-campaign",
            "environments": ["leduc-holdem", "limit-holdem"],
            "seed": 3,
        }
    )
    run_ids = [job["run_id"] for job in result["jobs"]]
    for run_id, environment in zip(run_ids, ["leduc-holdem", "limit-holdem"]):
        Trainer(
            TrainerConfig(
                run_id=run_id,
                environment=environment,
                run_directory=str(runs.runs_directory),
                episode_limit=1,
            )
        ).run()
    # A job hidden by an older build still occupies disk and must be swept up.
    with store.connect() as connection:
        connection.execute(
            "UPDATE jobs SET removed = 1 WHERE run_id = ?", (run_ids[1],)
        )
    assert len(store.list_campaigns()[0]["jobs"]) == 1

    summary = service.delete_campaign(result["campaign_id"])
    assert summary["jobs"] == 2
    assert summary["bytes_freed"] > 0
    assert all(not (runs.runs_directory / run_id).exists() for run_id in run_ids)
    assert store.list_campaigns() == []
    assert store.run_campaign_metadata() == {}
    assert runs.list_runs() == []


def test_delete_campaign_rejects_unknown_id(tmp_path):
    import pytest

    store = DashboardStore(tmp_path / "dashboard.sqlite")
    runs = RunsService(tmp_path / "runs")
    service = CampaignService(tmp_path, runs, store)
    with pytest.raises(FileNotFoundError):
        service.delete_campaign("campaign-missing")


def test_delete_run_rejects_path_traversal(tmp_path):
    import pytest

    runs = RunsService(tmp_path / "runs")
    keep = tmp_path / "keep.txt"
    keep.write_text("important", encoding="utf-8")
    for candidate in ["..", "../..", "a/../..", "", "with space"]:
        with pytest.raises(ValueError):
            runs.delete_run(candidate)
    assert keep.exists()
    assert runs.runs_directory.exists()


def test_delete_run_is_a_no_op_for_a_missing_directory(tmp_path):
    runs = RunsService(tmp_path / "runs")
    assert runs.delete_run("never-existed") == 0


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
                            "full_vision": True,
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
            "eval_interval": 5,
            "league_reroll_interval": 0,
        }
    )
    run_id = result["jobs"][0]["run_id"]
    checkpoint = runs.runs_directory / run_id / "checkpoint.json"
    assert checkpoint.exists()
    assert fake.submitted[0]["seat_configs"][0]["snapshot_id"] == snapshot["id"]
    assert fake.submitted[0]["seat_configs"][1]["mode"] == "evaluate"
    assert fake.submitted[0]["eval_interval"] == 5
    # Per-seat vision travels with the seat, not the table.
    assert fake.submitted[0]["seat_configs"][0]["full_vision"] is True
    assert fake.submitted[0]["seat_configs"][1]["full_vision"] is False
    import json

    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert [seat["full_vision"] for seat in saved["seat_configs"]] == [True, False]
    assert "observation_mode" not in saved["config"]


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


def test_resume_campaign_skips_completed_and_live_jobs(tmp_path):
    store = DashboardStore(tmp_path / "dashboard.sqlite")
    runs = RunsService(tmp_path / "runs")
    service = CampaignService(tmp_path, runs, store)
    fake = FakeSlurm()
    service.slurm = fake
    result = service.create(
        {
            "name": "resume-filter",
            "environments": ["leduc-holdem", "limit-holdem"],
            "seed": 1,
        }
    )
    jobs = result["jobs"]
    for job in jobs:
        Trainer(
            TrainerConfig(
                run_id=job["run_id"],
                environment=job["environment"],
                run_directory=str(runs.runs_directory),
                episode_limit=1,
            )
        ).run()
        store.update_job(job["run_id"], "COMPLETED")

    assert service.resume_campaign(result["campaign_id"]) == []

    runs.database(jobs[0]["run_id"]).update_run(status="stopped")
    store.update_job(jobs[0]["run_id"], "RUNNING")
    assert service.resume_campaign(result["campaign_id"]) == []

    store.update_job(jobs[0]["run_id"], "CANCELLED")
    resumed = service.resume_campaign(result["campaign_id"])
    assert len(resumed) == 1
    assert fake.submitted[-1]["run_id"] == jobs[0]["run_id"]
    assert runs.database(jobs[0]["run_id"]).get_run()["status"] == "running"
    assert service.resume_campaign(result["campaign_id"]) == []

    runs.database(jobs[1]["run_id"]).update_run(status="failed")
    store.update_job(jobs[1]["run_id"], "FAILED")
    resumed = service.resume_campaign(result["campaign_id"])
    assert len(resumed) == 1
    assert fake.submitted[-1]["run_id"] == jobs[1]["run_id"]
