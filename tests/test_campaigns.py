from web.services.campaigns import CampaignService
from web.services.dashboard_store import DashboardStore
from web.services.runs import RunsService
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
            "environments": ["leduc-holdem"],
            "seed": 9,
            "agent_snapshots": {"leduc-holdem": snapshot["id"]},
        }
    )
    run_id = result["jobs"][0]["run_id"]
    checkpoint = runs.runs_directory / run_id / "checkpoint.json"
    assert checkpoint.exists()
    assert fake.submitted[0]["shared_agents"] is True
