from fastapi.testclient import TestClient

import web.app as web_app
from spark.storage import AgentSnapshotStore
from spark.training.trainer import Trainer, TrainerConfig
from web.services.campaigns import CampaignService
from web.services.dashboard_store import DashboardStore
from web.services.runs import RunsService


def test_dashboard_serves_statistics_and_full_replay(tmp_path, monkeypatch):
    Trainer(
        TrainerConfig(
            run_id="web-test",
            environment="leduc-holdem",
            run_directory=str(tmp_path),
            episode_limit=1,
        )
    ).run()
    monkeypatch.setattr(web_app, "runs", RunsService(tmp_path))
    monkeypatch.setattr(
        web_app, "dashboard_store", DashboardStore(tmp_path / "dashboard.sqlite")
    )
    client = TestClient(web_app.app)

    page = client.get("/")
    assert page.status_code == 200
    assert 'id="performance-interpretation"' not in page.text
    assert 'id="run-environment"' in page.text
    assert "Advanced diagnostics" in page.text
    assert 'id="save-agent-form"' in page.text
    config = client.get("/api/config")
    assert config.status_code == 200
    assert set(config.json()["agents"]) == {
        "leduc-holdem",
        "limit-holdem",
        "no-limit-holdem",
    }
    runs = client.get("/api/runs").json()
    assert runs[0]["id"] == "web-test"
    assert "campaign_name" not in runs[0]
    stats = client.get("/api/runs/web-test/statistics").json()
    assert stats["episodes"] == 1
    assert "seat_payoffs" in stats
    assert "ci_low" in stats["seat_payoffs"]["0"]
    assert "win_rate" in stats["seat_outcomes"]["0"]
    summaries = client.get("/api/runs/web-test/episodes").json()
    episode = client.get(
        f"/api/runs/web-test/episodes/{summaries[0]['id']}"
    ).json()
    assert episode["steps"][0]["hidden_before"]["hand_cards"]
    assert episode["steps"][0]["visible_before"]["raw"]["hand"]


def test_run_list_uses_campaign_name_for_each_environment(tmp_path, monkeypatch):
    for run_id, environment in (
        ("friendly-leduc", "leduc-holdem"),
        ("friendly-limit", "limit-holdem"),
        ("standalone", "leduc-holdem"),
    ):
        Trainer(
            TrainerConfig(
                run_id=run_id,
                environment=environment,
                run_directory=str(tmp_path),
                episode_limit=1,
            )
        ).run()

    store = DashboardStore(tmp_path / "dashboard.sqlite")
    store.create_campaign(
        "campaign-friendly",
        "Friendly experiment",
        {"environments": ["leduc-holdem", "limit-holdem"]},
    )
    store.add_job(
        "friendly-leduc",
        "campaign-friendly",
        "leduc-holdem",
        None,
        "LOCAL",
    )
    store.add_job(
        "friendly-limit",
        "campaign-friendly",
        "limit-holdem",
        None,
        "LOCAL",
    )
    monkeypatch.setattr(web_app, "runs", RunsService(tmp_path))
    monkeypatch.setattr(web_app, "dashboard_store", store)

    values = TestClient(web_app.app).get("/api/runs").json()
    by_id = {item["id"]: item for item in values}
    assert by_id["friendly-leduc"]["campaign_name"] == "Friendly experiment"
    assert by_id["friendly-leduc"]["environment"] == "leduc-holdem"
    assert by_id["friendly-limit"]["campaign_name"] == "Friendly experiment"
    assert "campaign_name" not in by_id["standalone"]


def test_agent_snapshot_and_job_removal_endpoints(tmp_path, monkeypatch):
    runs = RunsService(tmp_path / "runs")
    Trainer(
        TrainerConfig(
            run_id="snapshot-source",
            environment="leduc-holdem",
            run_directory=str(runs.runs_directory),
            episode_limit=1,
        )
    ).run()
    store = DashboardStore(tmp_path / "dashboard.sqlite")
    service = CampaignService(tmp_path, runs, store)
    campaign = service.create(
        {
            "name": "removable",
            "environments": ["limit-holdem"],
            "seed": 2,
        }
    )
    snapshot_store = AgentSnapshotStore(runs.runs_directory)
    monkeypatch.setattr(web_app, "runs", runs)
    monkeypatch.setattr(web_app, "dashboard_store", store)
    monkeypatch.setattr(web_app, "campaigns", service)
    monkeypatch.setattr(web_app, "agent_snapshots", snapshot_store)
    client = TestClient(web_app.app)

    saved = client.post(
        "/api/agent-snapshots",
        json={"name": "Web saved agent", "run_id": "snapshot-source"},
    )
    assert saved.status_code == 200
    snapshot_id = saved.json()["id"]
    assert client.get("/api/agent-snapshots").json()[0]["id"] == snapshot_id
    assert client.delete(f"/api/agent-snapshots/{snapshot_id}").status_code == 200
    assert client.get("/api/agent-snapshots").json() == []

    run_id = campaign["jobs"][0]["run_id"]
    removed = client.delete(f"/api/jobs/{run_id}")
    assert removed.status_code == 200
    assert removed.json()["status"] == "removed"
    assert store.list_campaigns()[0]["jobs"] == []
