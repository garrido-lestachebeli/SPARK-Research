from fastapi.testclient import TestClient

import app.app as web_app
from spark.storage import AgentSnapshotStore, LeagueMemberStore
from spark.training.trainer import Trainer, TrainerConfig
from app.services.campaigns import CampaignService
from app.services.dashboard_store import DashboardStore
from app.services.runs import RunsService


def test_dashboard_serves_statistics_without_episode_replay_logs(tmp_path, monkeypatch):
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
    assert 'data-tab="replay"' not in page.text
    assert 'id="replay"' not in page.text
    assert "Advanced diagnostics" in page.text
    assert 'id="save-agent-form"' in page.text
    assert 'data-tab="league"' in page.text
    assert 'id="league-create-form"' in page.text
    assert 'name="eval_interval"' in page.text
    assert 'name="eval_episodes"' in page.text
    assert 'id="eval-chart"' in page.text
    assert 'id="eval-opponent-lineup"' in page.text
    assert 'name="starting_stack"' in page.text
    assert 'name="small_blind"' in page.text
    assert 'name="big_blind"' in page.text
    assert 'name="episode_mode"' in page.text
    assert 'name="observation_mode"' not in page.text
    script = client.get("/static/app.js")
    assert 'class="seat-full-vision"' in script.text
    assert 'name="league_reroll_interval"' in page.text
    assert 'data-preset="learner-vs-league"' in page.text
    config = client.get("/api/config")
    assert config.status_code == 200
    assert {
        "leduc-random",
        "leduc-q",
        "leduc-cfr",
        "leduc-rule-v1",
        "leduc-rule-v2",
        "limit-random",
        "limit-rule-v1",
        "no-limit-random",
    } <= {value["id"] for value in config.json()["agents"]}
    assert config.json()["agent_discovery_errors"] == []
    environments = {value["id"]: value for value in config.json()["environments"]}
    assert environments["limit-holdem"]["max_players"] == 10
    runs = client.get("/api/runs").json()
    assert runs[0]["id"] == "web-test"
    assert "campaign_name" not in runs[0]
    stats = client.get("/api/runs/web-test/statistics").json()
    assert stats["episodes"] == 1
    assert "seat_payoffs" in stats
    assert stats["evaluation"] == {"seats": {}, "rounds": 0, "episodes": 0}
    assert "ci_low" in stats["seat_payoffs"]["0"]
    assert "win_rate" in stats["seat_outcomes"]["0"]
    assert len(stats["seats"]) >= 2
    assert stats["seats"][0]["seat"] == 0
    assert stats["seats"][0]["agent_id"]
    assert stats["seats"][0]["short_label"]
    assert stats["seats"][0]["display_name"]
    assert "seat 0" in stats["seats"][0]["label"].lower()
    assert stats["focus_seat"] in {"0", "1"}
    assert stats["focus_seat_label"]
    assert stats["actions"]
    assert stats["average_steps"] > 0
    summaries = client.get("/api/runs/web-test/episodes").json()
    assert summaries[0]["step_count"] > 0
    episode = client.get(
        f"/api/runs/web-test/episodes/{summaries[0]['id']}"
    ).json()
    assert episode["payoffs"]
    assert episode["steps"] == []


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


def test_campaign_api_accepts_multiplayer_seat_configuration(tmp_path, monkeypatch):
    runs = RunsService(tmp_path / "runs")
    store = DashboardStore(tmp_path / "dashboard.sqlite")
    service = CampaignService(tmp_path, runs, store)
    monkeypatch.setattr(web_app, "campaigns", service)
    response = TestClient(web_app.app).post(
        "/api/campaigns",
        json={
            "name": "three-handed",
            "tables": {
                "limit-holdem": {
                    "table_size": 3,
                    "seats": [
                        {"agent_id": "limit-random", "mode": "train"},
                        {"agent_id": "limit-random", "mode": "evaluate"},
                        {"agent_id": "limit-random", "mode": "evaluate"},
                    ],
                }
            },
        },
    )
    assert response.status_code == 200
    config = store.list_campaigns()[0]["config"]["tables"]["limit-holdem"]
    assert config["table_size"] == 3
    assert config["seats"][1]["mode"] == "evaluate"


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
    assert removed.json()["status"] == "deleted"
    assert store.list_campaigns()[0]["jobs"] == []
    assert not (runs.runs_directory / run_id).exists()
    assert client.delete(f"/api/jobs/{run_id}").status_code == 404

    deleted = client.delete(f"/api/campaigns/{campaign['campaign_id']}")
    assert deleted.status_code == 200
    assert deleted.json()["status"] == "deleted"
    assert store.list_campaigns() == []
    assert client.delete(f"/api/campaigns/{campaign['campaign_id']}").status_code == 404
    # The source run belongs to no campaign and must survive.
    assert (runs.runs_directory / "snapshot-source" / "run.sqlite").exists()


def test_league_api_endpoints(tmp_path, monkeypatch):
    runs = RunsService(tmp_path / "runs")
    Trainer(
        TrainerConfig(
            run_id="league-web-source",
            environment="leduc-holdem",
            run_directory=str(runs.runs_directory),
            episode_limit=1,
            checkpoint_interval=1,
        )
    ).run()
    snapshot_store = AgentSnapshotStore(runs.runs_directory)
    league_store = LeagueMemberStore(runs.runs_directory)
    monkeypatch.setattr(web_app, "runs", runs)
    monkeypatch.setattr(web_app, "agent_snapshots", snapshot_store)
    monkeypatch.setattr(web_app, "league_members", league_store)
    client = TestClient(web_app.app)

    created = client.post(
        "/api/league/members",
        json={
            "name": "Web blank",
            "environment": "leduc-holdem",
            "agent_id": "leduc-random",
            "status": "league",
        },
    )
    assert created.status_code == 200
    member_id = created.json()["id"]
    assert created.json()["elo"] == 1000

    seeded = client.post(
        "/api/league/seed-builtins",
        json={"environment": "leduc-holdem"},
    )
    assert seeded.status_code == 200
    assert seeded.json()["count"] >= 1

    listed = client.get("/api/league/members", params={"environment": "leduc-holdem"})
    assert listed.status_code == 200
    assert any(item["id"] == member_id for item in listed.json())

    moved = client.patch(
        f"/api/league/members/{member_id}",
        json={"status": "bench", "name": "Benched random"},
    )
    assert moved.status_code == 200
    assert moved.json()["status"] == "bench"
    assert moved.json()["name"] == "Benched random"

    mode_change = client.patch(
        f"/api/league/members/{member_id}",
        json={"mode": "train"},
    )
    assert mode_change.status_code == 200
    assert mode_change.json()["mode"] == "train"

    snapshot = snapshot_store.save_from_run("League import", "league-web-source")
    imported = client.post(
        "/api/league/members/import-snapshot",
        json={
            "name": "Imported web",
            "snapshot_id": snapshot["id"],
            "status": "league",
        },
    )
    assert imported.status_code == 200
    assert imported.json()["source_kind"] == "snapshot"

    copied = client.post(
        f"/api/league/members/{member_id}/copy",
        json={"name": "Copied web"},
    )
    assert copied.status_code == 200
    assert copied.json()["elo"] == 1000

    detail = client.get(f"/api/league/members/{member_id}")
    assert detail.status_code == 200
    assert "elo_note" in detail.json()

    deleted = client.delete(f"/api/league/members/{member_id}")
    assert deleted.status_code == 200
    assert client.get(f"/api/league/members/{member_id}").status_code == 404


def test_invalid_run_id_returns_400(tmp_path, monkeypatch):
    monkeypatch.setattr(web_app, "runs", RunsService(tmp_path))
    client = TestClient(web_app.app)
    for path in (
        "/api/runs/bad.id/statistics",
        "/api/runs/bad.id/episodes",
        "/api/runs/bad.id/episodes/1",
        "/api/runs/bad.id/export",
    ):
        response = client.get(path)
        assert response.status_code == 400
        assert response.json()["detail"] == "Invalid run ID."
