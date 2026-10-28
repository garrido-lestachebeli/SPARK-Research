"""FastAPI application for training, statistics, and full-state replay."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from spark.agents import discovery_errors, list_agents
from spark.environments import list_environments
from spark.storage import (
    AgentSnapshotError,
    AgentSnapshotNotFound,
    AgentSnapshotStore,
    LeagueError,
    LeagueMemberNotFound,
    LeagueMemberStore,
)

from .services.campaigns import CampaignService
from .services.dashboard_store import DashboardStore
from .services.runs import RunsService

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIRECTORY = Path(os.getenv("SPARK_RUNS_DIRECTORY", PROJECT_ROOT / "data"))
ACCESS_TOKEN = os.getenv("SPARK_ACCESS_TOKEN", "")

runs = RunsService(RUNS_DIRECTORY)
dashboard_store = DashboardStore(RUNS_DIRECTORY / "dashboard.sqlite")
campaigns = CampaignService(PROJECT_ROOT, runs, dashboard_store)
agent_snapshots = AgentSnapshotStore(RUNS_DIRECTORY)
league_members = LeagueMemberStore(RUNS_DIRECTORY)

app = FastAPI(title="SPARK ROSIE Training Dashboard", version="0.1.0")
app.mount("/static", StaticFiles(directory=PROJECT_ROOT / "app" / "static"), name="static")


class SeatConfigurationRequest(BaseModel):
    source: Literal["fresh", "league_member", "league_random"] = "fresh"
    agent_id: str | None = Field(default=None, max_length=100)
    mode: Literal["train", "evaluate"] = "train"
    snapshot_id: str | None = None
    league_member_id: str | None = None
    # Show this seat every opponent's hole cards (learning sanity check).
    full_vision: bool = False


class TableConfigurationRequest(BaseModel):
    table_size: int = Field(ge=2, le=10)
    seats: list[SeatConfigurationRequest]


class CampaignRequest(BaseModel):
    name: str = ""
    environments: list[str] = Field(default_factory=list)
    tables: dict[str, TableConfigurationRequest] = Field(default_factory=dict)
    seed: int = 0
    episodes: int | None = Field(default=None, ge=1)
    checkpoint_interval: int = Field(default=100, ge=1)
    eval_interval: int = Field(default=0, ge=0)
    eval_episodes: int = Field(default=20, ge=1)
    league_reroll_interval: int = Field(default=0, ge=0)
    starting_stack: float = Field(default=50.0, ge=1)
    small_blind: int = Field(default=1, ge=1)
    big_blind: int = Field(default=2, ge=1)
    episode_mode: Literal["session", "hand"] = "session"
    max_raises: int = Field(default=0, ge=0)
    partition: str | None = None
    cpus: int = Field(default=1, ge=1, le=256)
    memory: str = "4G"
    wall_time: str = "04:00:00"
    agent_memory_size: int = Field(default=100, ge=0)
    agent_snapshots: dict[str, str] = Field(default_factory=dict)


class AgentSnapshotRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    run_id: str = Field(min_length=1, max_length=200)
    seat: int = Field(default=0, ge=0, le=9)


class LeagueMemberCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    environment: str = Field(min_length=1, max_length=100)
    agent_id: str = Field(min_length=1, max_length=100)
    status: Literal["league", "bench"] = "league"
    mode: Literal["train", "snapshot"] | None = None
    seed: int = 0


class LeagueMemberImportRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    snapshot_id: str = Field(min_length=1, max_length=100)
    status: Literal["league", "bench"] = "league"
    mode: Literal["train", "snapshot"] | None = None


class LeagueMemberCopyRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    # None keeps the source's mode; "snapshot" freezes the copy.
    mode: Literal["train", "snapshot"] | None = None


class LeagueMemberPatchRequest(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    status: Literal["league", "bench"] | None = None
    mode: Literal["train", "snapshot"] | None = None


class LeagueSeedRequest(BaseModel):
    environment: str = Field(min_length=1, max_length=100)


@app.middleware("http")
async def token_auth(request: Request, call_next):
    # Static assets must load without the token query string behind OnDemand.
    if request.url.path.startswith("/static/"):
        return await call_next(request)
    if not ACCESS_TOKEN:
        return await call_next(request)
    supplied = (
        request.query_params.get("token")
        or request.headers.get("X-SPARK-Token")
        or request.cookies.get("spark_token")
    )
    if supplied != ACCESS_TOKEN:
        return JSONResponse({"detail": "Invalid or missing access token."}, status_code=401)
    response = await call_next(request)
    if request.query_params.get("token") == ACCESS_TOKEN:
        response.set_cookie(
            "spark_token",
            ACCESS_TOKEN,
            httponly=True,
            samesite="lax",
            secure=False,
            path="/",
        )
    return response


@app.get("/")
def index() -> FileResponse:
    return FileResponse(PROJECT_ROOT / "app" / "templates" / "index.html")


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "slurm_available": campaigns.slurm.available,
        "runs_directory": str(RUNS_DIRECTORY),
    }


@app.get("/api/config")
def config() -> dict[str, Any]:
    return {
        "environments": [
            {
                "id": item.id,
                "name": item.name,
                "enabled": item.enabled,
                "num_players": item.num_players,
                "action_names": item.action_names,
                "reason_disabled": item.reason_disabled,
                "min_players": item.min_players,
                "max_players": item.max_players,
            }
            for item in list_environments()
        ],
        "agents": list_agents(),
        "agent_discovery_errors": discovery_errors(),
        "slurm_available": campaigns.slurm.available,
        "local_jobs_enabled": campaigns.local_jobs,
    }


@app.get("/api/agent-snapshots")
def list_agent_snapshots() -> list[dict[str, Any]]:
    return agent_snapshots.list()


@app.post("/api/agent-snapshots")
def save_agent_snapshot(request: AgentSnapshotRequest) -> dict[str, Any]:
    try:
        return agent_snapshots.save_from_run(request.name, request.run_id, request.seat)
    except AgentSnapshotNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except AgentSnapshotError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.delete("/api/agent-snapshots/{snapshot_id}")
def delete_agent_snapshot(snapshot_id: str) -> dict[str, str]:
    try:
        agent_snapshots.delete(snapshot_id)
    except AgentSnapshotNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except AgentSnapshotError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"status": "deleted", "id": snapshot_id}


@app.get("/api/league/members")
def list_league_members(
    environment: str | None = None,
    status: str | None = None,
) -> list[dict[str, Any]]:
    return league_members.list(environment=environment, status=status)


@app.get("/api/league/members/{member_id}")
def get_league_member(member_id: str) -> dict[str, Any]:
    try:
        return league_members.detail(member_id)
    except LeagueMemberNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except LeagueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/league/members")
def create_league_member(request: LeagueMemberCreateRequest) -> dict[str, Any]:
    try:
        return league_members.create_blank(
            request.name,
            request.environment,
            request.agent_id,
            status=request.status,
            mode=request.mode,
            seed=request.seed,
        )
    except LeagueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/league/members/import-snapshot")
def import_league_member(request: LeagueMemberImportRequest) -> dict[str, Any]:
    try:
        return league_members.create_from_snapshot(
            request.name,
            request.snapshot_id,
            status=request.status,
            mode=request.mode,
        )
    except LeagueMemberNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except LeagueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/league/members/{member_id}/copy")
def copy_league_member(member_id: str, request: LeagueMemberCopyRequest) -> dict[str, Any]:
    try:
        return league_members.copy_member(member_id, request.name, mode=request.mode)
    except LeagueMemberNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except LeagueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.patch("/api/league/members/{member_id}")
def patch_league_member(
    member_id: str,
    request: LeagueMemberPatchRequest,
) -> dict[str, Any]:
    try:
        if request.name is None and request.status is None and request.mode is None:
            raise LeagueError("No league member fields to update.")
        summary = None
        if request.name is not None:
            summary = league_members.rename(member_id, request.name)
        if request.status is not None:
            summary = league_members.set_status(member_id, request.status)
        if request.mode is not None:
            summary = league_members.set_mode(member_id, request.mode)
        return summary or league_members.detail(member_id)
    except LeagueMemberNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except LeagueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.delete("/api/league/members/{member_id}")
def delete_league_member(member_id: str) -> dict[str, str]:
    try:
        league_members.delete(member_id)
    except LeagueMemberNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except LeagueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"status": "deleted", "id": member_id}


@app.post("/api/league/seed-builtins")
def seed_league_builtins(request: LeagueSeedRequest) -> dict[str, Any]:
    try:
        created = league_members.seed_builtins(request.environment)
    except LeagueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"created": created, "count": len(created)}


@app.post("/api/campaigns")
def create_campaign(request: CampaignRequest) -> dict[str, Any]:
    try:
        return campaigns.create(request.model_dump())
    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/campaigns")
def list_campaigns() -> list[dict[str, Any]]:
    return campaigns.refresh()


@app.post("/api/campaigns/{campaign_id}/stop")
def stop_campaign(campaign_id: str) -> dict[str, Any]:
    try:
        count = campaigns.stop_campaign(campaign_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Campaign not found.") from error
    return {"status": "stop_requested", "jobs": count}


@app.post("/api/campaigns/{campaign_id}/resume")
def resume_campaign(campaign_id: str) -> dict[str, Any]:
    try:
        job_ids = campaigns.resume_campaign(campaign_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Campaign not found.") from error
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"status": "requeued", "scheduler_ids": job_ids}


@app.delete("/api/campaigns/{campaign_id}")
def delete_campaign(campaign_id: str) -> dict[str, Any]:
    try:
        result = campaigns.delete_campaign(campaign_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Campaign not found.") from error
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {"status": "deleted", **result}


@app.post("/api/jobs/{run_id}/stop")
def stop_job(run_id: str) -> dict[str, str]:
    try:
        campaigns.stop(run_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Job not found.") from error
    return {"status": "stop_requested"}


@app.delete("/api/jobs/{run_id}")
def remove_job(run_id: str) -> dict[str, Any]:
    try:
        result = campaigns.remove(run_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Job not found.") from error
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {"status": "deleted", **result}


@app.post("/api/jobs/{run_id}/resume")
def resume_job(run_id: str) -> dict[str, str]:
    try:
        job_id = campaigns.resume(run_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Job not found.") from error
    except RuntimeError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"status": "requeued", "scheduler_id": job_id}


@app.get("/api/jobs/{run_id}/logs")
def job_logs(run_id: str) -> dict[str, str]:
    try:
        return {"logs": campaigns.logs(run_id)}
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Job not found.") from error


@app.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
    values = runs.list_runs()
    metadata = dashboard_store.run_campaign_metadata()
    for run in values:
        run.update(metadata.get(run["id"], {}))
    return values


@app.get("/api/runs/{run_id}/statistics")
def run_statistics(run_id: str) -> dict[str, Any]:
    try:
        return runs.statistics(run_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Run not found.") from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/runs/{run_id}/episodes")
def run_episodes(
    run_id: str,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    outcome: str | None = None,
) -> list[dict[str, Any]]:
    try:
        return runs.list_episodes(run_id, limit, offset, outcome)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Run not found.") from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/runs/{run_id}/episodes/{episode_id}")
def get_episode(run_id: str, episode_id: int) -> dict[str, Any]:
    try:
        episode = runs.episode(run_id, episode_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Run not found.") from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if episode is None:
        raise HTTPException(status_code=404, detail="Episode not found.")
    return episode


@app.get("/api/runs/{run_id}/export")
def export_run(run_id: str) -> FileResponse:
    try:
        path = runs.export(run_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="Run not found.") from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return FileResponse(path, filename=path.name, media_type="application/json")
