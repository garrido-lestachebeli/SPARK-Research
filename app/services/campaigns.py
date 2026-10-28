"""Create, monitor, and stop multi-environment campaigns."""

from __future__ import annotations

import json
import os
import random
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from spark.agents import create_agent, default_agent_id
from spark.environments import list_environments
from spark.environments.rlcard_poker import normalize_episode_mode
from spark.slurm.client import SlurmClient
from spark.storage import AgentSnapshotStore, LeagueError, LeagueMemberStore

from .dashboard_store import DashboardStore
from .runs import RunsService


def _league_mode_to_seat_mode(mode: str) -> str:
    return "train" if str(mode).lower() == "train" else "evaluate"


class CampaignService:
    def __init__(
        self,
        project_root: str | Path,
        runs: RunsService,
        store: DashboardStore,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.runs = runs
        self.store = store
        self.slurm = SlurmClient(self.project_root)
        self.agent_snapshots = AgentSnapshotStore(self.runs.runs_directory)
        self.league = LeagueMemberStore(self.runs.runs_directory)
        self.local_jobs = os.getenv("SPARK_LOCAL_JOBS", "0") == "1"

    def create(self, config: dict[str, Any]) -> dict[str, Any]:
        tables = self._normalize_tables(config)
        environments = list(tables)
        if not environments:
            raise ValueError("Select at least one environment.")
        config = dict(config)
        config["environments"] = environments
        config["tables"] = tables
        config["eval_interval"] = max(0, int(config.get("eval_interval", 0)))
        config["eval_episodes"] = max(1, int(config.get("eval_episodes", 20)))
        config["league_reroll_interval"] = max(
            0, int(config.get("league_reroll_interval", 0))
        )
        config["starting_stack"] = max(1.0, float(config.get("starting_stack", 50)))
        config["small_blind"] = max(1, int(config.get("small_blind", 1)))
        config["big_blind"] = max(config["small_blind"], int(config.get("big_blind", 2)))
        if config["starting_stack"] < config["big_blind"]:
            raise ValueError(
                "Starting stack must be at least the big blind."
            )
        config["episode_mode"] = normalize_episode_mode(config.get("episode_mode", "session"))
        config["max_raises"] = max(0, int(config.get("max_raises", 0)))

        campaign_id = f"campaign-{uuid.uuid4().hex[:10]}"
        name = str(config.get("name") or campaign_id)
        self.store.create_campaign(campaign_id, name, config)
        submitted = []
        for index, environment in enumerate(environments):
            run_id = f"{campaign_id}-{environment}"
            job_config = dict(config)
            job_config.update(
                {
                    "run_id": run_id,
                    "environment": environment,
                    "seed": int(config.get("seed", 0)) + index * 1_000_000,
                    "runs_directory": str(self.runs.runs_directory),
                    "table_size": tables[environment]["table_size"],
                    "seat_configs": tables[environment]["seats"],
                }
            )
            self.agent_snapshots.materialize_lineup_checkpoint(
                run_id=run_id,
                environment=environment,
                table_size=job_config["table_size"],
                seat_configs=job_config["seat_configs"],
                seed=job_config["seed"],
                runs_directory=self.runs.runs_directory,
                checkpoint_interval=int(config.get("checkpoint_interval", 100)),
                episode_limit=config.get("episodes"),
                agent_memory_size=int(config.get("agent_memory_size", 100)),
                eval_interval=int(config["eval_interval"]),
                eval_episodes=int(config["eval_episodes"]),
                league_reroll_interval=int(config["league_reroll_interval"]),
                starting_stack=float(config.get("starting_stack", 50)),
                small_blind=int(config.get("small_blind", 1)),
                big_blind=int(config.get("big_blind", 2)),
                episode_mode=str(config.get("episode_mode", "session")),
                max_raises=int(config.get("max_raises", 0)),
                league_store=self.league,
            )
            if self.slurm.available:
                scheduler_id = self.slurm.submit_training(job_config)
                status = "SUBMITTED"
            elif self.local_jobs:
                scheduler_id = self._start_local(job_config)
                status = "LOCAL"
            else:
                scheduler_id = None
                status = "NOT_SUBMITTED"
            self.store.add_job(
                run_id, campaign_id, environment, scheduler_id, status
            )
            submitted.append(
                {
                    "run_id": run_id,
                    "environment": environment,
                    "scheduler_id": scheduler_id,
                    "status": status,
                    "table_size": job_config["table_size"],
                    "seats": job_config["seat_configs"],
                }
            )
        return {"campaign_id": campaign_id, "name": name, "jobs": submitted}

    def _normalize_tables(self, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
        metadata = {value.id: value for value in list_environments() if value.enabled}
        raw_tables = dict(config.get("tables") or {})
        seed = int(config.get("seed", 0))
        if not raw_tables:
            for environment in config.get("environments", []):
                info = metadata.get(str(environment))
                if info is None:
                    raise ValueError(f"Unknown environment {environment!r}.")
                default_id = default_agent_id(info.id)
                legacy_snapshot = dict(config.get("agent_snapshots", {})).get(info.id)
                raw_tables[info.id] = {
                    "table_size": info.num_players,
                    "seats": [
                        {
                            "source": "fresh",
                            "agent_id": default_id,
                            "mode": "train",
                            "snapshot_id": legacy_snapshot,
                        }
                        for _ in range(info.num_players)
                    ],
                }
        normalized: dict[str, dict[str, Any]] = {}
        for env_index, (environment, value) in enumerate(raw_tables.items()):
            info = metadata.get(environment)
            if info is None:
                raise ValueError(f"Unknown or disabled environment {environment!r}.")
            table_size = int(value.get("table_size", info.num_players))
            if not info.min_players <= table_size <= info.max_players:
                raise ValueError(
                    f"{environment} supports {info.min_players}-{info.max_players} players."
                )
            seats = list(value.get("seats", []))
            if len(seats) != table_size:
                raise ValueError(
                    f"{environment} requires {table_size} seat configurations."
                )
            normalized_seats: list[dict[str, Any]] = []
            for seat, seat_value in enumerate(seats):
                normalized_seats.append(
                    self._normalize_seat(
                        seat_value,
                        environment=environment,
                        seat=seat,
                        seed=seed + env_index * 10_000 + seat,
                    )
                )
            normalized[environment] = {
                "table_size": table_size,
                "seats": normalized_seats,
            }
        return normalized

    def _normalize_seat(
        self,
        seat_value: dict[str, Any],
        *,
        environment: str,
        seat: int,
        seed: int,
    ) -> dict[str, Any]:
        source = str(seat_value.get("source") or "fresh").lower()
        if source not in {"fresh", "league_member", "league_random"}:
            raise ValueError(f"Seat {seat} has invalid source {source!r}.")
        # Per-seat perfect information: this seat is shown every opponent's
        # hole cards while the rest of the table plays normal poker.
        full_vision = bool(seat_value.get("full_vision", False))

        if source == "fresh":
            agent_id = str(seat_value.get("agent_id", "")).strip()
            mode = str(seat_value.get("mode", "train")).lower()
            if not agent_id:
                raise ValueError(f"Seat {seat} has no agent_id.")
            if mode not in {"train", "evaluate"}:
                raise ValueError(f"Seat {seat} has invalid mode {mode!r}.")
            agent = create_agent(agent_id)
            if agent.environment_id != environment:
                raise ValueError(
                    f"Agent {agent_id!r} is not compatible with {environment!r}."
                )
            snapshot_id = seat_value.get("snapshot_id") or None
            if snapshot_id:
                self.agent_snapshots.state_for(
                    str(snapshot_id),
                    environment=environment,
                    agent_id=agent_id,
                    legacy_seat=seat,
                )
            return {
                "source": "fresh",
                "agent_id": agent_id,
                "mode": mode,
                "snapshot_id": snapshot_id,
                "league_member_id": None,
                "full_vision": full_vision,
            }

        if source == "league_member":
            member_id = str(seat_value.get("league_member_id") or "").strip()
            if not member_id:
                raise ValueError(f"Seat {seat} needs a league_member_id.")
            try:
                member = self.league.get(member_id)
                self.league.state_for(member_id, environment=environment)
            except LeagueError as error:
                raise ValueError(str(error)) from error
            return {
                "source": "league_member",
                "agent_id": str(member["agent_id"]),
                "mode": _league_mode_to_seat_mode(str(member.get("mode") or "snapshot")),
                "snapshot_id": None,
                "league_member_id": member_id,
                "full_vision": full_vision,
            }

        # league_random — resolve once for a reproducible starting lineup.
        try:
            member = self.league.sample_active(
                environment,
                rng=random.Random(seed),
            )
        except LeagueError as error:
            raise ValueError(str(error)) from error
        return {
            "source": "league_random",
            "agent_id": str(member["agent_id"]),
            "mode": _league_mode_to_seat_mode(str(member.get("mode") or "snapshot")),
            "snapshot_id": None,
            "league_member_id": str(member["id"]),
            "full_vision": full_vision,
        }

    def refresh(self) -> list[dict[str, Any]]:
        campaigns = self.store.list_campaigns()
        numeric_ids = [
            str(job["scheduler_id"])
            for campaign in campaigns
            for job in campaign["jobs"]
            if str(job.get("scheduler_id") or "").isdigit()
        ]
        statuses: dict[str, dict[str, str]] = {}
        if numeric_ids and self.slurm.available:
            statuses = self.slurm.statuses(numeric_ids)
        for campaign in campaigns:
            for job in campaign["jobs"]:
                scheduler_id = str(job.get("scheduler_id") or "")
                if scheduler_id in statuses:
                    status = statuses[scheduler_id]["state"]
                    self.store.update_job(job["run_id"], status)
                    job["status"] = status
                try:
                    run = self.runs.database(job["run_id"]).get_run()
                except (FileNotFoundError, sqlite3.Error, OSError):
                    # Missing, empty, or corrupt run.sqlite should not 500 the list.
                    run = None
                job["run"] = run
        return campaigns

    def stop(self, run_id: str) -> None:
        # Hidden jobs can still have a live worker, so they stay stoppable.
        job = self.store.get_job(run_id, include_removed=True)
        if not job:
            raise FileNotFoundError(run_id)
        try:
            self.runs.request_stop(run_id)
        except FileNotFoundError:
            pass
        scheduler_id = job.get("scheduler_id")
        if scheduler_id and scheduler_id.isdigit() and self.slurm.available:
            self.slurm.cancel(scheduler_id)
        self.store.update_job(run_id, "STOP_REQUESTED")

    def remove(self, run_id: str) -> dict[str, Any]:
        """Stop a job and permanently delete its run data from disk."""
        job = self.store.get_job(run_id, include_removed=True)
        if not job:
            raise FileNotFoundError(run_id)
        result = self._delete_job(job)
        self.store.delete_job(run_id)
        return result

    def delete_campaign(self, campaign_id: str) -> dict[str, Any]:
        """Delete every run directory in a campaign, then drop the campaign."""
        campaign = self.store.get_campaign(campaign_id)
        if not campaign:
            raise FileNotFoundError(campaign_id)
        # Already-hidden jobs still occupy disk, so sweep them up too.
        jobs = self.store.list_campaign_jobs(campaign_id, include_removed=True)
        results = [self._delete_job(job) for job in jobs]
        self.store.delete_campaign(campaign_id)
        return {
            "campaign_id": campaign_id,
            "jobs": len(results),
            "stopped": sum(1 for item in results if item["stopped"]),
            "bytes_freed": sum(item["bytes_freed"] for item in results),
        }

    def _delete_job(self, job: dict[str, Any]) -> dict[str, Any]:
        run_id = str(job["run_id"])
        stopped = False
        if self._job_is_active(job):
            self.stop(run_id)
            stopped = True
        freed = self.runs.delete_run(run_id)
        freed += self._delete_scheduler_logs(job.get("scheduler_id"))
        return {"run_id": run_id, "stopped": stopped, "bytes_freed": freed}

    def _delete_scheduler_logs(self, scheduler_id: Any) -> int:
        """Remove the Slurm stdout/stderr files that live beside the run folders."""
        identifier = str(scheduler_id or "")
        if not identifier.isdigit():
            return 0
        freed = 0
        for suffix in ("out", "err"):
            path = self.runs.runs_directory / f"slurm-{identifier}.{suffix}"
            if not path.is_file():
                continue
            size = path.stat().st_size
            try:
                path.unlink()
            except OSError:
                continue
            freed += size
        return freed

    def stop_campaign(self, campaign_id: str) -> int:
        campaign = next(
            (item for item in self.store.list_campaigns() if item["id"] == campaign_id),
            None,
        )
        if not campaign:
            raise FileNotFoundError(campaign_id)
        for job in campaign["jobs"]:
            self.stop(job["run_id"])
        return len(campaign["jobs"])

    def resume(self, run_id: str) -> str:
        job = self.store.get_job(run_id)
        if not job:
            raise FileNotFoundError(run_id)
        campaign = self.store.get_campaign(job["campaign_id"])
        if not campaign:
            raise FileNotFoundError(job["campaign_id"])
        config = dict(campaign["config"])
        config.update(
            {
                "run_id": run_id,
                "environment": job["environment"],
                "runs_directory": str(self.runs.runs_directory),
            }
        )
        table = dict(config.get("tables", {})).get(job["environment"], {})
        config["table_size"] = int(table.get("table_size", 2))
        config["seat_configs"] = list(table.get("seats", [])) or None
        run = self.runs.database(run_id).get_run()
        if run:
            config["seed"] = run["seed"]
            run_config = run.get("config", {})
            config["table_size"] = int(
                run_config.get("table_size", config["table_size"])
            )
            config["seat_configs"] = run_config.get(
                "seat_configs", config["seat_configs"]
            )
            config["eval_interval"] = int(
                run_config.get("eval_interval", config.get("eval_interval", 0))
            )
            config["eval_episodes"] = int(
                run_config.get("eval_episodes", config.get("eval_episodes", 20))
            )
            config["league_reroll_interval"] = int(
                run_config.get(
                    "league_reroll_interval",
                    config.get("league_reroll_interval", 0),
                )
            )
            config["starting_stack"] = float(
                run_config.get("starting_stack", config.get("starting_stack", 50))
            )
            config["small_blind"] = int(
                run_config.get("small_blind", config.get("small_blind", 1))
            )
            config["big_blind"] = int(
                run_config.get("big_blind", config.get("big_blind", 2))
            )
            config["episode_mode"] = normalize_episode_mode(
                run_config.get("episode_mode", config.get("episode_mode", "session"))
            )
            config["max_raises"] = max(
                0, int(run_config.get("max_raises", config.get("max_raises", 0)))
            )
        self.runs.clear_stop(run_id)
        if self.slurm.available:
            scheduler_id = self.slurm.submit_training(config)
            self.store.update_job(run_id, "RESUBMITTED", scheduler_id)
            self._mark_run_resumed(run_id)
            return scheduler_id
        if self.local_jobs:
            scheduler_id = self._start_local(config)
            self.store.update_job(run_id, "LOCAL", scheduler_id)
            self._mark_run_resumed(run_id)
            return scheduler_id
        raise RuntimeError("Slurm/local job execution is unavailable on this host.")

    def resume_campaign(self, campaign_id: str) -> list[str]:
        campaign = next(
            (item for item in self.store.list_campaigns() if item["id"] == campaign_id),
            None,
        )
        if not campaign:
            raise FileNotFoundError(campaign_id)
        resumed: list[str] = []
        live_scheduler = {
            "PENDING",
            "RUNNING",
            "CONFIGURING",
            "COMPLETING",
            "SUSPENDED",
            "REQUEUED",
        }
        for job in campaign["jobs"]:
            # Completed runs stay put. Sticky labels such as SUBMITTED, LOCAL,
            # and STOP_REQUESTED outlive the worker, so they do not block a
            # stopped or failed run. A live Slurm state still does.
            if self._run_status(job) not in {"stopped", "failed"}:
                continue
            if str(job.get("status", "")).upper() in live_scheduler:
                continue
            resumed.append(self.resume(job["run_id"]))
        return resumed

    def logs(self, run_id: str, *, max_bytes: int = 65_536) -> str:
        job = self.store.get_job(run_id)
        if not job:
            raise FileNotFoundError(run_id)
        candidates = [self.runs.runs_directory / run_id / "worker.log"]
        scheduler_id = job.get("scheduler_id")
        if scheduler_id and scheduler_id.isdigit():
            candidates.extend(
                [
                    self.runs.runs_directory / f"slurm-{scheduler_id}.out",
                    self.runs.runs_directory / f"slurm-{scheduler_id}.err",
                ]
            )
        sections = []
        for path in candidates:
            if path.exists():
                with path.open("rb") as handle:
                    handle.seek(max(0, path.stat().st_size - max_bytes))
                    content = handle.read().decode("utf-8", errors="replace")
                sections.append(f"--- {path.name} ---\n{content}")
        return "\n".join(sections) or "No log output is available yet."

    def _start_local(self, config: dict[str, Any]) -> str:
        command = [
            sys.executable,
            "-m",
            "spark.training.cli",
            "--environment",
            config["environment"],
            "--table-size",
            str(config.get("table_size", 2)),
            "--run-id",
            config["run_id"],
            "--run-directory",
            str(self.runs.runs_directory),
            "--seed",
            str(config["seed"]),
            "--checkpoint-interval",
            str(config.get("checkpoint_interval", 100)),
            "--eval-interval",
            str(config.get("eval_interval", 0)),
            "--eval-episodes",
            str(config.get("eval_episodes", 20)),
            "--league-reroll-interval",
            str(config.get("league_reroll_interval", 0)),
            "--agent-memory-size",
            str(config.get("agent_memory_size", 100)),
            "--starting-stack",
            str(config.get("starting_stack", 50)),
            "--small-blind",
            str(config.get("small_blind", 1)),
            "--big-blind",
            str(config.get("big_blind", 2)),
            "--episode-mode",
            str(config.get("episode_mode", "session")),
            "--max-raises",
            str(config.get("max_raises", 0)),
        ]
        if config.get("seat_configs"):
            command.extend(
                [
                    "--seat-config",
                    json.dumps(config["seat_configs"], separators=(",", ":")),
                ]
            )
        if config.get("episodes"):
            command.extend(["--episodes", str(int(config["episodes"]))])
        log_path = self.runs.runs_directory / config["run_id"] / "worker.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("a", encoding="utf-8")
        try:
            process = subprocess.Popen(
                command,
                cwd=self.project_root,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        finally:
            # The child keeps its own copy of the handle. Closing here stops
            # the dashboard from leaking one file per local worker.
            log.close()
        return f"local-{process.pid}"

    def _mark_run_resumed(self, run_id: str) -> None:
        """Mark a resubmitted run running so Resume all cannot queue it twice."""
        try:
            self.runs.database(run_id).update_run(status="running")
        except (FileNotFoundError, sqlite3.Error, OSError, ValueError):
            return

    def _run_status(self, job: dict[str, Any]) -> str:
        try:
            run = self.runs.database(str(job["run_id"])).get_run()
        except (FileNotFoundError, sqlite3.Error, OSError, ValueError):
            return ""
        if not run:
            return ""
        return str(run.get("status", "")).lower()

    def _job_is_active(self, job: dict[str, Any]) -> bool:
        status = str(job.get("status", "")).upper()
        active = {
            "SUBMITTED",
            "RESUBMITTED",
            "LOCAL",
            "PENDING",
            "RUNNING",
            "CONFIGURING",
            "COMPLETING",
            "SUSPENDED",
            "REQUEUED",
            "STOP_REQUESTED",
        }
        if status in active:
            return True
        return self._run_status(job) == "running"
