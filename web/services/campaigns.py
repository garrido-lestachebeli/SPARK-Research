"""Create, monitor, and stop multi-environment campaigns."""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from spark.slurm.client import SlurmClient
from spark.storage import AgentSnapshotError, AgentSnapshotStore

from .dashboard_store import DashboardStore
from .runs import RunsService


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
        self.local_jobs = os.getenv("SPARK_LOCAL_JOBS", "0") == "1"

    def create(self, config: dict[str, Any]) -> dict[str, Any]:
        environments = list(config.get("environments", []))
        if not environments:
            raise ValueError("Select at least one environment.")
        allowed = {"leduc-holdem", "limit-holdem", "no-limit-holdem"}
        if any(environment not in allowed for environment in environments):
            raise ValueError("Only registered poker environments are currently enabled.")
        snapshot_ids = dict(config.get("agent_snapshots", {}))
        unknown_snapshot_environments = set(snapshot_ids) - set(environments)
        if unknown_snapshot_environments:
            raise ValueError(
                "Agent snapshots were provided for unselected environments: "
                + ", ".join(sorted(unknown_snapshot_environments))
            )
        for environment, snapshot_id in snapshot_ids.items():
            snapshot = self.agent_snapshots.get(str(snapshot_id))
            if snapshot["environment"] != environment:
                raise AgentSnapshotError(
                    f"Snapshot {snapshot['name']!r} is for "
                    f"{snapshot['environment']}, not {environment}."
                )

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
                }
            )
            snapshot_id = snapshot_ids.get(environment)
            if snapshot_id:
                materialized = self.agent_snapshots.materialize_run_checkpoint(
                    str(snapshot_id),
                    run_id=run_id,
                    environment=environment,
                    seed=job_config["seed"],
                    checkpoint_interval=int(
                        config.get("checkpoint_interval", 100)
                    ),
                    episode_limit=config.get("episodes"),
                    agent_memory_size=int(config.get("agent_memory_size", 100)),
                )
                job_config["shared_agents"] = materialized["shared_agents"]
            else:
                job_config["shared_agents"] = True
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
                }
            )
        return {"campaign_id": campaign_id, "name": name, "jobs": submitted}

    def refresh(self) -> list[dict[str, Any]]:
        campaigns = self.store.list_campaigns()
        for campaign in campaigns:
            for job in campaign["jobs"]:
                scheduler_id = job.get("scheduler_id")
                if scheduler_id and scheduler_id.isdigit() and self.slurm.available:
                    status = self.slurm.status(scheduler_id)["state"]
                    self.store.update_job(job["run_id"], status)
                    job["status"] = status
                try:
                    run = self.runs.database(job["run_id"]).get_run()
                except FileNotFoundError:
                    run = None
                job["run"] = run
        return campaigns

    def stop(self, run_id: str) -> None:
        job = self.store.get_job(run_id)
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
        """Remove campaign controls for a job without deleting its run data."""
        job = self.store.get_job(run_id)
        if not job:
            raise FileNotFoundError(run_id)
        stopped = False
        if self._job_is_active(job):
            self.stop(run_id)
            stopped = True
        if not self.store.remove_job(run_id):
            raise FileNotFoundError(run_id)
        return {"run_id": run_id, "stopped": stopped}

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
        run = self.runs.database(run_id).get_run()
        if run:
            config["seed"] = run["seed"]
            config["shared_agents"] = bool(
                run.get("config", {}).get("shared_agents", True)
            )
        else:
            snapshot_id = dict(config.get("agent_snapshots", {})).get(
                job["environment"]
            )
            config["shared_agents"] = (
                bool(self.agent_snapshots.get(str(snapshot_id))["shared_agents"])
                if snapshot_id
                else True
            )
        self.runs.clear_stop(run_id)
        if self.slurm.available:
            scheduler_id = self.slurm.submit_training(config)
            self.store.update_job(run_id, "RESUBMITTED", scheduler_id)
            return scheduler_id
        if self.local_jobs:
            scheduler_id = self._start_local(config)
            self.store.update_job(run_id, "LOCAL", scheduler_id)
            return scheduler_id
        raise RuntimeError("Slurm/local job execution is unavailable on this host.")

    def resume_campaign(self, campaign_id: str) -> list[str]:
        campaign = next(
            (item for item in self.store.list_campaigns() if item["id"] == campaign_id),
            None,
        )
        if not campaign:
            raise FileNotFoundError(campaign_id)
        return [self.resume(job["run_id"]) for job in campaign["jobs"]]

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
            "--run-id",
            config["run_id"],
            "--run-directory",
            str(self.runs.runs_directory),
            "--seed",
            str(config["seed"]),
            "--checkpoint-interval",
            str(config.get("checkpoint_interval", 100)),
            "--agent-memory-size",
            str(config.get("agent_memory_size", 100)),
        ]
        if config.get("episodes"):
            command.extend(["--episodes", str(int(config["episodes"]))])
        if not config.get("shared_agents", True):
            command.append("--separate-seat-agents")
        log_path = self.runs.runs_directory / config["run_id"] / "worker.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("a", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=self.project_root,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        return f"local-{process.pid}"

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
        try:
            run = self.runs.database(str(job["run_id"])).get_run()
        except FileNotFoundError:
            return False
        return bool(run and str(run.get("status", "")).lower() == "running")
