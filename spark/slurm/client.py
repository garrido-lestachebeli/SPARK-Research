"""Safe subprocess wrapper for Slurm commands."""

from __future__ import annotations

import base64
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

JOB_ID_PATTERN = re.compile(r"^(?P<id>\d+)(?:;.*)?$")
SAFE_EXPORT_PATTERN = re.compile(r"^[A-Za-z0-9_./:@+=,-]+$")


class SlurmUnavailable(RuntimeError):
    pass


class SlurmClient:
    def __init__(self, project_root: str | Path) -> None:
        self.project_root = Path(project_root).resolve()

    @property
    def available(self) -> bool:
        return shutil.which("sbatch") is not None

    def submit_training(self, config: dict[str, Any]) -> str:
        self._require("sbatch")
        script = self.project_root / "app" / "hpc" / "train.sbatch"
        exports = {
            "SPARK_PROJECT_ROOT": str(self.project_root),
            "SPARK_RUN_ID": str(config["run_id"]),
            "SPARK_ENVIRONMENT": str(config["environment"]),
            "SPARK_TABLE_SIZE": str(int(config.get("table_size", 2))),
            "SPARK_SEED": str(int(config.get("seed", 0))),
            "SPARK_RUNS_DIRECTORY": str(config.get("runs_directory", "data")),
            "SPARK_CHECKPOINT_INTERVAL": str(
                int(config.get("checkpoint_interval", 100))
            ),
            "SPARK_EVAL_INTERVAL": str(int(config.get("eval_interval", 0))),
            "SPARK_EVAL_EPISODES": str(int(config.get("eval_episodes", 20))),
            "SPARK_LEAGUE_REROLL_INTERVAL": str(
                int(config.get("league_reroll_interval", 0))
            ),
            "SPARK_STARTING_STACK": str(float(config.get("starting_stack", 50))),
            "SPARK_EPISODE_MODE": str(config.get("episode_mode", "session")),
            "SPARK_EPISODES": str(config.get("episodes") or ""),
            "SPARK_AGENT_MEMORY_SIZE": str(
                int(config.get("agent_memory_size", 100))
            ),
            "SPARK_SEAT_CONFIG_B64": base64.urlsafe_b64encode(
                json.dumps(
                    config.get("seat_configs") or [],
                    separators=(",", ":"),
                ).encode("utf-8")
            ).decode("ascii"),
        }
        export_text = "ALL," + ",".join(
            f"{key}={self._safe_export(value)}" for key, value in exports.items()
        )
        command = [
            "sbatch",
            "--parsable",
            f"--job-name=spark-{config['environment']}",
            f"--cpus-per-task={int(config.get('cpus', 1))}",
            f"--mem={self._safe_export(str(config.get('memory', '4G')))}",
            f"--time={self._safe_export(str(config.get('wall_time', '04:00:00')))}",
            f"--export={export_text}",
        ]
        partition = config.get("partition")
        if partition:
            command.append(f"--partition={self._safe_export(str(partition))}")
        command.append(str(script))
        output = self._run(command)
        match = JOB_ID_PATTERN.match(output.strip())
        if not match:
            raise RuntimeError(f"Unexpected sbatch response: {output!r}")
        return match.group("id")

    def submit_web(
        self,
        *,
        port: int,
        token: str,
        partition: str | None = None,
        wall_time: str = "08:00:00",
    ) -> str:
        self._require("sbatch")
        script = self.project_root / "app" / "hpc" / "web.sbatch"
        exports = {
            "SPARK_PROJECT_ROOT": str(self.project_root),
            "SPARK_WEB_PORT": str(port),
            "SPARK_ACCESS_TOKEN": token,
        }
        command = [
            "sbatch",
            "--parsable",
            "--job-name=spark-web",
            f"--time={self._safe_export(wall_time)}",
            "--mem=2G",
            "--cpus-per-task=1",
            "--export=ALL,"
            + ",".join(f"{key}={self._safe_export(value)}" for key, value in exports.items()),
        ]
        if partition:
            command.append(f"--partition={self._safe_export(partition)}")
        command.append(str(script))
        output = self._run(command)
        match = JOB_ID_PATTERN.match(output.strip())
        if not match:
            raise RuntimeError(f"Unexpected sbatch response: {output!r}")
        return match.group("id")

    def status(self, job_id: str) -> dict[str, str]:
        job_id = self._job_id(job_id)
        self._require("squeue")
        output = self._run(
            ["squeue", "--noheader", "--jobs", job_id, "--format=%i|%T|%M|%N"],
            check=False,
        ).strip()
        if output:
            identifier, state, elapsed, node = (output.split("|", 3) + [""])[:4]
            return {
                "job_id": identifier,
                "state": state,
                "elapsed": elapsed,
                "node": node,
            }
        if shutil.which("sacct"):
            accounting = self._run(
                [
                    "sacct",
                    "--noheader",
                    "--parsable2",
                    "--jobs",
                    job_id,
                    "--format=JobIDRaw,State,Elapsed,NodeList",
                ],
                check=False,
            )
            for line in accounting.splitlines():
                values = line.strip().split("|")
                if values and values[0] == job_id:
                    return {
                        "job_id": job_id,
                        "state": values[1],
                        "elapsed": values[2],
                        "node": values[3],
                    }
        return {"job_id": job_id, "state": "UNKNOWN", "elapsed": "", "node": ""}

    def cancel(self, job_id: str) -> None:
        self._require("scancel")
        self._run(["scancel", self._job_id(job_id)])

    def requeue(self, job_id: str) -> None:
        self._require("scontrol")
        self._run(["scontrol", "requeue", self._job_id(job_id)])

    @staticmethod
    def parse_job_id(output: str) -> str:
        match = JOB_ID_PATTERN.match(output.strip())
        if not match:
            raise ValueError(f"Invalid Slurm job ID output: {output!r}")
        return match.group("id")

    @staticmethod
    def _job_id(value: str) -> str:
        if not value.isdigit():
            raise ValueError(f"Invalid Slurm job ID: {value!r}")
        return value

    @staticmethod
    def _safe_export(value: str) -> str:
        if not value or not SAFE_EXPORT_PATTERN.fullmatch(value):
            if value == "":
                return "none"
            raise ValueError(f"Unsafe Slurm value: {value!r}")
        return value

    @staticmethod
    def _require(command: str) -> None:
        if shutil.which(command) is None:
            raise SlurmUnavailable(f"{command} is not available on this host.")

    @staticmethod
    def _run(command: list[str], *, check: bool = True) -> str:
        result = subprocess.run(
            command,
            check=check,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout.strip()
