from pathlib import Path

import pytest

from spark.slurm.client import SlurmClient


def test_parse_sbatch_job_id():
    assert SlurmClient.parse_job_id("12345") == "12345"
    assert SlurmClient.parse_job_id("12345;cluster") == "12345"


def test_reject_invalid_job_id_and_export_values():
    with pytest.raises(ValueError):
        SlurmClient.parse_job_id("Submitted batch job 123")
    with pytest.raises(ValueError):
        SlurmClient._job_id("12; rm -rf /")
    with pytest.raises(ValueError):
        SlurmClient._safe_export("unsafe value")


def test_training_submission_propagates_separate_agent_topology(
    tmp_path, monkeypatch
):
    client = SlurmClient(tmp_path)
    client.project_root = Path("project")
    commands = []
    monkeypatch.setattr(client, "_require", lambda _command: None)
    monkeypatch.setattr(
        client,
        "_run",
        lambda command: commands.append(command) or "12345",
    )

    assert client.submit_training(
        {
            "run_id": "separate",
            "environment": "leduc-holdem",
            "shared_agents": False,
        }
    ) == "12345"
    export = next(value for value in commands[0] if value.startswith("--export="))
    assert "SPARK_SEPARATE_SEAT_AGENTS=1" in export
