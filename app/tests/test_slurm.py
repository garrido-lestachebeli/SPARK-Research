import base64
import json
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


def test_training_submission_propagates_table_and_seat_configuration(
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
            "table_size": 2,
            "seat_configs": [
                {"agent_id": "leduc-random", "mode": "train"},
                {"agent_id": "leduc-q", "mode": "evaluate"},
            ],
        }
    ) == "12345"
    export = next(value for value in commands[0] if value.startswith("--export="))
    assert "SPARK_TABLE_SIZE=2" in export
    encoded = export.split("SPARK_SEAT_CONFIG_B64=", 1)[1].split(",", 1)[0]
    seats = json.loads(base64.urlsafe_b64decode(encoded).decode("utf-8"))
    assert seats[1] == {"agent_id": "leduc-q", "mode": "evaluate"}
