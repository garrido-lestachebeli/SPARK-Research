"""Save and load Blackjack episode reports to a single JSON file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from main import EpisodeReport

STORAGE_VERSION = 1
DEFAULT_EPISODES_FILE = Path(__file__).resolve().parent / "episodes.json"


def save_episodes(
    reports: list[EpisodeReport],
    path: str | Path = DEFAULT_EPISODES_FILE,
) -> Path:
    """Write all episode reports to a single JSON file."""
    file_path = Path(path)
    payload = {
        "version": STORAGE_VERSION,
        "game": "Blackjack-v1",
        "episode_count": len(reports),
        "episodes": [report.to_dict() for report in reports],
    }
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return file_path


def load_episodes(path: str | Path = DEFAULT_EPISODES_FILE) -> list[EpisodeReport]:
    """Load all episode reports from a JSON file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Episode file not found: {file_path}")

    payload = json.loads(file_path.read_text(encoding="utf-8"))
    episodes_data = _extract_episodes(payload)
    return [EpisodeReport.from_dict(episode) for episode in episodes_data]


def append_episode(
    report: EpisodeReport,
    path: str | Path = DEFAULT_EPISODES_FILE,
) -> Path:
    """Append one episode to an existing file, or create the file if needed."""
    file_path = Path(path)
    if file_path.exists():
        reports = load_episodes(file_path)
        reports.append(report)
    else:
        reports = [report]
    return save_episodes(reports, file_path)


def _extract_episodes(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload

    if not isinstance(payload, dict):
        raise ValueError("Episode file must contain a JSON object or episode list.")

    if "episodes" in payload:
        episodes = payload["episodes"]
        if not isinstance(episodes, list):
            raise ValueError("'episodes' must be a list.")
        return episodes

    raise ValueError("Episode file is missing an 'episodes' list.")
