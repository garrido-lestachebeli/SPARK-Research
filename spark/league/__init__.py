"""League research algorithms: ratings and population evolution.

Persistence for league members lives in ``spark.storage.league``.
This package is the home for future ELO updates and Lamarckian evolution.
"""

from .elo import (
    DEFAULT_ELO,
    DEFAULT_K_FACTOR,
    expected_score,
    transfer_points,
    update_ratings,
)
from .evolution import LamarckianEvolution

__all__ = [
    "DEFAULT_ELO",
    "DEFAULT_K_FACTOR",
    "LamarckianEvolution",
    "expected_score",
    "transfer_points",
    "update_ratings",
]
