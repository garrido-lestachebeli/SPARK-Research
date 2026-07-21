from spark.storage.stats import mean_ci, wilson_ci
from spark.training.trainer import Trainer, TrainerConfig
from spark.storage import RunDatabase


def test_mean_ci_and_wilson_ci_helpers():
    summary = mean_ci([1.0, 2.0, 3.0, 4.0])
    assert summary["mean"] == 2.5
    assert summary["ci_low"] < summary["mean"] < summary["ci_high"]

    rate = wilson_ci(50, 100)
    assert abs(rate["rate"] - 0.5) < 1e-9
    assert 0.0 <= rate["ci_low"] <= rate["rate"] <= rate["ci_high"] <= 1.0


def test_statistics_include_confidence_intervals(tmp_path):
    Trainer(
        TrainerConfig(
            run_id="stats-rich",
            environment="leduc-holdem",
            run_directory=str(tmp_path),
            episode_limit=8,
            seed=11,
        )
    ).run()
    stats = RunDatabase(tmp_path / "stats-rich" / "run.sqlite", initialize=False).statistics()
    assert stats["episodes"] == 8
    assert "ci_low" in stats["seat_payoffs"]["0"]
    assert "win_rate" in stats["seat_outcomes"]["0"]
    assert stats["rolling"][-1]["seat_0_ci_low"] <= stats["rolling"][-1]["seat_0_average_payoff"]
    assert stats["highlights"]["seat_0_win_rate_ci"][0] <= stats["highlights"]["seat_0_win_rate"]
