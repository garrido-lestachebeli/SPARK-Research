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
    assert stats["rolling"][-1]["seat_0_average_payoff"] is not None
    assert "0" in stats["rolling"][-1]["seats"]
    assert "1" in stats["rolling"][-1]["seats"]
    assert "average_payoff" in stats["rolling"][-1]["seats"]["1"]
    assert stats["highlights"]["seat_0_win_rate_ci"][0] <= stats["highlights"]["seat_0_win_rate"]


def test_q_table_size_series_reports_distinct_states(tmp_path):
    Trainer(
        TrainerConfig(
            run_id="q-size",
            environment="leduc-holdem",
            run_directory=str(tmp_path),
            episode_limit=1,
        )
    ).run()
    database = RunDatabase(tmp_path / "q-size" / "run.sqlite", initialize=False)
    for episode in range(900):
        database.save_metrics(episode, {"seat_0.table_size": float(episode)})

    series = database.q_table_size_series()
    assert series["unit"] == "distinct Q-states"
    assert series["x_unit"] == "training episode"
    points = series["seats"]["0"]
    assert len(points) <= 401
    assert points[0] == {"episode": 0, "states": 0}
    assert points[-1] == {"episode": 899, "states": 899}
    stats = database.statistics()
    assert stats["q_table_size"]["seats"]["0"][-1]["states"] == 899


def evaluation_rows(payoffs, *, seat=0, member="league-a", name="A"):
    return [
        {
            "round_index": 0,
            "training_episode": 10,
            "agent_version": 10,
            "seat": seat,
            "opponent_member_id": member,
            "opponent_name": name,
            "opponent_agent_id": "leduc-random",
            "episode_index": index,
            "seed": 1_000 + index,
            "payoff": payoff,
            "outcome": "win" if payoff > 0 else "loss" if payoff < 0 else "draw",
        }
        for index, payoff in enumerate(payoffs)
    ]


def test_evaluation_summary_reports_box_plot_quartiles(tmp_path):
    Trainer(
        TrainerConfig(
            run_id="eval-stats",
            environment="leduc-holdem",
            run_directory=str(tmp_path),
            episode_limit=1,
        )
    ).run()
    database = RunDatabase(tmp_path / "eval-stats" / "run.sqlite", initialize=False)
    # 1..9 plus a far outlier: IQR is 3..8, so the Tukey fence sits well below 100.
    database.save_evaluation_episodes(
        evaluation_rows([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
    )

    summary = database.evaluation_summary()
    assert summary["rounds"] == 1
    assert summary["episodes"] == 10
    assert summary["seats"]["0"]["opponents"][0]["name"] == "A"
    opponent = summary["seats"]["0"]["rounds"][0]["opponents"][0]
    assert opponent["name"] == "A"
    assert opponent["n"] == 10
    assert opponent["q1"] == 3.25
    assert opponent["median"] == 5.5
    assert opponent["q3"] == 7.75
    assert opponent["whisker_high"] == 9.0
    assert opponent["whisker_low"] == 1.0
    assert opponent["outliers"] == [100.0]
    assert opponent["max"] == 100.0
    assert opponent["outcomes"]["win"] == 10


def test_evaluation_summary_keeps_one_column_per_testing_round(tmp_path):
    Trainer(
        TrainerConfig(
            run_id="eval-rounds",
            environment="leduc-holdem",
            run_directory=str(tmp_path),
            episode_limit=1,
        )
    ).run()
    database = RunDatabase(tmp_path / "eval-rounds" / "run.sqlite", initialize=False)
    database.save_evaluation_episodes(
        evaluation_rows([1, 2, 3], member="league-a", name="A")
        + evaluation_rows([10, 20, 30], member="league-b", name="B")
    )
    later = evaluation_rows([4, 5, 6], member="league-a", name="A")
    for row in later:
        row["round_index"] = 1
        row["training_episode"] = 20
    later_b = evaluation_rows([40, 50, 60], member="league-b", name="B")
    for row in later_b:
        row["round_index"] = 1
        row["training_episode"] = 20
    database.save_evaluation_episodes(later + later_b)

    summary = database.evaluation_summary()
    seat = summary["seats"]["0"]
    assert summary["rounds"] == 2
    assert [item["name"] for item in seat["opponents"]] == ["A", "B"]
    assert [item["round_index"] for item in seat["rounds"]] == [0, 1]
    assert seat["rounds"][0]["training_episode"] == 10
    assert seat["rounds"][1]["training_episode"] == 20
    assert seat["rounds"][0]["opponents"][0]["median"] == 2.0
    assert seat["rounds"][1]["opponents"][0]["median"] == 5.0
    assert seat["rounds"][0]["opponents"][1]["median"] == 20.0


def test_evaluation_episodes_are_replaced_not_duplicated(tmp_path):
    Trainer(
        TrainerConfig(
            run_id="eval-idempotent",
            environment="leduc-holdem",
            run_directory=str(tmp_path),
            episode_limit=1,
        )
    ).run()
    database = RunDatabase(tmp_path / "eval-idempotent" / "run.sqlite", initialize=False)
    database.save_evaluation_episodes(evaluation_rows([1, 2, 3]))
    database.save_evaluation_episodes(evaluation_rows([4, 5, 6]))

    opponent = database.evaluation_summary()["seats"]["0"]["rounds"][0]["opponents"][0]
    assert opponent["n"] == 3
    assert opponent["median"] == 5.0
