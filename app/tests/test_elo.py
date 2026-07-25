"""Tests for win-rate pairwise ELO transfers."""

from spark.league.elo import expected_score, transfer_points, update_ratings
from spark.storage import LeagueMemberStore
from spark.training.trainer import Trainer, TrainerConfig


def test_equal_winrates_transfer_little_when_elos_equal():
    delta = transfer_points(0.5, 0.5, 1000, 1000, k_factor=32)
    assert abs(delta) < 1e-9


def test_beating_weaker_opponent_transfers_less_than_upset():
    # Same strong win-rate gap; favorite gains less than an underdog upset.
    favorite_beats_weak = transfer_points(0.95, 0.05, 1200, 800, k_factor=32)
    underdog_beats_strong = transfer_points(0.95, 0.05, 800, 1200, k_factor=32)
    assert favorite_beats_weak > 0
    assert underdog_beats_strong > favorite_beats_weak


def test_near_tie_transfers_less_than_lopsided_result():
    near = abs(transfer_points(0.52, 0.48, 1000, 1000, k_factor=32))
    lopsided = abs(transfer_points(0.9, 0.1, 1000, 1000, k_factor=32))
    assert lopsided > near


def test_update_ratings_pairwise_zero_sum_and_no_self_pairs():
    ratings = {"a": 1000.0, "b": 1000.0, "c": 1000.0}
    winrates = {"a": 0.7, "b": 0.4, "c": 0.2}
    updated = update_ratings(ratings, winrates, k_factor=32)
    # Zero-sum across the three pairwise transfers.
    assert abs(sum(updated.values()) - sum(ratings.values())) < 1e-6
    assert updated["a"] > updated["b"] > updated["c"]


def test_expected_score_symmetric():
    assert abs(expected_score(1000, 1000) - 0.5) < 1e-9
    assert abs(expected_score(1200, 800) + expected_score(800, 1200) - 1.0) < 1e-9


def test_trainer_updates_league_elo_after_n_episodes(tmp_path):
    league = LeagueMemberStore(tmp_path)
    a = league.create_blank("A", "leduc-holdem", "leduc-random", status="league")
    b = league.create_blank("B", "leduc-holdem", "leduc-random", status="league")
    assert a["elo"] == 1000
    assert b["elo"] == 1000

    Trainer(
        TrainerConfig(
            run_id="elo-window",
            environment="leduc-holdem",
            run_directory=str(tmp_path),
            episode_limit=6,
            update_interval=3,
            checkpoint_interval=100,
            seat_configs=[
                {
                    "source": "league_member",
                    "agent_id": a["agent_id"],
                    "mode": "evaluate",
                    "league_member_id": a["id"],
                },
                {
                    "source": "league_member",
                    "agent_id": b["agent_id"],
                    "mode": "evaluate",
                    "league_member_id": b["id"],
                },
            ],
        )
    ).run()

    a_after = league.get(a["id"])
    b_after = league.get(b["id"])
    # Ratings should have moved (or stayed equal only if every window was tied).
    assert a_after["stats"]["matches"] == 6
    assert b_after["stats"]["matches"] == 6
    assert a_after["elo"] + b_after["elo"] == 2000
    assert {a_after["elo"], b_after["elo"]} != {1000} or a_after["stats"]["wins"] == b_after[
        "stats"
    ]["wins"]
