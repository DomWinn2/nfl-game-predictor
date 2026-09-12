import pandas as pd
import pytest

from nfl_predictor.features.team_form import build_team_form_features


def test_team_form_features_use_only_prior_games() -> None:
    games = pd.DataFrame(
        {
            "game_id": ["2025_01_A_B", "2025_02_C_A", "2025_03_A_D"],
            "season": [2025, 2025, 2025],
            "week": [1, 2, 3],
            "gameday": ["2025-09-07", "2025-09-14", "2025-09-21"],
            "away_team": ["A", "C", "A"],
            "away_score": [10, 14, 21],
            "away_rest": [7, 7, 7],
            "home_team": ["B", "A", "D"],
            "home_score": [20, 30, 17],
            "home_rest": [7, 7, 7],
        }
    )

    features = build_team_form_features(games, window=3)

    first_game = features.iloc[0]
    second_game = features.iloc[1]
    third_game = features.iloc[2]

    assert first_game["away_games_played_before"] == 0
    assert pd.isna(first_game["away_avg_points_for_last_3"])

    assert second_game["home_team"] == "A"
    assert second_game["home_games_played_before"] == 1
    assert second_game["home_avg_points_for_last_3"] == pytest.approx(10)
    assert second_game["home_avg_points_allowed_last_3"] == pytest.approx(20)

    assert third_game["away_team"] == "A"
    assert third_game["away_games_played_before"] == 2
    assert third_game["away_avg_points_for_last_3"] == pytest.approx(20)
    assert third_game["away_avg_points_allowed_last_3"] == pytest.approx(17)