import pandas as pd
import pytest

from nfl_predictor.features.model_dataset import build_model_dataset


def test_build_model_dataset_creates_expected_features() -> None:
    elo_history = pd.DataFrame(
        {
            "game_id": ["2025_01_A_B"],
            "home_elo_pre": [1550.0],
            "away_elo_pre": [1450.0],
            "expected_home_win_probability": [0.64],
            "home_result": [1.0],
        }
    )

    team_form = pd.DataFrame(
        {
            "game_id": ["2025_01_A_B"],
            "season": [2025],
            "week": [1],
            "gameday": ["2025-09-07"],
            "away_team": ["A"],
            "away_rest": [6],
            "home_team": ["B"],
            "home_rest": [7],
            "home_avg_points_for_last_3": [30.0],
            "home_avg_points_allowed_last_3": [20.0],
            "away_avg_points_for_last_3": [20.0],
            "away_avg_points_allowed_last_3": [25.0],
        }
    )

    model_data = build_model_dataset(elo_history, team_form)

    assert len(model_data) == 1
    assert model_data.loc[0, "elo_difference"] == pytest.approx(100)
    assert model_data.loc[0, "rest_difference"] == pytest.approx(1)
    assert model_data.loc[
        0,
        "point_differential_advantage_last_3",
    ] == pytest.approx(15)