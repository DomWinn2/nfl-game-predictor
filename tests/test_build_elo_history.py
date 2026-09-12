import pandas as pd
import pytest

from nfl_predictor.build_elo_history import build_elo_history


def test_elo_history_uses_prior_game_rating() -> None:
    games = pd.DataFrame(
        {
            "game_id": ["2025_02_B_C", "2025_01_A_B"],
            "season": [2025, 2025],
            "week": [2, 1],
            "gameday": ["2025-09-14", "2025-09-07"],
            "away_team": ["B", "A"],
            "away_score": [21, 24],
            "home_team": ["C", "B"],
            "home_score": [14, 17],
        }
    )

    history = build_elo_history(
        games,
        home_field_advantage=0,
        k_factor=20,
    )

    assert history["game_id"].tolist() == [
        "2025_01_A_B",
        "2025_02_B_C",
    ]

    first_game = history.iloc[0]
    second_game = history.iloc[1]

    assert first_game["home_elo_pre"] == pytest.approx(1500)
    assert first_game["home_elo_post"] == pytest.approx(1490)

    assert second_game["away_team"] == "B"
    assert second_game["away_elo_pre"] == pytest.approx(1490)