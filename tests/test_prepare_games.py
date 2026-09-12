import pandas as pd

from nfl_predictor.data.prepare_games import (
  prepare_completed_regular_season_games,
)


def test_prepare_games_keeps_only_completed_regular_season_games() -> None:
  raw_games = pd.DataFrame(
    {
      "game_id": [
        "2025_01_AWAY_HOME",
        "2025_18_AWAY_HOME",
        "2026_01_AWAY_HOME",
        "2009_01_AWAY_HOME",
      ],
      "season": [2025, 2025, 2026, 2009],
      "game_type": ["REG", "POST", "REG", "REG"],
      "week": [1, 18, 1, 1],
      "gameday": ["2025-09-07"] * 3 + ["2009-09-13"],
      "away_team": ["AWY"] * 4,
      "away_score": [17, 20, None, 21],
      "home_team": ["HME"] * 4,
      "home_score": [24, 27, None, 21],
      "away_rest": [7] * 4,
      "home_rest": [7] * 4,
    }
  )

  prepared_games = prepare_completed_regular_season_games(raw_games)

  assert prepared_games["game_id"].tolist() == ["2025_01_AWAY_HOME"]
  assert prepared_games.loc[0, "home_win"] == 1
  assert prepared_games.loc[0, "home_margin"] == 7
  assert prepared_games.loc[0, "total_points"] == 41