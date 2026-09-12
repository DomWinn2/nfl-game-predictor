"""Create clean, model ready table of completed NFl regular-season games"""

from pathlib import Path

import pandas as pd

RAW_GAMES_PATH = Path("data/raw/games.csv")
PROCESSED_GAMES_PATH = Path("data/processed/completed_regular_games.parquet")

REQUIRED_COLUMNS = {
  "game_id",
  "season",
  "game_type",
  "week",
  "gameday",
  "away_team",
  "away_score",
  "home_team",
  "home_score",
  "away_rest",
  "home_rest",
}


def prepare_completed_regular_season_games(
    games: pd.DataFrame,
    minimum_season: int = 2010,
) -> pd.DataFrame:
  """Return completed regular season games within forecast targets"""
  missing_columns = REQUIRED_COLUMNS - set(games.columns)
  if missing_columns:
    raise ValueError(f"Missing required columns: {sorted(missing_columns)}")


  completed_games = games.loc[
    (games["season"] >= minimum_season)
    & (games["game_type"] == "REG")
    & games["home_score"].notna()
    & games["away_score"].notna()
  ].copy()

  completed_games["gameday"] = pd.to_datetime(completed_games["gameday"])
  completed_games["home_win"] = (
    completed_games["home_score"] > completed_games["away_score"]
  ).astype(int)
  completed_games["home_margin"] = (
    completed_games["home_score"] - completed_games["away_score"]
  )
  completed_games["total_points"] = (
    completed_games["home_score"] + completed_games["away_score"]
  )

  columns = [
    "game_id",
    "season",
    "week",
    "gameday",
    "away_team",
    "away_score",
    "away_rest",
    "home_team",
    "home_score",
    "home_rest",
    "home_win",
    "home_margin",
    "total_points",
  ]

  return (
    completed_games.loc[:, columns]
    .sort_values(["gameday", "game_id"])
    .reset_index(drop=True)
  )


def main() -> None:
  """Read raw games, prepare them, and save results"""
  games = pd.read_csv(RAW_GAMES_PATH)
  prepared_games = prepare_completed_regular_season_games(games)

  PROCESSED_GAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
  prepared_games.to_parquet(PROCESSED_GAMES_PATH, index=False)

  print(f"Saved {len(prepared_games):,} completed regular-season games.")
  print(f"Output: {PROCESSED_GAMES_PATH}")
  print(prepared_games.head())


if __name__ == "__main__":
  main()