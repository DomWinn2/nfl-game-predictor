"""Download and save NFL game schedule data from nflverse"""

from pathlib import Path

import pandas as pd

GAMES_URL = (
  "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
)
RAW_GAMES_PATH = Path("data/raw/games.csv")


def download_games(
    url: str = GAMES_URL,
    destination: Path = RAW_GAMES_PATH,
) -> pd.DataFrame:
  """Download the game dataset and save it locally"""
  games = pd.read_csv(url)

  destination.parent.mkdir(parents=True, exist_ok=True)
  games.to_csv(destination, index=False)

  return games


def main() -> None:
  """Download games and display a quick inspection summary"""
  games = download_games()

  print(f"Saved {len(games):,} games to {RAW_GAMES_PATH}")
  print(f"Dataset shape: {games.shape}")
  print("\nColumns:")
  print(games.columns.tolist())
  print("\nFirst five rows:")
  print(games.head())


if __name__ == "__main__":
  main()