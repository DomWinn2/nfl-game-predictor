"""Build pregame and postgame Elo ratings for historical NFL games."""

from collections import defaultdict
from pathlib import Path

import pandas as pd

from nfl_predictor.elo import (
    DEFAULT_HOME_FIELD_ADVANTAGE,
    expected_home_win_probability,
    update_elo_ratings,
)

PROCESSED_GAMES_PATH = Path("data/processed/completed_regular_games.parquet")
ELO_HISTORY_PATH = Path("data/processed/elo_history.parquet")

INITIAL_ELO = 1500.0
K_FACTOR = 20.0


def build_elo_history(
    games: pd.DataFrame,
    initial_elo: float = INITIAL_ELO,
    home_field_advantage: float = DEFAULT_HOME_FIELD_ADVANTAGE,
    k_factor: float = K_FACTOR,
) -> pd.DataFrame:
    """Calculate each team's Elo before and after every completed game."""
    games = games.sort_values(["gameday", "game_id"]).reset_index(drop=True)

    ratings: defaultdict[str, float] = defaultdict(lambda: initial_elo)
    records: list[dict[str, object]] = []

    for game in games.itertuples(index=False):
        home_elo_pre = ratings[game.home_team]
        away_elo_pre = ratings[game.away_team]

        expected_home_win = expected_home_win_probability(
            home_elo=home_elo_pre,
            away_elo=away_elo_pre,
            home_field_advantage=home_field_advantage,
        )

        if game.home_score > game.away_score:
            home_result = 1.0
        elif game.home_score < game.away_score:
            home_result = 0.0
        else:
            home_result = 0.5

        home_elo_post, away_elo_post = update_elo_ratings(
            home_elo=home_elo_pre,
            away_elo=away_elo_pre,
            home_result=home_result,
            home_field_advantage=home_field_advantage,
            k_factor=k_factor,
        )

        ratings[game.home_team] = home_elo_post
        ratings[game.away_team] = away_elo_post

        records.append(
            {
                "game_id": game.game_id,
                "season": game.season,
                "week": game.week,
                "gameday": game.gameday,
                "away_team": game.away_team,
                "away_score": game.away_score,
                "away_elo_pre": away_elo_pre,
                "away_elo_post": away_elo_post,
                "home_team": game.home_team,
                "home_score": game.home_score,
                "home_elo_pre": home_elo_pre,
                "home_elo_post": home_elo_post,
                "expected_home_win_probability": expected_home_win,
                "home_result": home_result,
            }
        )

    return pd.DataFrame(records)


def main() -> None:
    """Load cleaned games, calculate Elo history, and save it."""
    games = pd.read_parquet(PROCESSED_GAMES_PATH)
    elo_history = build_elo_history(games)

    ELO_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    elo_history.to_parquet(ELO_HISTORY_PATH, index=False)

    print(f"Saved Elo history for {len(elo_history):,} games.")
    print(f"Output: {ELO_HISTORY_PATH}")
    print("\nFirst five games:")
    print(elo_history.head())


if __name__ == "__main__":
    main()