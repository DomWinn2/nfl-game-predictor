"""Create time-safe rolling team-form features."""

from pathlib import Path

import pandas as pd

INPUT_GAMES_PATH = Path("data/processed/completed_regular_games.parquet")
OUTPUT_FEATURES_PATH = Path(
    "data/processed/games_with_rolling_features.parquet"
)

REQUIRED_COLUMNS = {
    "game_id",
    "season",
    "week",
    "gameday",
    "away_team",
    "away_score",
    "home_team",
    "home_score",
    "away_rest",
    "home_rest",
}


def build_team_form_features(
    games: pd.DataFrame,
    window: int = 3,
) -> pd.DataFrame:
    """Return game-level features based on each team's prior games."""
    if window < 1:
        raise ValueError("window must be at least 1.")

    missing_columns = REQUIRED_COLUMNS - set(games.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    games = games.sort_values(["gameday", "game_id"]).reset_index(drop=True)

    game_columns = [
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
    ]

    home_games = games.loc[:, game_columns].copy()
    home_games["team"] = games["home_team"]
    home_games["opponent"] = games["away_team"]
    home_games["is_home"] = 1
    home_games["points_for"] = games["home_score"]
    home_games["points_allowed"] = games["away_score"]

    away_games = games.loc[:, game_columns].copy()
    away_games["team"] = games["away_team"]
    away_games["opponent"] = games["home_team"]
    away_games["is_home"] = 0
    away_games["points_for"] = games["away_score"]
    away_games["points_allowed"] = games["home_score"]

    team_games = pd.concat([home_games, away_games], ignore_index=True)
    team_games = team_games.sort_values(
        ["team", "gameday", "game_id"]
    ).reset_index(drop=True)

    team_games["games_played_before"] = team_games.groupby("team").cumcount()

    team_games[f"avg_points_for_last_{window}"] = (
        team_games.groupby("team")["points_for"]
        .transform(
            lambda values: values.shift(1).rolling(
                window=window,
                min_periods=1,
            ).mean()
        )
    )

    team_games[f"avg_points_allowed_last_{window}"] = (
        team_games.groupby("team")["points_allowed"]
        .transform(
            lambda values: values.shift(1).rolling(
                window=window,
                min_periods=1,
            ).mean()
        )
    )

    feature_columns = [
        "game_id",
        "games_played_before",
        f"avg_points_for_last_{window}",
        f"avg_points_allowed_last_{window}",
    ]

    home_features = team_games.loc[
        team_games["is_home"] == 1,
        feature_columns,
    ].rename(
        columns={
            "games_played_before": "home_games_played_before",
            f"avg_points_for_last_{window}": (
                f"home_avg_points_for_last_{window}"
            ),
            f"avg_points_allowed_last_{window}": (
                f"home_avg_points_allowed_last_{window}"
            ),
        }
    )

    away_features = team_games.loc[
        team_games["is_home"] == 0,
        feature_columns,
    ].rename(
        columns={
            "games_played_before": "away_games_played_before",
            f"avg_points_for_last_{window}": (
                f"away_avg_points_for_last_{window}"
            ),
            f"avg_points_allowed_last_{window}": (
                f"away_avg_points_allowed_last_{window}"
            ),
        }
    )

    return (
        games.merge(home_features, on="game_id", how="left", validate="one_to_one")
        .merge(away_features, on="game_id", how="left", validate="one_to_one")
    )


def main() -> None:
    """Build rolling features and save them as a Parquet file."""
    games = pd.read_parquet(INPUT_GAMES_PATH)
    features = build_team_form_features(games)

    OUTPUT_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_FEATURES_PATH, index=False)

    print(f"Saved rolling features for {len(features):,} games.")
    print(f"Output: {OUTPUT_FEATURES_PATH}")
    print(features.head())


if __name__ == "__main__":
    main()