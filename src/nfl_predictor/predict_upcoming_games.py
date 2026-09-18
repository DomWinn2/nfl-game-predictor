"""Generate winner predictions for upcoming NFL games."""

from pathlib import Path

import pandas as pd
from joblib import load

from nfl_predictor.train_win_model import FEATURE_COLUMNS

RAW_GAMES_PATH = Path("data/raw/games.csv")
COMPLETED_GAMES_PATH = Path("data/processed/completed_regular_games.parquet")
ELO_HISTORY_PATH = Path("data/processed/elo_history.parquet")
MODEL_PATH = Path("models/production_win_model.joblib")
PREDICTIONS_PATH = Path("data/processed/upcoming_predictions.csv")

TARGET_SEASON = 2026
INITIAL_ELO = 1500.0
ROLLING_WINDOW = 3


def get_latest_elo_ratings(elo_history: pd.DataFrame) -> pd.DataFrame:
    """Return each team's most recent postgame Elo rating."""
    home_ratings = elo_history.loc[
        :,
        ["gameday", "game_id", "home_team", "home_elo_post"],
    ].rename(
        columns={
            "home_team": "team",
            "home_elo_post": "elo_rating",
        }
    )

    away_ratings = elo_history.loc[
        :,
        ["gameday", "game_id", "away_team", "away_elo_post"],
    ].rename(
        columns={
            "away_team": "team",
            "away_elo_post": "elo_rating",
        }
    )

    ratings = pd.concat([home_ratings, away_ratings], ignore_index=True)

    return (
        ratings.sort_values(["gameday", "game_id"])
        .drop_duplicates("team", keep="last")
        .loc[:, ["team", "elo_rating"]]
    )


def get_recent_team_form(completed_games: pd.DataFrame) -> pd.DataFrame:
    """Return each team's average scoring over its most recent three games."""
    home_games = pd.DataFrame(
        {
            "game_id": completed_games["game_id"],
            "gameday": completed_games["gameday"],
            "team": completed_games["home_team"],
            "points_for": completed_games["home_score"],
            "points_allowed": completed_games["away_score"],
        }
    )

    away_games = pd.DataFrame(
        {
            "game_id": completed_games["game_id"],
            "gameday": completed_games["gameday"],
            "team": completed_games["away_team"],
            "points_for": completed_games["away_score"],
            "points_allowed": completed_games["home_score"],
        }
    )

    team_games = pd.concat([home_games, away_games], ignore_index=True)
    team_games = team_games.sort_values(["team", "gameday", "game_id"])

    recent_games = team_games.groupby("team", group_keys=False).tail(
        ROLLING_WINDOW
    )

    return (
        recent_games.groupby("team", as_index=False)
        .agg(
            avg_points_for_last_3=("points_for", "mean"),
            avg_points_allowed_last_3=("points_allowed", "mean"),
        )
    )


def build_upcoming_feature_data(
    raw_games: pd.DataFrame,
    completed_games: pd.DataFrame,
    elo_history: pd.DataFrame,
    forecast_date: pd.Timestamp,
) -> pd.DataFrame:
    """Build model features for scheduled games on or after forecast_date."""
    raw_games = raw_games.copy()
    raw_games["gameday"] = pd.to_datetime(raw_games["gameday"])

    upcoming_games = raw_games.loc[
        (raw_games["season"] == TARGET_SEASON)
        & (raw_games["game_type"] == "REG")
        & raw_games["home_score"].isna()
        & raw_games["away_score"].isna()
        & (raw_games["gameday"] >= forecast_date),
        [
            "game_id",
            "season",
            "week",
            "gameday",
            "away_team",
            "away_rest",
            "home_team",
            "home_rest",
        ],
    ].copy()

    latest_ratings = get_latest_elo_ratings(elo_history)
    recent_form = get_recent_team_form(completed_games)

    upcoming_games = upcoming_games.merge(
        latest_ratings.rename(
            columns={
                "team": "home_team",
                "elo_rating": "home_elo_pre",
            }
        ),
        on="home_team",
        how="left",
    ).merge(
        latest_ratings.rename(
            columns={
                "team": "away_team",
                "elo_rating": "away_elo_pre",
            }
        ),
        on="away_team",
        how="left",
    ).merge(
        recent_form.rename(
            columns={
                "team": "home_team",
                "avg_points_for_last_3": "home_avg_points_for_last_3",
                "avg_points_allowed_last_3": (
                    "home_avg_points_allowed_last_3"
                ),
            }
        ),
        on="home_team",
        how="left",
    ).merge(
        recent_form.rename(
            columns={
                "team": "away_team",
                "avg_points_for_last_3": "away_avg_points_for_last_3",
                "avg_points_allowed_last_3": (
                    "away_avg_points_allowed_last_3"
                ),
            }
        ),
        on="away_team",
        how="left",
    )

    upcoming_games["home_elo_pre"] = upcoming_games["home_elo_pre"].fillna(
        INITIAL_ELO
    )
    upcoming_games["away_elo_pre"] = upcoming_games["away_elo_pre"].fillna(
        INITIAL_ELO
    )

    upcoming_games["elo_difference"] = (
        upcoming_games["home_elo_pre"] - upcoming_games["away_elo_pre"]
    )
    upcoming_games["rest_difference"] = (
        upcoming_games["home_rest"] - upcoming_games["away_rest"]
    )
    upcoming_games["home_point_differential_last_3"] = (
        upcoming_games["home_avg_points_for_last_3"]
        - upcoming_games["home_avg_points_allowed_last_3"]
    )
    upcoming_games["away_point_differential_last_3"] = (
        upcoming_games["away_avg_points_for_last_3"]
        - upcoming_games["away_avg_points_allowed_last_3"]
    )
    upcoming_games["point_differential_advantage_last_3"] = (
        upcoming_games["home_point_differential_last_3"]
        - upcoming_games["away_point_differential_last_3"]
    )

    return upcoming_games.sort_values(["gameday", "game_id"]).reset_index(
        drop=True
    )


def main() -> None:
    """Generate and save upcoming-game winner predictions."""
    forecast_date = pd.Timestamp.now(
        tz="America/New_York"
    ).tz_localize(None).normalize()

    raw_games = pd.read_csv(RAW_GAMES_PATH)
    completed_games = pd.read_parquet(COMPLETED_GAMES_PATH)
    elo_history = pd.read_parquet(ELO_HISTORY_PATH)

    prediction_data = build_upcoming_feature_data(
        raw_games,
        completed_games,
        elo_history,
        forecast_date,
    )

    model = load(MODEL_PATH)
    home_win_probability = model.predict_proba(
        prediction_data[FEATURE_COLUMNS]
    )[:, 1]

    predictions = prediction_data.loc[
        :,
        ["gameday", "week", "away_team", "home_team"],
    ].copy()
    predictions["home_win_probability"] = home_win_probability
    predictions["away_win_probability"] = 1 - home_win_probability
    predictions["predicted_winner"] = predictions["home_team"].where(
        predictions["home_win_probability"] >= 0.5,
        predictions["away_team"],
    )

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(PREDICTIONS_PATH, index=False)

    print(f"Saved {len(predictions):,} upcoming predictions.")
    print(f"Output: {PREDICTIONS_PATH}")
    print(predictions.head(10).to_string(index=False))


if __name__ == "__main__":
    main()