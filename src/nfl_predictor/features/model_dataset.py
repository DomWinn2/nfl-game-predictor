"""Combine Elo and rolling-form features into a model dataset."""

from pathlib import Path

import pandas as pd

ELO_HISTORY_PATH = Path("data/processed/elo_history.parquet")
TEAM_FORM_PATH = Path("data/processed/games_with_rolling_features.parquet")
MODEL_DATASET_PATH = Path("data/processed/win_model_dataset.parquet")

REQUIRED_ELO_COLUMNS = {
    "game_id",
    "home_elo_pre",
    "away_elo_pre",
    "expected_home_win_probability",
    "home_result",
}

REQUIRED_FORM_COLUMNS = {
    "game_id",
    "season",
    "week",
    "gameday",
    "away_team",
    "away_rest",
    "home_team",
    "home_rest",
    "home_avg_points_for_last_3",
    "home_avg_points_allowed_last_3",
    "away_avg_points_for_last_3",
    "away_avg_points_allowed_last_3",
}


def build_model_dataset(
    elo_history: pd.DataFrame,
    team_form: pd.DataFrame,
) -> pd.DataFrame:
    """Return a game-level dataset for training a win-probability model."""
    missing_elo_columns = REQUIRED_ELO_COLUMNS - set(elo_history.columns)
    if missing_elo_columns:
        raise ValueError(
            f"Missing Elo columns: {sorted(missing_elo_columns)}"
        )

    missing_form_columns = REQUIRED_FORM_COLUMNS - set(team_form.columns)
    if missing_form_columns:
        raise ValueError(
            f"Missing team-form columns: {sorted(missing_form_columns)}"
        )

    form_columns = [
        "game_id",
        "season",
        "week",
        "gameday",
        "away_team",
        "away_rest",
        "home_team",
        "home_rest",
        "home_avg_points_for_last_3",
        "home_avg_points_allowed_last_3",
        "away_avg_points_for_last_3",
        "away_avg_points_allowed_last_3",
    ]

    elo_columns = [
        "game_id",
        "home_elo_pre",
        "away_elo_pre",
        "expected_home_win_probability",
        "home_result",
    ]

    model_data = team_form.loc[:, form_columns].merge(
        elo_history.loc[:, elo_columns],
        on="game_id",
        how="inner",
        validate="one_to_one",
    )

    model_data["elo_difference"] = (
        model_data["home_elo_pre"] - model_data["away_elo_pre"]
    )
    model_data["rest_difference"] = (
        model_data["home_rest"] - model_data["away_rest"]
    )
    model_data["home_point_differential_last_3"] = (
        model_data["home_avg_points_for_last_3"]
        - model_data["home_avg_points_allowed_last_3"]
    )
    model_data["away_point_differential_last_3"] = (
        model_data["away_avg_points_for_last_3"]
        - model_data["away_avg_points_allowed_last_3"]
    )
    model_data["point_differential_advantage_last_3"] = (
        model_data["home_point_differential_last_3"]
        - model_data["away_point_differential_last_3"]
    )

    return model_data.sort_values(["gameday", "game_id"]).reset_index(drop=True)


def main() -> None:
    """Load feature tables, build model data, and save it."""
    elo_history = pd.read_parquet(ELO_HISTORY_PATH)
    team_form = pd.read_parquet(TEAM_FORM_PATH)

    model_data = build_model_dataset(elo_history, team_form)

    MODEL_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_data.to_parquet(MODEL_DATASET_PATH, index=False)

    print(f"Saved model dataset for {len(model_data):,} games.")
    print(f"Output: {MODEL_DATASET_PATH}")
    print(model_data.head())


if __name__ == "__main__":
    main()