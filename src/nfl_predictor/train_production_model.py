"""Train the production NFL win-probability model."""

from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from nfl_predictor.train_win_model import FEATURE_COLUMNS

MODEL_DATASET_PATH = Path("data/processed/win_model_dataset.parquet")
MODEL_OUTPUT_PATH = Path("models/production_win_model.joblib")

PRODUCTION_TRAIN_END_SEASON = 2025


def train_production_model(model_data: pd.DataFrame) -> Pipeline:
    """Train on every completed non-tied game through 2025."""
    training_games = model_data.loc[
        model_data["home_result"].isin([0.0, 1.0])
        & (model_data["season"] <= PRODUCTION_TRAIN_END_SEASON)
    ].copy()

    x_train = training_games[FEATURE_COLUMNS]
    y_train = training_games["home_result"].astype(int)

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)
    return model


def main() -> None:
    """Train and save the production model."""
    model_data = pd.read_parquet(MODEL_DATASET_PATH)
    model = train_production_model(model_data)

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(model, MODEL_OUTPUT_PATH)

    training_games = model_data.loc[
        model_data["home_result"].isin([0.0, 1.0])
        & (model_data["season"] <= PRODUCTION_TRAIN_END_SEASON)
    ]

    print(
        f"Trained production model on {len(training_games):,} completed games."
    )
    print(f"Saved model: {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()