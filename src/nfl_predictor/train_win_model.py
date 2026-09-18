"""Train and evaluate a logistic-regression NFL win model."""

from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_DATASET_PATH = Path("data/processed/win_model_dataset.parquet")
MODEL_OUTPUT_PATH = Path("models/logistic_win_model.joblib")

TRAIN_END_SEASON = 2024
TEST_SEASON = 2025

FEATURE_COLUMNS = [
    "elo_difference",
    "rest_difference",
    "home_avg_points_for_last_3",
    "home_avg_points_allowed_last_3",
    "away_avg_points_for_last_3",
    "away_avg_points_allowed_last_3",
    "point_differential_advantage_last_3",
]


def train_and_evaluate(
    model_data: pd.DataFrame,
) -> tuple[Pipeline, dict[str, float]]:
    """Train on past seasons and evaluate on the held-out 2025 season."""
    completed_games = model_data.loc[
        model_data["home_result"].isin([0.0, 1.0])
    ].copy()

    training_games = completed_games.loc[
        completed_games["season"] <= TRAIN_END_SEASON
    ]
    test_games = completed_games.loc[
        completed_games["season"] == TEST_SEASON
    ]

    x_train = training_games[FEATURE_COLUMNS]
    y_train = training_games["home_result"].astype(int)

    x_test = test_games[FEATURE_COLUMNS]
    y_test = test_games["home_result"].astype(int)

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

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "training_games": float(len(training_games)),
        "test_games": float(len(test_games)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "log_loss": float(log_loss(y_test, probabilities)),
    }

    return model, metrics


def main() -> None:
    """Train the win model and print 2025 test performance."""
    model_data = pd.read_parquet(MODEL_DATASET_PATH)
    model, metrics = train_and_evaluate(model_data)
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(model, MODEL_OUTPUT_PATH)

    print("Logistic-regression win model: 2025 test season")
    print("-" * 50)
    print(f"Training games: {metrics['training_games']:,.0f}")
    print(f"Test games: {metrics['test_games']:,.0f}")
    print(f"Winner accuracy: {metrics['accuracy']:.1%}")
    print(f"Brier score: {metrics['brier_score']:.4f}")
    print(f"Log loss: {metrics['log_loss']:.4f}")
    print(f"Saved model: {MODEL_OUTPUT_PATH}")

    classifier = model.named_steps["classifier"]
    coefficients = pd.Series(
        classifier.coef_[0],
        index=FEATURE_COLUMNS,
    ).sort_values()

    print("\nFeature coefficients:")
    print(coefficients)


if __name__ == "__main__":
    main()