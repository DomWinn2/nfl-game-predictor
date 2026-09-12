"""Evaluate historical Elo win-probability predictions."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

ELO_HISTORY_PATH = Path("data/processed/elo_history.parquet")


def evaluate_elo_predictions(elo_history: pd.DataFrame) -> dict[str, float]:
    """Return accuracy and probability-quality metrics for Elo forecasts."""
    completed_non_ties = elo_history.loc[
        elo_history["home_result"].isin([0.0, 1.0])
    ].copy()

    actual_results = completed_non_ties["home_result"].astype(int)
    probabilities = completed_non_ties["expected_home_win_probability"].clip(
        lower=0.001,
        upper=0.999,
    )
    predicted_results = (probabilities >= 0.5).astype(int)

    return {
        "games_evaluated": float(len(completed_non_ties)),
        "accuracy": float((predicted_results == actual_results).mean()),
        "brier_score": float(brier_score_loss(actual_results, probabilities)),
        "log_loss": float(log_loss(actual_results, probabilities)),
    }


def main() -> None:
    """Load Elo history and print baseline performance."""
    elo_history = pd.read_parquet(ELO_HISTORY_PATH)
    metrics = evaluate_elo_predictions(elo_history)

    print("Elo baseline performance")
    print("-" * 30)
    print(f"Games evaluated: {metrics['games_evaluated']:,.0f}")
    print(f"Winner accuracy: {metrics['accuracy']:.1%}")
    print(f"Brier score: {metrics['brier_score']:.4f}")
    print(f"Log loss: {metrics['log_loss']:.4f}")


if __name__ == "__main__":
    main()