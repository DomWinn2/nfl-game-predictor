"""Evaluate simple NFL forecasting baselines."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

ELO_HISTORY_PATH = Path("data/processed/elo_history.parquet")
TEST_SEASON = 2025


def evaluate_home_team_baseline(
    elo_history: pd.DataFrame,
    season: int,
) -> dict[str, float]:
    """Evaluate always selecting the home team as the winner."""
    season_games = elo_history.loc[
        (elo_history["season"] == season)
        & elo_history["home_result"].isin([0.0, 1.0])
    ].copy()

    actual_results = season_games["home_result"].astype(int)

    # This baseline always selects the home team for accuracy.
    predicted_results = pd.Series(1, index=season_games.index)

    # For probability metrics, use a neutral 50% forecast.
    probabilities = pd.Series(0.5, index=season_games.index)

    return {
        "games_evaluated": float(len(season_games)),
        "accuracy": float((predicted_results == actual_results).mean()),
        "brier_score": float(brier_score_loss(actual_results, probabilities)),
        "log_loss": float(log_loss(actual_results, probabilities)),
    }


def main() -> None:
    """Load Elo history and print the home-team baseline."""
    elo_history = pd.read_parquet(ELO_HISTORY_PATH)
    metrics = evaluate_home_team_baseline(elo_history, TEST_SEASON)

    print(f"Home-team baseline: {TEST_SEASON} season")
    print("-" * 35)
    print(f"Games evaluated: {metrics['games_evaluated']:,.0f}")
    print(f"Winner accuracy: {metrics['accuracy']:.1%}")
    print(f"Brier score: {metrics['brier_score']:.4f}")
    print(f"Log loss: {metrics['log_loss']:.4f}")


if __name__ == "__main__":
    main()