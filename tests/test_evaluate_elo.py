import pandas as pd
import pytest

from nfl_predictor.evaluate_elo import evaluate_elo_predictions


def test_evaluate_elo_predictions_excludes_ties() -> None:
    elo_history = pd.DataFrame(
        {
            "home_result": [1.0, 0.0, 1.0, 0.5],
            "expected_home_win_probability": [0.8, 0.2, 0.5, 0.9],
        }
    )

    metrics = evaluate_elo_predictions(elo_history)

    assert metrics["games_evaluated"] == 3
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert metrics["brier_score"] == pytest.approx(0.11)
    assert metrics["log_loss"] > 0