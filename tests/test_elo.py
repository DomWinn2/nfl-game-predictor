import pytest

from nfl_predictor.elo import expected_home_win_probability
from nfl_predictor.elo import update_elo_ratings

def test_equal_teams_without_home_field_advantage() -> None:
    probability = expected_home_win_probability(
        home_elo=1500,
        away_elo=1500,
        home_field_advantage=0
    )

    assert probability == pytest.approx(0.5)


def test_stronger_home_team_has_higher_win_probability() -> None:
    probability = expected_home_win_probability(
        home_elo=1600,
        away_elo=1500,
        home_field_advantage=0
    )

    assert probability > 0.5


def test_home_win_updates_equal_ratings_without_home_field_advantage() -> None:
    updated_home_elo, updated_away_elo = update_elo_ratings(
        home_elo=1500,
        away_elo=1500,
        home_result=1.0,
        home_field_advantage=0,
    )

    assert updated_home_elo == pytest.approx(1510)
    assert updated_away_elo == pytest.approx(1490)


def test_tie_preserves_equal_ratings_without_home_field_advantage() -> None:
    updated_home_elo, updated_away_elo = update_elo_ratings(
        home_elo=1500,
        away_elo=1500,
        home_result=0.5,
        home_field_advantage=0,
    )

    assert updated_home_elo == pytest.approx(1500)
    assert updated_away_elo == pytest.approx(1500)