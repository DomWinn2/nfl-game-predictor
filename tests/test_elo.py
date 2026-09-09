import pytest

from nfl_predictor.elo import expected_home_win_probability

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