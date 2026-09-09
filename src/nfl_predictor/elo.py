"""Elo rating calculations for NFL win probability forecasts"""

DEFAULT_HOME_FIELD_ADVANTAGE = 65.0

def expected_home_win_probability(
    home_elo: float,
    away_elo: float,
    home_field_advantage: float = DEFAULT_HOME_FIELD_ADVANTAGE,
) -> float:
  """Return the home teams expected chnace of winning, from 0 to 1"""
  adjusted_home_elo = home_elo + home_field_advantage
  rating_difference = adjusted_home_elo - away_elo

  return 1 / (1 + 10 ** (-rating_difference / 400))