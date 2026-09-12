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


def update_elo_ratings(
    home_elo: float,
    away_elo: float,
    home_result: float,
    home_field_advantage: float = DEFAULT_HOME_FIELD_ADVANTAGE,
    k_factor: float = 20.0,
) -> tuple[float, float]:
  """Return updated ratings after a game.

  home_result is 1.0 for a home win, 0.0 for an away win, 0.5 for a tie
  """
  if not 0 <= home_result <= 1:
    raise ValueError("home_result must be between 0 and 1.")

  expected_home_win = expected_home_win_probability(
    home_elo=home_elo,
    away_elo=away_elo,
    home_field_advantage=home_field_advantage,
  )


  rating_change = k_factor * (home_result - expected_home_win)

  return home_elo + rating_change, away_elo - rating_change