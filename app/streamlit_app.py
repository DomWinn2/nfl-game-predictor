"""Interactive dashboard for NFL game predictions."""

from pathlib import Path

import pandas as pd
import streamlit as st

PREDICTIONS_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "upcoming_predictions.csv"
)

st.set_page_config(
    page_title="NFL Game Predictor",
    page_icon="🏈",
    layout="wide",
)


@st.cache_data
def load_predictions() -> pd.DataFrame:
    """Load the most recently generated predictions."""
    predictions = pd.read_csv(PREDICTIONS_PATH)
    predictions["gameday"] = pd.to_datetime(predictions["gameday"])
    return predictions


st.title("🏈 NFL Game Predictor")
st.caption(
    "Pre-game win probabilities from Elo ratings, recent team form, "
    "and logistic regression."
)

if not PREDICTIONS_PATH.exists():
    st.error("No prediction file found. Run the prediction pipeline first:")
    st.code(
        "PYTHONPATH=src python -m nfl_predictor.predict_upcoming_games"
    )
    st.stop()

predictions = load_predictions()

available_weeks = sorted(predictions["week"].unique())
selected_week = st.sidebar.selectbox(
    "Select week",
    available_weeks,
)

team_options = sorted(
    set(predictions["home_team"]).union(predictions["away_team"])
)
selected_team = st.sidebar.selectbox(
    "Filter by team",
    ["All teams", *team_options],
)

filtered_predictions = predictions.loc[
    predictions["week"] == selected_week
].copy()

if selected_team != "All teams":
    filtered_predictions = filtered_predictions.loc[
        (filtered_predictions["home_team"] == selected_team)
        | (filtered_predictions["away_team"] == selected_team)
    ]

st.subheader(f"Week {selected_week} predictions")

if filtered_predictions.empty:
    st.info("No games match the selected filters.")
else:
    for game in filtered_predictions.itertuples(index=False):
        away_probability = game.away_win_probability
        home_probability = game.home_win_probability

        with st.container(border=True):
            matchup_column, winner_column, probabilities_column = st.columns(
                [2, 1, 2]
            )

            with matchup_column:
                st.subheader(f"{game.away_team} @ {game.home_team}")
                st.write(game.gameday.strftime("%A, %B %d, %Y"))

            with winner_column:
                st.metric(
                    "Predicted winner",
                    game.predicted_winner,
                )

            with probabilities_column:
                away_column, home_column = st.columns(2)
                away_column.metric(
                    game.away_team,
                    f"{away_probability:.1%}",
                )
                home_column.metric(
                    game.home_team,
                    f"{home_probability:.1%}",
                )

st.divider()
st.subheader("Model performance")

accuracy_column, brier_column, log_loss_column = st.columns(3)

accuracy_column.metric(
    "2025 test accuracy",
    "64.2%",
    help="Percentage of 2025 winners correctly predicted.",
)
brier_column.metric(
    "2025 Brier score",
    "0.2239",
    help="Probability error; lower is better.",
)
log_loss_column.metric(
    "2025 log loss",
    "0.6378",
    help="Penalty for inaccurate or overconfident probabilities; lower is better.",
)

with st.expander("How the model works"):
    st.write(
        """
        The model predicts the probability that the home team wins.
        It combines pregame Elo ratings, recent three-game scoring form,
        points allowed, and days of rest. The selected winner is the team
        with the higher predicted probability.
        """
    )