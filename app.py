"""
Football Forecasting System - Interactive Demo
Clean, professional Streamlit interface
"""

import streamlit as st
import pandas as pd
import numpy as np
from models.poisson_predictor import DixonColesPredictor

st.set_page_config(
    page_title="Football Forecast Pro",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Football Forecast Pro")
st.caption("Dixon-Coles Bivariate Poisson Prediction Engine")

# Sidebar - just the important stuff
with st.sidebar:
    st.header("Match Setup")

    home_team = st.selectbox("Home Team", [
        "Manchester City", "Liverpool", "Arsenal", "Chelsea",
        "Manchester United", "Tottenham", "Newcastle", "Brighton",
        "Real Madrid", "Barcelona", "Bayern Munich", "PSG"
    ])

    away_team = st.selectbox("Away Team", [
        "Arsenal", "Liverpool", "Chelsea", "Manchester City",
        "Manchester United", "Tottenham", "Newcastle", "Brighton",
        "Real Madrid", "Barcelona", "Bayern Munich", "PSG"
    ], index=0)

    st.divider()

    home_xg = st.slider("Home xG", 0.5, 4.0, 1.85, 0.05)
    away_xg = st.slider("Away xG", 0.5, 4.0, 1.25, 0.05)

    st.divider()

    rho = st.slider("Dixon-Coles ρ", -0.30, 0.00, -0.13, 0.01,
                    help="Correlation parameter for low-scoring matches")

# Prediction
predictor = DixonColesPredictor(rho=rho)
pred = predictor.predict_match(home_team, away_team, home_xg, away_xg)

# Match header
st.markdown(f"### {home_team}  vs  {away_team}")
st.markdown(f"**Expected Goals:** {home_xg:.2f} - {away_xg:.2f}")

st.divider()

# 1X2 Results
st.subheader("Match Result (1X2)")

col1, col2, col3 = st.columns(3)
col1.metric("Home Win", f"{pred.home_win:.1%}", f"@ {1/pred.home_win:.2f}")
col2.metric("Draw", f"{pred.draw:.1%}", f"@ {1/pred.draw:.2f}")
col3.metric("Away Win", f"{pred.away_win:.1%}", f"@ {1/pred.away_win:.2f}")

st.divider()

# Goals Markets
left, right = st.columns(2)

with left:
    st.subheader("Goals Markets")

    # BTTS
    st.markdown("**Both Teams To Score**")
    b1, b2 = st.columns(2)
    b1.metric("BTTS Yes", f"{pred.btts_yes:.1%}", f"@ {1/pred.btts_yes:.2f}")
    b2.metric("BTTS No", f"{pred.btts_no:.1%}", f"@ {1/pred.btts_no:.2f}")

    # Over/Under
    st.markdown("**Over/Under**")
    ou_data = []
    for line in [1.5, 2.5, 3.5, 4.5]:
        over_prob, under_prob = pred.over_under[line]
        ou_data.append({
            "Line": line,
            "Over": f"{over_prob:.1%}",
            "Over Odds": round(1/over_prob, 2),
            "Under": f"{under_prob:.1%}",
            "Under Odds": round(1/under_prob, 2),
        })
    st.dataframe(pd.DataFrame(ou_data), hide_index=True, use_container_width=True)

with right:
    st.subheader("Correct Score")
    cs_data = []
    for score, prob in list(pred.correct_score.items())[:12]:
        cs_data.append({
            "Score": score,
            "Probability": f"{prob:.1%}",
            "Odds": round(1/prob, 1),
        })
    st.dataframe(pd.DataFrame(cs_data), hide_index=True, use_container_width=True)

st.divider()

# Score Matrix
st.subheader("Score Probability Matrix")

matrix_df = pd.DataFrame(
    pred.score_matrix[:6, :6] * 100,
    index=[f"Home {i}" for i in range(6)],
    columns=[f"Away {i}" for i in range(6)]
)

st.dataframe(
    matrix_df.style.background_gradient(cmap='Blues', axis=None).format("{:.1f}%"),
    use_container_width=True
)

# Technical details
with st.expander("Technical Details"):
    st.markdown("""
    **Dixon-Coles Model (1997)**

    Improves on independent Poisson by adjusting for correlation in low-scoring matches:
    - 0-0, 1-1 draws occur more than independent Poisson predicts
    - The ρ parameter corrects for this (typical range: -0.13 to -0.05)

    **Markets derived from score matrix:**
    - 1X2: Sum of matrix regions
    - BTTS: Probability both teams score ≥1
    - O/U: Cumulative totals across matrix
    """)

st.divider()
st.caption("Football Forecast Pro | Dixon-Coles Poisson Model")
