"""
Football Forecasting System - Interactive Demo
Streamlit web interface for match predictions
"""

import streamlit as st
import pandas as pd
from models.poisson_predictor import DixonColesPredictor

st.set_page_config(
    page_title="Football Forecast",
    page_icon="",
    layout="wide"
)

st.title("Football Forecasting System")
st.markdown("*Dixon-Coles Poisson Model with Correlation Adjustment*")

# Initialize predictor
predictor = DixonColesPredictor(rho=-0.13)

# Sidebar inputs
st.sidebar.header("Match Setup")

col1, col2 = st.sidebar.columns(2)
with col1:
    home_team = st.text_input("Home Team", "Manchester City")
    home_xg = st.number_input("Home xG", 0.5, 4.0, 1.85, 0.05)
with col2:
    away_team = st.text_input("Away Team", "Arsenal")
    away_xg = st.number_input("Away xG", 0.5, 4.0, 1.25, 0.05)

rho = st.sidebar.slider("Dixon-Coles rho", -0.3, 0.0, -0.13, 0.01,
                        help="Correlation adjustment for low-scoring games")

# Update predictor with custom rho
predictor.rho = rho

# Generate prediction
pred = predictor.predict_match(home_team, away_team, home_xg, away_xg)

# Display results
st.header(f"{home_team} vs {away_team}")
st.markdown(f"**Expected Goals:** {home_xg:.2f} - {away_xg:.2f}")

# 1X2 Market
st.subheader("1X2 Market")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Home Win", f"{pred.home_win:.1%}", f"@ {1/pred.home_win:.2f}")
with col2:
    st.metric("Draw", f"{pred.draw:.1%}", f"@ {1/pred.draw:.2f}")
with col3:
    st.metric("Away Win", f"{pred.away_win:.1%}", f"@ {1/pred.away_win:.2f}")

# Goals Markets
st.subheader("Goals Markets")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**BTTS (Both Teams To Score)**")
    btts_col1, btts_col2 = st.columns(2)
    with btts_col1:
        st.metric("Yes", f"{pred.btts_yes:.1%}", f"@ {1/pred.btts_yes:.2f}")
    with btts_col2:
        st.metric("No", f"{pred.btts_no:.1%}", f"@ {1/pred.btts_no:.2f}")

with col2:
    st.markdown("**Over/Under**")
    ou_data = []
    for line in [1.5, 2.5, 3.5, 4.5]:
        over_prob, under_prob = pred.over_under[line]
        ou_data.append({
            "Line": line,
            "Over %": f"{over_prob:.1%}",
            "Over Odds": f"{1/over_prob:.2f}",
            "Under %": f"{under_prob:.1%}",
            "Under Odds": f"{1/under_prob:.2f}",
        })
    st.dataframe(pd.DataFrame(ou_data), hide_index=True, use_container_width=True)

# Correct Score
st.subheader("Top 10 Correct Scores")
cs_data = []
for score, prob in list(pred.correct_score.items())[:10]:
    cs_data.append({
        "Score": score,
        "Probability": f"{prob:.1%}",
        "Odds": f"{1/prob:.1f}",
    })
col1, col2 = st.columns([1, 2])
with col1:
    st.dataframe(pd.DataFrame(cs_data), hide_index=True, use_container_width=True)

# Score Matrix Heatmap
with col2:
    st.markdown("**Score Probability Matrix**")
    matrix_df = pd.DataFrame(
        pred.score_matrix[:6, :6],
        index=[f"H:{i}" for i in range(6)],
        columns=[f"A:{i}" for i in range(6)]
    )
    st.dataframe(
        matrix_df.style.background_gradient(cmap='YlOrRd').format("{:.1%}"),
        use_container_width=True
    )

# Footer
st.markdown("---")
st.markdown("""
**Technical Details:**
- Dixon-Coles adjustment corrects for correlation in low-scoring matches (0-0, 1-0, 0-1, 1-1)
- Score matrix generated via bivariate Poisson with correlation parameter
- All markets derived from the probability matrix
""")
