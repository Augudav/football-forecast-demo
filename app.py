"""
Football Forecasting System - Demo
Shows the prediction engine with sample data
"""

import streamlit as st
import pandas as pd
from models.poisson_predictor import DixonColesPredictor

st.set_page_config(page_title="Football Forecast - Demo", page_icon="⚽", layout="wide")

st.title("⚽ Football Forecast Pro")
st.caption("Dixon-Coles Prediction Engine - Demo")

# Explanation box
st.info("""
**This is a demo of the core prediction engine.**

In the full system:
- xG values are calculated automatically from historical match data (shots, shot quality, positions)
- Team ratings update after each matchday using decay-weighted averages
- Data is pulled from your preferred provider (Opta, StatsBomb, FBref, etc.)

Below are **sample fixtures** with pre-calculated xG to demonstrate the prediction outputs.
""")

st.divider()

# Sample fixtures - simulating what would come from the database
SAMPLE_FIXTURES = [
    {"home": "Manchester City", "away": "Arsenal", "home_xg": 1.85, "away_xg": 1.25, "league": "Premier League"},
    {"home": "Liverpool", "away": "Chelsea", "home_xg": 1.72, "away_xg": 1.18, "league": "Premier League"},
    {"home": "Real Madrid", "away": "Barcelona", "home_xg": 1.55, "away_xg": 1.45, "league": "La Liga"},
    {"home": "Bayern Munich", "away": "Dortmund", "home_xg": 2.05, "away_xg": 1.35, "league": "Bundesliga"},
    {"home": "PSG", "away": "Marseille", "home_xg": 2.15, "away_xg": 0.95, "league": "Ligue 1"},
]

# Sidebar
with st.sidebar:
    st.header("Select Fixture")

    fixture_names = [f"{f['home']} vs {f['away']}" for f in SAMPLE_FIXTURES]
    selected_idx = st.selectbox("Sample Fixtures", range(len(fixture_names)),
                                 format_func=lambda x: fixture_names[x])

    fixture = SAMPLE_FIXTURES[selected_idx]

    st.divider()
    st.markdown("**xG from historical data:**")
    st.markdown(f"- {fixture['home']}: **{fixture['home_xg']}**")
    st.markdown(f"- {fixture['away']}: **{fixture['away_xg']}**")

    st.divider()
    st.markdown("**Model parameter:**")
    rho = st.slider("Dixon-Coles ρ", -0.30, 0.00, -0.13, 0.01)

    st.divider()
    st.caption("In production: xG calculated from team ratings, opponent strength, and venue factors")

# Run prediction
predictor = DixonColesPredictor(rho=rho)
pred = predictor.predict_match(fixture['home'], fixture['away'], fixture['home_xg'], fixture['away_xg'])

# Display
st.subheader(f"{fixture['home']} vs {fixture['away']}")
st.caption(f"{fixture['league']} | xG: {fixture['home_xg']} - {fixture['away_xg']}")

st.divider()

# 1X2
st.markdown("### Match Result (1X2)")
c1, c2, c3 = st.columns(3)
c1.metric("Home Win", f"{pred.home_win:.1%}", f"@ {1/pred.home_win:.2f}")
c2.metric("Draw", f"{pred.draw:.1%}", f"@ {1/pred.draw:.2f}")
c3.metric("Away Win", f"{pred.away_win:.1%}", f"@ {1/pred.away_win:.2f}")

st.divider()

# Two columns
left, right = st.columns(2)

with left:
    st.markdown("### Goals Markets")

    # BTTS
    b1, b2 = st.columns(2)
    b1.metric("BTTS Yes", f"{pred.btts_yes:.1%}", f"@ {1/pred.btts_yes:.2f}")
    b2.metric("BTTS No", f"{pred.btts_no:.1%}", f"@ {1/pred.btts_no:.2f}")

    # O/U table
    ou_data = []
    for line in [1.5, 2.5, 3.5, 4.5]:
        over_prob, under_prob = pred.over_under[line]
        ou_data.append({
            "Line": line,
            "Over": f"{over_prob:.1%}",
            "Odds": round(1/over_prob, 2),
            "Under": f"{under_prob:.1%}",
            "Odds ": round(1/under_prob, 2),
        })
    st.dataframe(pd.DataFrame(ou_data), hide_index=True, use_container_width=True)

with right:
    st.markdown("### Correct Score")
    cs_data = [{"Score": s, "Prob": f"{p:.1%}", "Odds": round(1/p, 1)}
               for s, p in list(pred.correct_score.items())[:10]]
    st.dataframe(pd.DataFrame(cs_data), hide_index=True, use_container_width=True)

st.divider()

# Score matrix
st.markdown("### Score Probability Matrix")
matrix_df = pd.DataFrame(
    pred.score_matrix[:6, :6] * 100,
    index=[f"Home {i}" for i in range(6)],
    columns=[f"Away {i}" for i in range(6)]
)
st.dataframe(
    matrix_df.style.background_gradient(cmap='Blues', axis=None).format("{:.1f}%"),
    use_container_width=True
)

st.divider()

# What the full system includes
st.markdown("### Full System Includes")
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Data Layer**
    - Historical match database
    - Automated xG data ingestion
    - Team ratings with decay weighting
    - Fixture scheduling integration
    """)

with col2:
    st.markdown("""
    **Prediction Layer**
    - Dixon-Coles Poisson engine ✓ *(this demo)*
    - Monte Carlo season simulations
    - Asian handicap markets
    - Value bet detection vs bookmaker odds
    """)

st.divider()

# API example
with st.expander("API Output Example (JSON)"):
    st.code(f"""{{
  "fixture": "{fixture['home']} vs {fixture['away']}",
  "league": "{fixture['league']}",
  "xg": {{"home": {fixture['home_xg']}, "away": {fixture['away_xg']}}},
  "predictions": {{
    "1x2": {{
      "home_win": {{"prob": {pred.home_win:.4f}, "odds": {1/pred.home_win:.2f}}},
      "draw": {{"prob": {pred.draw:.4f}, "odds": {1/pred.draw:.2f}}},
      "away_win": {{"prob": {pred.away_win:.4f}, "odds": {1/pred.away_win:.2f}}}
    }},
    "btts": {{"yes": {pred.btts_yes:.4f}, "no": {pred.btts_no:.4f}}},
    "over_under": {{
      "2.5": {{"over": {pred.over_under[2.5][0]:.4f}, "under": {pred.over_under[2.5][1]:.4f}}}
    }},
    "correct_score": {dict(list(pred.correct_score.items())[:5])}
  }}
}}""", language="json")

st.divider()
st.caption("Demo only - Full system delivers via REST API, not Streamlit UI")
