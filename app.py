"""
Football Forecasting System - Interactive Demo
Professional Streamlit interface for match predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
from models.poisson_predictor import DixonColesPredictor

# Page config
st.set_page_config(
    page_title="Football Forecast Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for cleaner look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6c757d;
        margin-top: 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 600;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1e3a5f;
        border-bottom: 2px solid #667eea;
        padding-bottom: 0.5rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .highlight-box {
        background: #e8f4f8;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .stDataFrame {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">⚽ Football Forecast Pro</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Dixon-Coles Bivariate Poisson Prediction Engine</p>', unsafe_allow_html=True)
st.markdown("---")

# Initialize predictor
@st.cache_resource
def get_predictor(rho):
    return DixonColesPredictor(rho=rho)

# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Match Configuration")
    st.markdown("---")

    st.markdown("### 🏠 Home Team")
    home_team = st.text_input("Team Name", "Manchester City", key="home")
    home_xg = st.slider("Expected Goals (xG)", 0.5, 4.0, 1.85, 0.05, key="home_xg")

    st.markdown("### ✈️ Away Team")
    away_team = st.text_input("Team Name", "Arsenal", key="away")
    away_xg = st.slider("Expected Goals (xG)", 0.5, 4.0, 1.25, 0.05, key="away_xg")

    st.markdown("---")
    st.markdown("### 🔧 Model Parameters")
    rho = st.slider(
        "Dixon-Coles ρ",
        -0.30, 0.00, -0.13, 0.01,
        help="Correlation adjustment for low-scoring matches. Typical: -0.13"
    )

    st.markdown("---")
    st.markdown("""
    <div style="background: #f0f2f6; padding: 1rem; border-radius: 8px; font-size: 0.85rem;">
    <strong>About xG:</strong><br>
    Expected Goals measures shot quality.
    League average is ~1.3 xG per team.
    </div>
    """, unsafe_allow_html=True)

# Get prediction
predictor = get_predictor(rho)
pred = predictor.predict_match(home_team, away_team, home_xg, away_xg)

# Match Header
col1, col2, col3 = st.columns([2, 1, 2])
with col1:
    st.markdown(f"### 🏠 {home_team}")
    st.markdown(f"**xG: {home_xg:.2f}**")
with col2:
    st.markdown("<h2 style='text-align: center; color: #6c757d;'>VS</h2>", unsafe_allow_html=True)
with col3:
    st.markdown(f"### ✈️ {away_team}")
    st.markdown(f"**xG: {away_xg:.2f}**")

st.markdown("---")

# 1X2 Market - Main Focus
st.markdown('<p class="section-header">📊 Match Result (1X2)</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(
        label=f"🏠 {home_team}",
        value=f"{pred.home_win:.1%}",
        delta=f"@ {1/pred.home_win:.2f}"
    )
with col2:
    st.metric(
        label="🤝 Draw",
        value=f"{pred.draw:.1%}",
        delta=f"@ {1/pred.draw:.2f}"
    )
with col3:
    st.metric(
        label=f"✈️ {away_team}",
        value=f"{pred.away_win:.1%}",
        delta=f"@ {1/pred.away_win:.2f}"
    )

# Probability bar
prob_data = pd.DataFrame({
    'Outcome': [home_team, 'Draw', away_team],
    'Probability': [pred.home_win, pred.draw, pred.away_win]
})
st.bar_chart(prob_data.set_index('Outcome'), height=150, color="#667eea")

# Two column layout for goals markets
col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<p class="section-header">⚽ Goals Markets</p>', unsafe_allow_html=True)

    # BTTS
    st.markdown("**Both Teams To Score (BTTS)**")
    btts_col1, btts_col2 = st.columns(2)
    with btts_col1:
        st.metric("Yes", f"{pred.btts_yes:.1%}", f"@ {1/pred.btts_yes:.2f}")
    with btts_col2:
        st.metric("No", f"{pred.btts_no:.1%}", f"@ {1/pred.btts_no:.2f}")

    # Over/Under Table
    st.markdown("**Over/Under Lines**")
    ou_data = []
    for line in [1.5, 2.5, 3.5, 4.5]:
        over_prob, under_prob = pred.over_under[line]
        ou_data.append({
            "Line": f"{line} Goals",
            "Over": f"{over_prob:.1%}",
            "O.Odds": f"{1/over_prob:.2f}",
            "Under": f"{under_prob:.1%}",
            "U.Odds": f"{1/under_prob:.2f}",
        })
    st.dataframe(
        pd.DataFrame(ou_data),
        hide_index=True,
        use_container_width=True
    )

with col_right:
    st.markdown('<p class="section-header">🎯 Correct Score</p>', unsafe_allow_html=True)

    # Top correct scores
    cs_data = []
    for i, (score, prob) in enumerate(list(pred.correct_score.items())[:10]):
        cs_data.append({
            "#": i + 1,
            "Score": score,
            "Prob": f"{prob:.1%}",
            "Odds": f"{1/prob:.1f}",
        })
    st.dataframe(
        pd.DataFrame(cs_data),
        hide_index=True,
        use_container_width=True,
        height=350
    )

# Score Matrix Heatmap
st.markdown('<p class="section-header">🔥 Score Probability Matrix</p>', unsafe_allow_html=True)

matrix_df = pd.DataFrame(
    pred.score_matrix[:7, :7] * 100,  # Convert to percentage
    index=[f"{home_team[:3].upper()} {i}" for i in range(7)],
    columns=[f"{away_team[:3].upper()} {i}" for i in range(7)]
)

st.dataframe(
    matrix_df.style
        .background_gradient(cmap='YlOrRd', axis=None)
        .format("{:.2f}%"),
    use_container_width=True,
    height=300
)

# Technical Details (Collapsible)
with st.expander("📚 Technical Details & Methodology"):
    st.markdown("""
    ### Dixon-Coles Model

    This prediction engine uses the **Dixon-Coles bivariate Poisson model** (1997),
    which improves upon independent Poisson by accounting for correlation in low-scoring matches.

    #### The Problem with Independent Poisson
    Standard Poisson assumes home and away goals are independent. In reality:
    - 0-0 draws occur more often than expected
    - 1-1 draws also show positive correlation

    #### Dixon-Coles Adjustment
    The model applies a correction factor τ to low-scoring outcomes:

    | Score | Adjustment Factor |
    |-------|-------------------|
    | 0-0 | `1 - λ₁ × λ₂ × ρ` |
    | 1-0 | `1 + λ₂ × ρ` |
    | 0-1 | `1 + λ₁ × ρ` |
    | 1-1 | `1 - ρ` |

    Where:
    - λ₁ = Home team expected goals
    - λ₂ = Away team expected goals
    - ρ = Correlation parameter (typically -0.13 to -0.05)

    #### Market Derivation
    All betting markets are derived from the score probability matrix:
    - **1X2**: Sum of relevant matrix regions
    - **BTTS**: 1 - P(home=0) - P(away=0) + P(0-0)
    - **Over/Under**: Cumulative sums across total goals
    - **Correct Score**: Direct matrix values
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6c757d; font-size: 0.9rem;">
    <strong>Football Forecast Pro</strong> | Powered by Dixon-Coles Poisson Model<br>
    Built for accurate match prediction and betting market analysis
</div>
""", unsafe_allow_html=True)
