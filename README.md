# ⚽ Football Forecast Pro

A professional-grade football match prediction system using the Dixon-Coles bivariate Poisson model. Generates accurate probabilities for all major betting markets.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://football-forecast-demo.streamlit.app)

---

## 🎯 Features

### Current Implementation (Demo)
- **Dixon-Coles Poisson Model** - Industry-standard correlation adjustment for low-scoring matches
- **All Major Markets** - 1X2, BTTS, Over/Under (all lines), Correct Score
- **Score Probability Matrix** - Full bivariate distribution visualization
- **Interactive Web Interface** - Real-time predictions with adjustable parameters
- **Team Rating Engine** - xG-based ratings with decay weighting

### Roadmap (Full Version)
- [ ] **Monte Carlo Season Simulator** - League table predictions, title/relegation probabilities
- [ ] **Historical Database** - PostgreSQL storage for match data, odds history
- [ ] **Data Source Integration** - API connections for live xG data
- [ ] **Value Bet Detection** - Compare model odds vs bookmaker odds
- [ ] **Asian Handicap Markets** - Full AH line coverage
- [ ] **Player-Level xG** - Individual player expected goals integration

---

## 🚀 Quick Start

### Option 1: Live Demo
Visit the deployed app: **[football-forecast-demo.streamlit.app](https://football-forecast-demo.streamlit.app)**

### Option 2: Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/football-forecast-demo.git
cd football-forecast-demo

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

# Or run the CLI demo
python demo.py
```

---

## 📊 Usage

### Python API

```python
from models.poisson_predictor import DixonColesPredictor

# Initialize with Dixon-Coles correlation parameter
predictor = DixonColesPredictor(rho=-0.13)

# Generate prediction from expected goals
pred = predictor.predict_match(
    home_team="Manchester City",
    away_team="Arsenal",
    home_xg=1.85,
    away_xg=1.25
)

# Access all markets
print(f"Home Win: {pred.home_win:.1%} @ {1/pred.home_win:.2f}")
print(f"Draw: {pred.draw:.1%} @ {1/pred.draw:.2f}")
print(f"Away Win: {pred.away_win:.1%} @ {1/pred.away_win:.2f}")

print(f"BTTS Yes: {pred.btts_yes:.1%}")
print(f"Over 2.5: {pred.over_under[2.5][0]:.1%}")

# Top correct scores
for score, prob in list(pred.correct_score.items())[:5]:
    print(f"{score}: {prob:.1%} @ {1/prob:.1f}")
```

### Team Ratings

```python
from models.team_ratings import TeamRatingEngine
import pandas as pd

# Load historical xG data
df = pd.read_csv("match_data.csv")

# Calculate ratings
engine = TeamRatingEngine(decay_rate=0.95)
ratings = engine.update_from_xg_data(df, league="Premier League")

# Get match xG from ratings
home_xg, away_xg = engine.get_match_xg("Man City", "Arsenal")
```

---

## 🔬 Model Details

### Dixon-Coles Adjustment

The standard Poisson model assumes home and away goals are independent. The Dixon-Coles (1997) adjustment corrects for observed correlation in low-scoring matches:

| Scoreline | Adjustment Factor |
|-----------|-------------------|
| 0-0 | `1 - λ_home × λ_away × ρ` |
| 1-0 | `1 + λ_away × ρ` |
| 0-1 | `1 + λ_home × ρ` |
| 1-1 | `1 - ρ` |

**Parameters:**
- `λ_home` = Home team expected goals
- `λ_away` = Away team expected goals
- `ρ` = Correlation parameter (typical range: -0.13 to -0.05)

### Market Derivation

All betting markets are derived from the 11×11 score probability matrix:

- **1X2**: Sum matrix regions (lower triangle = home win, diagonal = draw, upper = away)
- **BTTS**: `1 - P(home=0) - P(away=0) + P(0-0)`
- **Over/Under**: Cumulative probability across total goals
- **Correct Score**: Direct matrix cell values

### Team Ratings

Ratings blend multiple metrics with configurable weights:
- xG (40%) - Expected goals from shot quality
- npxG (25%) - Non-penalty expected goals
- xGOT (20%) - Expected goals on target
- Shots (5%) - Volume metric
- Goals (10%) - Actual outcome

Decay weighting (`0.95^n`) gives recent matches more influence.

---

## 📁 Project Structure

```
football_forecast/
├── app.py                      # Streamlit web interface
├── demo.py                     # CLI demonstration
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
└── models/
    ├── __init__.py
    ├── poisson_predictor.py    # Dixon-Coles prediction engine
    └── team_ratings.py         # Team rating calculations
```

---

## 📋 Requirements

- Python 3.9+
- pandas >= 2.0.0
- numpy >= 1.24.0
- scipy >= 1.10.0
- streamlit >= 1.28.0

---

## 📄 License

MIT License - Free to use and modify.

---

## 🤝 Contact

Built by [Your Name] - Available for custom development and extensions.
