# Football Forecasting System

A statistical prediction engine for football matches using the Dixon-Coles bivariate Poisson model.

## Features

- **Dixon-Coles Model**: Corrects for correlation in low-scoring matches
- **All Betting Markets**: 1X2, BTTS, Over/Under, Correct Score
- **Team Ratings**: xG-based ratings with decay weighting
- **Interactive Demo**: Streamlit web interface

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Usage

```python
from models.poisson_predictor import DixonColesPredictor

predictor = DixonColesPredictor(rho=-0.13)
pred = predictor.predict_match("Man City", "Arsenal", 1.85, 1.25)

print(f"Home Win: {pred.home_win:.1%}")
print(f"Draw: {pred.draw:.1%}")
print(f"Away Win: {pred.away_win:.1%}")
```

## Model Details

The Dixon-Coles adjustment modifies independent Poisson probabilities for low-scoring outcomes:
- 0-0: `P *= (1 - lambda_h * lambda_a * rho)`
- 1-0: `P *= (1 + lambda_a * rho)`
- 0-1: `P *= (1 + lambda_h * rho)`
- 1-1: `P *= (1 - rho)`

Typical rho values range from -0.13 to -0.05.

## Project Structure

```
football_forecast/
├── app.py                    # Streamlit web interface
├── demo.py                   # CLI demo
├── models/
│   ├── poisson_predictor.py  # Dixon-Coles prediction engine
│   └── team_ratings.py       # Team rating calculations
└── requirements.txt
```
