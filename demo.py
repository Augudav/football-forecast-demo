#!/usr/bin/env python3
"""
Football Forecasting System - Demo
Quick demonstration of core prediction engine

Run: python demo.py
"""

import pandas as pd
from models.poisson_predictor import DixonColesPredictor
from models.team_ratings import TeamRatingEngine

def main():
    print("=" * 60)
    print("FOOTBALL FORECASTING SYSTEM - DEMO")
    print("=" * 60)

    # Initialize predictor with Dixon-Coles adjustment
    predictor = DixonColesPredictor(rho=-0.13)

    # Sample fixtures with expected goals
    fixtures = [
        ("Manchester City", "Arsenal", 1.85, 1.25),
        ("Liverpool", "Chelsea", 1.72, 1.18),
        ("Real Madrid", "Barcelona", 1.55, 1.45),
        ("Bayern Munich", "Dortmund", 2.05, 1.35),
        ("PSG", "Marseille", 2.15, 0.95),
    ]

    # Generate predictions
    results = []

    for home, away, home_xg, away_xg in fixtures:
        pred = predictor.predict_match(home, away, home_xg, away_xg)

        results.append({
            'fixture': f"{home} vs {away}",
            'home_xg': home_xg,
            'away_xg': away_xg,
            'home_win_%': f"{pred.home_win:.1%}",
            'draw_%': f"{pred.draw:.1%}",
            'away_win_%': f"{pred.away_win:.1%}",
            'home_odds': round(1/pred.home_win, 2),
            'draw_odds': round(1/pred.draw, 2),
            'away_odds': round(1/pred.away_win, 2),
            'btts_yes_%': f"{pred.btts_yes:.1%}",
            'over_2.5_%': f"{pred.over_under[2.5][0]:.1%}",
            'under_2.5_%': f"{pred.over_under[2.5][1]:.1%}",
        })

        # Print detailed prediction
        print(f"\n{'─' * 60}")
        print(f"📊 {home} vs {away}")
        print(f"   Expected Goals: {home_xg:.2f} - {away_xg:.2f}")
        print(f"\n   1X2 Market:")
        print(f"   ├─ Home Win: {pred.home_win:.1%} → {1/pred.home_win:.2f}")
        print(f"   ├─ Draw:     {pred.draw:.1%} → {1/pred.draw:.2f}")
        print(f"   └─ Away Win: {pred.away_win:.1%} → {1/pred.away_win:.2f}")
        print(f"\n   Goals Markets:")
        print(f"   ├─ BTTS Yes: {pred.btts_yes:.1%} → {1/pred.btts_yes:.2f}")
        print(f"   ├─ Over 2.5: {pred.over_under[2.5][0]:.1%} → {1/pred.over_under[2.5][0]:.2f}")
        print(f"   └─ Under 2.5: {pred.over_under[2.5][1]:.1%} → {1/pred.over_under[2.5][1]:.2f}")
        print(f"\n   Top Correct Scores:")
        for i, (score, prob) in enumerate(list(pred.correct_score.items())[:5]):
            prefix = "   └─" if i == 4 else "   ├─"
            print(f"{prefix} {score}: {prob:.1%} → {1/prob:.1f}")

    # Create summary DataFrame
    df = pd.DataFrame(results)

    print(f"\n{'=' * 60}")
    print("SUMMARY OUTPUT (CSV FORMAT)")
    print("=" * 60)
    print(df.to_string(index=False))

    # Save to CSV
    df.to_csv('predictions_demo.csv', index=False)
    print(f"\n✓ Saved to predictions_demo.csv")

    # Show over/under for all lines
    print(f"\n{'=' * 60}")
    print("OVER/UNDER ALL LINES - Manchester City vs Arsenal")
    print("=" * 60)

    pred = predictor.predict_match("Manchester City", "Arsenal", 1.85, 1.25)
    for line in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]:
        over_prob, under_prob = pred.over_under[line]
        print(f"   {line}: Over {over_prob:.1%} ({1/over_prob:.2f}) | Under {under_prob:.1%} ({1/under_prob:.2f})")


if __name__ == "__main__":
    main()
