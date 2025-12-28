"""
Poisson Match Prediction Engine with Dixon-Coles Adjustment
Generates probabilities for all betting markets from expected goals
"""

import numpy as np
from scipy.stats import poisson
from scipy.optimize import minimize
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class MatchPrediction:
    """Complete match prediction with all markets"""
    home_team: str
    away_team: str
    home_xg: float
    away_xg: float

    # Core probabilities
    home_win: float
    draw: float
    away_win: float

    # Score matrix (0-10 goals each)
    score_matrix: np.ndarray

    # Markets
    btts_yes: float
    btts_no: float
    over_under: Dict[float, Tuple[float, float]]  # {line: (over_prob, under_prob)}
    correct_score: Dict[str, float]  # {"1-0": prob, ...}

    def to_odds(self) -> Dict[str, float]:
        """Convert probabilities to decimal odds"""
        def prob_to_odds(p: float) -> float:
            return round(1 / p, 2) if p > 0.01 else 100.0

        return {
            "home_win": prob_to_odds(self.home_win),
            "draw": prob_to_odds(self.draw),
            "away_win": prob_to_odds(self.away_win),
            "btts_yes": prob_to_odds(self.btts_yes),
            "btts_no": prob_to_odds(self.btts_no),
            **{f"over_{line}": prob_to_odds(probs[0])
               for line, probs in self.over_under.items()},
            **{f"under_{line}": prob_to_odds(probs[1])
               for line, probs in self.over_under.items()},
        }


class DixonColesPredictor:
    """
    Poisson-based match predictor with Dixon-Coles low-score correlation adjustment.

    The Dixon-Coles adjustment corrects for the observed correlation in low-scoring
    matches (0-0, 1-0, 0-1, 1-1) that standard independent Poisson doesn't capture.
    """

    def __init__(self, rho: float = -0.13):
        """
        Args:
            rho: Dixon-Coles correlation parameter (typically -0.13 to -0.05)
                 Negative rho increases 0-0 and 1-1 probabilities
        """
        self.rho = rho
        self.max_goals = 11  # Score matrix size

    def _dixon_coles_adjustment(
        self,
        home_goals: int,
        away_goals: int,
        home_xg: float,
        away_xg: float
    ) -> float:
        """
        Calculate Dixon-Coles adjustment factor for low-scoring outcomes.
        Only applies to scores: 0-0, 1-0, 0-1, 1-1
        """
        if home_goals == 0 and away_goals == 0:
            return 1 - home_xg * away_xg * self.rho
        elif home_goals == 0 and away_goals == 1:
            return 1 + home_xg * self.rho
        elif home_goals == 1 and away_goals == 0:
            return 1 + away_xg * self.rho
        elif home_goals == 1 and away_goals == 1:
            return 1 - self.rho
        else:
            return 1.0

    def generate_score_matrix(
        self,
        home_xg: float,
        away_xg: float
    ) -> np.ndarray:
        """
        Generate probability matrix for all scorelines with Dixon-Coles adjustment.

        Returns:
            ndarray of shape (max_goals, max_goals) where [i,j] = P(home=i, away=j)
        """
        matrix = np.zeros((self.max_goals, self.max_goals))

        for home_goals in range(self.max_goals):
            for away_goals in range(self.max_goals):
                # Base Poisson probability
                home_prob = poisson.pmf(home_goals, home_xg)
                away_prob = poisson.pmf(away_goals, away_xg)

                # Apply Dixon-Coles adjustment
                dc_adj = self._dixon_coles_adjustment(
                    home_goals, away_goals, home_xg, away_xg
                )

                matrix[home_goals, away_goals] = home_prob * away_prob * dc_adj

        # Normalize to ensure probabilities sum to 1
        matrix = matrix / matrix.sum()

        return matrix

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        home_xg: float,
        away_xg: float
    ) -> MatchPrediction:
        """
        Generate complete match prediction with all betting markets.
        """
        # Generate score matrix
        matrix = self.generate_score_matrix(home_xg, away_xg)

        # 1X2 probabilities
        home_win = np.sum(np.tril(matrix, -1))  # Below diagonal
        draw = np.trace(matrix)  # Diagonal
        away_win = np.sum(np.triu(matrix, 1))  # Above diagonal

        # BTTS
        btts_yes = 1 - matrix[0, :].sum() - matrix[:, 0].sum() + matrix[0, 0]
        btts_no = 1 - btts_yes

        # Over/Under for various lines
        over_under = {}
        for line in [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]:
            total_goals_probs = np.zeros(self.max_goals * 2)
            for i in range(self.max_goals):
                for j in range(self.max_goals):
                    total = i + j
                    if total < len(total_goals_probs):
                        total_goals_probs[total] += matrix[i, j]

            under_prob = sum(total_goals_probs[:int(line) + 1])
            over_prob = 1 - under_prob
            over_under[line] = (over_prob, under_prob)

        # Correct score (top 20 most likely)
        correct_score = {}
        for i in range(min(6, self.max_goals)):
            for j in range(min(6, self.max_goals)):
                score = f"{i}-{j}"
                correct_score[score] = matrix[i, j]

        # Sort by probability
        correct_score = dict(
            sorted(correct_score.items(), key=lambda x: x[1], reverse=True)[:20]
        )

        return MatchPrediction(
            home_team=home_team,
            away_team=away_team,
            home_xg=home_xg,
            away_xg=away_xg,
            home_win=home_win,
            draw=draw,
            away_win=away_win,
            score_matrix=matrix,
            btts_yes=btts_yes,
            btts_no=btts_no,
            over_under=over_under,
            correct_score=correct_score,
        )


def estimate_rho(
    results: list,  # List of (home_goals, away_goals, home_xg, away_xg)
) -> float:
    """
    Estimate optimal Dixon-Coles rho parameter from historical data.
    Uses maximum likelihood estimation.
    """
    def neg_log_likelihood(rho: float) -> float:
        predictor = DixonColesPredictor(rho=rho[0])
        ll = 0
        for home_g, away_g, home_xg, away_xg in results:
            matrix = predictor.generate_score_matrix(home_xg, away_xg)
            prob = matrix[int(home_g), int(away_g)]
            if prob > 0:
                ll += np.log(prob)
        return -ll

    result = minimize(neg_log_likelihood, [-0.1], bounds=[(-0.5, 0.5)])
    return result.x[0]


# Quick demo
if __name__ == "__main__":
    predictor = DixonColesPredictor(rho=-0.13)

    # Example: Man City vs Arsenal
    pred = predictor.predict_match(
        home_team="Manchester City",
        away_team="Arsenal",
        home_xg=1.85,
        away_xg=1.25
    )

    print(f"\n{pred.home_team} vs {pred.away_team}")
    print(f"Expected Goals: {pred.home_xg:.2f} - {pred.away_xg:.2f}")
    print(f"\n1X2 Probabilities:")
    print(f"  Home: {pred.home_win:.1%} (odds: {1/pred.home_win:.2f})")
    print(f"  Draw: {pred.draw:.1%} (odds: {1/pred.draw:.2f})")
    print(f"  Away: {pred.away_win:.1%} (odds: {1/pred.away_win:.2f})")
    print(f"\nBTTS: Yes {pred.btts_yes:.1%} | No {pred.btts_no:.1%}")
    print(f"\nOver/Under 2.5: Over {pred.over_under[2.5][0]:.1%} | Under {pred.over_under[2.5][1]:.1%}")
    print(f"\nTop 5 Correct Scores:")
    for score, prob in list(pred.correct_score.items())[:5]:
        print(f"  {score}: {prob:.1%} (odds: {1/prob:.1f})")
