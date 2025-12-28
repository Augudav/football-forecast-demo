"""
Team Rating Engine
Blends xG-based ratings with market-implied ratings
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime, timedelta


@dataclass
class TeamRating:
    """Complete team rating"""
    team_name: str
    league: str

    # Attack/Defense ratings (goals per game vs league average)
    attack_home: float = 1.0
    attack_away: float = 1.0
    defense_home: float = 1.0  # Lower is better
    defense_away: float = 1.0

    # Overall
    supremacy: float = 0.0  # Expected goal difference vs average team
    home_advantage: float = 0.25  # Team-specific home boost

    # Metadata
    matches_played: int = 0
    manager: str = ""
    manager_start_date: Optional[str] = None
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

    def expected_goals(self, is_home: bool, opponent: "TeamRating", league_avg: float = 1.35) -> float:
        """Calculate expected goals against a specific opponent"""
        if is_home:
            attack = self.attack_home
            defense = opponent.defense_away
            ha_boost = 1 + self.home_advantage
        else:
            attack = self.attack_away
            defense = opponent.defense_home
            ha_boost = 1.0

        return attack * defense * league_avg * ha_boost


class TeamRatingEngine:
    """
    Calculates team ratings from xG data with decay weighting.

    Blends:
    1. Data-based ratings (xG, shots, goals)
    2. Market-implied ratings (from historical odds)
    """

    def __init__(
        self,
        decay_rate: float = 0.95,  # Per-match decay
        manager_boost_matches: int = 5,  # Extra weight for new manager games
        manager_boost_factor: float = 1.5,
    ):
        self.decay_rate = decay_rate
        self.manager_boost_matches = manager_boost_matches
        self.manager_boost_factor = manager_boost_factor
        self.ratings: Dict[str, TeamRating] = {}
        self.league_averages: Dict[str, float] = {}

    def calculate_match_weight(
        self,
        matches_ago: int,
        manager_games: Optional[int] = None
    ) -> float:
        """Calculate weight for a historical match"""
        weight = self.decay_rate ** matches_ago

        # Boost recent games under new manager
        if manager_games is not None and manager_games <= self.manager_boost_matches:
            weight *= self.manager_boost_factor

        return weight

    def update_from_xg_data(
        self,
        df: pd.DataFrame,
        league: str,
        xg_weight: float = 0.4,
        npxg_weight: float = 0.25,
        xgot_weight: float = 0.2,
        shots_weight: float = 0.05,
        goals_weight: float = 0.1,
    ) -> Dict[str, TeamRating]:
        """
        Update team ratings from xG match data.

        Expected columns: home_team, away_team, date,
                         home_xg, away_xg, home_npxg, away_npxg,
                         home_xgot, away_xgot, home_shots, away_shots,
                         home_goals, away_goals
        """
        # Calculate league average
        league_avg_goals = (df['home_goals'].mean() + df['away_goals'].mean()) / 2
        self.league_averages[league] = league_avg_goals

        # Get unique teams
        teams = set(df['home_team'].unique()) | set(df['away_team'].unique())

        for team in teams:
            # Home matches
            home_matches = df[df['home_team'] == team].copy()
            away_matches = df[df['away_team'] == team].copy()

            if len(home_matches) + len(away_matches) == 0:
                continue

            # Calculate blended metrics for home games
            home_attack_samples = []
            home_defense_samples = []
            away_attack_samples = []
            away_defense_samples = []

            for idx, (_, match) in enumerate(home_matches.iterrows()):
                weight = self.calculate_match_weight(idx)

                # Blended attack metric
                blended_attack = (
                    xg_weight * match.get('home_xg', 0) +
                    npxg_weight * match.get('home_npxg', match.get('home_xg', 0)) +
                    xgot_weight * match.get('home_xgot', match.get('home_xg', 0)) +
                    shots_weight * match.get('home_shots', 0) / 10 +  # Normalize shots
                    goals_weight * match.get('home_goals', 0)
                )

                blended_defense = (
                    xg_weight * match.get('away_xg', 0) +
                    npxg_weight * match.get('away_npxg', match.get('away_xg', 0)) +
                    xgot_weight * match.get('away_xgot', match.get('away_xg', 0)) +
                    shots_weight * match.get('away_shots', 0) / 10 +
                    goals_weight * match.get('away_goals', 0)
                )

                home_attack_samples.append((blended_attack, weight))
                home_defense_samples.append((blended_defense, weight))

            for idx, (_, match) in enumerate(away_matches.iterrows()):
                weight = self.calculate_match_weight(idx)

                blended_attack = (
                    xg_weight * match.get('away_xg', 0) +
                    npxg_weight * match.get('away_npxg', match.get('away_xg', 0)) +
                    xgot_weight * match.get('away_xgot', match.get('away_xg', 0)) +
                    shots_weight * match.get('away_shots', 0) / 10 +
                    goals_weight * match.get('away_goals', 0)
                )

                blended_defense = (
                    xg_weight * match.get('home_xg', 0) +
                    npxg_weight * match.get('home_npxg', match.get('home_xg', 0)) +
                    xgot_weight * match.get('home_xgot', match.get('home_xg', 0)) +
                    shots_weight * match.get('home_shots', 0) / 10 +
                    goals_weight * match.get('home_goals', 0)
                )

                away_attack_samples.append((blended_attack, weight))
                away_defense_samples.append((blended_defense, weight))

            # Weighted averages
            def weighted_avg(samples):
                if not samples:
                    return league_avg_goals
                total_weight = sum(w for _, w in samples)
                if total_weight == 0:
                    return league_avg_goals
                return sum(v * w for v, w in samples) / total_weight

            home_attack = weighted_avg(home_attack_samples)
            home_defense = weighted_avg(home_defense_samples)
            away_attack = weighted_avg(away_attack_samples)
            away_defense = weighted_avg(away_defense_samples)

            # Convert to ratings relative to league average
            attack_home = home_attack / league_avg_goals if league_avg_goals > 0 else 1.0
            attack_away = away_attack / league_avg_goals if league_avg_goals > 0 else 1.0
            defense_home = home_defense / league_avg_goals if league_avg_goals > 0 else 1.0
            defense_away = away_defense / league_avg_goals if league_avg_goals > 0 else 1.0

            # Calculate supremacy
            avg_attack = (attack_home + attack_away) / 2
            avg_defense = (defense_home + defense_away) / 2
            supremacy = (avg_attack - avg_defense) * league_avg_goals

            # Estimate home advantage
            home_advantage = 0.25  # Default, would be calculated from data

            self.ratings[team] = TeamRating(
                team_name=team,
                league=league,
                attack_home=attack_home,
                attack_away=attack_away,
                defense_home=defense_home,
                defense_away=defense_away,
                supremacy=supremacy,
                home_advantage=home_advantage,
                matches_played=len(home_matches) + len(away_matches),
            )

        return self.ratings

    def get_match_xg(
        self,
        home_team: str,
        away_team: str,
        neutral: bool = False
    ) -> tuple:
        """Get expected goals for a fixture"""
        home_rating = self.ratings.get(home_team)
        away_rating = self.ratings.get(away_team)

        if not home_rating or not away_rating:
            return (1.35, 1.15)  # Default

        league = home_rating.league
        league_avg = self.league_averages.get(league, 1.35)

        # Home xG = home_attack * away_defense * league_avg * home_advantage
        if neutral:
            home_xg = home_rating.attack_home * away_rating.defense_away * league_avg
            away_xg = away_rating.attack_away * home_rating.defense_home * league_avg
        else:
            home_xg = (home_rating.attack_home * away_rating.defense_away *
                      league_avg * (1 + home_rating.home_advantage))
            away_xg = away_rating.attack_away * home_rating.defense_home * league_avg

        return (home_xg, away_xg)

    def to_dataframe(self) -> pd.DataFrame:
        """Export ratings to DataFrame"""
        data = []
        for team, rating in self.ratings.items():
            data.append({
                'team': team,
                'league': rating.league,
                'attack_home': round(rating.attack_home, 3),
                'attack_away': round(rating.attack_away, 3),
                'defense_home': round(rating.defense_home, 3),
                'defense_away': round(rating.defense_away, 3),
                'supremacy': round(rating.supremacy, 2),
                'home_advantage': round(rating.home_advantage, 2),
                'matches': rating.matches_played,
            })
        return pd.DataFrame(data).sort_values('supremacy', ascending=False)
