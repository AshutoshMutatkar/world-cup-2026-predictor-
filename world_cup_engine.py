"""
world_cup_engine.py

FIFA World Cup 2026 Predictive Engine -- Phase 1 Match Engine
==============================================================

Core Bivariate Poisson match simulator, Elo update logic, and group-stage
tiebreaker resolution. Consumed by world_cup_sim.py and main.py.

This module:
  - Simulates individual matches using a Bivariate Poisson model, with squad-
    strength inputs from world_cup_data.py and random volatility injection
  - Updates team Elo ratings post-match per FIDE chess rules (K=32)
  - Resolves group-stage tiebreakers in priority order
  - Computes group standings given completed match results
  - Handles host-nation Elo bonus for on-soil group fixtures
"""

import math
import random
import numpy as np
from typing import Dict, Tuple, List, Optional
from world_cup_data import (
    TEAM_METRICS,
    GROUPS,
    GROUP_IDS,
    COMPLETED_MATCHES,
    HOST_NATIONS,
    HOST_ELO_BONUS,
    SQUAD_STRENGTH_VOLATILITY,
    EXTRA_TIME_LAMBDA_SCALAR,
    PENALTY_SHOOTOUT_BASELINE,
    GROUP_TIEBREAKER_ORDER,
    WIN_POINTS,
    DRAW_POINTS,
)


def elo_to_expected_win_rate(elo_diff: float) -> float:
    """
    Convert an Elo difference (home_elo - away_elo) to an expected win
    probability via the standard formula: 1 / (1 + 10^(-d/400)).
    """
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))


def simulate_match(
    home_team: str,
    away_team: str,
    home_elo: float,
    away_elo: float,
    home_squad: float,
    away_squad: float,
    is_home_soil: bool = False,
    volatility_scale: float = SQUAD_STRENGTH_VOLATILITY,
) -> Tuple[int, int]:
    """
    Simulate a single group-stage match using Bivariate Poisson.
    
    Args:
        home_team, away_team: Team codes
        home_elo, away_elo: Current Elo ratings
        home_squad, away_squad: Squad strength (0-100 scale)
        is_home_soil: Whether home team is a host nation playing at home
        volatility_scale: +/- range for squad-strength random adjustment
        
    Returns:
        Tuple of (home_goals, away_goals)
    """
    
    # Apply host-nation Elo bonus if applicable
    if is_home_soil and home_team in HOST_NATIONS:
        home_elo += HOST_ELO_BONUS
    
    # Compute Poisson lambda parameters from Elo and squad strength
    # Elo diff contributes ~0.05 goals per 100 points
    elo_contribution = (home_elo - away_elo) / 2000.0
    
    # Squad strength (normalized to ~0-100) contributes ~0.02 goals per point
    squad_normalized_home = home_squad / 100.0
    squad_normalized_away = away_squad / 100.0
    squad_contribution = (squad_normalized_home - squad_normalized_away) * 0.5
    
    # Apply random volatility (±5%)
    home_volatility = 1.0 + random.uniform(-volatility_scale, volatility_scale)
    away_volatility = 1.0 + random.uniform(-volatility_scale, volatility_scale)
    
    # Base lambda (expected goals) per side. International football averages
    # ~2.5 total goals per match, so ~1.15 per team is the realistic baseline.
    home_lambda = 1.15 * (1.0 + elo_contribution + squad_contribution) * home_volatility
    away_lambda = 1.15 * (1.0 - elo_contribution - squad_contribution) * away_volatility
    
    # Clamp to non-negative
    home_lambda = max(0.1, home_lambda)
    away_lambda = max(0.1, away_lambda)
    
    # Draw Poisson variates
    home_goals = int(np.random.poisson(home_lambda))
    away_goals = int(np.random.poisson(away_lambda))
    
    return home_goals, away_goals


def update_elo(
    home_elo: float,
    away_elo: float,
    home_goals: int,
    away_goals: int,
    k: float = 32.0,
) -> Tuple[float, float]:
    """
    Update Elo ratings post-match using standard chess rating formula.
    
    Args:
        home_elo, away_elo: Pre-match Elo ratings
        home_goals, away_goals: Match result
        k: K-factor (controls rating adjustment magnitude)
        
    Returns:
        Tuple of (new_home_elo, new_away_elo)
    """
    
    # Determine match outcome (1=win, 0.5=draw, 0=loss)
    if home_goals > away_goals:
        home_outcome, away_outcome = 1.0, 0.0
    elif home_goals < away_goals:
        home_outcome, away_outcome = 0.0, 1.0
    else:
        home_outcome, away_outcome = 0.5, 0.5
    
    # Expected win probabilities
    home_expected = elo_to_expected_win_rate(home_elo - away_elo)
    away_expected = 1.0 - home_expected
    
    # Rating updates
    new_home_elo = home_elo + k * (home_outcome - home_expected)
    new_away_elo = away_elo + k * (away_outcome - away_expected)
    
    return new_home_elo, new_away_elo


def compute_group_standings(
    group_id: str,
    team_metrics: Dict[str, Dict[str, float]],
    match_list: List[Dict] = None,
) -> List[Tuple[str, Dict]]:
    """
    Compute group-stage standings given a list of match results.
    Applies tiebreaker logic per GROUP_TIEBREAKER_ORDER.

    Args:
        group_id: Group identifier (A-L)
        team_metrics: Current team metrics (Elo, squad strength)
        match_list: List of match dicts to score against. When called from
                    main.py for display purposes, pass None to use the real
                    COMPLETED_MATCHES global. When called from within a
                    Monte Carlo simulation, pass the combined real-plus-
                    simulated match list so future simulated results feed
                    into standings correctly.

    Returns:
        Sorted list of (team_code, standing_dict) for the group,
        with standing_dict containing: points, wins, draws, losses,
        goals_for, goals_against, goal_difference
    """
    if match_list is None:
        match_list = COMPLETED_MATCHES

    teams_in_group = GROUPS[group_id]
    standings = {}

    # Initialize
    for team in teams_in_group:
        standings[team] = {
            "points": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_difference": 0,
            "elo_rating": team_metrics[team]["elo_rating"],
        }

    # Aggregate matches
    for match in match_list:
        if match["group"] != group_id:
            continue
        
        home, away = match["home"], match["away"]
        home_goals, away_goals = match["home_goals"], match["away_goals"]
        
        standings[home]["goals_for"] += home_goals
        standings[home]["goals_against"] += away_goals
        standings[away]["goals_for"] += away_goals
        standings[away]["goals_against"] += home_goals
        
        if home_goals > away_goals:
            standings[home]["points"] += 3
            standings[home]["wins"] += 1
            standings[away]["losses"] += 1
        elif home_goals < away_goals:
            standings[away]["points"] += 3
            standings[away]["wins"] += 1
            standings[home]["losses"] += 1
        else:
            standings[home]["points"] += 1
            standings[away]["points"] += 1
            standings[home]["draws"] += 1
            standings[away]["draws"] += 1
    
    # Compute goal difference
    for team in standings:
        standings[team]["goal_difference"] = (
            standings[team]["goals_for"] - standings[team]["goals_against"]
        )
    
    # Sort via tiebreaker chain
    def tiebreaker_key(item: Tuple[str, Dict]) -> tuple:
        team, standing = item
        key_tuple = []
        for criterion in GROUP_TIEBREAKER_ORDER:
            if criterion == "points":
                key_tuple.append(-standing["points"])
            elif criterion == "head_to_head":
                # Simplified: not yet implemented; use 0 as placeholder
                key_tuple.append(0)
            elif criterion == "goal_difference":
                key_tuple.append(-standing["goal_difference"])
            elif criterion == "goals_for":
                key_tuple.append(-standing["goals_for"])
            elif criterion == "elo_rating":
                key_tuple.append(-standing["elo_rating"])
        return tuple(key_tuple)
    
    sorted_teams = sorted(standings.items(), key=tiebreaker_key)
    return sorted_teams


def get_group_qualifiers(
    group_id: str,
    team_metrics: Dict,
    match_list: List[Dict] = None,
) -> Tuple[List[str], str]:
    """
    Get the top 2 teams (auto-qualifiers) and third-place finisher for a group.

    Returns:
        Tuple of ([team1, team2], team3_code)
    """
    standings = compute_group_standings(group_id, team_metrics, match_list)
    qualifiers = [standings[0][0], standings[1][0]]
    third_place = standings[2][0]
    return qualifiers, third_place


def rank_third_place_teams(
    third_place_teams: List[Tuple[str, str]],
    team_metrics: Dict,
    match_list: List[Dict] = None,
) -> List[str]:
    """
    Rank third-place finishers across all groups to determine which 8
    advance to the Round of 32.

    Args:
        third_place_teams: List of (group_id, team_code) tuples
        team_metrics: Current team metrics
        match_list: Combined real + simulated match list (see compute_group_standings)

    Returns:
        Sorted list of team codes of the 8 advancing third-place finishers
    """
    third_place_standings = {}

    for group_id, team_code in third_place_teams:
        group_standings = compute_group_standings(group_id, team_metrics, match_list)
        team_standing = next(s for s in group_standings if s[0] == team_code)[1]
        third_place_standings[team_code] = {
            "group": group_id,
            "points": team_standing["points"],
            "goal_difference": team_standing["goal_difference"],
            "goals_for": team_standing["goals_for"],
            "elo_rating": team_standing["elo_rating"],
        }
    
    def third_place_key(team_code: str) -> tuple:
        data = third_place_standings[team_code]
        return (
            -data["points"],
            -data["goal_difference"],
            -data["goals_for"],
            -data["elo_rating"],
        )
    
    ranked = sorted(third_place_standings.keys(), key=third_place_key)
    return ranked[:8]  # Top 8


def simulate_knockout_match(
    home_team: str,
    away_team: str,
    home_elo: float,
    away_elo: float,
    home_squad: float,
    away_squad: float,
    allow_extra_time: bool = True,
) -> Tuple[str, Tuple[int, int], Optional[Tuple[int, int]], Optional[bool]]:
    """
    Simulate a knockout-stage match, including extra time and penalties if tied.
    
    Args:
        home_team, away_team: Team codes
        home_elo, away_elo: Current Elo ratings
        home_squad, away_squad: Squad strength (0-100)
        allow_extra_time: Whether to play ET/penalties if drawn at 90 min
        
    Returns:
        Tuple of (winner_code, (home_90, away_90), (home_et, away_et) or None, penalty_winner_was_home or None)
    """
    
    # Simulate 90-minute match
    home_90, away_90 = simulate_match(
        home_team, away_team, home_elo, away_elo, home_squad, away_squad
    )
    
    if home_90 != away_90 or not allow_extra_time:
        winner = home_team if home_90 > away_90 else away_team
        return winner, (home_90, away_90), None, None
    
    # Extra time: Poisson with reduced lambda (30 mins, scaled accordingly)
    home_et = int(np.random.poisson(1.15 * EXTRA_TIME_LAMBDA_SCALAR))
    away_et = int(np.random.poisson(1.15 * EXTRA_TIME_LAMBDA_SCALAR))
    
    if home_90 + home_et != away_90 + away_et:
        winner = home_team if (home_90 + home_et) > (away_90 + away_et) else away_team
        return winner, (home_90, away_90), (home_et, away_et), None
    
    # Penalty shootout: 50/50 baseline, adjusted by squad strength
    home_strength_adj = (home_squad - away_squad) / 200.0
    penalty_prob = PENALTY_SHOOTOUT_BASELINE + home_strength_adj
    penalty_prob = max(0.2, min(0.8, penalty_prob))
    
    home_wins_penalties = random.random() < penalty_prob
    winner = home_team if home_wins_penalties else away_team
    
    return winner, (home_90, away_90), (home_et, away_et), home_wins_penalties
