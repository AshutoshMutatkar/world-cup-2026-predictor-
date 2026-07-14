"""
main.py

FIFA World Cup 2026 Predictive Engine -- Phase 4 CLI Orchestration
===================================================================

Command-line entry point for the World Cup 2026 prediction system.
Orchestrates data loading, match simulation, and statistical analysis.

Usage:
  python main.py [--sims N] [--quick]

Options:
  --sims N      Run N Monte Carlo simulations (default: 10000)
  --quick       Quick mode: 1000 simulations instead of 10000
  --check       Validate data integrity only
  --standings   Show current group standings
"""

import sys
import os
import json
import argparse
import random
from typing import Optional
from world_cup_data import (
    TEAM_METRICS,
    GROUPS,
    COMPLETED_MATCHES,
    REMAINING_FIXTURES,
    GROUP_IDS,
    PLAYER_METRICS,
)
from world_cup_engine import compute_group_standings, get_group_qualifiers
from world_cup_sim import MonteCarloSimulator


def print_header(text: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_subheader(text: str):
    """Print a formatted subsection header."""
    print(f"\n>>> {text}")
    print("-" * 70)


def bar(prob: float, width: int) -> str:
    """ASCII progress bar -- avoids UnicodeEncodeError on Windows cmd,
    whose default CP1252 codepage can't render block-drawing characters
    like the previous version's block/shade glyphs."""
    filled = int(prob * width)
    return "#" * filled + "-" * (width - filled)


def show_current_standings():
    """Display current group-stage standings."""
    print_header("CURRENT GROUP STAGE STANDINGS")
    
    for group_id in GROUP_IDS:
        standings = compute_group_standings(group_id, TEAM_METRICS)
        print(f"\nGroup {group_id}:")
        print(
            f"  {'Pos':<4} {'Team':<6} {'Pts':<4} {'W-D-L':<10} "
            f"{'GF':<4} {'GA':<4} {'GD':<5}"
        )
        
        for pos, (team, standing) in enumerate(standings, start=1):
            w = standing["wins"]
            d = standing["draws"]
            l = standing["losses"]
            gf = standing["goals_for"]
            ga = standing["goals_against"]
            gd = standing["goal_difference"]
            pts = standing["points"]
            
            print(
                f"  {pos:<4} {team:<6} {pts:<4} {w}-{d}-{l:<7} "
                f"{gf:<4} {ga:<4} {gd:+<5}"
            )


def show_upcoming_fixtures():
    """Display upcoming group-stage fixtures."""
    print_header("UPCOMING GROUP-STAGE FIXTURES")

    if not REMAINING_FIXTURES:
        print("\n  None -- the group stage is complete. Tournament is in the")
        print("  knockout stage; see the Quarterfinal predictions below.")
        return
    
    # Group by date
    fixtures_by_date = {}
    for fixture in REMAINING_FIXTURES:
        date = fixture["date"]
        if date not in fixtures_by_date:
            fixtures_by_date[date] = []
        fixtures_by_date[date].append(fixture)
    
    for date in sorted(fixtures_by_date.keys()):
        print(f"\n{date}:")
        for fixture in fixtures_by_date[date]:
            group = fixture["group"]
            home = fixture["home"]
            away = fixture["away"]
            print(f"  Group {group}: {home:>3} vs {away:<3}")


def run_monte_carlo(num_sims: int = 10000, verbose: bool = True):
    """Run Monte Carlo tournament simulations."""
    print_header(f"RUNNING {num_sims:,} MONTE CARLO SIMULATIONS")
    
    # Seed for reproducibility (optional)
    random.seed(42)
    
    simulator = MonteCarloSimulator(num_simulations=num_sims)
    print(f"\nSimulating remaining tournament {num_sims:,} times...")
    simulator.run()
    print("[OK] Simulations complete.")
    
    return simulator


def show_champion_predictions(simulator: MonteCarloSimulator):
    """Display predicted tournament winner probabilities."""
    print_header("TOURNAMENT WINNER PREDICTIONS")
    
    top_teams = simulator.top_winners(n=16)
    print(
        f"\n  {'Rank':<6} {'Team':<6} {'Win Probability':<20} {'Odds':<10}"
    )
    
    for rank, (team, prob) in enumerate(top_teams, start=1):
        odds = prob_to_odds(prob)
        bar_str = bar(prob, 50)
        
        print(
            f"  {rank:<6} {team:<6} {prob:>6.2%} [{bar_str}] {odds:>8}"
        )


def show_group_winner_predictions(simulator: MonteCarloSimulator):
    """Display predicted group winners."""
    print_header("GROUP WINNER PREDICTIONS")
    
    group_winner_probs = simulator.statistics["group_winner_probability"]
    
    for group_id in GROUP_IDS:
        probs = group_winner_probs[group_id]
        top_teams = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        
        print(f"\nGroup {group_id}:")
        for team, prob in top_teams:
            bar_str = bar(prob, 30)
            print(f"  {team}: {prob:>6.2%} [{bar_str}]")


def show_knockout_predictions(simulator: MonteCarloSimulator):
    """Display knockout-stage appearance probabilities."""
    print_header("KNOCKOUT-STAGE APPEARANCE PREDICTIONS")
    
    ko_data = simulator.statistics["knockout_appearance"]
    
    for stage in ["Round of 16", "Quarterfinals", "Semifinals", "Final"]:
        probs = ko_data[stage]
        top_teams = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:8]
        
        print(f"\n>>> {stage}:")
        for team, prob in top_teams:
            bar_str = bar(prob, 40)
            print(f"  {team}: {prob:>6.2%} [{bar_str}]")


def show_medal_predictions(simulator: MonteCarloSimulator):
    """Display Gold/Silver/Bronze medal predictions."""
    print_header("MEDAL PREDICTIONS")
    
    gold = simulator.statistics["win_probability"]
    silver = simulator.statistics["runner_up_probability"]
    bronze = simulator.statistics["third_place_probability"]
    
    # Top 3 for each medal
    top_gold = sorted(gold.items(), key=lambda x: x[1], reverse=True)[:3]
    top_silver = sorted(silver.items(), key=lambda x: x[1], reverse=True)[:3]
    top_bronze = sorted(bronze.items(), key=lambda x: x[1], reverse=True)[:3]
    
    print("\n>>> GOLD MEDAL (Tournament Winner):")
    for team, prob in top_gold:
        print(f"  {team}: {prob:>6.2%}")
    
    print("\n>>> SILVER MEDAL (Runner-up):")
    for team, prob in top_silver:
        print(f"  {team}: {prob:>6.2%}")
    
    print("\n>>> BRONZE MEDAL (Third Place):")
    for team, prob in top_bronze:
        print(f"  {team}: {prob:>6.2%}")


def show_player_predictions(simulator: MonteCarloSimulator):
    """Display Golden Boot and Player of the Tournament predictions.

    NOTE: only players in PLAYER_METRICS are tracked (real goal/assist
    tallies for ~25 notable names, not full squads). Goals scored by
    untracked players in simulated future matches are intentionally left
    unattributed rather than forced onto a tracked name -- so these odds
    reflect known contenders, not a claim of full-squad coverage.
    """
    print_header("GOLDEN BOOT PREDICTIONS (Top Scorer)")
    top_scorers = simulator.top_golden_boot(n=10)
    print(f"\n  {'Player':<20} {'Team':<6} {'Real Goals':<12} {'Win Prob':<10}")
    for name, prob in top_scorers:
        team = PLAYER_METRICS[name]["team"]
        real_goals = PLAYER_METRICS[name]["tournament_goals"]
        status = "" if PLAYER_METRICS[name].get("team_alive") else " (eliminated, final)"
        print(f"  {name:<20} {team:<6} {real_goals:<12} {prob:>7.2%}{status}")

    print_header("PLAYER OF THE TOURNAMENT PREDICTIONS")
    print("  (heuristic: goals + assists + how far the player's team goes --")
    print("   not fitted to real Golden Ball voting data, see PLAYER_METRICS)")
    top_pott = simulator.top_pott(n=10)
    print(f"\n  {'Player':<20} {'Team':<6} {'Win Prob':<10}")
    for name, prob in top_pott:
        team = PLAYER_METRICS[name]["team"]
        print(f"  {name:<20} {team:<6} {prob:>7.2%}")


def prob_to_odds(prob: float) -> str:
    """Convert probability to decimal odds format."""
    if prob <= 0 or prob >= 1:
        return "N/A"
    odds = 1.0 / prob
    return f"{odds:.2f}"


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="FIFA World Cup 2026 Predictive Engine"
    )
    parser.add_argument(
        "--sims",
        type=int,
        default=10000,
        help="Number of Monte Carlo simulations (default: 10000)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 1000 simulations",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Data validation only",
    )
    parser.add_argument(
        "--standings",
        action="store_true",
        help="Show current standings and upcoming fixtures",
    )
    
    args = parser.parse_args()
    
    num_sims = 1000 if args.quick else args.sims
    
    # Data validation
    print_header("DATA VALIDATION")
    errors = validate_data()
    if errors:
        print(f"\n[ERROR] {len(errors)} integrity issue(s) found:")
        for error in errors:
            print(f"  - {error}")
        return 1
    else:
        print("[OK] All data integrity checks passed.")
        print(f"  Teams: {len(TEAM_METRICS)}")
        print(f"  Groups: {len(GROUPS)}")
        print(f"  Completed matches: {len(COMPLETED_MATCHES)}")
        print(f"  Remaining fixtures: {len(REMAINING_FIXTURES)}")

    # Live-data check: if world_cup_scraper.py has produced a fresh
    # live_updates.json, note that here. Not yet wired into the simulation
    # itself -- this is a heads-up for the user, not a data source swap.
    live_updates_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_updates.json")
    if os.path.exists(live_updates_path):
        try:
            with open(live_updates_path, "r", encoding="utf-8") as f:
                live_data = json.load(f)
            fetched_at = live_data.get("fetched_at", "unknown time")
            print(f"\n[INFO] Found live_updates.json (fetched {fetched_at}).")
            print("  NOTE: not yet consumed by the simulator -- hardcoded")
            print("  world_cup_data.py values are still what's being used.")
        except (json.JSONDecodeError, OSError) as e:
            print(f"\n[WARN] live_updates.json exists but couldn't be read ({e}).")
            print("  Falling back to hardcoded data, as designed.")
    
    if args.check:
        return 0
    
    # Current standings
    show_current_standings()
    show_upcoming_fixtures()
    
    if args.standings:
        return 0
    
    # Monte Carlo simulation
    simulator = run_monte_carlo(num_sims=num_sims)
    
    # Display predictions
    show_champion_predictions(simulator)
    show_medal_predictions(simulator)
    show_group_winner_predictions(simulator)
    show_knockout_predictions(simulator)
    show_player_predictions(simulator)
    
    print_header("SIMULATION COMPLETE")
    print(f"[OK] Ran {num_sims:,} independent tournament simulations.")
    print(f"[OK] All predictions computed from aggregate statistics.")
    
    return 0


def validate_data():
    """Validate data integrity. Returns list of errors."""
    errors = []
    
    if len(TEAM_METRICS) != 48:
        errors.append(f"TEAM_METRICS has {len(TEAM_METRICS)} teams, expected 48.")
    
    if len(GROUPS) != 12:
        errors.append(f"GROUPS has {len(GROUPS)} groups, expected 12.")
    
    all_group_teams = []
    for group_id, teams in GROUPS.items():
        if len(teams) != 4:
            errors.append(f"Group {group_id} has {len(teams)} teams, expected 4.")
        all_group_teams.extend(teams)
    
    duplicate_teams = sorted(
        {t for t in all_group_teams if all_group_teams.count(t) > 1}
    )
    if duplicate_teams:
        errors.append(f"Teams in multiple groups: {duplicate_teams}")
    
    missing_metrics = sorted(set(all_group_teams) - set(TEAM_METRICS))
    if missing_metrics:
        errors.append(f"Teams in GROUPS but not in TEAM_METRICS: {missing_metrics}")
    
    orphan_metrics = sorted(set(TEAM_METRICS) - set(all_group_teams))
    if orphan_metrics:
        errors.append(
            f"Teams in TEAM_METRICS but not in any group: {orphan_metrics}"
        )
    
    for match_list, label in [
        (COMPLETED_MATCHES, "COMPLETED_MATCHES"),
        (REMAINING_FIXTURES, "REMAINING_FIXTURES"),
    ]:
        for m in match_list:
            for side in ("home", "away"):
                if m[side] not in TEAM_METRICS:
                    errors.append(
                        f"{label} references unknown team '{m[side]}' ({m})"
                    )
            if m["group"] not in GROUPS:
                errors.append(
                    f"{label} references unknown group '{m['group']}' ({m})"
                )
    
    return errors


if __name__ == "__main__":
    sys.exit(main())
