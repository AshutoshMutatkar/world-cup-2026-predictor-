"""
world_cup_sim.py

FIFA World Cup 2026 Predictive Engine -- Phase 3 Tournament Simulation
======================================================================

Monte Carlo tournament simulator for group-stage resolution and knockout-
bracket progression. Runs N independent simulations of the remaining
tournament, producing statistical predictions (win probabilities, medal
odds, knockout stage appearance rates, etc.).

Consumes world_cup_data.py for team metrics and completed results.
Consumes world_cup_engine.py for match simulation and Elo updates.
"""

import random
import copy
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from world_cup_data import (
    TEAM_METRICS,
    GROUPS,
    GROUP_IDS,
    COMPLETED_MATCHES,
    REMAINING_FIXTURES,
    TOTAL_TEAMS,
    KNOCKOUT_BRACKET_SIZE,
    GROUP_STAGE_AUTO_QUALIFIERS_PER_GROUP,
    THIRD_PLACE_TEAMS_ADVANCING,
    REAL_QUARTERFINAL_FIELD,
    SEMIFINAL_FIELD,
    PLAYER_METRICS,
)
from world_cup_engine import (
    simulate_match,
    simulate_knockout_match,
    update_elo,
    compute_group_standings,
    get_group_qualifiers,
    rank_third_place_teams,
)


class TournamentSimulation:
    """
    Single-run tournament simulator. Simulates remaining group-stage
    matches, computes final group standings, runs knockout brackets.
    """
    
    def __init__(self, team_metrics: Dict = None):
        """
        Initialize with a copy of team metrics to avoid modifying global state.
        """
        self.team_metrics = copy.deepcopy(team_metrics or TEAM_METRICS)
        self.tournament_log = []
        self.final_standings = {}
        self.knockout_results = {}
        self.winner = None
        self.runner_up = None
        self.third_place = None

        # Player-level tracking for Golden Boot / POTT (added with the
        # real-data update; only meaningful for teams still alive, since
        # eliminated players' real tournament totals are already final).
        # simulated_goals / simulated_assists accumulate ONLY the goals this
        # particular simulation run projects for the remaining matches.
        self.simulated_goals = defaultdict(int)
        self.simulated_assists = defaultdict(int)
        # Track each alive team's furthest stage reached in this run. All
        # four teams in SEMIFINAL_FIELD have, for real, already reached the
        # Semifinals -- that's the floor now, not something to simulate.
        self.team_furthest_stage = {team: "Semifinal" for team in SEMIFINAL_FIELD}
    
    def _attribute_goal(self, team: str):
        """
        Pick a scorer for one simulated goal by `team`, weighted by each
        tracked player's goals_per_game. If the team has no tracked players
        (or the random draw lands outside all tracked probabilities), the
        goal is deliberately left unattributed rather than forced onto a
        tracked player -- see the handoff discussion on why: forcing it
        would inflate a couple of names' Golden Boot odds using goals that
        were never really theirs to begin with.
        """
        candidates = [
            (name, data["goals_per_game"])
            for name, data in PLAYER_METRICS.items()
            if data["team"] == team and data.get("team_alive")
        ]
        if not candidates:
            return  # unattributed -- team goal total is still correct elsewhere
        total_rate = sum(rate for _, rate in candidates)
        if total_rate <= 0:
            return
        pick = random.uniform(0, total_rate)
        cumulative = 0.0
        for name, rate in candidates:
            cumulative += rate
            if pick <= cumulative:
                self.simulated_goals[name] += 1
                return
    
    def _attribute_assist(self, team: str):
        """Same idea as _attribute_goal, but for assists."""
        candidates = [
            (name, data["assists_per_game"])
            for name, data in PLAYER_METRICS.items()
            if data["team"] == team and data.get("team_alive")
        ]
        if not candidates:
            return
        total_rate = sum(rate for _, rate in candidates)
        if total_rate <= 0:
            return
        pick = random.uniform(0, total_rate)
        cumulative = 0.0
        for name, rate in candidates:
            cumulative += rate
            if pick <= cumulative:
                self.simulated_assists[name] += 1
                return
    
    def simulate_remaining_group_matches(self):
        """
        Simulate all remaining group-stage fixtures, updating Elo ratings
        and logging results.
        """
        for fixture in REMAINING_FIXTURES:
            group = fixture["group"]
            home = fixture["home"]
            away = fixture["away"]
            date = fixture["date"]
            
            home_elo = self.team_metrics[home]["elo_rating"]
            away_elo = self.team_metrics[away]["elo_rating"]
            home_squad = self.team_metrics[home]["squad_strength"]
            away_squad = self.team_metrics[away]["squad_strength"]
            
            # Determine if home team is playing on home soil
            is_home_soil = (home in ["USA", "MEX", "CAN"])
            
            # Simulate match
            home_goals, away_goals = simulate_match(
                home, away, home_elo, away_elo, home_squad, away_squad,
                is_home_soil=is_home_soil
            )
            
            # Update Elo ratings
            new_home_elo, new_away_elo = update_elo(
                home_elo, away_elo, home_goals, away_goals
            )
            self.team_metrics[home]["elo_rating"] = new_home_elo
            self.team_metrics[away]["elo_rating"] = new_away_elo
            
            # Log result
            self.tournament_log.append({
                "stage": "group",
                "group": group,
                "home": home,
                "away": away,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "date": date,
            })
    
    def compute_group_stage_results(self):
        """
        Compute final group standings and identify qualifiers + third-place
        finishers. Uses the combined real + simulated match list so that
        simulated fixture outcomes actually feed into the standings.
        """
        # Build the combined match list: real confirmed results + all group
        # matches simulated in this run. This is the fix for the critical bug
        # where standings only saw COMPLETED_MATCHES global data.
        simulated_group_matches = [
            {
                "group": entry["group"],
                "home": entry["home"],
                "away": entry["away"],
                "home_goals": entry["home_goals"],
                "away_goals": entry["away_goals"],
            }
            for entry in self.tournament_log
            if entry["stage"] == "group"
        ]
        self.all_matches = list(COMPLETED_MATCHES) + simulated_group_matches

        self.qualifiers = {}
        self.third_place_finishers = []

        for group_id in GROUP_IDS:
            qualifiers, third = get_group_qualifiers(
                group_id, self.team_metrics, self.all_matches
            )
            self.qualifiers[group_id] = qualifiers
            self.third_place_finishers.append((group_id, third))

            # Store group standings
            self.final_standings[group_id] = compute_group_standings(
                group_id, self.team_metrics, self.all_matches
            )
    
    def simulate_knockout_stage(self):
        """
        Simulate the knockout bracket from the Semifinals through the Final.

        NOTE: as of the July 13, 2026 real-data update, the group stage, the
        Round of 32, the Round of 16, AND the Quarterfinals have all already
        been played out in reality. The July 8 version of this file still
        simulated the Quarterfinals from REAL_QUARTERFINAL_FIELD -- that's
        now also real history (see QUARTERFINAL_RESULTS in world_cup_data.py)
        rather than something to simulate. The bracket now starts from
        SEMIFINAL_FIELD -- the actual four semifinalists (France, Spain,
        England, Argentina) -- the true first point of remaining uncertainty.
        """
        sf_teams = list(SEMIFINAL_FIELD)

        # Kept for backward-compat with any code reading
        # knockout_results["Round of 32"] / ["Round of 16"] / ["Quarterfinals"]
        # -- all three rounds are real and complete, so all three are simply
        # this same fixed field (all 4 of these teams have, for certain,
        # already reached the Semifinals).
        self.knockout_results["Round of 32"] = sf_teams
        self.knockout_results["Round of 16"] = sf_teams
        self.knockout_results["Quarterfinals"] = sf_teams
        self.knockout_results["Semifinals"] = self._simulate_knockout_round(
            sf_teams, "Semifinals"
        )
        
        # Finals
        final_teams = self.knockout_results["Semifinals"]
        final_result = self._simulate_knockout_round(
            final_teams, "Final"
        )
        
        self.winner = final_result[0]

        # Runner-up: the finalist who lost the Final.
        finalists = final_teams
        self.runner_up = finalists[1] if self.winner == finalists[0] else finalists[0]

        # Third place: the higher-Elo team among the two semifinal losers.
        # sf_teams has exactly 4 entries -- the four real semifinalists.
        # Subtract the two finalists to get the two teams that lost their
        # semifinals.
        finalists_set = set(finalists)
        sf_losers = [t for t in sf_teams if t not in finalists_set]
        self.third_place = max(
            sf_losers, key=lambda t: self.team_metrics[t]["elo_rating"]
        )

        self._compute_player_awards()

    def _compute_player_awards(self):
        """
        Golden Boot: real tournament_goals (locked for eliminated players)
        plus this run's simulated_goals (only possible for alive-team
        players, since only they can play more matches).

        POTT (Player of the Tournament): a constructed heuristic, not
        fitted to real award-voting data (see PLAYER_METRICS docstring in
        world_cup_data.py for that caveat in full). Formula:
            4 x goals + 2 x assists + stage_bonus(team's furthest stage)
        where stage_bonus rewards deep team runs, matching the real-world
        pattern that Golden Ball winners often aren't the top scorer
        (Modric 2018, Messi 2022).
        """
        stage_bonus = {
            "Group stage": 0, "Round of 32": 1, "Round of 16": 2,
            "Quarterfinal": 4, "Semifinal": 7, "Runner-up": 12, "Champion": 18,
        }
        self.golden_boot_totals = {}
        self.pott_scores = {}
        for name, data in PLAYER_METRICS.items():
            total_goals = data["tournament_goals"] + self.simulated_goals.get(name, 0)
            total_assists = data["tournament_assists"] + self.simulated_assists.get(name, 0)
            self.golden_boot_totals[name] = total_goals
            team_stage = self.team_furthest_stage.get(data["team"], "Round of 16")
            bonus = stage_bonus.get(team_stage, 0)
            self.pott_scores[name] = 4 * total_goals + 2 * total_assists + bonus
    
    def _simulate_knockout_round(
        self, teams: List[str], round_name: str
    ) -> List[str]:
        """
        Simulate a knockout round (pairs of teams, winners advance).
        """
        winners = []
        
        for i in range(0, len(teams), 2):
            home = teams[i]
            away = teams[i + 1]
            
            home_elo = self.team_metrics[home]["elo_rating"]
            away_elo = self.team_metrics[away]["elo_rating"]
            home_squad = self.team_metrics[home]["squad_strength"]
            away_squad = self.team_metrics[away]["squad_strength"]
            
            winner, score_90, score_et, penalty_result = simulate_knockout_match(
                home, away, home_elo, away_elo, home_squad, away_squad
            )
            
            # Update Elo post-knockout match (use 90-min result)
            new_home_elo, new_away_elo = update_elo(
                home_elo, away_elo, score_90[0], score_90[1], k=40.0  # K=40 for KO
            )
            self.team_metrics[home]["elo_rating"] = new_home_elo
            self.team_metrics[away]["elo_rating"] = new_away_elo
            
            # Attribute goals (90-min + extra-time, if played) to tracked
            # players on the winning-eligible squads. Unattributed goals
            # (no tracked player on that team, or team not alive) simply
            # don't add to any individual's simulated tally -- see
            # _attribute_goal for why that's the honest choice here.
            home_goals_total = score_90[0] + (score_et[0] if score_et else 0)
            away_goals_total = score_90[1] + (score_et[1] if score_et else 0)
            for _ in range(home_goals_total):
                self._attribute_goal(home)
                self._attribute_assist(home)
            for _ in range(away_goals_total):
                self._attribute_goal(away)
                self._attribute_assist(away)

            # Update furthest-stage-reached for the loser (used by POTT
            # stage_bonus). The winner's stage gets set/overwritten in a
            # later round, or explicitly as Champion/Runner-up below.
            loser = away if winner == home else home
            if round_name == "Semifinals":
                self.team_furthest_stage[loser] = "Semifinal"
            elif round_name == "Final":
                self.team_furthest_stage[loser] = "Runner-up"
                self.team_furthest_stage[winner] = "Champion"

            self.tournament_log.append({
                "stage": "knockout",
                "round": round_name,
                "home": home,
                "away": away,
                "score_90": score_90,
                "score_et": score_et,
                "penalty_shootout": penalty_result,
                "winner": winner,
            })
            
            winners.append(winner)
        
        return winners


class MonteCarloSimulator:
    """
    Run N independent tournament simulations, aggregate statistics.
    """
    
    def __init__(self, num_simulations: int = 10000):
        self.num_simulations = num_simulations
        self.results = []
        self.statistics = {}
    
    def run(self):
        """
        Execute all simulations and compute aggregate statistics.
        """
        for i in range(self.num_simulations):
            sim = TournamentSimulation()
            sim.simulate_remaining_group_matches()
            sim.compute_group_stage_results()
            sim.simulate_knockout_stage()
            self.results.append(sim)
        
        self._compute_statistics()
    
    def _compute_statistics(self):
        """
        Aggregate results across all simulations to compute probabilities.
        """
        # Win probability for each team
        win_counts = defaultdict(int)
        runner_up_counts = defaultdict(int)
        third_place_counts = defaultdict(int)
        group_winner_counts = defaultdict(lambda: defaultdict(int))
        r16_appearance = defaultdict(int)
        qf_appearance = defaultdict(int)
        sf_appearance = defaultdict(int)
        final_appearance = defaultdict(int)
        golden_boot_counts = defaultdict(float)
        pott_counts = defaultdict(float)
        
        for sim in self.results:
            win_counts[sim.winner] += 1
            runner_up_counts[sim.runner_up] += 1
            third_place_counts[sim.third_place] += 1
            
            # Group winners
            for group_id, qualifiers in sim.qualifiers.items():
                group_winner_counts[group_id][qualifiers[0]] += 1
            
            # Knockout appearances
            r32_teams = sim.knockout_results["Round of 32"]
            for team in r32_teams:
                r16_appearance[team] += 1
            
            r16_teams = sim.knockout_results["Round of 16"]
            for team in r16_teams:
                qf_appearance[team] += 1
            
            qf_teams = sim.knockout_results["Quarterfinals"]
            for team in qf_teams:
                sf_appearance[team] += 1
            
            sf_teams = sim.knockout_results["Semifinals"]
            for team in sf_teams:
                final_appearance[team] += 1

            # Golden Boot: whoever has strictly the most total goals wins
            # outright in this sim. FIFA's real tiebreakers (assists, then
            # fewest minutes) aren't modeled here -- minutes-played isn't
            # tracked at all -- so ties are split evenly among the tied
            # leaders rather than guessing a winner. Flagging that
            # simplification rather than silently picking one.
            max_goals = max(sim.golden_boot_totals.values())
            goal_leaders = [n for n, g in sim.golden_boot_totals.items() if g == max_goals]
            for name in goal_leaders:
                golden_boot_counts[name] += 1.0 / len(goal_leaders)

            # POTT: highest constructed pott_score wins outright in this sim.
            max_pott = max(sim.pott_scores.values())
            pott_leaders = [n for n, s in sim.pott_scores.items() if s == max_pott]
            for name in pott_leaders:
                pott_counts[name] += 1.0 / len(pott_leaders)
        
        # Convert to probabilities
        self.statistics = {
            "win_probability": {
                team: count / self.num_simulations
                for team, count in win_counts.items()
            },
            "runner_up_probability": {
                team: count / self.num_simulations
                for team, count in runner_up_counts.items()
            },
            "third_place_probability": {
                team: count / self.num_simulations
                for team, count in third_place_counts.items()
            },
            "group_winner_probability": {
                group_id: {
                    team: count / self.num_simulations
                    for team, count in group_winners.items()
                }
                for group_id, group_winners in group_winner_counts.items()
            },
            "knockout_appearance": {
                "Round of 16": {
                    team: count / self.num_simulations
                    for team, count in r16_appearance.items()
                },
                "Quarterfinals": {
                    team: count / self.num_simulations
                    for team, count in qf_appearance.items()
                },
                "Semifinals": {
                    team: count / self.num_simulations
                    for team, count in sf_appearance.items()
                },
                "Final": {
                    team: count / self.num_simulations
                    for team, count in final_appearance.items()
                },
            },
            "golden_boot_probability": {
                name: count / self.num_simulations
                for name, count in golden_boot_counts.items()
            },
            "pott_probability": {
                name: count / self.num_simulations
                for name, count in pott_counts.items()
            },
            "expected_final_goals": {
                name: sum(sim.golden_boot_totals[name] for sim in self.results) / self.num_simulations
                for name in PLAYER_METRICS
            },
        }
    
    def top_winners(self, n: int = 10) -> List[Tuple[str, float]]:
        """Return the n teams most likely to win."""
        return sorted(
            self.statistics["win_probability"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:n]
    
    def top_golden_boot(self, n: int = 10) -> List[Tuple[str, float]]:
        """Return the n players most likely to win the Golden Boot."""
        return sorted(
            self.statistics["golden_boot_probability"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:n]
    
    def top_pott(self, n: int = 10) -> List[Tuple[str, float]]:
        """Return the n players most likely to be named Player of the Tournament."""
        return sorted(
            self.statistics["pott_probability"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:n]
        """Return the n teams most likely to win."""
        return sorted(
            self.statistics["win_probability"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:n]
    
    def top_semifinalists(self, n: int = 16) -> List[Tuple[str, float]]:
        """Return the n teams most likely to reach the semifinals."""
        return sorted(
            self.statistics["knockout_appearance"]["Semifinals"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:n]
