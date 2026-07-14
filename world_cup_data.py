"""
world_cup_data.py

FIFA World Cup 2026 Predictive Engine -- Phase 2 Data Layer
=============================================================

Zero-API, self-contained, manually-updatable state layer for the World Cup
2026 prediction system. This module holds no simulation logic -- it is pure
data, intended to be imported by:

    - world_cup_engine.py  (Bivariate Poisson match engine, Elo updates,
      group tie-breaker resolution, host-nation Elo modifier)
    - world_cup_sim.py     (Monte Carlo tournament simulation, knockout
      bracket progression, extra-time / penalty-shootout resolution)
    - main.py              (CLI / orchestration entry point)

UPDATING THIS FILE
-------------------
As real-world results come in: move the corresponding fixture out of
REMAINING_FIXTURES, into COMPLETED_MATCHES, and fill in home_goals /
away_goals. TEAM_METRICS (elo_rating, squad_strength) can be hand-edited at
any time to reflect post-match Elo movement, late squad news, injuries, or
suspensions -- downstream modules re-read this file on every run, so no
other code needs to change.

No network calls, no scraping, no external dependencies. Every value below
is a plain Python literal so this file can be opened and edited in any
text editor.
"""

# ---------------------------------------------------------------------------
# Tournament-wide constants
# ---------------------------------------------------------------------------

# Group-stage points awarded per match outcome.
WIN_POINTS = 3
DRAW_POINTS = 1
LOSS_POINTS = 0

# Tournament shape.
TOTAL_TEAMS = 48
TOTAL_GROUPS = 12
TEAMS_PER_GROUP = 4
MATCHES_PER_GROUP = 6  # round-robin of 4 teams: C(4,2) = 6
TOTAL_GROUP_STAGE_MATCHES = TOTAL_GROUPS * MATCHES_PER_GROUP  # 72

GROUP_IDS = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L")

# Knockout structure: top 2 from each of the 12 groups (24) plus the best 8
# of the 12 third-place finishers (8) = 32-team Round of 32.
GROUP_STAGE_AUTO_QUALIFIERS_PER_GROUP = 2
THIRD_PLACE_TEAMS_ADVANCING = 8
KNOCKOUT_BRACKET_SIZE = 32

# Host nations receive a flat Elo bump applied ONLY to group-stage fixtures
# played on home soil (per system architecture spec). The bonus itself is
# applied inside world_cup_engine.py; it lives here because it modifies the
# team data this file owns.
HOST_NATIONS = ("USA", "MEX", "CAN")
HOST_ELO_BONUS = 100

# Per-match random volatility the simulation engine applies to squad-
# strength inputs. Defined here, consumed in world_cup_engine.py /
# world_cup_sim.py -- not applied within this data module itself.
SQUAD_STRENGTH_VOLATILITY = 0.05  # +/- 5%

# Knockout-stage extra-time goal-rate scaling (Poisson lambda multiplier)
# and the baseline 50/50 penalty-shootout coin-flip, before squad-metric
# adjustment. Defined here for downstream convenience.
EXTRA_TIME_LAMBDA_SCALAR = 0.33
PENALTY_SHOOTOUT_BASELINE = 0.50

# Multi-tiered group tie-breaker chain, in priority order. Consumed by
# world_cup_engine.py when ranking teams within a group and when ranking
# third-place finishers against each other.
GROUP_TIEBREAKER_ORDER = (
    "points",
    "head_to_head",
    "goal_difference",
    "goals_for",
    "elo_rating",
)


# ---------------------------------------------------------------------------
# 1. TEAM_METRICS
# ---------------------------------------------------------------------------
# Baseline, pre-tournament strength inputs for all 48 qualified nations.
#
#   elo_rating      : int   - World Football Elo rating benchmark, June 2026
#                              (eloratings.net scale; anchored to the most
#                              recent confirmed snapshot, hand-correctable
#                              as official ratings move).
#   squad_strength  : float - Aggregated EA Sports FC-style squad rating,
#                              0-100 scale, reflecting current player-level
#                              quality and recent form.
#
# Keys are standard FIFA 3-letter team codes, identical to the codes used
# in GROUPS, COMPLETED_MATCHES, and REMAINING_FIXTURES.
TEAM_METRICS = {
    # --- Group A ---
    "MEX": {"elo_rating": 1792, "squad_strength": 78.5},
    "KOR": {"elo_rating": 1805, "squad_strength": 76.0},
    "RSA": {"elo_rating": 1478, "squad_strength": 58.0},
    "CZE": {"elo_rating": 1658, "squad_strength": 66.5},

    # --- Group B ---
    "CAN": {"elo_rating": 1702, "squad_strength": 73.5},
    "BIH": {"elo_rating": 1648, "squad_strength": 65.0},
    "QAT": {"elo_rating": 1542, "squad_strength": 60.5},
    "SUI": {"elo_rating": 1897, "squad_strength": 79.5},

    # --- Group C ---
    "BRA": {"elo_rating": 1979, "squad_strength": 88.5},
    "MAR": {"elo_rating": 1812, "squad_strength": 80.0},
    "HAI": {"elo_rating": 1378, "squad_strength": 50.0},
    "SCO": {"elo_rating": 1701, "squad_strength": 71.5},

    # --- Group D ---
    "USA": {"elo_rating": 1789, "squad_strength": 79.0},
    "PAR": {"elo_rating": 1558, "squad_strength": 61.0},
    "AUS": {"elo_rating": 1698, "squad_strength": 68.5},
    "TUR": {"elo_rating": 1880, "squad_strength": 78.0},

    # --- Group E ---
    "GER": {"elo_rating": 1910, "squad_strength": 85.5},
    "CIV": {"elo_rating": 1742, "squad_strength": 76.5},
    "ECU": {"elo_rating": 1933, "squad_strength": 78.5},
    "CUW": {"elo_rating": 1382, "squad_strength": 48.0},

    # --- Group F ---
    "NED": {"elo_rating": 1959, "squad_strength": 85.0},
    "JPN": {"elo_rating": 1879, "squad_strength": 79.5},
    "SWE": {"elo_rating": 1702, "squad_strength": 70.0},
    "TUN": {"elo_rating": 1618, "squad_strength": 64.5},

    # --- Group G ---
    "BEL": {"elo_rating": 1849, "squad_strength": 81.5},
    "EGY": {"elo_rating": 1682, "squad_strength": 70.5},
    "IRN": {"elo_rating": 1722, "squad_strength": 68.0},
    "NZL": {"elo_rating": 1378, "squad_strength": 52.0},

    # --- Group H ---
    "ESP": {"elo_rating": 2171, "squad_strength": 91.0},
    "URU": {"elo_rating": 1890, "squad_strength": 81.0},
    "KSA": {"elo_rating": 1622, "squad_strength": 62.5},
    "CPV": {"elo_rating": 1478, "squad_strength": 56.5},

    # --- Group I ---
    "FRA": {"elo_rating": 2063, "squad_strength": 90.5},
    "SEN": {"elo_rating": 1869, "squad_strength": 78.0},
    "NOR": {"elo_rating": 1922, "squad_strength": 80.5},
    "IRQ": {"elo_rating": 1498, "squad_strength": 56.0},

    # --- Group J ---
    "ARG": {"elo_rating": 2113, "squad_strength": 89.5},
    "ALG": {"elo_rating": 1698, "squad_strength": 71.0},
    "AUT": {"elo_rating": 1790, "squad_strength": 75.5},
    "JOR": {"elo_rating": 1518, "squad_strength": 54.0},

    # --- Group K ---
    "POR": {"elo_rating": 1976, "squad_strength": 86.5},
    "COL": {"elo_rating": 1998, "squad_strength": 80.0},
    "UZB": {"elo_rating": 1518, "squad_strength": 55.5},
    "COD": {"elo_rating": 1562, "squad_strength": 60.0},

    # --- Group L ---
    "ENG": {"elo_rating": 2042, "squad_strength": 87.5},
    "CRO": {"elo_rating": 1933, "squad_strength": 81.0},
    "GHA": {"elo_rating": 1638, "squad_strength": 65.0},
    "PAN": {"elo_rating": 1558, "squad_strength": 58.5},
}


# ---------------------------------------------------------------------------
# 2. GROUPS
# ---------------------------------------------------------------------------
# Official 2026 FIFA World Cup group draw. 12 groups (A-L), 4 teams each,
# 48 teams total. Team order within each list carries no significance.
GROUPS = {
    "A": ["MEX", "KOR", "RSA", "CZE"],
    "B": ["CAN", "BIH", "QAT", "SUI"],
    "C": ["BRA", "MAR", "HAI", "SCO"],
    "D": ["USA", "PAR", "AUS", "TUR"],
    "E": ["GER", "CIV", "ECU", "CUW"],
    "F": ["NED", "JPN", "SWE", "TUN"],
    "G": ["BEL", "EGY", "IRN", "NZL"],
    "H": ["ESP", "URU", "KSA", "CPV"],
    "I": ["FRA", "SEN", "NOR", "IRQ"],
    "J": ["ARG", "ALG", "AUT", "JOR"],
    "K": ["POR", "COL", "UZB", "COD"],
    "L": ["ENG", "CRO", "GHA", "PAN"],
}


# ---------------------------------------------------------------------------
# 3. COMPLETED_MATCHES
# ---------------------------------------------------------------------------
# Flat, append-only log of real group-stage results since kickoff on
# June 11, 2026. Each entry:
#
#   group       : str  - group identifier, "A" - "L"
#   home        : str  - home team code
#   away        : str  - away team code
#   home_goals  : int  - full-time goals, home team
#   away_goals  : int  - full-time goals, away team
#   date        : str  - fixture date, e.g. "June 11"
#
# Move a fixture here from REMAINING_FIXTURES once its real-world result is
# confirmed; downstream modules recompute standings and Elo from this list
# on every run, so this is the single source of truth for "what happened."
COMPLETED_MATCHES = [
    # ==========================================================================
    # REAL-WORLD DATA -- verified July 4, 2026 via cross-referenced web search
    # (Wikipedia, ESPN, CBS Sports, Yahoo Sports, FIFA.com, UEFA.com, NBC Sports).
    # The entire group stage (June 11-27, 2026) is complete in reality. All 72
    # group-stage matches below are real results, not simulated.
    #
    # Caveats, stated plainly rather than glossed over:
    #   - Sourced from news-aggregator search snippets, not one single official
    #     feed. Cross-checked final standings/qualifiers against final group
    #     tables (Group A, B, E spot-verified by hand against reported
    #     qualifiers) and all matched, but not every one of the 72 scorelines
    #     was independently re-verified goal-by-goal.
    #   - One real discrepancy was found and resolved: the Switzerland-Canada
    #     Group B finale was reported as both 3-1 and 2-1 by different Yahoo
    #     articles. Went with 2-1, corroborated by two independent sources
    #     (NBC Sports recap + a separate Yahoo daily-schedule article) against
    #     one outlier. Recorded as SUI 2-1 CAN below.
    #   - Home/away labels for a handful of matchday-3 fixtures may not exactly
    #     match which team's original fixture slot said "home" (doesn't affect
    #     standings math, which is symmetric -- flagged for data-hygiene only).
    # ==========================================================================

    # --- Group A: Mexico, South Korea, South Africa, Czechia ---
    {"group": "A", "home": "MEX", "away": "RSA", "home_goals": 2, "away_goals": 0, "date": "June 11"},
    {"group": "A", "home": "KOR", "away": "CZE", "home_goals": 2, "away_goals": 1, "date": "June 11"},
    {"group": "A", "home": "CZE", "away": "RSA", "home_goals": 1, "away_goals": 1, "date": "June 18"},
    {"group": "A", "home": "MEX", "away": "KOR", "home_goals": 1, "away_goals": 0, "date": "June 18"},
    {"group": "A", "home": "MEX", "away": "CZE", "home_goals": 3, "away_goals": 0, "date": "June 24"},
    {"group": "A", "home": "RSA", "away": "KOR", "home_goals": 1, "away_goals": 0, "date": "June 24"},

    # --- Group B: Canada, Bosnia and Herzegovina, Qatar, Switzerland ---
    {"group": "B", "home": "CAN", "away": "BIH", "home_goals": 1, "away_goals": 1, "date": "June 12"},
    {"group": "B", "home": "SUI", "away": "QAT", "home_goals": 1, "away_goals": 1, "date": "June 13"},
    {"group": "B", "home": "SUI", "away": "BIH", "home_goals": 4, "away_goals": 1, "date": "June 18"},
    {"group": "B", "home": "CAN", "away": "QAT", "home_goals": 6, "away_goals": 0, "date": "June 18"},
    {"group": "B", "home": "SUI", "away": "CAN", "home_goals": 2, "away_goals": 1, "date": "June 24"},
    {"group": "B", "home": "BIH", "away": "QAT", "home_goals": 3, "away_goals": 1, "date": "June 24"},

    # --- Group C: Brazil, Morocco, Haiti, Scotland ---
    {"group": "C", "home": "BRA", "away": "MAR", "home_goals": 1, "away_goals": 1, "date": "June 13"},
    {"group": "C", "home": "SCO", "away": "HAI", "home_goals": 1, "away_goals": 0, "date": "June 13"},
    {"group": "C", "home": "SCO", "away": "MAR", "home_goals": 0, "away_goals": 1, "date": "June 19"},
    {"group": "C", "home": "BRA", "away": "HAI", "home_goals": 3, "away_goals": 0, "date": "June 19"},
    {"group": "C", "home": "SCO", "away": "BRA", "home_goals": 0, "away_goals": 3, "date": "June 24"},
    {"group": "C", "home": "MAR", "away": "HAI", "home_goals": 4, "away_goals": 2, "date": "June 24"},

    # --- Group D: USA, Paraguay, Australia, Turkiye ---
    {"group": "D", "home": "USA", "away": "PAR", "home_goals": 4, "away_goals": 1, "date": "June 12"},
    {"group": "D", "home": "AUS", "away": "TUR", "home_goals": 2, "away_goals": 0, "date": "June 13"},
    {"group": "D", "home": "USA", "away": "AUS", "home_goals": 2, "away_goals": 0, "date": "June 19"},
    {"group": "D", "home": "TUR", "away": "PAR", "home_goals": 0, "away_goals": 1, "date": "June 19"},
    {"group": "D", "home": "TUR", "away": "USA", "home_goals": 3, "away_goals": 2, "date": "June 25"},
    {"group": "D", "home": "PAR", "away": "AUS", "home_goals": 0, "away_goals": 0, "date": "June 25"},

    # --- Group E: Germany, Ivory Coast, Ecuador, Curacao ---
    {"group": "E", "home": "GER", "away": "CUW", "home_goals": 7, "away_goals": 1, "date": "June 14"},
    {"group": "E", "home": "CIV", "away": "ECU", "home_goals": 1, "away_goals": 0, "date": "June 14"},
    {"group": "E", "home": "GER", "away": "CIV", "home_goals": 2, "away_goals": 1, "date": "June 20"},
    {"group": "E", "home": "ECU", "away": "CUW", "home_goals": 0, "away_goals": 0, "date": "June 20"},
    {"group": "E", "home": "ECU", "away": "GER", "home_goals": 2, "away_goals": 1, "date": "June 25"},
    {"group": "E", "home": "CUW", "away": "CIV", "home_goals": 0, "away_goals": 2, "date": "June 25"},

    # --- Group F: Netherlands, Japan, Sweden, Tunisia ---
    {"group": "F", "home": "NED", "away": "JPN", "home_goals": 2, "away_goals": 2, "date": "June 14"},
    {"group": "F", "home": "SWE", "away": "TUN", "home_goals": 5, "away_goals": 1, "date": "June 14"},
    {"group": "F", "home": "NED", "away": "SWE", "home_goals": 5, "away_goals": 1, "date": "June 20"},
    {"group": "F", "home": "JPN", "away": "TUN", "home_goals": 4, "away_goals": 0, "date": "June 20"},
    {"group": "F", "home": "JPN", "away": "SWE", "home_goals": 1, "away_goals": 1, "date": "June 25"},
    {"group": "F", "home": "NED", "away": "TUN", "home_goals": 3, "away_goals": 1, "date": "June 25"},

    # --- Group G: Belgium, Egypt, Iran, New Zealand ---
    {"group": "G", "home": "BEL", "away": "EGY", "home_goals": 1, "away_goals": 1, "date": "June 15"},
    {"group": "G", "home": "IRN", "away": "NZL", "home_goals": 2, "away_goals": 2, "date": "June 15"},
    {"group": "G", "home": "BEL", "away": "IRN", "home_goals": 0, "away_goals": 0, "date": "June 21"},
    {"group": "G", "home": "EGY", "away": "NZL", "home_goals": 3, "away_goals": 1, "date": "June 21"},
    {"group": "G", "home": "EGY", "away": "IRN", "home_goals": 1, "away_goals": 1, "date": "June 26"},
    {"group": "G", "home": "BEL", "away": "NZL", "home_goals": 5, "away_goals": 1, "date": "June 26"},

    # --- Group H: Spain, Uruguay, Saudi Arabia, Cabo Verde ---
    {"group": "H", "home": "ESP", "away": "CPV", "home_goals": 0, "away_goals": 0, "date": "June 15"},
    {"group": "H", "home": "KSA", "away": "URU", "home_goals": 1, "away_goals": 1, "date": "June 15"},
    {"group": "H", "home": "ESP", "away": "KSA", "home_goals": 4, "away_goals": 0, "date": "June 21"},
    {"group": "H", "home": "URU", "away": "CPV", "home_goals": 2, "away_goals": 2, "date": "June 21"},
    {"group": "H", "home": "CPV", "away": "KSA", "home_goals": 0, "away_goals": 0, "date": "June 26"},
    {"group": "H", "home": "ESP", "away": "URU", "home_goals": 1, "away_goals": 0, "date": "June 26"},

    # --- Group I: France, Senegal, Norway, Iraq ---
    {"group": "I", "home": "FRA", "away": "SEN", "home_goals": 3, "away_goals": 1, "date": "June 16"},
    {"group": "I", "home": "NOR", "away": "IRQ", "home_goals": 4, "away_goals": 1, "date": "June 16"},
    {"group": "I", "home": "FRA", "away": "IRQ", "home_goals": 3, "away_goals": 0, "date": "June 22"},
    {"group": "I", "home": "NOR", "away": "SEN", "home_goals": 3, "away_goals": 2, "date": "June 22"},
    {"group": "I", "home": "FRA", "away": "NOR", "home_goals": 4, "away_goals": 1, "date": "June 26"},
    {"group": "I", "home": "SEN", "away": "IRQ", "home_goals": 5, "away_goals": 0, "date": "June 26"},

    # --- Group J: Argentina, Algeria, Austria, Jordan ---
    {"group": "J", "home": "ARG", "away": "ALG", "home_goals": 3, "away_goals": 0, "date": "June 16"},
    {"group": "J", "home": "AUT", "away": "JOR", "home_goals": 3, "away_goals": 1, "date": "June 17"},
    {"group": "J", "home": "ARG", "away": "AUT", "home_goals": 2, "away_goals": 0, "date": "June 22"},
    {"group": "J", "home": "JOR", "away": "ALG", "home_goals": 1, "away_goals": 2, "date": "June 22"},
    {"group": "J", "home": "ARG", "away": "JOR", "home_goals": 3, "away_goals": 1, "date": "June 27"},
    {"group": "J", "home": "AUT", "away": "ALG", "home_goals": 3, "away_goals": 3, "date": "June 27"},

    # --- Group K: Portugal, Colombia, Uzbekistan, DR Congo ---
    {"group": "K", "home": "POR", "away": "COD", "home_goals": 1, "away_goals": 1, "date": "June 17"},
    {"group": "K", "home": "UZB", "away": "COL", "home_goals": 1, "away_goals": 3, "date": "June 17"},
    {"group": "K", "home": "POR", "away": "UZB", "home_goals": 5, "away_goals": 0, "date": "June 23"},
    {"group": "K", "home": "COL", "away": "COD", "home_goals": 1, "away_goals": 0, "date": "June 23"},
    {"group": "K", "home": "COL", "away": "POR", "home_goals": 0, "away_goals": 0, "date": "June 27"},
    {"group": "K", "home": "COD", "away": "UZB", "home_goals": 3, "away_goals": 1, "date": "June 27"},

    # --- Group L: England, Croatia, Ghana, Panama ---
    {"group": "L", "home": "ENG", "away": "CRO", "home_goals": 4, "away_goals": 2, "date": "June 17"},
    {"group": "L", "home": "GHA", "away": "PAN", "home_goals": 1, "away_goals": 0, "date": "June 17"},
    {"group": "L", "home": "ENG", "away": "GHA", "home_goals": 0, "away_goals": 0, "date": "June 23"},
    {"group": "L", "home": "PAN", "away": "CRO", "home_goals": 0, "away_goals": 1, "date": "June 23"},
    {"group": "L", "home": "ENG", "away": "PAN", "home_goals": 2, "away_goals": 0, "date": "June 27"},
    {"group": "L", "home": "CRO", "away": "GHA", "home_goals": 2, "away_goals": 1, "date": "June 27"},
]

# ---------------------------------------------------------------------------
# 3b. ROUND_OF_32_RESULTS (real, completed June 28 - July 3, 2026)
# ---------------------------------------------------------------------------
# The Round of 32 has also been played out in reality. This is a plain
# record of what happened (for reference/audit), separate from the group
# match list since it isn't consumed by compute_group_standings. What the
# simulator actually uses going forward is REAL_ROUND_OF_16_FIELD below.
ROUND_OF_32_RESULTS = [
    {"winner": "CAN", "loser": "RSA", "score": "1-0", "date": "June 28"},
    {"winner": "BRA", "loser": "JPN", "score": "2-1", "date": "June 29"},
    {"winner": "PAR", "loser": "GER", "score": "1-1 (4-3 pens)", "date": "June 29"},
    {"winner": "MAR", "loser": "NED", "score": "1-1 (3-2 pens)", "date": "June 29"},
    {"winner": "NOR", "loser": "CIV", "score": "2-1", "date": "June 30"},
    {"winner": "FRA", "loser": "SWE", "score": "3-0", "date": "June 30"},
    {"winner": "MEX", "loser": "ECU", "score": "2-0", "date": "June 30"},
    {"winner": "ENG", "loser": "COD", "score": "2-1", "date": "July 1"},
    {"winner": "BEL", "loser": "SEN", "score": "3-2 (a.e.t.)", "date": "July 1"},
    {"winner": "USA", "loser": "BIH", "score": "2-0", "date": "July 1"},
    {"winner": "ESP", "loser": "AUT", "score": "3-0", "date": "July 2"},
    {"winner": "POR", "loser": "CRO", "score": "2-1", "date": "July 2"},
    {"winner": "SUI", "loser": "ALG", "score": "2-0", "date": "July 2"},
    {"winner": "EGY", "loser": "AUS", "score": "1-1 (4-2 pens)", "date": "July 3"},
    {"winner": "ARG", "loser": "CPV", "score": "3-2 (a.e.t.)", "date": "July 3"},
    {"winner": "COL", "loser": "GHA", "score": "1-0", "date": "July 3"},
]

# ---------------------------------------------------------------------------
# 3c. ROUND_OF_16_RESULTS (real, completed July 4-7, 2026)
# ---------------------------------------------------------------------------
# The Round of 16 has ALSO been played out in full as of this update (checked
# July 8, 2026). Cross-verified against FIFA.com match reports, ESPN, Sky
# Sports, Al Jazeera and AP -- all consistent on every score below, including
# exact goal-scorers for the Mexico-England match where two initial sources
# briefly disagreed (2-1 vs 3-2; five independent detailed match reports
# confirm 3-2 to England).
ROUND_OF_16_RESULTS = [
    {"winner": "MAR", "loser": "CAN", "score": "3-0", "date": "July 4"},
    {"winner": "FRA", "loser": "PAR", "score": "1-0", "date": "July 4"},
    {"winner": "NOR", "loser": "BRA", "score": "2-1", "date": "July 5"},
    {"winner": "ENG", "loser": "MEX", "score": "3-2", "date": "July 5"},
    {"winner": "ESP", "loser": "POR", "score": "1-0", "date": "July 6"},
    {"winner": "BEL", "loser": "USA", "score": "4-1", "date": "July 6"},
    {"winner": "ARG", "loser": "EGY", "score": "3-2", "date": "July 7"},
    {"winner": "SUI", "loser": "COL", "score": "0-0 (4-3 pens)", "date": "July 7"},
]

# ---------------------------------------------------------------------------
# 3d. REAL_QUARTERFINAL_FIELD -- COMPLETE as of this update (was "unplayed"
#     in the July 8 version of this file; corrected July 13, 2026)
# ---------------------------------------------------------------------------
# Kept as a historical record (used by the dashboard's bracket view) of
# which 8 teams reached the Quarterfinals. All four QF matches have now
# been played in reality -- see QUARTERFINAL_RESULTS below for scores.
REAL_QUARTERFINAL_FIELD = [
    "FRA", "MAR",   # July 9, Foxborough (Boston)
    "ESP", "BEL",   # July 10, Inglewood (LA)
    "NOR", "ENG",   # July 11, Miami Gardens
    "ARG", "SUI",   # July 11, Kansas City
]

QUARTERFINAL_RESULTS = [
    {"winner": "FRA", "loser": "MAR", "score": "2-0", "date": "July 9"},
    {"winner": "ESP", "loser": "BEL", "score": "2-1", "date": "July 10"},
    {"winner": "ENG", "loser": "NOR", "score": "2-1", "date": "July 11"},
    {"winner": "ARG", "loser": "SUI", "score": "3-1 (a.e.t.)", "date": "July 11"},
]

# ---------------------------------------------------------------------------
# 3e. SEMIFINAL_FIELD -- the ACTUAL remaining uncertainty (July 13, 2026)
# ---------------------------------------------------------------------------
# As of today, only four teams remain: France, Spain, England, Argentina.
# This is what the simulator now actually projects: two semifinals, a
# third-place match, and the Final. Pairing order matches
# world_cup_sim.py's _simulate_knockout_round (index 0 vs 1, 2 vs 3).
#   Semifinal 1: France vs Spain      -- July 14, Dallas (AT&T Stadium)
#   Semifinal 2: England vs Argentina -- July 15, Atlanta
SEMIFINAL_FIELD = ["FRA", "ESP", "ENG", "ARG"]


# ---------------------------------------------------------------------------
# 4. REMAINING_FIXTURES
# ---------------------------------------------------------------------------
# Unplayed group-stage matches awaiting simulation. Each entry:
#
#   group : str - group identifier, "A" - "L"
#   home  : str - home team code
#   away  : str - away team code
#   date  : str - scheduled fixture date
#
# Includes the last few Matchday-1 fixtures (Groups K-L) whose real-world
# results had not yet been confirmed as of this file's last update -- this
# includes England vs Croatia, which was in progress at time of writing --
# plus the full Matchday 2 / Matchday 3 schedule for every group (June 18 -
# June 27). Move an entry to COMPLETED_MATCHES with final goals once it has
# actually been played.
REMAINING_FIXTURES = [
    # The group stage is complete in reality (last match played June 27, 2026).
    # No group-stage fixtures remain -- this list is intentionally empty.
    # The tournament is now in the knockout stage; see REAL_ROUND_OF_16_FIELD
    # above for what the simulator actually projects going forward.
]


# ---------------------------------------------------------------------------
# 5. PLAYER_METRICS
# ---------------------------------------------------------------------------
# Real tournament goal/assist tallies, verified July 13, 2026 (FIFA.com,
# ESPN, Sky Sports, Goal.com, FOX Sports Golden Boot trackers, Al Jazeera,
# AP, worldcupwiki.com -- cross-checked, not single-sourced).
#
# HONESTY NOTES, stated plainly rather than glossed over:
#   - tournament_goals / tournament_assists are REAL. For eliminated
#     players these totals are FINAL and cannot change further.
#   - This update folds in the Quarterfinals (July 9-11), which the prior
#     (July 8) version of this file predated. Two things changed:
#       (1) Only France, Spain, England, and Argentina remain alive --
#           Morocco, Belgium, Norway, and Switzerland were all eliminated
#           in the QFs, so their players' team_alive flips to False and
#           their goal tallies lock in.
#       (2) Some leaders' tallies grew in the QFs: Mbappe scored against
#           Morocco (FIFA.com: his "eighth goal of the tournament," in the
#           QF), and Dembele scored the decisive goal in that same 2-0 win
#           (Goal.com, putting him on 5). Bellingham scored a brace against
#           Norway in England's QF win (multiple sources), moving him up.
#           NOT every player's exact QF involvement was individually
#           re-verified goal-by-goal -- only the headline scorers reported
#           across multiple outlets were updated with confidence. Numbers
#           for less-prominent alive-team players (e.g. Oyarzabal) are
#           carried over from the July 8 snapshot and may be stale by a
#           goal or two if they scored in the QF without it being a
#           headline in the sources checked.
#   - goals_per_game / assists_per_game are a rough per-match RATE used
#     ONLY to drive Monte Carlo goal attribution for players on the four
#     teams still alive (France, Spain, England, Argentina). Approximate,
#     flagged as such, not presented as precise.
#   - pott_rating is a constructed heuristic (base 5.0 + weighted
#     goals/assists), NOT fitted to real Ballon d'Or/Golden Ball voting
#     data. It's a tiebreak/display proxy, not a calibrated prediction.
#   - "team_alive" marks whether the player's team remains in the
#     tournament as of this update (semifinals: France, Spain, England,
#     Argentina only).
PLAYER_METRICS = {
    # --- Still alive (Semifinals: France, Spain, England, Argentina) ---
    "Lionel Messi":        {"team": "ARG", "goals_per_game": 1.6, "assists_per_game": 0.2, "pott_rating": 9.6, "tournament_goals": 8, "tournament_assists": 1, "team_alive": True},
    "Kylian Mbappe":       {"team": "FRA", "goals_per_game": 1.4, "assists_per_game": 0.4, "pott_rating": 9.4, "tournament_goals": 8, "tournament_assists": 2, "team_alive": True},
    "Harry Kane":          {"team": "ENG", "goals_per_game": 1.2, "assists_per_game": 0.2, "pott_rating": 8.4, "tournament_goals": 6, "tournament_assists": 1, "team_alive": True},
    "Jude Bellingham":     {"team": "ENG", "goals_per_game": 0.8, "assists_per_game": 0.0, "pott_rating": 8.3, "tournament_goals": 6, "tournament_assists": 0, "team_alive": True},
    "Ousmane Dembele":     {"team": "FRA", "goals_per_game": 0.8, "assists_per_game": 0.0, "pott_rating": 7.9, "tournament_goals": 5, "tournament_assists": 0, "team_alive": True},
    "Mikel Oyarzabal":     {"team": "ESP", "goals_per_game": 0.8, "assists_per_game": 0.0, "pott_rating": 7.6, "tournament_goals": 4, "tournament_assists": 0, "team_alive": True},
    "Alexis Mac Allister": {"team": "ARG", "goals_per_game": 0.4, "assists_per_game": 0.1, "pott_rating": 7.3, "tournament_goals": 2, "tournament_assists": 1, "team_alive": True},
    "Julian Alvarez":      {"team": "ARG", "goals_per_game": 0.4, "assists_per_game": 0.1, "pott_rating": 7.2, "tournament_goals": 2, "tournament_assists": 0, "team_alive": True},
    "Lamine Yamal":        {"team": "ESP", "goals_per_game": 0.4, "assists_per_game": 0.3, "pott_rating": 7.5, "tournament_goals": 2, "tournament_assists": 3, "team_alive": True},

    # --- Eliminated (Quarterfinals or earlier): totals FINAL, locked ---
    "Erling Haaland":      {"team": "NOR", "goals_per_game": 0.0, "assists_per_game": 0.0, "pott_rating": 8.7, "tournament_goals": 7, "tournament_assists": 0, "team_alive": False},
    "Ismael Saibari":      {"team": "MAR", "goals_per_game": 0.0, "assists_per_game": 0.0, "pott_rating": 7.1, "tournament_goals": 3, "tournament_assists": 0, "team_alive": False},
    "Johan Manzambi":      {"team": "SUI", "goals_per_game": 0.0, "assists_per_game": 0.0, "pott_rating": 7.1, "tournament_goals": 3, "tournament_assists": 0, "team_alive": False},
    "Romelu Lukaku":       {"team": "BEL", "goals_per_game": 0.0, "assists_per_game": 0.0, "pott_rating": 7.0, "tournament_goals": 3, "tournament_assists": 0, "team_alive": False},
    "Vinicius Junior":     {"team": "BRA", "goals_per_game": 0.0, "assists_per_game": 0.0, "pott_rating": 7.6, "tournament_goals": 4, "tournament_assists": 0, "team_alive": False},
    "Ismaila Sarr":        {"team": "SEN", "goals_per_game": 0.0, "assists_per_game": 0.0, "pott_rating": 7.6, "tournament_goals": 4, "tournament_assists": 0, "team_alive": False},
    "Julian Quinones":     {"team": "MEX", "goals_per_game": 0.0, "assists_per_game": 0.0, "pott_rating": 7.6, "tournament_goals": 4, "tournament_assists": 0, "team_alive": False},
    "Deniz Undav":         {"team": "GER", "goals_per_game": 0.0, "assists_per_game": 0.0, "pott_rating": 7.1, "tournament_goals": 3, "tournament_assists": 0, "team_alive": False},
    "Kai Havertz":         {"team": "GER", "goals_per_game": 0.0, "assists_per_game": 0.0, "pott_rating": 7.1, "tournament_goals": 3, "tournament_assists": 0, "team_alive": False},
    "Cody Gakpo":          {"team": "NED", "goals_per_game": 0.0, "assists_per_game": 0.0, "pott_rating": 7.1, "tournament_goals": 3, "tournament_assists": 0, "team_alive": False},
    "Brian Brobbey":       {"team": "NED", "goals_per_game": 0.0, "assists_per_game": 0.0, "pott_rating": 7.1, "tournament_goals": 3, "tournament_assists": 0, "team_alive": False},
    "Folarin Balogun":     {"team": "USA", "goals_per_game": 0.0, "assists_per_game": 0.0, "pott_rating": 7.1, "tournament_goals": 3, "tournament_assists": 0, "team_alive": False},
    "Matheus Cunha":       {"team": "BRA", "goals_per_game": 0.0, "assists_per_game": 0.0, "pott_rating": 7.1, "tournament_goals": 3, "tournament_assists": 0, "team_alive": False},
    "Jonathan David":      {"team": "CAN", "goals_per_game": 0.0, "assists_per_game": 0.0, "pott_rating": 7.1, "tournament_goals": 3, "tournament_assists": 0, "team_alive": False},
    "Cristiano Ronaldo":   {"team": "POR", "goals_per_game": 0.0, "assists_per_game": 0.0, "pott_rating": 7.1, "tournament_goals": 3, "tournament_assists": 0, "team_alive": False},
    "Yoane Wissa":         {"team": "COD", "goals_per_game": 0.0, "assists_per_game": 0.0, "pott_rating": 7.1, "tournament_goals": 3, "tournament_assists": 0, "team_alive": False},
    "Raul Jimenez":        {"team": "MEX", "goals_per_game": 0.0, "assists_per_game": 0.0, "pott_rating": 7.1, "tournament_goals": 3, "tournament_assists": 0, "team_alive": False},
}

# Teams still alive in the tournament as of this update (Semifinals stage).
# Used by the sim to decide which teams' unattributed-goal buckets can
# still grow, and which players can still add to their tallies.
TEAMS_STILL_ALIVE = ["FRA", "ESP", "ENG", "ARG"]

# ---------------------------------------------------------------------------
# Self-validation
# ---------------------------------------------------------------------------
# Run this file directly (`python world_cup_data.py`) after any manual edit
# to catch typos in team codes, duplicate teams, or malformed group sizes
# before downstream modules import this data.
if __name__ == "__main__":
    errors = []

    if len(TEAM_METRICS) != TOTAL_TEAMS:
        errors.append(f"TEAM_METRICS has {len(TEAM_METRICS)} teams, expected {TOTAL_TEAMS}.")

    if len(GROUPS) != TOTAL_GROUPS:
        errors.append(f"GROUPS has {len(GROUPS)} groups, expected {TOTAL_GROUPS}.")

    all_group_teams = []
    for group_id, teams in GROUPS.items():
        if len(teams) != TEAMS_PER_GROUP:
            errors.append(f"Group {group_id} has {len(teams)} teams, expected {TEAMS_PER_GROUP}.")
        all_group_teams.extend(teams)

    duplicate_teams = sorted({t for t in all_group_teams if all_group_teams.count(t) > 1})
    if duplicate_teams:
        errors.append(f"Teams appearing in more than one group: {duplicate_teams}")

    missing_metrics = sorted(set(all_group_teams) - set(TEAM_METRICS))
    if missing_metrics:
        errors.append(f"Teams in GROUPS but missing from TEAM_METRICS: {missing_metrics}")

    orphan_metrics = sorted(set(TEAM_METRICS) - set(all_group_teams))
    if orphan_metrics:
        errors.append(f"Teams in TEAM_METRICS but not placed in any GROUP: {orphan_metrics}")

    for match_list, label in (
        (COMPLETED_MATCHES, "COMPLETED_MATCHES"),
        (REMAINING_FIXTURES, "REMAINING_FIXTURES"),
    ):
        for m in match_list:
            for side in ("home", "away"):
                if m[side] not in TEAM_METRICS:
                    errors.append(f"{label} references unknown team code '{m[side]}' ({m}).")
            if m["group"] not in GROUPS:
                errors.append(f"{label} references unknown group '{m['group']}' ({m}).")

    if errors:
        print(f"world_cup_data.py: {len(errors)} integrity issue(s) found:\n")
        for e in errors:
            print(f"  - {e}")
    else:
        played = len(COMPLETED_MATCHES)
        pending = len(REMAINING_FIXTURES)
        print("world_cup_data.py: integrity check passed.")
        print(f"  Teams: {len(TEAM_METRICS)} | Groups: {len(GROUPS)}")
        print(f"  Group-stage matches completed: {played} | remaining: {pending} | total: {played + pending}")
