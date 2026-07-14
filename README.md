# FIFA World Cup 2026 Predictive Engine
> **Status: Working prototype, actively evolving.** Core simulation engine and CLI
> are fully functional and verified against real tournament data. The live-data
> scraper and dashboard visuals are built but not yet production-hardened — see
> [Known Limitations](#known-limitations) for the specific, honest list of what's
> left. This is a portfolio project I'm iterating on, not a finished product.

A zero-API, self-contained Monte Carlo simulator that predicts FIFA World Cup 2026
outcomes — champion odds, medal odds, group standings, knockout bracket progression,
Golden Boot, and Player of the Tournament — using a Bivariate Poisson match model
driven by Elo ratings, squad strength, and host-nation advantage.

Built to run entirely offline on hardcoded, hand-verified real-world data, with an
optional on-demand scraper for fresher inputs. No paid APIs, no API keys, no external
services required.

![Dashboard Screenshot](https://docs.google.com/document/d/1jZYMK3jwLwOTpRWYzQlK1TX_2CM9VdS849z5bK7yL4g/edit?usp=sharing)

## Features

- **Bivariate Poisson match engine** — simulates individual matches using Elo
  difference and squad-strength inputs, with random volatility and host-nation Elo
  bonus for group-stage fixtures on home soil.
- **Elo rating updates** — post-match rating adjustments using the standard chess
  K-factor formula (K=32 group stage, K=40 knockout).
- **Full group-stage tiebreaker resolution** — points, goal difference, goals for,
  Elo rating (head-to-head placeholder noted as a known simplification).
- **Monte Carlo tournament simulation** — thousands of independent full-tournament
  runs to produce probabilistic champion/medal/knockout-stage odds.
- **Golden Boot & Player of the Tournament prediction** — per-goal player attribution
  weighted by tracked players' real scoring rates, plus a transparent, documented
  POTT heuristic (`4×goals + 2×assists + stage_bonus`).
- **Interactive Streamlit dashboard** — 5 tabs: Champion Odds, Group Stage, Knockout
  Bracket, Player Predictions, Live Standings.
- **Optional live-data scraper** — pulls fresh Elo ratings (eloratings.net), match
  results (Wikipedia), and player stats (FBRef) on demand, with a clean fallback to
  hardcoded data if scraping fails or is stale.

## Architecture

```
world_cup_data.py       # Pure data layer: teams, groups, match results, player stats
world_cup_engine.py      # Match simulation, Elo updates, tiebreaker logic
world_cup_sim.py         # Monte Carlo tournament orchestration + player-award scoring
main.py                  # CLI entry point
world_cup_dashboard.py   # Streamlit visual dashboard
world_cup_scraper.py     # Optional live-data refresh (no API key required)
```

Each module has a single responsibility and no circular dependencies. `world_cup_data.py`
is the single source of truth — updating it after real matches is all that's needed to
keep every downstream module (CLI and dashboard) in sync.

## Installation

```bash
git clone https://github.com/<your-username>/world-cup-2026-predictor.git
cd world-cup-2026-predictor
pip install -r requirements.txt
```

## Usage

**Run the CLI predictor:**
```bash
python main.py --quick        # 1,000 simulations, fast
python main.py                # 10,000 simulations, default
python main.py --standings    # Just show current group standings, no simulation
python main.py --check        # Validate data integrity only
```

**Launch the dashboard:**
```bash
streamlit run world_cup_dashboard.py
```

**Refresh live data (optional, run on your own machine):**
```bash
python world_cup_scraper.py
```

## Data Honesty & Design Philosophy

This project treats data integrity as a first-class concern, not an afterthought:

- **All match results are real**, cross-verified against multiple independent sources
  (FIFA.com, ESPN, Sky Sports, Wikipedia, AP), not simulated placeholders.
- **No invented data.** Where exact figures aren't available (e.g. per-player scoring
  rates used to drive Monte Carlo attribution), values are explicitly labeled as
  estimates/heuristics in code comments — never presented as verified fact.
- **Only ~30 marquee players are individually tracked** for Golden Boot/POTT
  attribution, not full 48-team squads. Goals by untracked players are deliberately
  left unattributed rather than forced onto a tracked name, to avoid inflating a
  handful of players' odds with goals that were never really theirs.
- **The scraper fails loud, not silent-wrong.** If a target site changes its HTML
  structure, `world_cup_scraper.py` prints a clear warning and returns an empty
  result — callers fall back cleanly to the hardcoded dataset rather than risking a
  silently incorrect number.

## Known Limitations

- Group-stage tiebreaker logic has a placeholder for head-to-head record (not yet
  implemented; falls through to goal difference).
- `world_cup_scraper.py` has been reviewed and unit-tested for its fallback path, but
  has not yet been run against live HTML from a network that can reach
  eloratings.net / Wikipedia / FBRef — selectors may need adjustment on first real run.
- Player tracking covers marquee names only; full-squad stats are out of scope by
  design (see Data Honesty above).
- `world_cup_scraper.py` output (`live_updates.json`) is not yet wired into the
  simulation inputs — it's fetched and saved but not consumed by `main.py` /
  `world_cup_sim.py` yet.

## Tech Stack

Python 3.14, NumPy (Poisson sampling), Streamlit + Plotly (dashboard), no external
paid APIs.

## License

MIT
