"""
world_cup_dashboard.py

FIFA World Cup 2026 Predictive Engine -- Streamlit Dashboard
==============================================================

Run with:
    streamlit run world_cup_dashboard.py

Five tabs: Champion Odds, Group Stage, Knockout Bracket, Players,
Live Standings. Reads the same world_cup_data.py / world_cup_sim.py /
world_cup_engine.py modules as main.py -- no separate data source, so
anything true of the CLI output is true here too.
"""

import os
import random

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from world_cup_data import (
    GROUPS, GROUP_IDS, TEAM_METRICS, COMPLETED_MATCHES, REMAINING_FIXTURES,
    PLAYER_METRICS, REAL_QUARTERFINAL_FIELD, ROUND_OF_32_RESULTS,
    ROUND_OF_16_RESULTS,
)
from world_cup_engine import compute_group_standings
from world_cup_sim import MonteCarloSimulator


# ---------------------------------------------------------------------------
# Page config + theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="World Cup 2026 Predictor",
    page_icon="\U0001F3C6",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_BG = "#0b1120"
PANEL_BG = "#141c2f"
GOLD = "#d4af37"
AMBER = "#f5a623"
BLUE_ACCENT = "#3b82f6"
TEXT_MUTED = "#9aa5b8"

st.markdown(f"""
<style>
    .stApp {{
        background: linear-gradient(160deg, {DARK_BG} 0%, #0d1424 100%);
        color: #e8ecf4;
    }}
    section[data-testid="stSidebar"] {{
        background: {PANEL_BG};
        border-right: 1px solid rgba(212,175,55,0.15);
    }}
    .glass-card {{
        background: rgba(20, 28, 47, 0.65);
        border: 1px solid rgba(212,175,55,0.18);
        border-radius: 14px;
        padding: 1.1rem 1.4rem;
        backdrop-filter: blur(6px);
        margin-bottom: 0.8rem;
    }}
    h1, h2, h3 {{
        font-family: 'Georgia', serif;
        color: {GOLD} !important;
        letter-spacing: 0.3px;
    }}
    .metric-label {{ color: {TEXT_MUTED}; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; }}
    .stTabs [data-baseweb="tab"] {{ font-size: 1.0rem; padding: 10px 18px; }}
    .stTabs [aria-selected="true"] {{ color: {GOLD} !important; border-bottom-color: {GOLD} !important; }}
    .caveat {{ color: {TEXT_MUTED}; font-size: 0.82rem; font-style: italic; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached simulation
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_simulation(num_sims: int, seed: int):
    random.seed(seed)
    sim = MonteCarloSimulator(num_simulations=num_sims)
    sim.run()
    return sim.statistics


def load_live_data_status():
    """Mirrors main.py's check -- reports whether live_updates.json exists,
    without pretending the dashboard consumes it (it doesn't yet)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_updates.json")
    return os.path.exists(path)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown(f"## \U0001F3C6 World Cup 2026")
st.sidebar.markdown("**Prediction Dashboard**")
st.sidebar.markdown("---")
num_sims = st.sidebar.slider("Monte Carlo simulations", 200, 5000, 1000, step=200,
                              help="More = smoother probabilities, slower to compute.")
seed = st.sidebar.number_input("Random seed", value=42, step=1,
                                help="Same seed + same sim count = reproducible run.")

if load_live_data_status():
    st.sidebar.success("live_updates.json found (not yet wired into the sim).")
else:
    st.sidebar.info("Using hardcoded, hand-verified real data (no live_updates.json).")

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<span class="caveat">Data verified July 8, 2026 via cross-referenced '
    'web search. Group stage, Round of 32, and Round of 16 are real, '
    'completed results. Quarterfinals onward are Monte Carlo projections, '
    'not real outcomes.</span>',
    unsafe_allow_html=True,
)

with st.spinner("Running tournament simulations..."):
    stats = run_simulation(num_sims, seed)

st.title("FIFA World Cup 2026 -- Prediction Dashboard")
st.markdown(
    f'<span class="caveat">Projections from the Quarterfinals onward, based on {num_sims:,} '
    f'Monte Carlo simulations of Bivariate Poisson matches. Everything before the '
    f'Quarterfinals below is real, not simulated.</span>',
    unsafe_allow_html=True,
)

tab_champion, tab_groups, tab_bracket, tab_players, tab_live = st.tabs(
    ["\U0001F3C6 Champion Odds", "\U0001F4CA Group Stage", "\U0001F333 Knockout Bracket",
     "\u26BD Player Predictions", "\U0001F4CD Live Standings"]
)


# ---------------------------------------------------------------------------
# Tab 1: Champion Odds
# ---------------------------------------------------------------------------
with tab_champion:
    st.subheader("Tournament Winner Probability")
    win_probs = sorted(stats["win_probability"].items(), key=lambda x: x[1], reverse=True)
    df_win = pd.DataFrame(win_probs, columns=["Team", "Probability"])
    df_win["Probability (%)"] = df_win["Probability"] * 100
    df_win["Decimal Odds"] = df_win["Probability"].apply(lambda p: f"{1/p:.2f}" if p > 0 else "N/A")

    fig = px.bar(
        df_win, x="Probability (%)", y="Team", orientation="h",
        color="Probability (%)", color_continuous_scale=[BLUE_ACCENT, GOLD],
        text=df_win["Probability (%)"].apply(lambda v: f"{v:.1f}%"),
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#e8ecf4", yaxis={"categoryorder": "total ascending"},
        height=520, showlegend=False, coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig, width='stretch')

    st.markdown("#### Medal Odds (Top 5 each)")
    c1, c2, c3 = st.columns(3)
    for col, key, label, emoji in [
        (c1, "win_probability", "Gold (Champion)", "\U0001F947"),
        (c2, "runner_up_probability", "Silver (Runner-up)", "\U0001F948"),
        (c3, "third_place_probability", "Bronze (3rd Place)", "\U0001F949"),
    ]:
        with col:
            st.markdown(f"**{emoji} {label}**")
            top5 = sorted(stats[key].items(), key=lambda x: x[1], reverse=True)[:5]
            for team, prob in top5:
                st.markdown(f'<div class="glass-card">{team} &nbsp; <b style="color:{GOLD}">{prob:.1%}</b></div>',
                            unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 2: Group Stage
# ---------------------------------------------------------------------------
with tab_groups:
    st.subheader("Final Group Standings (real, completed results)")
    st.markdown('<span class="caveat">The group stage is fully complete in reality -- '
                'this is a readout, not a prediction.</span>', unsafe_allow_html=True)
    cols = st.columns(3)
    for idx, group_id in enumerate(GROUP_IDS):
        standings = compute_group_standings(group_id, TEAM_METRICS)
        rows = []
        for pos, (team, s) in enumerate(standings, start=1):
            qualified = pos <= 2
            rows.append({
                "Pos": pos, "Team": team, "Pts": s["points"],
                "W": s["wins"], "D": s["draws"], "L": s["losses"],
                "GF": s["goals_for"], "GA": s["goals_against"], "GD": s["goal_difference"],
                "Qualified": "\u2705" if qualified else "",
            })
        df = pd.DataFrame(rows)
        with cols[idx % 3]:
            st.markdown(f"**Group {group_id}**")
            st.dataframe(df, hide_index=True, width='stretch', height=175)


# ---------------------------------------------------------------------------
# Tab 3: Knockout Bracket
# ---------------------------------------------------------------------------
with tab_bracket:
    st.subheader("Path to the Final")
    st.markdown('<span class="caveat">Round of 32 and Round of 16 are real, completed '
                'results. Quarterfinals onward are simulated.</span>', unsafe_allow_html=True)

    st.markdown("##### Round of 32 (real results)")
    r32_df = pd.DataFrame(ROUND_OF_32_RESULTS)
    st.dataframe(r32_df, hide_index=True, width='stretch')

    st.markdown("##### Round of 16 (real results)")
    r16_df = pd.DataFrame(ROUND_OF_16_RESULTS)
    st.dataframe(r16_df, hide_index=True, width='stretch')

    st.markdown("##### Quarterfinals (simulated from here)")
    qf_pairs = [(REAL_QUARTERFINAL_FIELD[i], REAL_QUARTERFINAL_FIELD[i + 1])
                for i in range(0, len(REAL_QUARTERFINAL_FIELD), 2)]
    qf_cols = st.columns(4)
    for i, (a, b) in enumerate(qf_pairs):
        with qf_cols[i]:
            a_prob = stats["knockout_appearance"]["Semifinals"].get(a, 0)
            b_prob = stats["knockout_appearance"]["Semifinals"].get(b, 0)
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;">
                <b>{a}</b> vs <b>{b}</b><br>
                <span style="color:{AMBER}">{a}: {a_prob:.0%} to advance</span><br>
                <span style="color:{AMBER}">{b}: {b_prob:.0%} to advance</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("##### Semifinal & Final appearance probability")
    stage_choice = st.radio("Stage", ["Semifinals", "Final"], horizontal=True)
    stage_probs = sorted(stats["knockout_appearance"][stage_choice].items(), key=lambda x: x[1], reverse=True)
    df_stage = pd.DataFrame(stage_probs, columns=["Team", "Probability"])
    df_stage["Probability (%)"] = df_stage["Probability"] * 100
    fig2 = px.bar(df_stage, x="Team", y="Probability (%)", color="Probability (%)",
                  color_continuous_scale=[BLUE_ACCENT, GOLD])
    fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                        font_color="#e8ecf4", height=400, showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(fig2, width='stretch')


# ---------------------------------------------------------------------------
# Tab 4: Player Predictions
# ---------------------------------------------------------------------------
with tab_players:
    st.subheader("Golden Boot & Player of the Tournament")
    st.markdown(
        '<span class="caveat">Only ~25 notable players are tracked with real goal/assist '
        'data -- not full squads. Eliminated players\' totals are FINAL and locked in. '
        'POTT is a constructed heuristic (goals + assists + how far the team goes), '
        'not fitted to real award-voting data.</span>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### \U0001F45F Golden Boot Race")
        gb = sorted(stats["golden_boot_probability"].items(), key=lambda x: x[1], reverse=True)[:10]
        rows = []
        for name, prob in gb:
            d = PLAYER_METRICS[name]
            rows.append({
                "Player": name, "Team": d["team"], "Real Goals": d["tournament_goals"],
                "Status": "Alive" if d["team_alive"] else "Eliminated (final)",
                "Win Prob": f"{prob:.1%}",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

    with col2:
        st.markdown("#### \u2B50 Player of the Tournament")
        pott = sorted(stats["pott_probability"].items(), key=lambda x: x[1], reverse=True)[:10]
        rows = []
        for name, prob in pott:
            d = PLAYER_METRICS[name]
            rows.append({"Player": name, "Team": d["team"], "Win Prob": f"{prob:.1%}"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

    st.markdown("#### Current Real Golden Boot Standings (as of July 7-8, 2026)")
    all_players = sorted(PLAYER_METRICS.items(), key=lambda x: x[1]["tournament_goals"], reverse=True)
    rows = [{"Player": n, "Team": d["team"], "Goals": d["tournament_goals"],
             "Assists": d["tournament_assists"],
             "Status": "Alive" if d["team_alive"] else "Eliminated"} for n, d in all_players]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch', height=400)


# ---------------------------------------------------------------------------
# Tab 5: Live Standings
# ---------------------------------------------------------------------------
with tab_live:
    st.subheader("Live / Real-World Progress")
    if load_live_data_status():
        st.success("live_updates.json is present.")
        st.markdown('<span class="caveat">NOTE: this dashboard does not yet consume '
                    'live_updates.json values -- it displays the same hardcoded, '
                    'hand-verified data as everywhere else in the app. Wiring the '
                    'scraper output into the simulation is a pending next step.</span>',
                    unsafe_allow_html=True)
    else:
        st.info("No live_updates.json found. Run `python world_cup_scraper.py` "
                "on your own machine to generate one (this sandbox can't reach "
                "the target sites, so that script is untested against live data "
                "-- see its module docstring).")

    st.markdown("#### Tournament Status")
    st.markdown(f"""
    <div class="glass-card">
    <b>Group stage:</b> Complete (72/72 matches, June 11-27, 2026)<br>
    <b>Round of 32:</b> Complete (16/16 matches, June 28 - July 3, 2026)<br>
    <b>Round of 16:</b> Complete (8/8 matches, July 4-7, 2026)<br>
    <b>Quarterfinals:</b> Not yet played (July 9-11, 2026) -- simulated above<br>
    <b>Remaining group-stage fixtures:</b> {len(REMAINING_FIXTURES)}<br>
    <b>Completed matches on record:</b> {len(COMPLETED_MATCHES)}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Remaining Field")
    st.write(", ".join(REAL_QUARTERFINAL_FIELD))
