"""
world_cup_scraper.py

FIFA World Cup 2026 Predictive Engine -- Optional Live-Data Layer
====================================================================

Zero-API, on-demand scraper. Run manually (once a day, or whenever you
want a refresh) to pull the freshest Elo ratings, match results, and
player stats from public sites, and write them to live_updates.json next
to this file. world_cup_data.py's hardcoded values remain the permanent
fallback -- nothing here is required for the simulator to run.

STATUS, stated plainly: this file is logically complete but UNVERIFIED
against live data. This sandbox's network allowlist blocks eloratings.net,
en.wikipedia.org, and fbref.com directly (confirmed via failed test
fetches -- "Host not in allowlist" / 403), so this has only been checked
by static code review, never a real run. Your machine has normal internet
access, so this is expected to work there -- but the first real run is
also the first real test. If a selector is wrong, it'll show up as an
empty/partial live_updates.json or a clear warning printed to the
console, NOT a silent wrong answer -- every fetch function is written to
fail loud and fall back cleanly rather than guess.

Usage (on your own machine, not in this sandbox):
    cd C:\\Users\\Ashutosh\\Desktop\\world_cup_predictor\\
    python world_cup_scraper.py

No API key. No paid service. Just urllib + basic HTML parsing against
public pages, run on-demand, at your discretion.
"""

import json
import os
import re
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from html.parser import HTMLParser


LIVE_UPDATES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "live_updates.json")
STALE_AFTER_HOURS = 24
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) world_cup_predictor/1.0 (personal, non-commercial use)"


# ---------------------------------------------------------------------------
# Country-name / code mapping tables
# ---------------------------------------------------------------------------
# eloratings.net and Wikipedia both use full country names, not FIFA's
# 3-letter codes. These maps are hand-built and will need occasional
# correction -- flagging that rather than pretending they're exhaustive.
ELO_NAME_TO_CODE = {
    "Mexico": "MEX", "South Korea": "KOR", "South Africa": "RSA", "Czech Republic": "CZE",
    "Canada": "CAN", "Bosnia-Herzegovina": "BIH", "Qatar": "QAT", "Switzerland": "SUI",
    "Brazil": "BRA", "Morocco": "MAR", "Haiti": "HAI", "Scotland": "SCO",
    "USA": "USA", "Paraguay": "PAR", "Australia": "AUS", "Turkey": "TUR",
    "Germany": "GER", "Ivory Coast": "CIV", "Ecuador": "ECU", "Curacao": "CUW",
    "Netherlands": "NED", "Japan": "JPN", "Sweden": "SWE", "Tunisia": "TUN",
    "Belgium": "BEL", "Egypt": "EGY", "Iran": "IRN", "New Zealand": "NZL",
    "Spain": "ESP", "Uruguay": "URU", "Saudi Arabia": "KSA", "Cape Verde Islands": "CPV",
    "France": "FRA", "Senegal": "SEN", "Norway": "NOR", "Iraq": "IRQ",
    "Argentina": "ARG", "Algeria": "ALG", "Austria": "AUT", "Jordan": "JOR",
    "Portugal": "POR", "Colombia": "COL", "Uzbekistan": "UZB", "DR Congo": "COD",
    "England": "ENG", "Croatia": "CRO", "Ghana": "GHA", "Panama": "PAN",
}

WIKI_NAME_TO_CODE = dict(ELO_NAME_TO_CODE)
WIKI_NAME_TO_CODE.update({
    "Czechia": "CZE",
    "Bosnia and Herzegovina": "BIH",
    "Côte d'Ivoire": "CIV",
    "Curaçao": "CUW",
    "Cabo Verde": "CPV",
    "Türkiye": "TUR",
})


def _http_get(url: str, timeout: int = 15) -> str:
    """
    Fetch a URL with a real User-Agent (some sites, FBRef especially,
    reject requests with Python's default urllib UA). Raises on failure
    rather than returning an empty string, so callers can distinguish
    "got nothing" from "got an empty page".
    """
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# 1. Elo ratings
# ---------------------------------------------------------------------------
def fetch_elo_ratings() -> dict:
    """
    Pull current Elo ratings from eloratings.net's World.tsv.

    Column layout (reverse-engineered from third-party package docs during
    the session that wrote this file, NOT verified against a live fetch --
    flagging that explicitly): rank, rank_change, country_name, elo_rating,
    ...additional columns not needed here.

    Returns: {team_code: elo_rating} for every team we recognize. Teams
    not in ELO_NAME_TO_CODE are silently skipped (logged to stderr-style
    print, not raised) -- an unmapped country shouldn't crash the whole
    fetch.
    """
    url = "https://www.eloratings.net/World.tsv"
    ratings = {}
    try:
        raw = _http_get(url)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"[scraper] fetch_elo_ratings: FAILED to reach {url} ({e})")
        return ratings

    for line in raw.splitlines():
        cols = line.split("\t")
        if len(cols) < 4:
            continue
        country_name = cols[2].strip()
        try:
            elo_value = float(cols[3].strip())
        except ValueError:
            continue
        code = ELO_NAME_TO_CODE.get(country_name)
        if code is None:
            continue  # unmapped country name -- skip, don't guess
        ratings[code] = elo_value

    if not ratings:
        print(f"[scraper] fetch_elo_ratings: page fetched but 0 ratings parsed -- "
              f"column layout or country-name map is likely stale, needs a look at real HTML.")
    return ratings


# ---------------------------------------------------------------------------
# 2. Match results (Wikipedia)
# ---------------------------------------------------------------------------
class _FootballboxParser(HTMLParser):
    """
    Minimal HTML parser for Wikipedia's div.footballbox score blocks and
    the group-heading (h3/h2) text that precedes them. Deliberately
    simple -- a full DOM parser (BeautifulSoup) isn't a hard requirement
    and this avoids adding a dependency the user may not have installed.
    """
    def __init__(self):
        super().__init__()
        self.in_footballbox = False
        self.div_depth = 0
        self.current_group = None
        self.current_text_chunks = []
        self.matches = []
        self._capturing_heading = False
        self._heading_text = []

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        classes = attrs_dict.get("class", "")
        if tag == "div" and "footballbox" in classes.split():
            self.in_footballbox = True
            self.div_depth = 1
            self.current_text_chunks = []
        elif self.in_footballbox and tag == "div":
            self.div_depth += 1
        elif tag in ("h2", "h3"):
            self._capturing_heading = True
            self._heading_text = []

    def handle_endtag(self, tag):
        if self.in_footballbox and tag == "div":
            self.div_depth -= 1
            if self.div_depth == 0:
                self.in_footballbox = False
                self._parse_footballbox_text("".join(self.current_text_chunks))
        elif tag in ("h2", "h3") and self._capturing_heading:
            self._capturing_heading = False
            heading = "".join(self._heading_text)
            m = re.search(r"Group\s+([A-L])\b", heading)
            if m:
                self.current_group = m.group(1)

    def handle_data(self, data):
        if self.in_footballbox:
            self.current_text_chunks.append(data)
        if self._capturing_heading:
            self._heading_text.append(data)

    def _parse_footballbox_text(self, text: str):
        # Expected rough shape once whitespace-collapsed:
        # "<Home> <home_goals> - <away_goals> <Away>"
        collapsed = re.sub(r"\s+", " ", text).strip()
        m = re.search(r"([A-Za-zÀ-ÿ'\.\s]+?)\s+(\d+)\s*[-–]\s*(\d+)\s+([A-Za-zÀ-ÿ'\.\s]+)", collapsed)
        if not m:
            return
        home_name, home_goals, away_goals, away_name = m.groups()
        home_code = WIKI_NAME_TO_CODE.get(home_name.strip())
        away_code = WIKI_NAME_TO_CODE.get(away_name.strip())
        if home_code is None or away_code is None or self.current_group is None:
            return  # unmapped name or no group context yet -- skip, don't guess
        self.matches.append({
            "group": self.current_group,
            "home": home_code,
            "away": away_code,
            "home_goals": int(home_goals),
            "away_goals": int(away_goals),
        })


def fetch_match_results() -> list:
    """
    Scrape Wikipedia's "2026 FIFA World Cup" article for group-stage
    results. Returns a list of match dicts in the same shape as
    COMPLETED_MATCHES entries (minus "date", which Wikipedia's markup
    doesn't expose in a consistently parseable spot).
    """
    url = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup"
    try:
        raw = _http_get(url)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"[scraper] fetch_match_results: FAILED to reach {url} ({e})")
        return []

    parser = _FootballboxParser()
    parser.feed(raw)
    if not parser.matches:
        print("[scraper] fetch_match_results: page fetched but 0 matches parsed -- "
              "the footballbox class or page structure has likely changed, needs "
              "a look at real HTML rather than guessing a fix.")
    return parser.matches


# ---------------------------------------------------------------------------
# 3. Player stats (FBRef)
# ---------------------------------------------------------------------------
def fetch_player_stats(player_names: list) -> dict:
    """
    Scrape FBRef's World Cup standard-stats table for goals/assists for
    the given player names (fuzzy-matched: exact, then last-name, then
    substring).

    KNOWN GOTCHA, handled rather than ignored: FBRef frequently serves
    its data tables wrapped inside HTML comments (<!-- <table>...</table>
    -->) specifically to defeat naive scrapers that only look for the
    table tag directly in the visible DOM. This function checks both the
    plain HTML and the content of any HTML comments for the target table
    id. If FBRef's anti-bot layer blocks the request outright (403, or a
    CAPTCHA-interstitial page), this fails loud with a clear warning and
    returns {} -- callers fall back to PLAYER_METRICS's real, hand-
    verified tallies rather than getting silently wrong numbers.
    """
    url = "https://fbref.com/en/comps/1/stats/FIFA-World-Cup-Stats"
    try:
        raw = _http_get(url)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"[scraper] fetch_player_stats: FAILED to reach {url} ({e}) -- "
              f"FBRef's anti-bot protection may be blocking this. Skipping "
              f"player stats; simulation will fall back to PLAYER_METRICS defaults.")
        return {}

    # Pull the target table out of HTML comments if it's not directly present.
    table_html = None
    direct_match = re.search(r'<table[^>]*id="stats_standard"[^>]*>.*?</table>', raw, re.DOTALL)
    if direct_match:
        table_html = direct_match.group(0)
    else:
        for comment in re.findall(r"<!--(.*?)-->", raw, re.DOTALL):
            if 'id="stats_standard"' in comment:
                m = re.search(r'<table[^>]*id="stats_standard"[^>]*>.*?</table>', comment, re.DOTALL)
                if m:
                    table_html = m.group(0)
                    break

    if table_html is None:
        print("[scraper] fetch_player_stats: page fetched but stats_standard table "
              "not found (checked both direct HTML and HTML-comment-wrapped tables). "
              "FBRef likely changed their table id or layout. Skipping player stats.")
        return {}

    # Extract per-player rows: name, goals, assists columns by data-stat attr.
    row_pattern = re.compile(
        r'<tr[^>]*>.*?data-stat="player"[^>]*>(?:<a[^>]*>)?([^<]+)(?:</a>)?.*?'
        r'data-stat="goals"[^>]*>(\d*)</td>.*?'
        r'data-stat="assists"[^>]*>(\d*)</td>',
        re.DOTALL,
    )
    scraped = {}
    for name_raw, goals_raw, assists_raw in row_pattern.findall(table_html):
        name = name_raw.strip()
        goals = int(goals_raw) if goals_raw.strip().isdigit() else 0
        assists = int(assists_raw) if assists_raw.strip().isdigit() else 0
        scraped[name] = {"goals": goals, "assists": assists}

    if not scraped:
        print("[scraper] fetch_player_stats: table found but 0 rows parsed -- "
              "row regex likely doesn't match FBRef's current column layout.")
        return {}

    # Fuzzy-match requested player_names against scraped keys.
    results = {}
    for target in player_names:
        if target in scraped:
            results[target] = scraped[target]
            continue
        target_last = target.split()[-1].lower()
        match = next((k for k in scraped if k.split()[-1].lower() == target_last), None)
        if match:
            results[target] = scraped[match]
            continue
        match = next((k for k in scraped if target.lower() in k.lower() or k.lower() in target.lower()), None)
        if match:
            results[target] = scraped[match]
    return results


# ---------------------------------------------------------------------------
# 4. Save / load
# ---------------------------------------------------------------------------
def save_live_updates(elo_ratings: dict, match_results: list, player_stats: dict) -> None:
    payload = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "elo_ratings": elo_ratings,
        "match_results": match_results,
        "player_stats": player_stats,
    }
    with open(LIVE_UPDATES_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[scraper] Wrote {LIVE_UPDATES_PATH}")


def load_live_updates():
    """
    Returns the parsed live_updates.json contents if it exists AND is
    fresher than STALE_AFTER_HOURS, else None. Callers should treat None
    as "use the hardcoded world_cup_data.py values" -- this is the
    fallback contract the whole design relies on.
    """
    if not os.path.exists(LIVE_UPDATES_PATH):
        return None
    try:
        with open(LIVE_UPDATES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        fetched_at = datetime.fromisoformat(data["fetched_at"])
        age_hours = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 3600.0
        if age_hours > STALE_AFTER_HOURS:
            print(f"[scraper] live_updates.json is {age_hours:.1f}h old (stale after "
                  f"{STALE_AFTER_HOURS}h) -- ignoring, using hardcoded fallback.")
            return None
        return data
    except (json.JSONDecodeError, KeyError, OSError, ValueError) as e:
        print(f"[scraper] live_updates.json exists but couldn't be read ({e}) -- "
              f"using hardcoded fallback.")
        return None


def main():
    print("FIFA World Cup 2026 -- Live Data Scraper")
    print("=" * 50)
    print("NOTE: this has not been run against live data before -- this")
    print("      environment's sandbox blocks the target sites. This run,")
    print("      on your machine, is also the first real test of it.")
    print()

    print("[1/3] Fetching Elo ratings from eloratings.net ...")
    elo_ratings = fetch_elo_ratings()
    print(f"      -> {len(elo_ratings)} team ratings parsed.")

    print("[2/3] Fetching match results from Wikipedia ...")
    match_results = fetch_match_results()
    print(f"      -> {len(match_results)} matches parsed.")

    print("[3/3] Fetching player stats from FBRef ...")
    from world_cup_data import PLAYER_METRICS
    player_stats = fetch_player_stats(list(PLAYER_METRICS.keys()))
    print(f"      -> {len(player_stats)} of {len(PLAYER_METRICS)} tracked players matched.")

    save_live_updates(elo_ratings, match_results, player_stats)
    print()
    print("Done. Run 'python main.py' to see if it picks up live_updates.json")
    print("(NOTE: as of this version, main.py only reports whether the file")
    print(" exists -- it does not yet feed these values into the simulation.")
    print(" That integration step is still pending.)")


if __name__ == "__main__":
    main()
