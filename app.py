"""
UEFA Champions / Europa / Conference League Qualifying — Live Bracket Tracker
================================================================================
Reads a manually-curated bracket.csv (Team, League, Round, Path, Match) and
overlays live/final scores pulled from ESPN's public (unofficial) soccer
scoreboard JSON API.

Run locally:   streamlit run app.py
Deploy:        see DEPLOY.md / the chat message this shipped with.
"""

import re
import unicodedata
from datetime import datetime, timezone

import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

st.set_page_config(page_title="UEFA Qualifying Bracket Tracker", layout="wide")

BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"

# Order matters for tab display
COMPETITIONS = {
    "Champions League": "uefa.champions_qual",
    "Europa League": "uefa.europa_qual",
    "Conference League": "uefa.europa.conf_qual",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36 "
        "UEFABracketTracker/1.0 (+streamlit app)"
    )
}

REQUEST_TIMEOUT = 10  # seconds

# --------------------------------------------------------------------------
# Name normalization / matching helpers
# --------------------------------------------------------------------------

_STOPWORDS = {
    "fc", "cf", "sc", "ac", "afc", "cfc", "sk", "fk", "nk", "if", "bk",
    "the", "club", "de", "do", "da", "and", "united", "utd",
}


def normalize_name(name: str) -> str:
    """Lowercase, strip accents/punctuation, drop common club-name noise words."""
    if not name:
        return ""
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower()
    name = re.sub(r"[^a-z0-9 ]", " ", name)
    tokens = [t for t in name.split() if t and t not in _STOPWORDS]
    return " ".join(tokens).strip()


# --------------------------------------------------------------------------
# Data fetching (cached)
# --------------------------------------------------------------------------

@st.cache_data(ttl=180, show_spinner=False)
def fetch_scoreboard(slug: str):
    """Fetch a scoreboard JSON. Returns None on any failure or malformed body."""
    url = f"{BASE}/{slug}/scoreboard"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        return None
    if not isinstance(data, dict) or "events" not in data:
        return None
    return data


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_teams(slug: str):
    """Fetch the /teams list for a competition. Returns {} on failure."""
    url = f"{BASE}/{slug}/teams?limit=300"
    out = {}
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError):
        return out

    try:
        for sport in data.get("sports", []):
            for league in sport.get("leagues", []):
                for entry in league.get("teams", []):
                    t = entry.get("team", {})
                    tid = t.get("id")
                    if not tid:
                        continue
                    for field in ("displayName", "shortDisplayName", "name", "nickname", "abbreviation", "location"):
                        val = t.get(field)
                        if val:
                            out[normalize_name(val)] = tid
    except AttributeError:
        return {}
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def build_team_lookup():
    """Merge team name->id lookups across all three competitions (a team can
    appear in more than one, e.g. dropping from UCL quals into UEL/UECL)."""
    merged = {}
    for slug in COMPETITIONS.values():
        merged.update(fetch_teams(slug))
    return merged


def resolve_team_id(team_name: str, lookup: dict):
    """Exact normalized match first, then a loose substring/close-match fallback."""
    norm = normalize_name(team_name)
    if norm in lookup:
        return lookup[norm]

    # fallback: substring containment either direction
    for key, tid in lookup.items():
        if not key:
            continue
        if norm in key or key in norm:
            return tid

    # fallback: difflib close match
    import difflib
    candidates = difflib.get_close_matches(norm, lookup.keys(), n=1, cutoff=0.82)
    if candidates:
        return lookup[candidates[0]]

    return None


# --------------------------------------------------------------------------
# Scoreboard parsing / scoring
# --------------------------------------------------------------------------

def extract_finished_matches(scoreboard_json):
    """Pull out only FULL-TIME matches as a flat list of dicts."""
    matches = []
    if not scoreboard_json:
        return matches

    for event in scoreboard_json.get("events", []):
        comps = event.get("competitions") or []
        if not comps:
            continue
        comp = comps[0]

        status = comp.get("status", {}) or {}
        stype = status.get("type", {}) or {}
        state = stype.get("state")
        completed = bool(stype.get("completed", False))

        # Only fully-completed (full time / after ET / after pens) matches count.
        # Explicitly skip scheduled ("pre"), in-progress ("in"), and postponed/
        # cancelled matches (state "post" with completed False, e.g. some
        # abandoned-match edge cases).
        if state != "post" or not completed:
            continue

        competitors = comp.get("competitors") or []
        if len(competitors) != 2:
            continue

        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        try:
            home_score = int(float(home.get("score", 0)))
            away_score = int(float(away.get("score", 0)))
        except (TypeError, ValueError):
            continue

        home_team = home.get("team", {}) or {}
        away_team = away.get("team", {}) or {}
        if not home_team.get("id") or not away_team.get("id"):
            continue

        matches.append({
            "home_id": str(home_team["id"]),
            "away_id": str(away_team["id"]),
            "home_name": home_team.get("displayName", "Home"),
            "away_name": away_team.get("displayName", "Away"),
            "home_score": home_score,
            "away_score": away_score,
            "home_shootout": home.get("shootoutScore"),
            "away_shootout": away.get("shootoutScore"),
            "date": event.get("date"),
        })
    return matches


def find_legs(matches, team_id_a, team_id_b):
    """Return the finished legs (up to 2) between two specific team ids, sorted by date."""
    pair = {team_id_a, team_id_b}
    legs = [m for m in matches if {m["home_id"], m["away_id"]} == pair]
    legs.sort(key=lambda m: m.get("date") or "")
    return legs


def summarize_tie(legs, team_id_a, team_id_b):
    """Aggregate two legs into a result summary."""
    agg = {team_id_a: 0, team_id_b: 0}
    for leg in legs:
        agg[leg["home_id"]] += leg["home_score"]
        agg[leg["away_id"]] += leg["away_score"]

    result = {
        "legs_played": len(legs),
        "agg": agg,
        "winner_id": None,
        "decided_by": None,  # "aggregate" | "penalties" | None
    }

    if len(legs) < 2:
        return result

    if agg[team_id_a] != agg[team_id_b]:
        result["winner_id"] = max(agg, key=agg.get)
        result["decided_by"] = "aggregate"
        return result

    # Aggregate tied -> look for a shootout score on either leg
    for leg in legs:
        h_so, a_so = leg.get("home_shootout"), leg.get("away_shootout")
        if h_so is not None and a_so is not None:
            try:
                h_so, a_so = int(h_so), int(a_so)
            except (TypeError, ValueError):
                continue
            if h_so != a_so:
                result["winner_id"] = leg["home_id"] if h_so > a_so else leg["away_id"]
                result["decided_by"] = "penalties"
                result["penalty_score"] = (h_so, a_so) if h_so > a_so else (a_so, h_so)
                return result

    result["decided_by"] = "tied_unresolved"
    return result


# --------------------------------------------------------------------------
# Bracket loading
# --------------------------------------------------------------------------

REQUIRED_COLUMNS = {"Team", "League", "Round", "Path", "Match"}


@st.cache_data(show_spinner=False)
def load_bracket(path: str = "bracket.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"bracket.csv is missing required column(s): {sorted(missing)}")
    df["Path"] = df["Path"].fillna("").astype(str).str.strip()
    df["Team"] = df["Team"].astype(str).str.strip()
    df["League"] = df["League"].astype(str).str.strip()
    df["Round"] = df["Round"].astype(str).str.strip()
    df["Match"] = pd.to_numeric(df["Match"], errors="coerce")
    if df["Match"].isna().any():
        bad = df[df["Match"].isna()]
        raise ValueError(f"Non-numeric Match value(s) found:\n{bad}")
    df["Match"] = df["Match"].astype(int)
    return df


def round_sort_key(round_label: str):
    m = re.search(r"(\d+)", str(round_label))
    return int(m.group(1)) if m else 999


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def team_display(name, tid, agg_score, is_winner, legs_played):
    style = "font-weight:700;color:#1a7f37;" if is_winner else "font-weight:400;"
    unresolved = "" if tid else " ⚠️"
    return f"<span style='{style}'>{name}{unresolved}</span> — <b>{agg_score if legs_played else '–'}</b>"


def render_match_card(team_a, team_b, id_a, id_b, legs, summary, match_num):
    agg = summary["agg"]
    legs_played = summary["legs_played"]

    leg_lines = []
    for i, leg in enumerate(legs, start=1):
        leg_lines.append(
            f"Leg {i}: {leg['home_name']} {leg['home_score']} – {leg['away_score']} {leg['away_name']}"
        )
    for i in range(legs_played, 2):
        leg_lines.append(f"Leg {i + 1}: not yet played")

    winner_id = summary["winner_id"]
    decided_by = summary["decided_by"]

    a_line = team_display(team_a, id_a, agg.get(id_a, 0) if id_a else None, winner_id == id_a, legs_played)
    b_line = team_display(team_b, id_b, agg.get(id_b, 0) if id_b else None, winner_id == id_b, legs_played)

    footer = ""
    if decided_by == "penalties":
        pens = summary.get("penalty_score", ("?", "?"))
        footer = f"<div style='font-size:0.85em;color:#666;'>Decided on penalties ({pens[0]}–{pens[1]})</div>"
    elif decided_by == "tied_unresolved":
        footer = "<div style='font-size:0.85em;color:#b35c00;'>Aggregate tied — check penalty result (not available from this feed)</div>"
    elif legs_played < 2:
        footer = f"<div style='font-size:0.85em;color:#888;'>{legs_played}/2 legs played</div>"

    st.markdown(
        f"""
<div style="border:1px solid #e0e0e0;border-radius:10px;padding:10px 14px;margin-bottom:10px;background:#fafafa;">
  <div style="font-size:0.75em;color:#999;">Match {match_num}</div>
  <div>{a_line}</div>
  <div>{b_line}</div>
  <div style="font-size:0.8em;color:#777;margin-top:6px;">{"<br>".join(leg_lines)}</div>
  {footer}
</div>
""",
        unsafe_allow_html=True,
    )


def render_league(league_name: str, slug: str, bracket_df: pd.DataFrame, team_lookup: dict):
    league_df = bracket_df[bracket_df["League"].str.strip().str.lower() == league_name.lower()].copy()
    if league_df.empty:
        st.info(f"No rows found in bracket.csv for League == '{league_name}'.")
        return

    scoreboard = fetch_scoreboard(slug)
    if scoreboard is None:
        st.warning(
            f"⚠️ Live data temporarily unavailable for {league_name}. "
            "Showing the bracket structure without scores — try refreshing shortly."
        )
        finished_matches = []
    else:
        finished_matches = extract_finished_matches(scoreboard)

    # Resolve team ids
    league_df["TeamID"] = league_df["Team"].apply(lambda n: resolve_team_id(n, team_lookup))
    unresolved = sorted(league_df.loc[league_df["TeamID"].isna(), "Team"].unique())
    if unresolved:
        with st.expander(f"⚠️ {len(unresolved)} team name(s) in bracket.csv didn't match ESPN's team list", expanded=False):
            st.write(unresolved)
            st.caption(
                "These teams will still show in the bracket but can't be auto-scored until the "
                "name matches (or is close enough to) ESPN's team names for this competition."
            )

    rounds = sorted(league_df["Round"].unique(), key=round_sort_key)

    for rnd in rounds:
        st.markdown(f"### {rnd}")
        round_df = league_df[league_df["Round"] == rnd]
        paths = sorted(round_df["Path"].unique())  # "" sorts first (no split)

        if paths == [""]:
            render_matches_grid(round_df, finished_matches)
        else:
            cols = st.columns(len(paths))
            for col, path in zip(cols, paths):
                with col:
                    label = path if path else "Unsplit"
                    st.markdown(f"**{label}**")
                    render_matches_grid(round_df[round_df["Path"] == path], finished_matches)

        st.divider()


def render_matches_grid(round_path_df: pd.DataFrame, finished_matches):
    match_nums = sorted(round_path_df["Match"].unique())
    for match_num in match_nums:
        teams_in_match = round_path_df[round_path_df["Match"] == match_num]
        if len(teams_in_match) != 2:
            st.caption(f"Match {match_num}: expected 2 teams in bracket.csv, found {len(teams_in_match)}.")
            continue
        row_a, row_b = teams_in_match.iloc[0], teams_in_match.iloc[1]
        team_a, team_b = row_a["Team"], row_b["Team"]
        id_a, id_b = row_a["TeamID"], row_b["TeamID"]

        if id_a and id_b:
            legs = find_legs(finished_matches, id_a, id_b)
            summary = summarize_tie(legs, id_a, id_b)
        else:
            legs, summary = [], {"legs_played": 0, "agg": {}, "winner_id": None, "decided_by": None}

        render_match_card(team_a, team_b, id_a, id_b, legs, summary, match_num)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    st.title("🏆 UEFA Qualifying — Live Bracket Tracker")
    st.caption(
        "Champions League · Europa League · Conference League qualifying rounds. "
        "Scores pulled live from ESPN (unofficial API), cached 3 minutes."
    )

    top_col1, top_col2 = st.columns([1, 5])
    with top_col1:
        if st.button("🔄 Refresh now"):
            fetch_scoreboard.clear()
            fetch_teams.clear()
            build_team_lookup.clear()
            st.rerun()
    with top_col2:
        st.caption(f"Last app render: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    try:
        bracket_df = load_bracket("bracket.csv")
    except FileNotFoundError:
        st.error("Couldn't find bracket.csv next to app.py. Add it to the repo and redeploy.")
        return
    except ValueError as e:
        st.error(f"bracket.csv problem: {e}")
        return

    team_lookup = build_team_lookup()
    if not team_lookup:
        st.warning(
            "⚠️ Couldn't reach ESPN's /teams endpoints right now — team-name matching "
            "may fail until that's back. The bracket structure will still render."
        )

    tabs = st.tabs(list(COMPETITIONS.keys()))
    for tab, (league_name, slug) in zip(tabs, COMPETITIONS.items()):
        with tab:
            render_league(league_name, slug, bracket_df, team_lookup)


if __name__ == "__main__":
    main()
