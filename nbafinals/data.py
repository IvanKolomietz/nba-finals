"""Download raw NBA data from stats.nba.com (via the nba_api package) and cache it.

Every request is cached as a gzipped CSV under data/raw/, so each one only ever
hits the network once. Re-running is cheap and polite to the NBA's servers.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from .config import RAW_DIR, FIRST_SEASON, CURRENT_SEASON, season_str

REQUEST_PAUSE = 0.8  # seconds between live requests

# Columns kept from the (very wide) player game log, to keep the cache small.
PLAYER_LOG_COLS = [
    "SEASON_YEAR", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
    "GAME_ID", "GAME_DATE", "MIN", "FGM", "FGA", "FTM", "FTA", "OREB", "DREB",
    "AST", "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS",
]


def _cached(name: str, fetch, refresh: bool = False) -> pd.DataFrame:
    path = RAW_DIR / f"{name}.csv.gz"
    if path.exists() and not refresh:
        return pd.read_csv(path, dtype={"GAME_ID": str})
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            df = fetch()
            break
        except Exception as exc:  # network hiccups / rate limits
            wait = 5 * (attempt + 1)
            print(f"  {name}: {exc!r}; retrying in {wait}s")
            time.sleep(wait)
    else:
        raise RuntimeError(f"could not fetch {name}")
    df.to_csv(path, index=False, compression="gzip")
    time.sleep(REQUEST_PAUSE)
    return pd.read_csv(path, dtype={"GAME_ID": str})


def team_game_log(year: int, season_type: str, refresh: bool = False) -> pd.DataFrame:
    """One row per team per game. season_type: 'Regular Season' | 'Playoffs'."""
    from nba_api.stats.endpoints import leaguegamelog

    tag = {"Regular Season": "reg", "Playoffs": "po"}[season_type]

    def fetch():
        return leaguegamelog.LeagueGameLog(
            season=season_str(year), season_type_all_star=season_type,
            player_or_team_abbreviation="T", timeout=60,
        ).get_data_frames()[0]

    return _cached(f"team_games_{tag}_{year}", fetch, refresh)


def player_game_log(year: int, refresh: bool = False) -> pd.DataFrame:
    """One row per player per regular-season game."""
    from nba_api.stats.endpoints import playergamelogs

    def fetch():
        df = playergamelogs.PlayerGameLogs(
            season_nullable=season_str(year), season_type_nullable="Regular Season", timeout=120,
        ).get_data_frames()[0]
        return df[PLAYER_LOG_COLS]

    return _cached(f"player_games_{year}", fetch, refresh)


def player_season_stats(year: int, season_type: str, refresh: bool = False) -> pd.DataFrame:
    """Season totals per player (used for age and playoff games played)."""
    from nba_api.stats.endpoints import leaguedashplayerstats

    tag = {"Regular Season": "reg", "Playoffs": "po"}[season_type]

    def fetch():
        df = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season_str(year), season_type_all_star=season_type,
            per_mode_detailed="Totals", timeout=60,
        ).get_data_frames()[0]
        return df[["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "AGE", "GP", "MIN"]]

    return _cached(f"player_season_{tag}_{year}", fetch, refresh)


def standings(year: int, refresh: bool = False) -> pd.DataFrame:
    """Final regular-season standings, including conference seed (PlayoffRank)."""
    from nba_api.stats.endpoints import leaguestandingsv3

    def fetch():
        df = leaguestandingsv3.LeagueStandingsV3(season=season_str(year), timeout=60).get_data_frames()[0]
        return df[["TeamID", "TeamCity", "TeamName", "Conference", "PlayoffRank", "WINS", "LOSSES", "WinPCT"]]

    return _cached(f"standings_{year}", fetch, refresh)


def download_all(first: int = FIRST_SEASON, last: int = CURRENT_SEASON, refresh_current: bool = False) -> None:
    for year in range(first, last + 1):
        refresh = refresh_current and year == last
        print(f"season {season_str(year)}")
        team_game_log(year, "Regular Season", refresh)
        team_game_log(year, "Playoffs", refresh)
        player_game_log(year, refresh)
        player_season_stats(year, "Regular Season", refresh)
        player_season_stats(year, "Playoffs", refresh)
        standings(year, refresh)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Download and cache raw NBA data.")
    p.add_argument("--refresh-current", action="store_true", help="re-download the current season")
    args = p.parse_args()
    download_all(refresh_current=args.refresh_current)
