from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "processed"

# Seasons are named by the calendar year they start in: 2025 means 2025-26.
FIRST_SEASON = 1996      # earliest season stats.nba.com has player data for
FIRST_MODEL_SEASON = 2003  # earlier seasons only feed the playoff-experience feature
FIRST_BACKTEST_SEASON = 2008  # every backtest season has >= 5 earlier training seasons
CURRENT_SEASON = 2025

N_SIMS = 20_000
RANDOM_SEED = 2026


def season_str(year: int) -> str:
    return f"{year}-{str(year + 1)[-2:]}"
