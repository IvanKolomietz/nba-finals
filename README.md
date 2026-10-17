# NBA Finals title odds, with an honest backtest

Estimates each NBA team's chance of winning the championship by simulating the
playoff bracket 20,000 times. The main part of the project is the **model report card**: a
walk-forward backtest over 18 seasons that checks whether the model actually beats
simple baselines. (Mostly it doesn't, by a margin we can measure. See below.)

![tests](https://github.com/IvanKolomietz/nba-finals/actions/workflows/ci.yml/badge.svg)

## What it does

1. Downloads 30 seasons (1996-97 to 2025-26) of team and player game logs from stats.nba.com and caches them locally.
2. Builds 10 features per team from regular-season data only (see below).
3. Fits a logistic regression that predicts **individual playoff games** from the difference between two teams' features, plus home court.
4. Turns game odds into series odds (exactly, for the 2-2-1-1-1 home pattern) and runs a **Monte Carlo simulation of the full bracket** (20,000 runs) to get each team's chance of reaching each round.
5. Backtests the whole pipeline season by season against simple baselines.
6. Serves a local dashboard with:
   - **Title odds**: the 2025-26 playoff field, plus how the odds moved during the regular season
   - **Team pages**: features, league ranks, the top 8 players and their availability
   - **Model report card**: the backtest results, a calibration chart and feature importance

> **About "current" odds.** The 2025-26 season ended in June 2026 (Knicks over the Spurs, 4-1).
> The dashboard shows the odds the model gives on the eve of the 2025-26 playoffs, using a model
> trained only on 2003-04 to 2024-25. It had Oklahoma City as the favourite at 44%; the Knicks were
> 6th of 16 at 4.4%. 2025-26 is also included as an ordinary test season in the backtest.

## Data sources

| Data | Source | Notes |
|---|---|---|
| Team game logs (regular season + playoffs) | stats.nba.com `leaguegamelog` via [`nba_api`](https://github.com/swar/nba_api) | box score per team per game |
| Player game logs | `playergamelogs` | who played, minutes, box score |
| Player ages, career playoff games | `leaguedashplayerstats` | |
| Final standings and seeds | `leaguestandingsv3` | |

Basketball-Reference was **not** used: its terms restrict scraping. The NBA's stats site has no
published API terms; `nba_api` is a widely used open-source client. Requests are cached
(`data/raw/`, not committed) and spaced out, so a full download makes about 180 requests, once.
No injury-report data is used, because there is no free historical source. Health is estimated from
who actually played (see Limitations).

The two processed JSON files the dashboard reads (`data/processed/`, ~100 KB) **are** committed, so
the dashboard and tests run without downloading anything.

## Method

### Features (per team, from regular-season games only)

| Feature | Meaning |
|---|---|
| `adj_net` | Net rating (points per 100 possessions), adjusted for strength of schedule with a least-squares rating fit |
| `win_pct` | Win percentage |
| `net_last20` | Net rating over the last 20 games (recent form, e.g. after trades) |
| `def_rel` | Defensive rating vs league average |
| `age` | Roster age, weighted by minutes played |
| `star_power` | Average [Game Score](https://www.basketball-reference.com/about/glossary.html) of the team's two best players |
| `star_share` | Share of the top 8 players' production that comes from the top 2 (star concentration) |
| `health` | How often the top 8 players (by minutes) appeared in the team's last 15 games |
| `top3_missed` | Share of games the top 3 players missed this season |
| `playoff_exp` | Career playoff games of the roster, weighted by minutes |

A player counts for the team he most recently played for, so traded players move with the trade.

### Game model → series → bracket

- **Game model**: logistic regression on `features(A) − features(B)` plus a home-court term, with no
  intercept, so swapping the two teams exactly flips the probability. It is trained only on playoff
  games (about 80 per season), because regular-season features measured over the whole season would
  include the very game being predicted. The 2019-20 bubble playoffs have no home court.
- **Regularisation**: the penalty strength is chosen by leave-one-season-out cross-validation
  *inside the training seasons only*. An early version without this was badly overconfident, giving
  the 2010-11 champion Mavericks 0.01%.
- **Series**: exact best-of-7 probability from the home and road game probabilities. Home court goes to the better record.
- **Bracket**: the real seeds (including play-in qualifiers) are reconstructed from standings and
  first-round matchups, then the bracket is simulated 20,000 times. Tests check that the simulation
  matches an exact calculation to within 0.6 percentage points.
- **In-season trend**: at 2-week checkpoints the model uses only games played so far, pulls
  early-season stats toward league average, then simulates the rest of the regular season, the
  play-in and the playoffs. The trend chart is **not** backtested.

### Backtest

Walk-forward: for each season from 2008-09 to 2025-26, train on 2003-04 up to the previous season,
predict on the eve of the playoffs, then score. Models compared:

- **Full model**: all 10 features
- **Win % only**: the same pipeline, but the only feature is regular-season record
- **Net rating only**: the same pipeline, but the only feature is schedule-adjusted net rating
- **Best record wins**: 100% to the team with the best record
- **Uniform**: 1/16 to every playoff team

Betting odds were not compared because no free, legitimate source of historical title odds was found.

## Results (18 test seasons, 2008-09 to 2025-26)

**Predicting the champion** (one outcome per season)

| Model | Brier ↓ | Log loss ↓ | Champion was #1 pick | Champion in top 3 | Avg. chance given to champion |
|---|---|---|---|---|---|
| **Full model** | **0.740** | **1.708** | 33% | **78%** | **28.4%** |
| Win % only | 0.773 | 1.744 | 33% | 72% | 27.1% |
| Net rating only | 0.801 | 1.817 | 33% | 67% | 26.1% |
| Best record wins | 1.306 | ∞ | 33% | 69% | 33.3% |
| Uniform (1/16 each) | 0.938 | 2.773 | 6% | 19% | 6.3% |

Brier is summed over the 16 playoff teams (0 = perfect). Ties in "best record" are broken at random.

**Predicting series and games** (270 series, 1,506 games)

| Model | Series log loss ↓ | Series accuracy | Game log loss ↓ | Game accuracy |
|---|---|---|---|---|
| Full model | 0.542 | 69.3% | 0.630 | 65.5% |
| Win % only | **0.537** | **71.9%** | 0.630 | 64.5% |
| Net rating only | 0.565 | 69.6% | 0.633 | 65.9% |
| *Always pick the team with home court* | | *72.2%* | | *60.8%* |

**What this means, in plain terms**

- The model is clearly better than guessing, and it put the eventual champion in its top 3 in 14 of 18 seasons.
- But it is **not reliably better than knowing only each team's win %**. A paired bootstrap over
  seasons gives a 95% interval of −0.28 to +0.27 for the champion log-loss difference, which
  includes zero. At the series and game level the two are tied, and for picking series winners,
  "the team with home court wins" is just as accurate.
- Calibration: when the model gave a series favourite 90-100%, the favourite won 98% of the time
  (54 series). In the 50-60% bucket favourites won only 32% (47 series), so close calls are noisy.
- Big misses: 2010-11 Dallas (0.7%), 2020-21 Milwaukee (6.1%) and 2025-26 New York (4.4%).

### Which features matter

Drop-one-feature test (re-run the whole backtest without the feature; positive = the feature helps out of sample):

| Feature removed | Game log loss change | Series log loss change |
|---|---|---|
| Win % | +0.0018 | +0.0085 |
| Star power | +0.0011 | +0.0023 |
| Net rating (adjusted) | +0.0008 | −0.0068 |
| Playoff experience | +0.0006 | +0.0048 |
| Everything else (age, health, form, defense, star concentration) | ≤ +0.0002 or negative | mostly negative |

Win %, star power and playoff experience carry most of the signal, and every effect is small. The
health proxies, age and recent form don't help out of sample. The current model's standardised
coefficients agree: win % (0.26), net rating (0.23), star power (0.17) and playoff experience (0.15)
are the largest.

## Limitations

- **Small sample.** 18 champions can't separate models this close, which is why series and game metrics are also reported.
- **Health is a proxy.** "Missed games" can't tell injury from rest or load management. There is no information after the regular season ends, such as injuries during the playoffs.
- **Rosters.** A player belongs to the last team he played for. Bought-out players who didn't sign elsewhere look "injured".
- **Playoff experience** only counts games since 1996-97, so veterans are undercounted in early seasons.
- **Independence.** Games are treated as independent with fixed probabilities: no momentum, coaching adjustments or injuries within a series.
- **Format details simplified.** Every series uses 2-2-1-1-1 (the Finals were 2-3-2 before 2014). Pre-2006 seeding rules are taken from the recorded seeds.
- **The in-season trend is a heuristic.** Its early-season shrinkage (10 games) was not tuned or backtested.

## Roadmap: live tracking during the season

Right now the dashboard is a snapshot of the finished 2025-26 season. The next planned feature is
to **track title odds live through the 2026-27 season**, updating as games are played:

- [ ] Re-download the current season's data on each update (the cache currently keeps old files)
- [ ] Fetch the remaining schedule, so unplayed games can be simulated
- [ ] Before the playoffs, use the full-season simulation for the headline odds (rest of the regular season, then play-in, then playoffs) instead of the real bracket
- [ ] Rebuild automatically on a schedule (e.g. a daily scheduled job)
- [ ] Backtest the in-season odds on past seasons and tune the early-season shrinkage, which is currently a guess

## How to run

Requires Python 3.11.

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt

python app.py                   # dashboard at http://127.0.0.1:5000 (uses committed results)
pytest                          # unit tests

python -m nbafinals.build --fetch   # optional: download raw data (~20 MB, ~5 min) and rebuild everything
```

Project layout:

```
nbafinals/
  data.py       download + cache raw data (nba_api)
  features.py   team features as of any date
  bracket.py    real seeds, series results and champions from the logs
  model.py      logistic game model + cross-validated regularisation
  simulate.py   series maths, bracket and full-season Monte Carlo
  backtest.py   walk-forward backtest, scoring rules, ablation
  build.py      runs everything, writes data/processed/*.json
app.py          Flask dashboard
tests/          unit tests (features, simulation, probabilities, model, app)
```

## How this was built

Ivan Kolomiets set the goals and direction for this project: what to predict, which inputs to
consider, and that honest validation against baselines should be the centrepiece. The code,
analysis and this README were written by [Claude Code](https://claude.com/claude-code) (Anthropic's AI
coding assistant), working from that brief.
