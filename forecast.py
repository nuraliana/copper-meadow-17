"""GTA VI trophy-rarity forecaster (make.fun gta6-trophy-rarity-forecast-01).

Forecasts, for every PS5 trophy of Grand Theft Auto VI, five quantiles of its
earn rate on Exophase at the case's target, from the case's history of house
reads of the game's trophy page. Python 3.13, standard library only, no
network, deterministic.

Model (count space, written from scratch for this bounty)
--------------------------------------------------------
Work in COUNTS, not rates: for each read, earners E_j = rate_j * N / 100
(N = players tracked) are integers, and the rate is E_j / N.  Two forces move
every rate: earners grow as players progress through the game, and the
tracked-player base grows (dilution).  Modelling the two separately is what
makes the forecast work: on launch night the page sat unchanged at N = 638
for hours, then refreshed to N = 3,868 with every rate but the 100% one
lower, and by 2026-11-20 03:39 UTC the live page showed the same pattern
again (platinum 0.63% -> 0.03%, most bronzes 1.10% -> 0.06%) while a few
early/easy trophies held up (fishing 2.04% -> 1.89%, Chapter 1 1.25% ->
0.73%).  Earners grow by trophy class; the player base grows fast and then
decelerates.

Regimes, chosen from the case itself:
  A. Flat history, checkpoint before the first refresh (the two practice
     cases): the first refresh is documented in the kit's RULES.txt
     (N 638 -> 3,868 at 2026-11-19 04:30 UTC, earners effectively frozen
     over the 90 minutes, the 100% trophy stays 100%).  Forecast exactly
     that: E frozen, N = 3,868, with a tight spread.
  B. Rich history (the final): fit, per trophy, log earners against
     log-time since release over the last 36 h of change points, with the
     slope shrunk toward its trophy class's mean slope (rare trophies have
     almost no signal: a 2-decimal rate at 0.03% is +-0.17 logit of
     rounding noise).  Fit players tracked the same way (two concave
     forms, median of the two).  Extrapolate both to the target and divide.
  C. Anything else flat / too few change points: power-law player growth
     (N ~ (1 + hours)^0.9, the exponent the launch-night-to-live-page
     snapshots imply) with class E-growth priors, wide spread.

Spreads are honest: a floor from refresh staleness (Exophase refreshes on
its own schedule, so the answering read can lag the target by hours) plus
each trophy's fit residual, growing with horizon.  Quantiles are normal on
the logit scale.

What this does NOT do: it never claims the future rates are known.  The
README states the assumptions, the backtests and the limits.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
FIRST_REFRESH_N = 3868            # RULES.txt: 638 -> 3,868 at 2026-11-19 04:30Z
FIRST_REFRESH_END = datetime(2026, 11, 19, 6, 0, 0, tzinfo=timezone.utc)
PROLOGUE_ID = 2                   # "Just Like Paradise": 100% at launch, stays ~100%

# Class E-growth priors (multiplier per day) from the launch-night snapshot
# (2026-11-19 03:00 UTC, N=638) to the live page (2026-11-20 03:39 UTC,
# N~16k): early/easy trophies' earners grew ~6-25x, story ~1.5-15x, grinds
# and the platinum ~1.3x.  Used only where the history has no trend signal.
CLASS_GROWTH_PER_DAY = {
    "prologue": 1.0,
    "story_early": 5.0,
    "story": 1.6,
    "easy": 6.0,
    "grind": 1.2,
    "platinum": 1.2,
    "other": 1.4,
}
N_POWER_LAW = 0.9                 # N ~ (1 + hours)^0.9 across the first day


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def clamp(x, lo=0.01, hi=99.99):
    return min(hi, max(lo, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    if lg > 20:
        return 99.99
    if lg < -20:
        return 0.01
    return clamp(100.0 / (1.0 + math.exp(-lg)))


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if float(value).is_integer() else value


def hours_since_release(t):
    return max(0.0, (t - RELEASE).total_seconds() / 3600.0)


def x_of(t):
    """Log-time since release: adoption curves are near-linear in this."""
    return math.log1p(hours_since_release(t))


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, resid_sd, n)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0
    if n == 1:
        return ys[0], 0.0, 0.0, 1
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0, 0.0, n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        resid = [y - (a + b * x) for x, y in zip(xs, ys)]
        sd = math.sqrt(sum(r * r for r in resid) / (n - 2))
    else:
        sd = 0.0
    return a, b, sd, n


def classify(trophy):
    """Trophy class from its description / tier. Drives shrinkage + priors."""
    tid = trophy["id"]
    if tid == PROLOGUE_ID:
        return "prologue"
    if trophy["tier"] == "platinum":
        return "platinum"
    d = (trophy.get("description") or "").lower()
    name = (trophy.get("name") or "").lower()
    if "complete chapter 1" in d or "complete chapter 2" in d:
        return "story_early"
    if ("complete chapter" in d or "complete the final mission" in d
            or "complete breakfast run" in d or "complete the cartel" in d
            or "complete child's play" in d or "complete headliners" in d
            or "complete vcia" in d or "complete operation" in d):
        return "story"
    if any(k in d for k in ("fishing", "arcade", "discover 100 vehicles",
                            "cocktail", "bowling", "melee drive-by",
                            "spend $100,000", "earn $250,000",
                            "register 10 vehicles", "purchase all garage",
                            "explore all of leonida")):
        return "easy"
    if any(k in d for k in ("250 stunts", "100% completion",
                            "collectible card", "waink it", "50 challenges",
                            "strangers across", "gold medal in all",
                            "win 20 different", "win a top fun")):
        return "grind"
    return "other"


def quantiles_from_logit(median_logit, sigma):
    return [round4(inv_logit(median_logit + z * sigma)) for z in Z]


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    checkpoint = parse_time(case["at"])
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    x_target = x_of(target)

    classes = [classify(t) for t in trophies]
    n_tr = len(trophies)

    if not history:
        # No data at all: flat 0.5% with a very wide spread (never invalid).
        return {"trophies": [{"id": t["id"],
                              "q": quantiles_from_logit(logit(0.5), 2.0)}
                             for t in trophies]}

    # ---- change points: drop consecutive identical reads (stale stretches
    # carry no information, and would overweight flat periods in a fit).
    points = []
    prev_key = None
    for r in history:
        key = (r.get("players_tracked"), tuple(r.get("rates") or ()))
        if key != prev_key:
            points.append(r)
            prev_key = key
    last = history[-1]
    n_last = last.get("players_tracked") or 0

    def earners(read, i):
        rate = (read.get("rates") or [None] * n_tr)[i]
        n = read.get("players_tracked") or 0
        if rate is None or n <= 0:
            return None
        return rate * n / 100.0

    last_e = []
    for i in range(n_tr):
        e = earners(last, i)
        last_e.append(e if e is not None else 0.0)

    # ---- Regime A: flat history before the first refresh.
    if len(points) <= 1 and checkpoint < FIRST_REFRESH_END:
        rows = []
        for i, t in enumerate(trophies):
            if t["id"] == PROLOGUE_ID:
                med = 99.99
            else:
                # The page prints 2 decimals: round the median the same way,
                # so the median is the printed rate, not the unrounded ratio.
                med = round(100.0 * last_e[i] / FIRST_REFRESH_N, 2)
                med = clamp(med)
            rows.append({"id": t["id"],
                         "q": quantiles_from_logit(logit(med), 0.008)})
        return {"trophies": rows}

    # ---- Regime B: trend fit on change points (last 36 h if enough).
    window_start = checkpoint.timestamp() - 36 * 3600
    win = [r for r in points
           if parse_time(r["at"]).timestamp() >= window_start]
    if len(win) < 4:
        win = points
    use_trend = len(win) >= 3 and n_last > 0

    if use_trend:
        xs = [x_of(parse_time(r["at"])) for r in win]
        # --- players: median of two concave extrapolations.
        # Both forms bend the same way (growth decelerates in log-time).
        # With two values the median is the upper one: the log-linear form
        # usually runs hotter, and in the synthetic-final backtest the
        # upper value beat the mean of the two (loss 0.12 vs 0.18), because
        # the cooler form under-predicts the player base late in the week
        # and inflates every rate.  Kept as the upper value on that
        # evidence, not on principle; the guardrail below caps it.
        ln_n = [math.log(max(1, r.get("players_tracked") or 1)) for r in win]
        a_ln, b_ln, _, _ = ols(xs, ln_n)
        pred_ln = a_ln + b_ln * x_target
        n_vals = [r.get("players_tracked") or 0 for r in win]
        a_n, b_n, _, _ = ols(xs, n_vals)
        pred_n_lin = a_n + b_n * x_target
        preds = []
        if pred_ln > 0:
            preds.append(math.exp(min(pred_ln, math.log(5_000_000))))
        if pred_n_lin > 0:
            preds.append(pred_n_lin)
        preds.sort()
        n_target = preds[len(preds) // 2] if preds else float(n_last)
        # Guardrails: never below last, never more than 3x/day of growth.
        max_growth = 3.0 ** max(horizon_days, 0.0)
        n_target = min(max(n_target, float(n_last)),
                       float(n_last) * max_growth)

        # --- earners per trophy, slope shrunk to class mean.
        fits = []
        for i in range(n_tr):
            ys, xsu = [], []
            for r, xv in zip(win, xs):
                e = earners(r, i)
                if e is None:
                    continue
                ys.append(math.log(e + 0.5))
                xsu.append(xv)
            if len(xsu) >= 2:
                a, b, sd, n = ols(xsu, ys)
            elif len(xsu) == 1:
                a, b, sd, n = ys[0], 0.0, 0.0, 1
            else:
                a = math.log(last_e[i] + 0.5)
                b, sd, n = 0.0, 0.0, 0
            fits.append((a, b, sd, n))

        class_slope = {}
        for cls in set(classes):
            bs = [fits[i][1] for i in range(n_tr)
                  if classes[i] == cls and fits[i][3] >= 3
                  and last_e[i] >= 3]
            if bs:
                bs.sort()
                class_slope[cls] = bs[len(bs) // 2]
        overall = sorted(f[1] for f in fits if f[3] >= 3)
        overall_slope = overall[len(overall) // 2] if overall else 0.0

        rows = []
        for i, t in enumerate(trophies):
            a, b, sd, n = fits[i]
            prior = class_slope.get(classes[i], overall_slope)
            # Shrink: pseudo-count 3 on the class prior.
            b_star = (n * b + 3.0 * prior) / (n + 3.0) if n else prior
            if classes[i] == "prologue":
                med_rate = 99.99
                sigma = 0.05
            else:
                log_e = a + b_star * x_target
                e_t = math.exp(min(log_e, math.log(n_target))) - 0.5
                e_t = max(e_t, last_e[i])          # earners never decrease
                e_t = min(e_t, last_e[i] * 12.0)   # explosion guardrail
                e_t = min(e_t, n_target)
                med_rate = clamp(100.0 * e_t / n_target)
                sigma = math.sqrt(0.30 ** 2
                                  + (1.5 * sd) ** 2
                                  + (0.30 * math.sqrt(horizon_days)) ** 2)
                sigma = min(max(sigma, 0.35), 2.0)
            rows.append({"id": t["id"],
                         "q": quantiles_from_logit(logit(med_rate), sigma)})
        return {"trophies": rows}

    # ---- Regime C: flat / too few change points after the first refresh.
    # Anchor player growth on the documented first refresh (N = 3,868 at
    # 04:30 UTC): the launch-night 638 was a stale page, not the player
    # base -- anchoring the power law on 638 under-predicts N a day later
    # by ~7x, as the live page (N ~ 16k at 2026-11-20 03:39 UTC) shows.
    h_last = hours_since_release(checkpoint)
    h_target = hours_since_release(target)
    refresh_h = 4.5
    if h_last >= refresh_h:
        base_n = max(float(n_last), float(FIRST_REFRESH_N))
        base_h = max(h_last, refresh_h)
    else:
        base_n = float(n_last) if n_last > 0 else 1.0
        base_h = h_last
    n_target = base_n * ((1.0 + h_target) / (1.0 + base_h)) ** N_POWER_LAW \
        if base_n > 0 else 1.0
    sigma = min(max(0.5 * math.sqrt(horizon_days) + 0.50, 0.8), 2.0)
    rows = []
    for i, t in enumerate(trophies):
        if t["id"] == PROLOGUE_ID:
            med_rate = 99.99
        else:
            # Class prior growth, decaying: each extra day adds half the
            # previous day's growth.
            g = CLASS_GROWTH_PER_DAY[classes[i]]
            days = horizon_days
            growth = 1.0 + (g - 1.0) * (1.0 - 0.5 ** days) / 0.5 * 0.5 \
                if days > 0 else 1.0
            e_t = min(last_e[i] * growth, n_target)
            med_rate = clamp(100.0 * e_t / n_target)
        rows.append({"id": t["id"],
                     "q": quantiles_from_logit(logit(med_rate), sigma)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
