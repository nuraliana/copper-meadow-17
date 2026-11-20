# gta6-trophy-rarity-forecast-01

[![make.fun bounty · GTA VI trophy rarity forecast](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![leaderboard rank · copper-meadow-17](https://make.fun/badge/gta6-trophy-rarity-forecast-01/nuraliana/copper-meadow-17/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward · copper-meadow-17](https://make.fun/badge/gta6-trophy-rarity-forecast-01/nuraliana/copper-meadow-17/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, five
quantiles of its earn rate on Exophase at the case's target
(2026-11-26T00:00:00Z for the final). Python 3.13, standard library only,
no network, deterministic. Built for
[make.fun's GTA VI trophy-rarity forecast bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
($250 in USDC).

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` is the packed entrypoint (see `forecast.py`, the readable source
it was packed from with `makefun compress`). One `<id>.out` is written per
`<id>.in`: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}]}`.

## Model — count space, earners over players

Work in **counts, not rates**: for each read, earners `E = rate x N / 100`
(`N` = players tracked) are integers, and the rate is `E / N`. Two forces
move every rate: earners grow as players progress through the game, and the
tracked-player base grows (dilution). Modelling the two separately is what
makes the forecast work.

What the data showed (and the model is built on):

- Launch night the page sat unchanged at N = 638 for hours, then refreshed
  to N = 3,868 with every rate but the 100% one lower (documented in the
  kit's RULES.txt). By 2026-11-20 03:39 UTC the live page showed the same
  pattern again: platinum 0.63% -> 0.03%, most bronzes 1.10% -> 0.06%,
  while a few early/easy trophies held up (fishing 2.04% -> 1.89%,
  Chapter 1 1.25% -> 0.73%). Dilution dominates; earners grow by trophy
  class (easy activities ~6-25x in a day, story ~1.5-15x, grinds and the
  platinum ~1.3x). The median logit move over that first day was about
  -3. The live page is also visibly mid-refresh (some trophies' rates imply
  incompatible player counts), so the answering read can lag the target by
  hours: spreads must cover that.
- A rate is therefore **not** extrapolated as a rate. Earners and players
  are extrapolated separately, in log-time since release
  (`ln(1 + hours)`), and divided.

Three regimes, chosen from the case itself:

- **A. Flat history, checkpoint before the first refresh** (the two
  practice cases): the documented first refresh — earners frozen,
  N = 3,868, the 100% trophy stays 100%, medians rounded to the page's
  2 decimals, tight spread.
- **B. Rich history** (the final): per trophy, log earners against
  log-time over the last 36 h of change points (consecutive identical
  reads are dropped — stale stretches carry no information), slope shrunk
  toward its trophy class's mean slope (a 2-decimal rate at 0.03% is
  +/-0.17 logit of rounding noise, so rare trophies have almost no signal
  of their own). Players tracked the same way (median of two concave
  extrapolations). Extrapolate both to the target and divide. Guardrails:
  earners never decrease, never exceed players tracked, growth capped.
- **C. Flat / too few change points after the first refresh**: power-law
  player growth anchored on the documented first refresh (N = 3,868 at
  04:30 UTC — anchoring on the stale 638 under-predicts N a day later by
  ~7x), class E-growth priors, wide spread.

Spreads are honest: a floor from refresh staleness plus each trophy's fit
residual, growing with horizon; quantiles are normal on the logit scale.
No claim is made that the future rates are known — the intervals say how
unsure the forecast is, and the score (pinball loss on the logit scale)
rewards exactly that honesty.

## Local validation

| Test | This entry | House starter |
|---|---|---|
| Practice p00/p01 vs the documented-refresh answer (E frozen, N = 3,868) | **loss 0.0016**, coverage 0.98, typical miss 0.003 pp | ~1.51 |
| 24 h backtest: launch-night case -> live Exophase page 2026-11-20 03:39 UTC | **loss 0.31**, coverage 0.98, typical miss 0.07 pp | 2.32 |
| Synthetic final (2 days of 10-min reads with refresh staleness, 4.5-day horizon) | **loss 0.12**, coverage 1.00, typical miss 0.006 pp | 0.34 |

The synthetic final is a self-built world (its truth follows the same
log-time dynamics the model assumes), so it validates the machinery, not
the future. The 24 h backtest is real data: it is the only genuine
out-of-sample check available before the target, and it is where the
Regime C anchoring fix came from (first version: loss 1.30, coverage
0.13 — anchoring player growth on the stale 638; fixed by anchoring on
the documented 3,868 refresh).

## Limits (stated plainly)

- The final's answer is a single read in the hour from 2026-11-26T00:00Z;
  Exophase's refresh schedule is its own, so the answer can lag the
  target by hours. The spread prices this in; it cannot remove it.
- Earner growth over days 3-7 is extrapolated from days 1-2. If the bulk
  of players reaches mid-story content faster or slower than the fitted
  log-time trend, mid-story trophies will be off — in either direction,
  which is why the spread is wide there.
- Trophy 2 ("Just Like Paradise") is ~100% by construction; the platinum
  and grind trophies sit near the 0.01% floor, where a single earner moves
  the printed rate.
- Nothing here uses any data beyond the case file at run time (no network
  in the sandbox). The live-page snapshot above only calibrated priors
  and the validation; the program itself reads only its case file.
