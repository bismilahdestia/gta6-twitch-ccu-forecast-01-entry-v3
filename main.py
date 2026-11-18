"""gta6-twitch-ccu-forecast-01 entry: comparable-shape forecaster (v3).

Forecasts GTA VI's launch-day Twitch viewers hour by hour (hours 2..24) from
the case's own comparables: past big console/PC launches' hourly Twitch
curves (SullyGnome hourly category averages), each normalised by its hour-1
print into a log-shape, plus the case's hour-1 print.

v3 changes vs v2 (practice 0.136 -> 0.129; leave-one-out 0.634 -> 0.632;
console-only GTA-like subset 0.205 -> 0.201; rolling 0.237 -> 0.232;
big hour-1 0.237 -> 0.232):
1. The MEDIAN component now uses a CONSOLE-ONLY pool. PC launches
   (Baldur's Gate 3, Black Myth, Silksong, ...) have systematically
   different shapes -- huge simultaneous-unlock ramps -- that bias the
   all-pool median upward at mid/late hours; for a console target the
   console median is the right centre. This alone moves practice
   0.136 -> 0.129, improving all four practice cases. The regression
   component still uses the full comparable pool (its slope needs the
   PC launches' range of hour-1 prints). If fewer than 5 console
   comparables survive, the median falls back to the full pool.
2. Blend weight: wp -0.4 -> -0.3 (slightly more regression weight at huge
   hour-1 prints) and the hour groups moved to <=4 / 5-10 / >=11 (the
   late group now starts at hour 11, where the noon dip ends and the
   evening ramp begins).
3. Median spread is now three-group: k=0.7 early (hours <= 6), 0.5 mid
   (7-13), 0.25 late (>= 14), plus +0.15 at hours 11-13 (the noon dip
   window, where Spider-Man 2's idiosyncratic -1.00 log dip lives).

Two components, blended in log space (geometric mean of quantiles):
1. Regression component: per hour, weighted OLS of log-shape on log(hour-1
   print) across the pool. Launch curves scale sublinearly with their
   hour-1 print: across the comparables, log(p1) vs log(peak ratio)
   correlates at about -0.86 -- a launch whose hour 1 is already huge
   (875k, ELDEN RING) peaks at ~1.00x its hour 1, while one whose hour 1
   is small (23k, Marvel's Wolverine) ramps to ~9x. The prediction is
   clamped to the pool's log(p1) range. Spread = residual std of the fit.
2. Median-shape component: per hour, the (weighted) median log-shape
   across the CONSOLE pool with an IQR-based sigma. Robust to extreme shapes.

Pool filter (both components): tier "comparable" only (the comparable's
unlock text carries "Tier: comparable"; early-access-contaminated and weak
launches are excluded), and hour-1 print >= 3000 (a launch whose hour-1
print is below that had not started by hour 1 -- simultaneous unlock later
in the day, e.g. Silksong 164 viewers at hour 1 ramping 2120x -- a
different regime). The Grand Theft Auto VI category's own pre-release
history is not a launch and is excluded. If fewer than 5 comparables
survive, falls back to the house persistence baseline.

Backtest (local, against SullyGnome answers extracted for the 4 practice
cases + a 37-launch leave-one-out over the comparables):
  practice loss 0.129 (house baseline 0.387), per case 0.12/0.13/0.09/0.18;
  leave-one-out 0.632 overall, 0.201 on the console-only GTA-like subset,
  0.232 rolling, 0.232 big hour-1. Coverage 90% = 0.967.
  Known limits: Spider-Man 2 (p03, 0.175) still dominates the practice
  loss -- its h12 dip (-1.00 log) is idiosyncratic and no pool feature
  predicts which launches dip at noon. Small hour-1 console targets
  (Ghost of Yotei / Stellar Blade, 27k) remain the weakest LOO subset:
  their modest 2-3.5x ramps sit far below the regression's small-p1
  prediction, and nothing in hour 1 separates them from Wolverine's 9x
  ramp at the same hour-1 print. The 0.122 target was not reached;
  closing the remaining gap would need the p03 dip to be predictable,
  and it is not, from hour 1 alone.

Pure standard library, deterministic (no randomness, pool sorted by name),
no network, no model calls. Runtime is milliseconds per case.
"""

import json
import math
import os
import re
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# --- model constants (tuned on the practice cases + comparable LOO backtest)
P1_MIN = 3000        # comparables with a smaller hour-1 print are excluded
REG_K = 0.50         # spread multiplier, regression component
MED_K_EARLY = 0.70   # spread multiplier, median component, hours <= 6
MED_K_MID = 0.50     # spread multiplier, median component, hours 7-13
MED_K_LATE = 0.25    # spread multiplier, median component, hours >= 14
MED_K_MIDBUMP = 0.15  # extra median spread at hours 11-13 (noon dip window)
SIGMA_FLOOR = 0.05   # minimum log-spread per hour
MIN_POOL = 5         # below this, fall back to the persistence baseline
BASE_SIGMA = 0.15    # persistence baseline spread (house starter)
BLEND_W0 = 0.70      # base weight of the regression component in the log blend
BLEND_WH = 0.20      # extra reg weight at late hours
BLEND_WP = -0.30     # reg-weight reduction for very large hour-1 prints
P1_FRAC_LO = 11.5    # log(p1) below this: no p1 adjustment
P1_FRAC_HI = 13.0    # log(p1) above this: full p1 adjustment
HF_EARLY_END = 4     # hour groups for the blend: <=4 / 5-10 / >=11
HF_MID_END = 10


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def wpercentile(pairs, q):
    """Weighted percentile of [(value, weight)] pairs, sorted by value."""
    pairs = sorted(pairs)
    tot = sum(w for _, w in pairs)
    if tot <= 0:
        return 0.0
    thr = q * tot
    acc = 0.0
    for i, (v, w) in enumerate(pairs):
        acc += w
        if acc >= thr:
            if i == 0:
                return v
            pv, _ = pairs[i - 1]
            frac = (thr - (acc - w)) / w if w > 0 else 0.0
            return pv + (v - pv) * frac
    return pairs[-1][0]


def release_window(comp):
    """The comparable's release-day 24 hourly viewer counts, or None.

    The window starts at the hour whose `from` equals the comparable's
    day_start; all 24 values must be present and positive.
    """
    day_start = comp.get("day_start")
    if not day_start:
        return None
    hourly = comp.get("hourly") or []
    for i, row in enumerate(hourly):
        if row.get("from") == day_start:
            win = hourly[i:i + 24]
            if len(win) < 24:
                return None
            vals = [r.get("viewers") for r in win]
            if any(v is None or v <= 0 for v in vals):
                return None
            return vals
    return None


def tier_of(comp):
    m = re.search(r"Tier:\s*([a-z-]+)", comp.get("unlock", "") or "")
    return m.group(1) if m else "unknown"


def is_console_only(comp):
    return "PC" not in (comp.get("platforms") or "")


def pool_entries(comparables, console_only=False):
    """(p1, log-shape[23]) for each comparable that passes the pool filter.

    With console_only=True, only console-only launches are kept (used for
    the median component: PC launches have systematically different shapes).
    """
    entries = []
    for comp in sorted(comparables, key=lambda c: c.get("name", "")):
        if comp.get("name", "").startswith("Grand Theft Auto VI"):
            continue  # the target's own pre-release history, not a launch
        if tier_of(comp) != "comparable":
            continue  # early-access-contaminated or weak launches
        if console_only and not is_console_only(comp):
            continue
        win = release_window(comp)
        if win is None:
            continue
        if win[0] < P1_MIN:
            continue  # launch had not started by hour 1: different regime
        p1 = win[0]
        shapes = [math.log(v / p1) for v in win[1:]]
        entries.append((p1, shapes))
    return entries


def wols(xs, ys, ws):
    """Weighted least squares y = a + b*x; returns (a, b, residual std)."""
    tot = sum(ws)
    mx = sum(w * x for x, w in zip(xs, ws)) / tot
    my = sum(w * y for y, w in zip(ys, ws)) / tot
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx == 0:
        return my, 0.0, 0.0
    b = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws)) / sxx
    a = my - b * mx
    var = sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / tot
    return a, b, math.sqrt(var)


def blend_weight(hour, p1):
    """Regression-component weight: higher late in the day, lower for very
    large hour-1 prints (where the regression is dragged flat by
    simultaneous launches)."""
    hf = 0.0 if hour <= HF_EARLY_END else (0.5 if hour <= HF_MID_END else 1.0)
    logp = math.log(max(1, p1))
    if logp <= P1_FRAC_LO:
        pf = 0.0
    elif logp >= P1_FRAC_HI:
        pf = 1.0
    else:
        pf = (logp - P1_FRAC_LO) / (P1_FRAC_HI - P1_FRAC_LO)
    w = BLEND_W0 + BLEND_WH * hf + BLEND_WP * pf
    return min(0.95, max(0.05, w))


def med_k(hour):
    """Median-component spread multiplier: wide early, narrow late, with a
    bump at the noon-dip window (hours 11-13)."""
    if hour <= 6:
        k = MED_K_EARLY
    elif hour <= 13:
        k = MED_K_MID
    else:
        k = MED_K_LATE
    if 11 <= hour <= 13:
        k += MED_K_MIDBUMP
    return k


def regression_hours(p1, entries):
    """Per-hour OLS of log-shape on log(p1); returns {hour: [5 quantiles]}."""
    logp = [math.log(v) for v, _ in entries]
    lo, hi = min(logp), max(logp)
    x = min(max(math.log(max(1, p1)), lo), hi)  # clamp to the pool's range
    ws = [1.0] * len(entries)
    out = {}
    for hi_ in range(23):
        ys = [shapes[hi_] for _, shapes in entries]
        a, b, sig = wols(logp, ys, ws)
        sig = max(SIGMA_FLOOR, sig)
        med = a + b * x
        h = hi_ + 2
        out[h] = [p1 * math.exp(med + z * sig * REG_K) for z in Z]
    return out


def median_hours(p1, entries):
    """Per-hour median log-shape + IQR sigma; returns {hour: [5 quantiles]}."""
    out = {}
    for hi_ in range(23):
        pairs = [(shapes[hi_], 1.0) for _, shapes in entries]
        med = wpercentile(pairs, 0.5)
        sig = max(SIGMA_FLOOR, (wpercentile(pairs, 0.75) - wpercentile(pairs, 0.25)) / 1.349)
        h = hi_ + 2
        out[h] = [p1 * math.exp(med + z * sig * med_k(h)) for z in Z]
    return out


def persistence_hours(p1):
    """House baseline fallback: every hour stays at the hour-1 print."""
    out = {}
    for h in range(2, 25):
        out[h] = [p1 * math.exp(z * BASE_SIGMA * math.sqrt(h - 1)) for z in Z]
    return out


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    entries = pool_entries(case.get("comparables") or [])
    if len(entries) < MIN_POOL:
        hours = persistence_hours(p1)
    else:
        reg = regression_hours(p1, entries)
        # median component: console-only pool (falls back to the full pool
        # if too few console comparables survive)
        med_entries = pool_entries(case.get("comparables") or [], console_only=True)
        if len(med_entries) < MIN_POOL:
            med_entries = entries
        med = median_hours(p1, med_entries)
        hours = {}
        for h in range(2, 25):
            w = blend_weight(h, p1)
            hours[h] = [
                math.exp(w * math.log(reg[h][i]) + (1 - w) * math.log(med[h][i]))
                for i in range(5)
            ]
    rows = []
    for row in case["forecast"]:
        h = row["hour"]
        qs = []
        prev = 0.0
        for q in hours[h]:
            v = min(1e9, max(1.0, round2(q)))
            if v < prev:  # keep quantiles non-decreasing after rounding
                v = prev
            qs.append(v)
            prev = v
        rows.append({"hour": h, "q": qs})
    return {"hours": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
