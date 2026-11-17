"""Forecast GTA VI's launch-day Twitch viewers hour by hour (make.fun bounty
gta6-twitch-ccu-forecast-01). Python 3.13, standard library only, offline,
deterministic.

Model
-----
Per forecast hour h = 2..24, fit a local linear regression of the log ratio
log(viewers_h / viewers_1) on log(viewers_1) across the case file's own
comparable past launches:

    log(v_h / v_1) = a_h + b_h * log(v_1) + noise

weighted by a Gaussian kernel (bandwidth 1.5 in log space) centred on the
target's own log(hour-1 print), with the centre capped to the training range
so the fit never extrapolates beyond the largest/smallest comparable.

The median forecast is the fitted value at the target's log(v_1); the five
quantiles are normal quantiles around it with a calibrated residual spread.
The spread is the leave-one-out residual sd (see `loo_spread`): each training
point is refit without itself, so the spread is an honest out-of-sample
error estimate instead of the in-sample residual sd, which understates error
because every point pulls its own fit toward itself. The LOO sd runs ~1.08x
the in-sample sd, so its multiplier is 0.5 (0.5 x 1.08 ~= 0.54, matching the
old 0.55 x in-sample sd); the result is floored at 0.2 log units. The 0.5
factor was chosen on the four practice cases and checked against
leave-one-out over the comparables (see NOTES.md).

Training shapes are cleaned before fitting:
  - GTA VI's own pre-release category history (a comparable whose name
    contains "Grand Theft") is excluded: it is not a launch curve.
  - Early-access-contaminated launches (unlock text contains "contaminated")
    are excluded: their release-day curves are distorted because a large
    share of players already had access days before launch.
  - A launch is used only if it has a valid hour-1 print and at least 18 of
    the 23 forecast hours.

Practice-case validation (answers reconstructed from SullyGnome's public
hourly averages; hour-1 prints match the case files exactly):
  mean pinball loss 0.141535, versus 0.387 for the house persistence baseline.
  p00 0.128, p01 0.146, p02 0.115, p03 0.178; 90% coverage 0.902.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages.
Reads every <id>.in in the input directory and writes <id>.out to the output
directory: {"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]},
exactly 23 rows, hours 2..24 each once, each q five finite numbers above 0
and at most 1e9, non-decreasing, at most 65,536 bytes.

Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Local-linear kernel bandwidth, in log(viewers) units.
BANDWIDTH = 1.5
# Calibrated multiplier on the residual spread (see module docstring).
SPREAD_MULT = 0.55
# Multiplier on the leave-one-out residual spread. The LOO sd runs ~1.08x
# the in-sample residual sd, so 0.5 x 1.08 ~= 0.54, matching SPREAD_MULT's
# effective width while estimating the spread honestly per hour.
LOO_SPREAD_MULT = 0.5
# Floor / cap on the spread, in log units.
SD_FLOOR = 0.2
SD_CAP = 1.2
# Minimum training points for a local-linear fit at one hour; below this,
# fall back to the pooled median of the available log-ratios.
MIN_POINTS = 8
# Minimum forecast hours a comparable must have to be used at all.
MIN_HOURS = 18


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100 rounds it.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def norm_ppf(p: float) -> float:
    """Standard normal quantile (Acklam's approximation, ~1e-9 accurate)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def z_for_levels(levels: list[float]) -> list[float]:
    """Normal quantiles for the requested levels."""
    if levels == LEVELS:
        return list(Z)
    return [norm_ppf(p) for p in levels]


def release_day_curve(comp: dict) -> dict[int, float]:
    """hour -> viewers for the comparable's release (UTC) day."""
    day = comp["day_start"][:10]
    out: dict[int, float] = {}
    for p in comp.get("hourly") or []:
        if p.get("from", "")[:10] == day:
            v = p.get("viewers")
            if v is not None and v > 0:
                out[int(p["from"][11:13]) + 1] = v
    return out


def training_shapes(case: dict) -> list[dict]:
    """Comparable launches usable as training shapes for the shape model."""
    shapes = []
    for comp in case.get("comparables") or []:
        name = comp.get("name") or ""
        if "grand theft" in name.lower():
            continue  # GTA VI's own pre-release history: not a launch curve
        if "contaminated" in (comp.get("unlock") or "").lower():
            continue  # early-access-contaminated: distorted release-day curve
        curve = release_day_curve(comp)
        v1 = curve.get(1)
        if not v1 or v1 < 1:
            continue
        present = sum(1 for h in range(2, 25) if curve.get(h))
        if present < MIN_HOURS:
            continue
        shapes.append({"v1": v1, "curve": curve})
    return shapes


def wls_line(xs: list[float], ys: list[float], ws: list[float]):
    """Weighted least squares y ~ b0 + b1*x. Returns (b0, b1, resid_sd) or None."""
    s = sum(ws)
    if s <= 0:
        return None
    xm = sum(w * x for w, x in zip(ws, xs)) / s
    ym = sum(w * y for w, y in zip(ws, ys)) / s
    sxx = sum(w * (x - xm) ** 2 for w, x in zip(ws, xs))
    if sxx < 1e-12:
        return None
    sxy = sum(w * (x - xm) * (y - ym) for w, x, y in zip(ws, xs, ys))
    b1 = sxy / sxx
    b0 = ym - b1 * xm
    var = sum(w * (y - (b0 + b1 * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / s
    return b0, b1, math.sqrt(max(var, 0.0))


def loo_spread(pts: list[tuple[float, float]], bw: float, lx_eff: float) -> float | None:
    """Leave-one-out residual sd at one forecast hour.

    Each training point is refit without itself, with the kernel centred on
    that point itself (as if it were the target), and the squared residual is
    weighted by the point's proximity to the actual target. In-sample
    residuals understate the prediction error because every point pulls its
    own fit toward itself; LOO residuals do not have that bias, so this is
    the honest per-hour spread. Returns None if no refit succeeds.
    """
    num = 0.0
    den = 0.0
    for i, (xi, yi) in enumerate(pts):
        xs_i = [x for j, (x, _) in enumerate(pts) if j != i]
        ys_i = [y for j, (_, y) in enumerate(pts) if j != i]
        ws_i = [math.exp(-0.5 * ((x - xi) / bw) ** 2) for x in xs_i]
        fit_i = wls_line(xs_i, ys_i, ws_i)
        if fit_i is None:
            continue
        b0_i, b1_i, _ = fit_i
        w_tgt = math.exp(-0.5 * ((xi - lx_eff) / bw) ** 2)
        num += w_tgt * (yi - (b0_i + b1_i * xi)) ** 2
        den += w_tgt
    if den <= 0:
        return None
    return math.sqrt(num / den)


def hour_quantiles(shapes: list[dict], p1: float, h: int, zs: list[float]) -> list[float | int]:
    """The five quantiles for forecast hour h, given the training shapes."""
    lx = math.log(p1)
    train_lx = [math.log(s["v1"]) for s in shapes]
    if train_lx:
        lx_eff = min(max(lx, min(train_lx)), max(train_lx))
    else:
        lx_eff = lx
    pts = [
        (math.log(s["v1"]), math.log(s["curve"][h] / s["v1"]))
        for s in shapes
        if s["curve"].get(h) and s["curve"][h] > 0
    ]
    if len(pts) >= MIN_POINTS:
        ws = [math.exp(-0.5 * ((x - lx_eff) / BANDWIDTH) ** 2) for x, _ in pts]
        fit = wls_line([x for x, _ in pts], [y for _, y in pts], ws)
        if fit is not None:
            b0, b1, sd = fit
            mu = b0 + b1 * lx_eff
            sd_loo = loo_spread(pts, BANDWIDTH, lx_eff)
            sd_use = sd_loo if sd_loo is not None else sd
            mult_use = LOO_SPREAD_MULT if sd_loo is not None else SPREAD_MULT
            s_eff = min(max(sd_use * mult_use, SD_FLOOR), SD_CAP)
        else:
            mu = sum(y for _, y in pts) / len(pts)
            s_eff = min(max(0.4 * SPREAD_MULT, SD_FLOOR), SD_CAP)
    elif pts:
        ys = sorted(y for _, y in pts)
        mu = ys[len(ys) // 2]
        s_eff = min(max(0.4 * SPREAD_MULT, SD_FLOOR), SD_CAP)
    else:
        mu, s_eff = 0.0, min(max(0.4 * SPREAD_MULT, SD_FLOOR), SD_CAP)
    qs: list[float | int] = []
    for z in zs:
        q = p1 * math.exp(mu + z * s_eff)
        q = min(max(q, 1.0), 1e9)
        qs.append(round2(q))
    # Keep the row non-decreasing after rounding.
    for i in range(1, len(qs)):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def hour1_print(case: dict) -> float:
    """Hour 1's print, with fallbacks if `known` lacks it."""
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    if first is not None and first.get("viewers"):
        return max(1, first["viewers"])
    # Fall back to the most recent hourly history point before the checkpoint.
    hist = [p for p in case.get("history_hourly") or [] if p.get("viewers")]
    if hist:
        return max(1, hist[-1]["viewers"])
    known = [k for k in case.get("known") or [] if k.get("viewers")]
    if known:
        return max(1, known[-1]["viewers"])
    return 1.0


def forecast_case(case: dict, name: str) -> dict:
    levels = case.get("levels") or LEVELS
    zs = z_for_levels(levels)
    p1 = hour1_print(case)
    shapes = training_shapes(case)
    hours = [row["hour"] for row in case.get("forecast") or []] or list(range(2, 25))
    return {
        "hours": [
            {"hour": h, "q": hour_quantiles(shapes, p1, h, zs)}
            for h in sorted(hours)
        ]
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
