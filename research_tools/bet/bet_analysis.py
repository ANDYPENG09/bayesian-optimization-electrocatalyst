"""Deterministic BET analysis of standardized nitrogen adsorption CSVs."""
import argparse
import csv
import json
import math
from collections import defaultdict
from statistics import mean

AREA_FACTOR = 4.3535
LIQUID_FACTOR = 0.0015468


def bet(points, low=0.05, high=0.30):
    """Fit measured adsorption points only; never interpolate."""
    selected = sorted((p, v) for p, v in points if low <= p <= high)
    if len(selected) < 3:
        raise ValueError("At least three distinct measured pressures required")
    if len({p for p, _ in selected}) != len(selected):
        raise ValueError("Duplicate pressures in fitting interval")
    x = [p for p, _ in selected]
    y = [p / (v * (1 - p)) for p, v in selected]
    xm, ym = mean(x), mean(y)
    s = sum((a-xm)*(b-ym) for a,b in zip(x,y)) / sum((a-xm)**2 for a in x)
    b = ym - s*xm
    ss = sum((z-ym)**2 for z in y)
    r2 = 1 - sum((z-s*p-b)**2 for p,z in zip(x,y))/ss if ss else None
    c = 1+s/b if b else None
    vm = 1/(s+b) if s+b > 0 else None
    mono = 1/(1+math.sqrt(c)) if c is not None and c > 0 else None
    vals = [v*(1-p) for p,v in selected]
    flags = []
    if b <= 0 or c is None or c <= 0 or vm is None:
        flags.append("nonphysical_bet")
    if c is None or c < 50:
        flags.append("C_below_50_or_undefined")
    if r2 is None or r2 < 0.9995:
        flags.append("R2_below_threshold_or_undefined")
    if any(a > z for a,z in zip(vals, vals[1:])):
        flags.append("rouquerol_monotonicity_failed")
    if mono is None or not x[0] <= mono <= x[-1]:
        flags.append("monolayer_pressure_outside_fit")
    return dict(n_points=len(x), pressure_points=x, slope=s, intercept=b,
                C=c, Vm_cm3_stp_g=vm, S_m2_g=AREA_FACTOR*vm if vm else None,
                R2=r2, monolayer_p_rel=mono, flags=flags,
                fit_status="FAIL" if "nonphysical_bet" in flags else "WARNING" if flags else "PASS")


def analyze(path):
    groups = defaultdict(list)
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            mid = row["measurement_id"].strip()
            branch = row["branch"].strip().lower()
            p, v = float(row["p_rel"]), float(row["v_cm3_g_stp"])
            if not mid or branch not in {"adsorption", "desorption"} or not (math.isfinite(p) and math.isfinite(v) and 0 < p < 1 and v >= 0):
                raise ValueError("Invalid measurement ID, branch, pressure or uptake")
            groups.setdefault(mid, [])
            if branch == "adsorption":
                groups[mid].append((p, v))
    result = {}
    for mid, points in sorted(groups.items()):
        try:
            if any(v <= 0 for p,v in points if 0.05 <= p <= 0.30):
                raise ValueError("Nonpositive uptake in fitting interval")
            r = bet(points)
        except ValueError as exc:
            result[mid] = {"fit_status": "FAIL", "error": str(exc)}
            continue
        exact = [v for p,v in points if p == 0.95]
        r["V_liq_exact_p95_cm3_g"] = exact[0]*LIQUID_FACTOR if len(exact) == 1 else None
        q = r["V_liq_exact_p95_cm3_g"]
        area = r["S_m2_g"]
        r["D_4V_S_nm"] = 4000*q/area if q is not None and area and r["fit_status"] != "FAIL" else None
        r["p95_status"] = "measured" if len(exact) == 1 else "missing_or_ambiguous"
        result[mid] = r
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", help="Standardized isotherm CSV")
    args = parser.parse_args()
    print(json.dumps(analyze(args.csv), indent=2, allow_nan=False))
