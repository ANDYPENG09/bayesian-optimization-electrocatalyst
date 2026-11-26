"""Validated scan summaries and a nested repeatability model for the IrOx case."""

from itertools import permutations

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve, null_space
from scipy.optimize import minimize
from scipy.stats import rankdata, spearmanr

DESIGN_COLUMNS = (
    "batch_id", "role", "precursor", "temperature_c", "hold_min", "pore_family",
    "pore_ratio", "naoh_g", "ir_mg",
)
SCAN_COLUMNS = ("batch_id", "scan_order", "session_id", "eta10_mv")


def load_inputs(batch_path, scan_path):
    """Read IDs and empty fields without CSV type inference."""
    return tuple(pd.read_csv(path, dtype=str, keep_default_na=False)
                 for path in (batch_path, scan_path))


def validate_inputs(designs, scans):
    """Return normalized copies. Session IDs are local to a batch."""
    frames = []
    for frame, fields in ((designs, DESIGN_COLUMNS), (scans, SCAN_COLUMNS)):
        missing = set(fields).difference(frame.columns)
        if missing:
            raise ValueError(f"Missing input columns: {', '.join(sorted(missing))}")
        frame = frame.loc[:, list(fields)].copy()
        for col in fields:
            frame[col] = frame[col].astype("string").str.strip().fillna("")
        frames.append(frame)
    designs, scans = frames

    if designs["batch_id"].eq("").any() or designs["batch_id"].duplicated().any():
        raise ValueError("Designs require unique nonempty batch_id values")
    if not designs["role"].isin(["synthesis", "control"]).all():
        raise ValueError("role must be synthesis or control")
    if set(scans["batch_id"]) != set(designs["batch_id"]):
        raise ValueError("The scan and design inputs must contain the same batch IDs")
    if scans["session_id"].eq("").any():
        raise ValueError("Every scan requires a session_id")

    for col in ("temperature_c", "hold_min", "pore_ratio", "naoh_g", "ir_mg"):
        measured = designs[col].ne("")
        designs[col] = pd.to_numeric(designs[col].where(measured), errors="raise").astype(float)
        if not np.isfinite(designs.loc[measured, col]).all():
            raise ValueError(f"{col} must be finite when supplied")
    synthesis = designs.loc[designs["role"].eq("synthesis")]
    required = ["temperature_c", "hold_min", "naoh_g", "ir_mg"]
    if synthesis[required].isna().any().any():
        raise ValueError("Synthesis rows need temperature_c, hold_min, naoh_g and ir_mg")
    if synthesis[["precursor", "pore_family"]].eq("").any().any():
        raise ValueError("Synthesis rows need precursor and pore_family")
    if synthesis[["hold_min", "ir_mg"]].le(0).any().any():
        raise ValueError("hold_min and ir_mg must be positive")
    if synthesis[["naoh_g", "pore_ratio"]].lt(0).any().any():
        raise ValueError("naoh_g and pore_ratio must be nonnegative")

    for col in ("scan_order", "eta10_mv"):
        scans[col] = pd.to_numeric(scans[col], errors="raise")
        if not np.isfinite(scans[col].to_numpy(dtype=float)).all():
            raise ValueError(f"{col} must be finite")
    scans = scans.sort_values(["batch_id", "scan_order"]).reset_index(drop=True)
    if not scans["scan_order"].eq(scans.groupby("batch_id").cumcount() + 1).all():
        raise ValueError("scan_order must be consecutive from 1 within each batch")
    session_starts = scans.loc[
        scans["session_id"].ne(scans.groupby("batch_id")["session_id"].shift().fillna(""))
    ]
    if session_starts.duplicated(["batch_id", "session_id"]).any():
        raise ValueError("A session must occupy consecutive scans within its batch")
    return designs, scans


def summarize_batches(designs, scans):
    response = scans.groupby("batch_id")["eta10_mv"].agg(
        eta_first_mv="first", eta_mean_mv="mean", eta_sd_mv="std", n_scans="size"
    )
    return designs.merge(response, on="batch_id", validate="one_to_one")


def pooled_batch_variance(rows):
    repeats = rows.loc[rows["n_scans"].gt(1)]
    degrees = repeats["n_scans"].to_numpy() - 1
    if not degrees.size:
        raise ValueError("A training fold needs replicated batches to estimate a noise floor")
    return float(np.average(repeats["eta_sd_mv"] ** 2, weights=degrees))


def scan_trends(scans):
    """Whole-batch Spearman tests are descriptive when a batch spans sessions."""
    records = []
    for key, rows in scans.groupby("batch_id", sort=False):
        values = rows["eta10_mv"].to_numpy(dtype=float)
        difference = np.diff(values)
        record = {
            "batch_id": key, "n_scans": len(values), "n_sessions": rows["session_id"].nunique(),
            "change_mv": None, "pattern": "single_scan", "p_spearman": None,
            "test": "insufficient", "descriptive_only": rows["session_id"].nunique() > 1,
        }
        if difference.size:
            record["change_mv"] = values[-1] - values[0]
            pattern = "mixed_or_tied"
            if (difference < 0).all():
                pattern = "decreasing"
            elif (difference > 0).all():
                pattern = "increasing"
            elif (difference == 0).all():
                pattern = "constant"
            record["pattern"] = pattern
            if pattern != "constant" and len(values) <= 8:
                ranks = rankdata(values)
                order = np.arange(len(values)) - (len(values) - 1) / 2
                observed = abs(ranks @ order)
                distribution = np.asarray(list(permutations(ranks))) @ order
                record["p_spearman"] = float(np.mean(np.abs(distribution) >= observed - 1e-10))
                record["test"] = "exact_permutation"
            elif pattern != "constant":
                record["p_spearman"] = float(spearmanr(values, rows["scan_order"]).pvalue)
                record["test"] = "asymptotic"
        records.append(record)
    return pd.DataFrame(records)


def nested_repeatability(designs, scans):
    """REML with random batch/session intercepts and a role fixed effect."""
    frame = scans.merge(designs[["batch_id", "role"]], on="batch_id", validate="many_to_one")
    batch = pd.get_dummies(frame["batch_id"]).to_numpy(dtype=float)
    session = pd.get_dummies(frame.groupby(["batch_id", "session_id"]).ngroup()).to_numpy(dtype=float)
    fixed = [np.ones(len(frame))]
    if frame["role"].nunique() > 1:
        fixed.append(frame["role"].eq("control").to_numpy(dtype=float))
    df = {
        "same_session": len(frame) - session.shape[1],
        "across_session": session.shape[1] - batch.shape[1],
        "between_batch": batch.shape[1] - len(fixed),
    }
    result = {"degrees_of_freedom": df, "status": "insufficient", "components": {}}
    if min(df.values()) < 1:
        return result
    contrasts = null_space(np.vstack(fixed)).T
    z = contrasts @ frame["eta10_mv"].to_numpy()
    if np.linalg.norm(z) < 1e-10:
        return result
    batch_code, session_code = contrasts @ batch, contrasts @ session
    batch_cov = batch_code @ batch_code.T
    session_cov = session_code @ session_code.T
    identity = np.eye(z.size)

    def likelihood(log_ratios, output=False):
        r_batch, r_session = np.exp(log_ratios)
        cov = identity + r_batch * batch_cov + r_session * session_cov
        factor = cho_factor(cov, lower=True, check_finite=False)
        scale = float(z @ cho_solve(factor, z, check_finite=False)) / z.size
        if output:
            return np.array([scale, r_session * scale, r_batch * scale])
        return 2 * np.log(np.diag(factor[0])).sum() + z.size * np.log(scale)

    fits = [minimize(likelihood, start, method="L-BFGS-B", bounds=[(-18, 12)] * 2)
            for start in [(2.0, 0.0), (0.0, -5.0)]]
    successful = [fit for fit in fits if fit.success]
    if not successful:
        result["status"] = "fit_warning"
        return result
    best = min(successful, key=lambda fit: fit.fun)
    variances = likelihood(best.x, output=True)
    result["status"] = "estimated"
    result["components"] = {
        name: {"variance_mv2": float(v), "sd_mv": float(np.sqrt(v)),
               "percent": float(100 * v / variances.sum())}
        for name, v in zip(df, variances)
    }
    return result


def demo_inputs():
    """Seeded synthetic values drawn independently of the industrial case."""
    rng = np.random.default_rng(2026)
    designs, scans = [], []
    for index in range(19):
        batch_id = f"demo-{index + 1:02d}"
        control = index >= 16
        temperature = [350, 400, 450, 500, 550][index % 5]
        minutes = [25, 40, 55][(index // 2) % 3]
        precursor = ["precursor-A", "precursor-B"][(index // 3) % 2]
        family = ["NaHCO3", "NH3+NaCl+KCl"][(index // 4) % 2]
        ratio = float(rng.uniform(11, 46))
        naoh = float(index % 2)
        role = "control" if control else "synthesis"
        designs.append([batch_id, role, precursor, temperature, minutes, family, ratio, naoh, 150])
        mean = 320 + 0.16 * (temperature - 440) + 0.3 * (minutes - 40)
        mean += 12 * (precursor == "precursor-B") - 6 * naoh
        mean += (ratio - 28) * (0.2 if family == "NaHCO3" else -0.4)
        mean += rng.normal(0, 13)
        drift = rng.normal(1, 4)
        n_scans = 1 if index % 7 == 0 else 3
        second_session = rng.normal(0, 8)
        for scan in range(n_scans):
            new_session = scan == 2 and index % 3 == 0
            value = mean + scan * drift + new_session * second_session + rng.normal(0, 5)
            scans.append([batch_id, scan + 1, "B" if new_session else "A", round(value, 1)])
    return pd.DataFrame(designs, columns=DESIGN_COLUMNS), pd.DataFrame(scans, columns=SCAN_COLUMNS)
