"""IrOx small-sample screening benchmark. Run from the repository or this directory."""

import argparse
import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut

from measurements import (
    demo_inputs, load_inputs, nested_repeatability, pooled_batch_variance, scan_trends,
    summarize_batches, validate_inputs,
)

TARGETS = ("eta_first_mv", "eta_mean_mv")
MODELS = ("ridge", "matern_gp")
SCREEN_FAMILIES = ("NaHCO3", "NH3+NaCl+KCl")
TEMPERATURE_WINDOW = (350, 500)
MIN_RECORDED_SD_MV = 0.1


def design_matrices(train, test, precursors, families):
    values = pd.concat([train, test], ignore_index=True)
    columns = []
    for field in ("temperature_c", "hold_min"):
        numeric = values[field].to_numpy(dtype=float)
        if field == "hold_min":
            numeric = np.log(numeric)
        training = numeric[:len(train)]
        columns.append((numeric - training.mean()) / max(training.std(), 1e-9))

    columns.extend(values["precursor"].eq(p).to_numpy(dtype=float) for p in precursors[1:])
    columns.append(values["naoh_g"].to_numpy(dtype=float))
    reference = "NaHCO3" if "NaHCO3" in families else families[0]
    columns.extend(values["pore_family"].eq(f).to_numpy(dtype=float)
                   for f in families if f != reference)
    for family in families:
        train_ratio = train.loc[train["pore_family"].eq(family), "pore_ratio"].dropna()
        term = np.zeros(len(values))
        if len(train_ratio) >= 2 and train_ratio.std(ddof=0) > 1e-9:
            active = values["pore_family"].eq(family) & values["pore_ratio"].notna()
            term[active] = ((values.loc[active, "pore_ratio"] - train_ratio.mean())
                            / train_ratio.std(ddof=0))
        columns.append(term)
    matrix = np.column_stack(columns)
    return matrix[:len(train)], matrix[len(train):]


def fit_models(train, test, target, precursors, families):
    x, x_test = design_matrices(train, test, precursors, families)
    y = train[target].to_numpy(dtype=float)
    floor = max(pooled_batch_variance(train), MIN_RECORDED_SD_MV**2)
    noise = np.fmax(train["eta_sd_mv"].to_numpy(dtype=float)**2, floor)
    weights = 1 / noise
    weights /= weights.mean()
    ridge = Ridge(alpha=1.0).fit(x, y, sample_weight=weights)

    center = np.average(y, weights=weights)
    scale = max(float(np.sqrt(np.average((y - center)**2, weights=weights))), 1.0)
    kernel = (
        ConstantKernel(1.0, (0.02, 30.0))
        * Matern(length_scale=1.7, length_scale_bounds=(0.2, 25.0), nu=2.5)
        + WhiteKernel(0.1, (0.0001, 2.0))
    )
    gp = GaussianProcessRegressor(
        kernel=kernel, alpha=noise / scale**2, n_restarts_optimizer=3, random_state=22,
        normalize_y=False,
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        gp.fit(x, (y - center) / scale)
    notes = [str(note.message) for note in captured
             if issubclass(note.category, ConvergenceWarning)]
    hyperparameters = {
        "matern_nu": 2.5, "length_scale": float(gp.kernel_.k1.k2.length_scale),
        "signal_variance_mv2": float(gp.kernel_.k1.k1.constant_value * scale**2),
        "white_variance_mv2": float(gp.kernel_.k2.noise_level * scale**2),
        "training_pooled_sd_mv": float(np.sqrt(floor)),
        "warnings": notes,
    }
    return {
        "ridge": ridge.predict(x_test),
        "matern_gp": gp.predict(x_test) * scale + center,
    }, hyperparameters


def evaluate(rows, scope):
    """One held-out synthesis batch per fold, with common folds across models."""
    precursors = sorted(rows["precursor"].unique())
    families = sorted(rows["pore_family"].unique())
    predictions, scores, full = [], [], {}
    for target in TARGETS:
        estimates = {model: np.empty(len(rows)) for model in MODELS}
        warned = 0
        for train, test in LeaveOneOut().split(rows):
            result, diagnostics = fit_models(rows.iloc[train], rows.iloc[test], target, precursors, families)
            warned += bool(diagnostics["warnings"])
            for model in MODELS:
                estimates[model][test] = result[model]
        observed = rows[target].to_numpy(dtype=float)
        for model, predicted in estimates.items():
            rho = None
            if np.ptp(observed) > 0 and np.ptp(predicted) > 0:
                rho = float(spearmanr(observed, predicted).statistic)
            scores.append({
                "scope": scope, "target": target, "model": model, "n_batches": len(rows),
                "r2": r2_score(observed, predicted), "rmse_mv": np.sqrt(mean_squared_error(observed, predicted)),
                "mae_mv": mean_absolute_error(observed, predicted), "spearman_rho": rho,
                "warned_folds": warned if model == "matern_gp" else 0,
            })
            predictions.extend({
                "scope": scope, "target": target, "model": model, "batch_id": key,
                "temperature_c": temperature, "observed_mv": y, "predicted_mv": prediction,
                "residual_mv": y - prediction,
            } for key, temperature, y, prediction in zip(rows["batch_id"], rows["temperature_c"], observed, predicted))
        _, full[target] = fit_models(rows, rows.iloc[:1], target, precursors, families)
    return predictions, scores, full


def matched_temperature_pairs(eligible):
    factors = ["precursor", "hold_min", "pore_family", "pore_ratio", "naoh_g", "ir_mg"]
    records = []
    fields = ["lower_batch", "higher_batch", "lower_temp_c", "higher_temp_c",
              "higher_minus_lower_first_mv", "higher_minus_lower_mean_mv"]
    for _, group in eligible.dropna(subset=["pore_ratio"]).groupby(factors):
        ordered = group.sort_values("temperature_c")
        for low, high in combinations(ordered.itertuples(index=False), 2):
            if low.temperature_c == high.temperature_c:
                continue
            records.append((
                low.batch_id, high.batch_id, low.temperature_c, high.temperature_c,
                high.eta_first_mv - low.eta_first_mv, high.eta_mean_mv - low.eta_mean_mv,
            ))
    return pd.DataFrame(records, columns=fields)


def benchmark(designs, scans, dataset_kind):
    designs, scans = validate_inputs(designs, scans)
    batches = summarize_batches(designs, scans)
    syntheses = batches.loc[batches["role"].eq("synthesis")].reset_index(drop=True)
    windows = syntheses.loc[syntheses["temperature_c"].between(*TEMPERATURE_WINDOW)]
    scope_data = {"historical": syntheses, "temperature_window": windows}
    report = {
        "dataset_kind": dataset_kind,
        "counts": {"synthesis_batches": len(syntheses), "control_batches": len(batches) - len(syntheses),
                   "scans": len(scans), "sessions": int(scans.groupby(["batch_id", "session_id"]).ngroups)},
        "repeatability": {
            "all": nested_repeatability(designs, scans),
            "synthesis": nested_repeatability(designs.loc[designs["role"].eq("synthesis")],
                                             scans.loc[scans["batch_id"].isin(syntheses["batch_id"])]),
        },
        "gp_full_fit": {}, "validation_status": {},
        "scope": {"temperature_c": list(TEMPERATURE_WINDOW), "screened_pore_families": list(SCREEN_FAMILIES)},
        "intended_use": "retrospective screening and model validation",
    }
    predictions, metrics = [], []
    for scope, rows in scope_data.items():
        if len(rows) < 4 or rows["n_scans"].gt(1).sum() < 2:
            report["validation_status"][scope] = "insufficient batches or repeats"
            continue
        results, scores, hyperparameters = evaluate(rows, scope)
        report["validation_status"][scope] = "evaluated"
        report["gp_full_fit"][scope] = hyperparameters
        predictions.extend(results)
        metrics.extend(scores)
    if not predictions:
        raise ValueError("Need at least four synthesis batches and two replicated batches for validation")

    eligible = windows.loc[windows["pore_family"].isin(SCREEN_FAMILIES)].sort_values(
        ["eta_first_mv", "batch_id"]
    )
    report["counts"]["screened_batches"] = len(eligible)
    return report, {
        "metrics": pd.DataFrame(metrics), "loo_predictions": pd.DataFrame(predictions),
        "batch_summaries": batches, "scan_trends": scan_trends(scans),
        "screened_observations": eligible, "matched_temperature_pairs": matched_temperature_pairs(eligible),
    }


def write_report(report, tables, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(destination / f"{name}.csv", index=False, float_format="%.8g")
    (destination / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=Path, help="CSV of independent syntheses and controls")
    parser.add_argument("--scans", type=Path, help="CSV of annotated per-scan eta10 measurements")
    parser.add_argument("--output", type=Path, default=Path("irox_output"))
    args = parser.parse_args()
    if bool(args.batches) != bool(args.scans):
        parser.error("--batches and --scans must be supplied together")
    source = "user_supplied" if args.batches else "synthetic_demo"
    try:
        inputs = load_inputs(args.batches, args.scans) if args.batches else demo_inputs()
        report, tables = benchmark(*inputs, dataset_kind=source)
        write_report(report, tables, args.output)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"{source}: {report['counts']}")
    print(tables["metrics"].round(3).to_string(index=False))
    print(f"Output written to {args.output}")


if __name__ == "__main__":
    main()
