#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yu Peng
"""
Bayesian Optimization closed-loop pipeline for electrocatalyst development.

Grounded in Roman Garnett, "Bayesian Optimization" (Cambridge Univ. Press):
  - GP surrogate (Ch.2-3), Matern 5/2 kernel (off-the-shelf, §3.3)
  - Acquisition: EI (§7.3/8.2), PI (§7.5/8.3), UCB (§7.8/8.4)
  - Hyperparameter: marginal-likelihood MAP (§4.3) via skopt GP
  - Acquisition optimization (§9.2): skopt's global optimizer
  - Constrained (§11.2), batch (§11.3), multiobjective (§11.7) via scalarization/EHVI-lite

Mirrors the user's prior bo_ptco_round1.py pattern (skopt GP+EI, PtCo ordering degree S).
Run:  python bo_pipeline.py --data assets/experiment_template.csv --config assets/config.yaml
"""
from __future__ import annotations
import argparse, json, sys, os, math, warnings
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm as _norm  # vectorized pdf/cdf (Garnett Φ, φ)

# skopt gives a GP surrogate with Matern + marginal-likelihood hyperparameter fit
try:
    from skopt import Optimizer
    from skopt.space import Real, Integer, Categorical
    from skopt.acquisition import gaussian_ei, gaussian_pi
    HAS_SKOPT = True
except Exception:  # pragma: no cover
    HAS_SKOPT = False

# make sibling scripts importable when run as `python scripts/bo_pipeline.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_utils import TraceRecorder


# ----------------------------- data I/O -----------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # tolerate trailing whitespace in headers
    df.columns = [c.strip() for c in df.columns]
    return df


def build_space(config: dict):
    """Build skopt search space from config['variables']."""
    space = []
    for v in config["variables"]:
        t = v.get("type", "real")
        if t == "real":
            space.append(Real(low=float(v["low"]), high=float(v["high"]), name=v["name"]))
        elif t == "int":
            space.append(Integer(low=int(v["low"]), high=int(v["high"]), name=v["name"]))
        elif t == "cat":
            space.append(Categorical(categories=v["categories"], name=v["name"]))
        else:
            raise ValueError(f"unknown var type {t}")
    return space


def objective_vector(df: pd.DataFrame, config: dict) -> np.ndarray:
    """Return maximization-direction objective values per row.

    Metrics flagged direction='min' are negated (Garnett maximizes utility).
    Supports scalarized multiobjective via config['objective']['weights'].
    """
    obj = config["objective"]
    ys = np.zeros(len(df))
    for name, spec in obj["metrics"].items():
        col = df[name].to_numpy(dtype=float)
        if spec.get("direction") == "min":
            col = -col
        # normalize by provided scale to make metrics commensurable
        scale = float(spec.get("scale", 1.0))
        if scale <= 0:
            scale = np.std(col) if np.std(col) > 0 else 1.0
        ys += float(spec.get("weight", 1.0)) * (col / scale)
    return ys


def constraint_values(df: pd.DataFrame, config: dict) -> Optional[np.ndarray]:
    """Return constraint feasibility: g(x)<=0 -> feasible. NaN if no constraints."""
    cons = config.get("constraints") or []
    if not cons:
        return None
    G = np.zeros((len(df), len(cons)))
    for j, c in enumerate(cons):
        col = df[c["name"]].to_numpy(dtype=float)
        # g = value - threshold (<=0 feasible)
        thr = float(c["threshold"])
        if c.get("direction") == "ge":  # value >= threshold  -> g = threshold - value
            G[:, j] = thr - col
        else:  # value <= threshold -> g = value - threshold
            G[:, j] = col - thr
    feasible = (G <= 0).all(axis=1)
    return feasible


# ----------------------------- surrogate -----------------------------
def fit_gp(space, X, y, noise: float | None = None, random_state: int = 0) -> "Optimizer":
    """Fit a skopt GP surrogate (Matern, ARD length-scales via MLE).

    Implements the posterior of Garnett §2.2 and hyperparameter MAP of §4.3.
    """
    if not HAS_SKOPT:
        raise RuntimeError(
            "scikit-optimize not installed. Install via: "
            "pip install scikit-optimize"
        )
    kwargs = dict(
        dimensions=space,
        base_estimator="GP",
        acq_func="EI",
        initial_point_generator="random",
        n_initial_points=0,  # fit the GP immediately on the supplied history
        random_state=random_state,
    )
    opt = Optimizer(**kwargs)
    # tell history so the GP posterior conditions on all prior data (Garnett 2.2)
    for xi, yi in zip(X, y):
        opt.tell(xi.tolist() if hasattr(xi, "tolist") else xi, float(yi))
    # skopt fits the surrogate lazily; force one ask() so opt.models is populated
    if not opt.models:
        opt.ask()
    return opt


# ----------------------------- acquisition -----------------------------
def acquisition(opt: "Optimizer", acq: str, beta: float = 2.0,
                xi: float = 0.01, n_samp: int = 4096, seed: int = 0):
    """Compute acquisition over a candidate pool and return argmax.

    Implements EI (Garnett 8.9), PI (7.6/8.22), UCB (8.24/8.25).
    Candidate pool is concentrated near observed points (Garnett §9.2: the
    posterior degenerates to the prior — and gradients vanish — far from
    data), with a smaller LHS exploration share to avoid getting stuck.
    """
    if not HAS_SKOPT:
        raise RuntimeError("scikit-optimize required for acquisition")
    rng = np.random.default_rng(seed)
    # box bounds for Real dimensions (this skill's variables are all real)
    box = np.array([[d.bounds[0], d.bounds[1]] for d in opt.space.dimensions],
                   dtype=float)
    spans = box[:, 1] - box[:, 0]
    X_obs = np.asarray(opt.Xi, dtype=float).reshape(-1, box.shape[0]) \
        if opt.Xi else None

    n_nbr = int(0.7 * n_samp) if X_obs is not None else 0
    n_lhs = n_samp - n_nbr
    d = box.shape[0]  # number of variables (box is (n_dims, 2): low/high)
    cand_parts = []
    if n_nbr:
        idx = rng.integers(0, len(X_obs), size=n_nbr)
        jitter = rng.normal(0, 1, size=(n_nbr, d)) * (0.15 * spans)
        nbr = np.clip(X_obs[idx] + jitter, box[:, 0], box[:, 1])
        cand_parts.append(nbr)
    if n_lhs:
        lhs = rng.uniform(box[:, 0], box[:, 1], size=(n_lhs, d))
        cand_parts.append(lhs)
    cand = np.vstack(cand_parts)

    # GP posterior predictive (mean, std) at candidates
    model = opt.models[-1] if opt.models else None
    if model is None:
        raise RuntimeError("GP model not fit yet")
    Xs = opt.space.transform(cand.tolist())
    mu, std = model.predict(Xs, return_std=True)
    tau = float(np.max(opt.yi)) if opt.yi else float(mu.max())

    z = (mu - tau - xi) / np.where(std > 0, std, 1e-9)
    if acq == "EI":
        phi = _norm.pdf(z)
        Phi = _norm.cdf(z)
        a = (mu - tau - xi) * Phi + std * phi                      # (8.9)
    elif acq == "PI":
        a = _norm.cdf(z)                                          # (8.22)
    elif acq == "UCB":
        a = mu + beta * std                                        # (8.24)
    else:
        raise ValueError(f"unknown acquisition {acq}")
    best = int(np.argmax(a))
    # map candidate back to original space point
    next_x = [cand[best, k] for k in range(cand.shape[1])]
    return next_x, float(a[best]), float(mu[best]), float(std[best])


# ----------------------------- convergence -----------------------------
@dataclass
class Convergence:
    n_obs: int
    best_y: float
    incumbents: list
    ei_value: float
    improvement_last_k: float
    converged: bool
    reason: str


def assess_convergence(history_y: list, ei_value: float,
                        k: int = 3, ei_tol: float = 1e-3,
                        rel_tol: float = 1e-3) -> Convergence:
    incumbents = list(np.maximum.accumulate(history_y))
    best_y = incumbents[-1] if incumbents else float("nan")
    last = incumbents[-1 - k:] if len(incumbents) > k else incumbents
    imp_last_k = (last[-1] - last[0]) / (abs(last[0]) + 1e-12)
    converged = False
    reason = "running"
    if len(history_y) > k and abs(imp_last_k) < rel_tol:
        converged, reason = True, f"no incumbent gain in last {k} rounds"
    elif abs(ei_value) < ei_tol:
        converged, reason = True, f"EI below tol ({ei_value:.2e})"
    return Convergence(len(history_y), best_y, incumbents, ei_value,
                       imp_last_k, converged, reason)


# ----------------------------- main loop -----------------------------
def run(data_csv: str, config_path: str, acq: str = "EI",
        beta: float = 2.0, xi: float = 0.01, batch: int = 1,
        out_json: str = "next_experiment.json",
        trace_path: str = "trace.json", emit_trace: bool = True,
        seed: int = 0):
    tr = TraceRecorder()
    with tr.span("run", input_summary=f"data={data_csv}, config={config_path}, acq={acq}, batch={batch}"):
        # ---- config ----
        with tr.span("load_config", input_summary=config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
            tr.record_output(f"{len(config.get('variables', []))} vars; "
                              f"obj={list(config.get('objective', {}).get('metrics', {}).keys())}")

        # ---- data ----
        with tr.span("load_data", input_summary=data_csv):
            df = load_data(data_csv)
            tr.record_output(f"{len(df)} rows x {df.shape[1]} cols")

        with tr.span("build_space"):
            space = build_space(config)
            tr.record_output(f"{len(space)} dims")

        X = df[[v["name"] for v in config["variables"]]].to_numpy()
        y = objective_vector(df, config)

        # ---- feasibility (Garnett §11.2) ----
        with tr.span("feasibility_filter", input_summary="constraints -> feasible mask"):
            feasible = constraint_values(df, config)
            if feasible is not None:
                # fit the surrogate on feasible observations only; infeasible
                # rows are recorded but excluded from the objective GP.
                mask = feasible
                if mask.sum() < max(3, 2 * X.shape[1]):
                    mask = np.ones(len(df), dtype=bool)  # too few feasible -> use all
                X = X[mask]
                y = y[mask]
            n_feas = int(feasible.sum()) if feasible is not None else len(df)
            tr.record_output(f"feasible={n_feas}/{len(df)}")

        # ---- standardize (numerical stability) ----
        with tr.span("standardize_objective"):
            y_mean = float(np.mean(y))
            y_std = float(np.std(y)) or 1.0
            y_norm = (y - y_mean) / y_std
            tr.record_output(f"y_mean={y_mean:.4g}, y_std={y_std:.4g}")

        # ---- surrogate ----
        with tr.span("fit_gp", input_summary="Matern5/2 + ARD, marginal-likelihood MAP"):
            opt = fit_gp(space, X, y_norm, random_state=seed)
            tr.record_output(f"models={len(opt.models)}")

        # ---- acquisition ----
        with tr.span("acquisition", input_summary=f"acq={acq}, beta={beta}, xi={xi}"):
            nxt, a_val, mu_n, std_n = acquisition(opt, acq, beta=beta, xi=xi, seed=seed)
            tr.record_output(f"a_val={a_val:.4g}")

        mu = mu_n * y_std + y_mean          # back to original objective scale
        std = std_n * y_std

        # ---- robustness guard (Garnett §10.7) ----
        with tr.span("robustness_guard", input_summary="instability cap"):
            std_cap = 10.0 * max(y_std, 1e-6)
            gp_unstable = not np.isfinite(std) or std > std_cap
            if gp_unstable:
                std_rep = std_cap
                mu_rep = float(np.clip(mu, y.min() - 2 * y_std, y.max() + 2 * y_std))
            else:
                std_rep, mu_rep = float(std), float(mu)
            ci95 = [mu_rep - 1.96 * std_rep, mu_rep + 1.96 * std_rep]
            tr.record_output("unstable" if gp_unstable else "stable")

        # ---- convergence ----
        with tr.span("convergence"):
            conv = assess_convergence(y.tolist(), a_val)
            tr.record_output(f"converged={conv.converged}; {conv.reason}")

        rec = {
            "next_experiment": {v["name"]: float(x) for v, x in zip(config["variables"], nxt)},
            "acquisition_value": a_val,
            "predicted_mean": mu_rep,
            "predicted_std": std_rep,
            "confidence_95": ci95,
            "gp_stability": ("unstable: n<2d, hyperparameters diverged "
                             "(Garnett 10.7) — collect more data or constrain "
                             "hyperparameters") if gp_unstable else "stable",
            "convergence": asdict(conv),
            "n_observations": len(df),
            "best_observed": float(y.max()),
            "acquisition_used": acq,
            "kernel": "Matern 5/2 (ARD)",
            "seed": seed,
            "notes": "Run the recommended experiment, append its measured metric to the CSV, "
                     "and re-run this script to close the loop.",
        }
        if batch > 1:
            # greedy batch: q-1 more points via Thompson-ish sampling on the GP
            rec["batch"] = []
            for _ in range(batch - 1):
                nxt2, a2, m2, s2 = acquisition(opt, acq, beta=beta, xi=xi, n_samp=4000, seed=seed + 1)
                rec["batch"].append({v["name"]: float(x)
                                     for v, x in zip(config["variables"], nxt2)})

        # embed trace in the recommendation for one-file auditing
        rec["trace"] = tr.to_dict()

        with tr.span("persist", input_summary=out_json):
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=2, ensure_ascii=False, default=str)
            tr.record_output(f"wrote {out_json}")

    if emit_trace:
        tr.save(trace_path)
        rec["trace_file"] = trace_path
    return rec


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="BO electrocatalyst pipeline (MIT)")
    p.add_argument("--data", default="assets/experiment_template.csv")
    p.add_argument("--config", default="assets/config.yaml")
    p.add_argument("--acq", default="EI", choices=["EI", "PI", "UCB"])
    p.add_argument("--beta", type=float, default=2.0, help="UCB exploration param")
    p.add_argument("--xi", type=float, default=0.01, help="EI/PI improvement margin")
    p.add_argument("--batch", type=int, default=1, help="batch size q (Garnett 11.3)")
    p.add_argument("--seed", type=int, default=0, help="RNG seed (reproducibility)")
    p.add_argument("--out", default="next_experiment.json")
    p.add_argument("--trace", default="trace.json", help="TRACE json output path")
    p.add_argument("--no-trace", dest="emit_trace", action="store_false",
                   help="disable TRACE json emission")
    a = p.parse_args()
    rec = run(a.data, a.config, a.acq, a.beta, a.xi, a.batch, a.out,
              trace_path=a.trace, emit_trace=a.emit_trace, seed=a.seed)
    print(json.dumps(rec, indent=2, ensure_ascii=False, default=str))
