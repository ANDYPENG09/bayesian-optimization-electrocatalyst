#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Yu Peng
"""GP experiment planning. Batch selection is a separation heuristic, not q-EI."""
from __future__ import annotations
import argparse, itertools, json, math, os, sys, warnings
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from skopt.space import Real, Integer, Categorical, Space
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_utils import TraceRecorder


def load_data(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    if df.columns.duplicated().any():
        raise ValueError('Duplicate CSV column names')
    return df


def build_space(config):
    dims, names = [], set()
    for v in config['variables']:
        name, kind = v['name'], v.get('type', 'real')
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError('Variables require unique, nonempty names')
        names.add(name)
        if kind == 'cat':
            cats = v['categories']
            if not cats or len(set(cats)) != len(cats) or any(
                not isinstance(x, (str, int, float, bool)) or
                (isinstance(x, (int, float)) and not math.isfinite(x)) for x in cats):
                raise ValueError('Categories must be unique finite JSON scalars')
            dims.append(Categorical(cats, name=name, transform='onehot'))
        elif kind in ('real', 'int'):
            lo, hi = float(v['low']), float(v['high'])
            if not np.isfinite([lo, hi]).all() or lo >= hi:
                raise ValueError('Invalid variable bounds')
            if kind == 'int' and (not lo.is_integer() or not hi.is_integer()):
                raise ValueError('Integer bounds must be integers')
            cls = Integer if kind == 'int' else Real
            dims.append(cls(int(lo) if kind == 'int' else lo, int(hi) if kind == 'int' else hi,
                            name=name, transform='normalize'))
        else:
            raise ValueError(f'Unknown variable type: {kind}')
    if not dims:
        raise ValueError('Configure at least one variable')
    return dims


def validate_point(point, dims):
    if len(point) != len(dims):
        raise ValueError('Wrong number of variables')
    out = []
    for x, d in zip(point, dims):
        if isinstance(d, Categorical):
            if x not in d.categories:
                raise ValueError(f'Unknown category for {d.name}')
            out.append(x.item() if isinstance(x, np.generic) else x)
        else:
            if isinstance(x, bool):
                raise ValueError(f'{d.name} must be numeric')
            x = float(x)
            if not np.isfinite(x) or not d.low <= x <= d.high or (isinstance(d, Integer) and not x.is_integer()):
                raise ValueError(f'{d.name} must be finite, in bounds, and match its type')
            out.append(int(x) if isinstance(d, Integer) else x)
    return out


def finite_column(df, name):
    if name not in df:
        raise ValueError(f'Missing CSV column: {name}')
    col = pd.to_numeric(df[name], errors='raise').to_numpy(dtype=float)
    if not np.isfinite(col).all():
        raise ValueError(f'Missing/non-finite values in {name}')
    return col


def objective_vector(df, config):
    obj = config['objective']
    if not obj['metrics'] or obj.get('aggregation', 'weighted_sum') != 'weighted_sum':
        raise ValueError('Only nonempty weighted_sum objectives are implemented')
    y, active = np.zeros(len(df)), False
    for name, spec in obj['metrics'].items():
        scale, weight = float(spec.get('scale', 1)), float(spec.get('weight', 1))
        direction = spec.get('direction', 'max')
        if direction not in ('min', 'max') or not np.isfinite([scale, weight]).all() or scale <= 0 or weight < 0:
            raise ValueError('Objectives need min/max, positive scale and nonnegative weight')
        active |= weight > 0
        y += finite_column(df, name) / scale * weight * (-1 if direction == 'min' else 1)
    if not active:
        raise ValueError('At least one objective weight must be positive')
    if not np.isfinite(y).all():
        raise ValueError('Weighted objective overflowed; review scales')
    return y


def constraint_values(df, config):
    specs = config.get('constraints', [])
    if not specs:
        return None
    feasible = np.ones(len(df), dtype=bool)
    for spec in specs:
        threshold, direction = float(spec['threshold']), spec.get('direction', 'le')
        if not math.isfinite(threshold) or direction not in ('le', 'ge'):
            raise ValueError('Constraints require finite thresholds and le/ge direction')
        col = finite_column(df, spec['name'])
        feasible &= col >= threshold if direction == 'ge' else col <= threshold
    return feasible


@dataclass
class GPState:
    space: Space
    Xi: list
    yi: list
    models: list
    warnings: list


def fit_gp(space, X, y, noise=None, random_state=0):
    """One fit; marginal-likelihood optimization without a hyperparameter prior."""
    sp = Space(space)
    points = [validate_point(list(row), space) for row in X]
    xt = np.asarray(sp.transform(points), dtype=float)
    kernel = ConstantKernel(1, (1e-3, 1e3)) * Matern(np.ones(xt.shape[1]), (1e-2, 1e2), nu=2.5) + WhiteKernel(1e-5, (1e-8, .1))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-8 if noise is None else noise,
                                   normalize_y=False, n_restarts_optimizer=1, random_state=random_state)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        gp.fit(xt, y)
    return GPState(sp, points, list(y), [gp], sorted({str(w.message) for w in caught}))


def candidate_pool(opt, n_samp=4096, seed=0):
    dims = opt.space.dimensions
    sets = [list(d.categories) if isinstance(d, Categorical) else list(range(d.low, d.high+1))
            if isinstance(d, Integer) else None for d in dims]
    if all(s is not None for s in sets) and math.prod(map(len, sets)) <= n_samp:
        return [list(p) for p in itertools.product(*sets)]
    rng = np.random.default_rng(seed)
    pool = opt.space.rvs(n_samples=n_samp, random_state=seed)
    for i in range(int(.7*n_samp)):
        obs = opt.Xi[int(rng.integers(len(opt.Xi)))]
        for j, d in enumerate(dims):
            if isinstance(d, Categorical):
                if rng.random() < .7:
                    pool[i][j] = obs[j]
            else:
                x = np.clip(float(obs[j]) + rng.normal()*.15*(d.high-d.low), d.low, d.high)
                pool[i][j] = int(round(x)) if isinstance(d, Integer) else float(x)
    unique = {}
    for point in pool:
        point = validate_point(point, dims)
        unique[tuple(point)] = point
    return list(unique.values())


def posterior(opt, points):
    mu, std = opt.models[-1].predict(np.asarray(opt.space.transform(points), dtype=float), return_std=True)
    if not np.isfinite(mu).all() or not np.isfinite(std).all() or (std < 0).any():
        raise ValueError('Invalid GP posterior; no recommendation exported')
    return mu, std


def acquisition_values(mu, std, incumbent, acq, beta=2, xi=.01):
    improvement = mu-incumbent-xi
    z = improvement/np.maximum(std, 1e-15)
    ei = np.maximum(improvement*norm.cdf(z)+std*norm.pdf(z), 0)
    ei = np.where(std > 1e-15, ei, np.maximum(improvement, 0))
    if acq == 'EI': return ei, ei
    if acq == 'PI': return np.where(std > 1e-15, norm.cdf(z), (improvement > 0).astype(float)), ei
    if acq == 'UCB': return mu+beta*std, ei
    raise ValueError(f'Unknown acquisition: {acq}')


def separated(space, candidates, excluded, distance):
    keep = np.ones(len(candidates), dtype=bool)
    c = np.asarray(space.transform(candidates), dtype=float)
    for row in space.transform(excluded) if excluded else []:
        keep &= np.linalg.norm(c-row, axis=1) > distance
    return keep


def acquisition(opt, acq, beta=2, xi=.01, n_samp=4096, seed=0):
    """Compatibility helper for unconstrained single-point selection."""
    points = candidate_pool(opt, n_samp, seed)
    mask = separated(opt.space, points, opt.Xi, 1e-6)
    if not mask.any(): raise ValueError('No untested candidates remain')
    mu, std = posterior(opt, points)
    score, _ = acquisition_values(mu, std, max(opt.yi), acq, beta, xi)
    i = int(np.argmax(np.where(mask, score, -np.inf)))
    return points[i], float(score[i]), float(mu[i]), float(std[i])


@dataclass
class Convergence:
    n_obs: int
    best_y: float | None
    incumbents: list
    ei_value: float
    improvement_last_k: float | None
    converged: bool
    reason: str


def assess_convergence(history_y, ei_value, k=3, ei_tol=1e-3, rel_tol=1e-3):
    incumbent = np.maximum.accumulate(history_y).tolist()
    gain = None if len(incumbent) <= k else (incumbent[-1]-incumbent[-1-k])/max(abs(incumbent[-1-k]), 1e-12)
    done = gain is not None and gain < rel_tol and ei_value < ei_tol
    return Convergence(len(history_y), incumbent[-1] if incumbent else None, incumbent,
                       float(ei_value), gain, bool(done), 'Small gain AND low modeled EI (heuristic)' if done else 'Continue collecting evidence')


def run(data_csv, config_path, acq='EI', beta=2, xi=.01, batch=1,
        out_json='next_experiment.json', trace_path='trace.json', emit_trace=True, seed=0):
    if not isinstance(batch, int) or isinstance(batch, bool) or batch < 1:
        raise ValueError('Batch must be a positive integer')
    if not np.isfinite([beta, xi]).all() or min(beta, xi) < 0:
        raise ValueError('beta and xi must be finite and nonnegative')
    tr = TraceRecorder()
    with tr.span('run', input_summary=f'data={data_csv}, config={config_path}, batch={batch}'):
        with open(config_path, encoding='utf-8') as f: config = yaml.safe_load(f)
        dims, df = build_space(config), load_data(data_csv)
        original_count = len(df)
        history_points = [validate_point(row, dims) for row in df[[d.name for d in dims]].values.tolist()]
        if 'qc_status' in df:
            accepted = config.get('accepted_qc', ['Good'])
            if not accepted or not set(accepted) <= {'Good', 'Check recommended'}:
                raise ValueError('accepted_qc may include Good and/or Check recommended, never Invalid')
            df = df[df.qc_status.isin(accepted)].reset_index(drop=True)
        if len(df) < 3: raise ValueError('At least three complete accepted observations are required')
        X = [validate_point(row, dims) for row in df[[d.name for d in dims]].values.tolist()]
        if len({tuple(row) for row in X}) < 2: raise ValueError('At least two distinct designs are required')
        y = objective_vector(df, config)
        feasible = constraint_values(df, config)
        if feasible is None: feasible = np.ones(len(df), dtype=bool)
        pending = [validate_point([p[d.name] for d in dims], dims) for p in config.get('pending_experiments', [])]
        center, scale = float(y.mean()), float(y.std()) or 1
        with tr.span('fit_objective_gp', input_summary='Matern5/2; marginal likelihood; all accepted observations'):
            opt = fit_gp(dims, X, (y-center)/scale, random_state=seed)
        settings = config.get('recommendation', {})
        n = settings.get('n_candidates', 4096)
        prob_min = float(settings.get('min_feasibility_probability', .8))
        history_distance = float(settings.get('history_distance', 1e-6))
        batch_distance = float(settings.get('batch_distance', .02))
        if not isinstance(n, int) or n < max(16, batch) or not 0 < prob_min <= 1 or not np.isfinite([history_distance, batch_distance]).all() or min(history_distance, batch_distance) < 0:
            raise ValueError('Invalid candidate count, feasibility probability or separation distance')
        points = candidate_pool(opt, n, seed)
        mu, std = posterior(opt, points)
        allowed = separated(opt.space, points, history_points+pending, history_distance)
        probability, forecasts = np.ones(len(points)), []
        model_warnings = list(opt.warnings)
        with tr.span('candidate_constraints'):
            for i, spec in enumerate(config.get('constraints', [])):
                name, threshold, direction = spec['name'], float(spec['threshold']), spec.get('direction', 'le')
                if name in [d.name for d in dims]:
                    j = [d.name for d in dims].index(name)
                    if isinstance(dims[j], Categorical): raise ValueError('Numeric constraints cannot target categories')
                    cmu, cstd = np.array([p[j] for p in points]), np.zeros(len(points))
                    p = (cmu >= threshold if direction == 'ge' else cmu <= threshold).astype(float)
                    allowed &= p.astype(bool)
                    source = 'exact input constraint'
                else:
                    values = finite_column(df, name)
                    ccenter, cscale = float(values.mean()), float(values.std()) or 1
                    gp = fit_gp(dims, X, (values-ccenter)/cscale, random_state=seed+i+1)
                    cmu, cstd = posterior(gp, points)
                    cmu, cstd = cmu*cscale+ccenter, cstd*cscale
                    signed = cmu-threshold if direction == 'ge' else threshold-cmu
                    p = np.where(cstd > 1e-15, norm.cdf(signed/np.maximum(cstd, 1e-15)), (signed >= 0).astype(float))
                    source = 'GP forecast; experimentally verify'
                    model_warnings.extend(gp.warnings)
                probability *= p
                forecasts.append((spec, cmu, cstd, p, source))
        allowed &= probability >= prob_min
        if not allowed.any():
            raise ValueError('No untested candidates meet the feasibility probability. Add constraint data or explicitly review the threshold; no recommendation written.')
        mode = 'objective_optimization' if feasible.any() else 'feasibility_search'
        incumbent = float(np.max((y[feasible]-center)/scale)) if feasible.any() else float(max(opt.yi))
        score, ei = acquisition_values(mu, std, incumbent, acq, beta, xi)
        score = probability if mode == 'feasibility_search' else (score*probability if acq in ('EI', 'PI') else score)
        records, mask = [], allowed.copy()
        with tr.span('select_unique_batch', input_summary='Fixed posterior; greedy minimum-distance separation'):
            for _ in range(batch):
                if not mask.any(): raise ValueError('Not enough distinct eligible candidates; reduce batch/separation. No partial batch written.')
                i = int(np.argmax(np.where(mask, score, -np.inf)))
                mean, sigma = float(mu[i]*scale+center), float(std[i]*scale)
                records.append(dict(parameters=dict(zip([d.name for d in dims], points[i])),
                    predicted_mean=mean, predicted_std=sigma, confidence_95=[mean-1.96*sigma, mean+1.96*sigma],
                    acquisition_value=float(score[i]), feasibility_probability=float(probability[i]),
                    constraints=[dict(name=s['name'], threshold=float(s['threshold']), direction=s.get('direction', 'le'),
                        predicted_mean=float(m[i]), predicted_std=float(sd[i]), probability=float(p[i]), source=src)
                        for s, m, sd, p, src in forecasts]))
                mask &= separated(opt.space, points, [points[i]], batch_distance)
        cc = config.get('convergence', {})
        k, rel_tol, ei_tol = int(cc.get('k_rounds_no_gain', 3)), float(cc.get('relative_tol', 1e-3)), float(cc.get('ei_tol', 1e-3))
        if k < 1 or not np.isfinite([rel_tol, ei_tol]).all() or min(rel_tol, ei_tol) < 0: raise ValueError('Invalid convergence settings')
        conv = assess_convergence(y[feasible].tolist(), float(np.max((ei*probability)[allowed])), k, ei_tol, rel_tol)
        sparse = len(df) < 2*len(dims)+1
        if sparse or model_warnings or mode == 'feasibility_search':
            conv.converged, conv.reason = False, 'Continue: sparse data, model warnings, or no measured feasible design'
        first = records[0]
        rec = dict(schema_version='1.1.0', next_experiment=first['parameters'],
                   **{key: first[key] for key in ('acquisition_value', 'predicted_mean', 'predicted_std', 'confidence_95', 'feasibility_probability', 'constraints')},
                   recommendations=records, convergence=asdict(conv), n_observations=len(df),
                   excluded_qc_rows=original_count-len(df), n_feasible_observations=int(feasible.sum()),
                   best_observed=float(y[feasible].max()) if feasible.any() else None,
                   acquisition_used=acq, mode=mode, seed=seed,
                   kernel='Matern 5/2 on normalized numeric / one-hot categorical inputs',
                   gp_stability='review' if sparse or model_warnings else 'no fitting warnings',
                   model_warnings=sorted(set(model_warnings)), objective_definition=config['objective'],
                   objective_units='weighted, direction-adjusted, scaled utility',
                   batch_method='fixed-posterior greedy separation (not q-EI)', minimum_feasibility_probability=prob_min,
                   notes='Outcome feasibility is a forecast, not a guarantee. The 95% band is conditional GP uncertainty, never clipped. Verify proposed experiments.')
        if batch > 1: rec['batch'] = [r['parameters'] for r in records[1:]]
    rec['trace'] = tr.to_dict()
    if emit_trace:
        rec['trace_file'] = trace_path
        tr.save(trace_path)
    with open(out_json, 'w', encoding='utf-8') as f: json.dump(rec, f, indent=2, ensure_ascii=False, allow_nan=False)
    return rec


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Typed, constrained GP experiment planning')
    p.add_argument('--data', default='assets/experiment_template.csv')
    p.add_argument('--config', default='assets/config.yaml')
    p.add_argument('--acq', default='EI', choices=['EI', 'PI', 'UCB'])
    p.add_argument('--beta', type=float, default=2)
    p.add_argument('--xi', type=float, default=.01)
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='next_experiment.json')
    p.add_argument('--trace', default='trace.json')
    p.add_argument('--no-trace', dest='emit_trace', action='store_false')
    a = p.parse_args()
    try: result = run(a.data, a.config, a.acq, a.beta, a.xi, a.batch, a.out, a.trace, a.emit_trace, a.seed)
    except (ValueError, KeyError) as error: p.error(str(error))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
