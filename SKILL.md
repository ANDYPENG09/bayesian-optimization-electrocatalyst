---
name: bayesian-optimization-electrocatalyst
description: Recommend next electrocatalyst experiments from complete historical data using Gaussian-process EI/PI/UCB optimization, typed synthesis variables, outcome-feasibility forecasts and weighted objectives. Use for BO, next-experiment planning, catalyst optimization, uncertainty review, or GP experiment selection.
---

# Bayesian optimization for electrocatalyst R&D

## Workflow

1. Establish the reaction, measured objective units, variable ranges, experimental resolution and budget. Preserve constraints and previously tested baselines; do not schedule unchanged baseline designs again.
2. Inspect historical data and method provenance. Keep empirical XRD relative indices separate from calibrated long-range S. Join electrochemistry results to design variables by unique sample ID with `scripts/import_analyzer_results.py` when needed. Preserve QC. Only Good is accepted by default; enable Check recommended only after explicit review. Never enable Invalid.
3. Configure `assets/config.yaml` or the synthetic `assets/config_ptco_example.yaml`. Use finite complete objective/constraint columns, positive metric scales and nonnegative weights. Validate bounds/types. Record scheduled designs under `pending_experiments`.
4. Run `python scripts/bo_pipeline.py --data experiments.csv --config campaign.yaml --acq EI --batch 3 --seed 0 --out next_experiment.json`.
5. Inspect every recommendation's feasibility probability, constraint forecasts, unclipped conditional GP uncertainty and model warnings. If no candidate meets the threshold, collect more evidence or review the threshold explicitly; do not silently relax it.
6. Measure a chosen experiment, append its outcome and rerun. Report observed results separately from model predictions. Use `scripts/run_synthetic_closed_loop.py` only as an algorithm demonstration; do not present synthetic rewards as catalyst performance.

## Implemented methods

Fit a Matérn 5/2 GP with marginal-likelihood hyperparameter optimization, not MAP with an explicit prior. Normalize numeric variables and one-hot encode categories. Use direction-adjusted weighted-sum utility; this implementation does not perform EHVI or Pareto-front optimization.

Filter known input inequalities exactly. Model measured outcome inequalities with separate GPs and combine probabilities under an independence approximation. A forecast probability is not a physical guarantee. With no observed feasible design, label the result feasibility_search and prioritize forecast feasibility.

Select distinct batch candidates using fixed-posterior greedy separation. This is not joint q-EI or a fantasy-updated batch. Exclude historical and pending designs. Keep numeric integer and categorical types in outputs.

Retain actual posterior mean/std without clipping. Interpret the 95% band as conditional on the fitted model and fixed hyperparameters. Review sparse/high-dimensional data and fitting warnings; do not infer reliable calibration from a synthetic example.

Treat convergence as a heuristic requiring small recent incumbent gain AND low modeled EI. PI/UCB runs still evaluate EI separately for stopping. Do not stop merely because a few rounds have no gain, and do not claim a final global optimum.

## Resources

- `README.md`: CLI, schema, configuration rules and implementation limits.
- `docs/verified-closed-loop.md`: executed synthetic example.
- `scripts/bo_pipeline.py`: validated planning pipeline.
- `scripts/import_analyzer_results.py`: strict ID-based CSV handoff.
- `scripts/data_io.py`: optional external-source conversion helpers; inspect their actual input formats rather than assuming a live connection.
- `references/bo_theory.md`: general BO theory; not a claim that every described algorithm is implemented.
- `references/electrocatalyst_metrics.md`: metric definitions and unit considerations.

Use `next_experiment.json` schema 1.1.0. `recommendations` contains all q predictions; legacy `batch` contains q−1 additional parameter sets. Join completed results to sample IDs externally. Keep TRACE outputs with the experiment record.
