# Bayesian Optimization in 5 Minutes — A Visual Guide

> Companion walkthrough for the closed-loop pipeline in this skill.
> Read order: this guide → `README.md` (CLI reference) → `references/bo_theory.md` (derivations).
> Author: Yu Peng · MIT

---

## 1 · The problem this solves

Electrocatalyst development is a **black-box optimization** problem:

- Many variables — reduction / alloying temperature, hold time, cooling rate,
  metal loading, catalyst loading (the PtCo example uses a 7-D search space);
- Expensive evaluations — one synthesis + characterization round costs days;
- Conflicting objectives — activity, stability, particle size, cost;
- No closed-form objective — you can sample it, but not differentiate it.

Classical DoE lays out a fixed grid up front; the experiment count explodes
with dimensionality. **Bayesian optimization (BO) is the sequential, adaptive
alternative**: run one (or a small batch) of *most worthwhile* experiments per
round, learn from each result, and iterate until further experiments no longer
pay off.

## 2 · The closed loop

```
        ┌────────────────────────────────────────────────────────────┐
        │                                                            │
        ▼                                                            │
 ① Seed data (DoE)            ② Surrogate (GP)           ③ Acquisition (EI)
 LHS / fractional factorial   predict μ(x) and σ(x)      pick argmax EI(x):
 2d+2 points                  for every candidate        exploit μ ↑ vs explore σ ↑
        │                            ▲                            │
        │                            │                            ▼
        ▼                            │                    ④ Run experiment,
 ⑤ Append result to CSV ─────────────┘                    measure metrics (≥3 reps)
        │
        ▼
 ⑥ Convergence check: incumbent stagnant k rounds & EI < tol
        └──> deliver argmax μ with 95% credible band
```

Each round costs **exactly one (or q) experiments**. This is the core
difference from one-shot DoE: **DoE does not learn; BO learns.**

## 3 · The building blocks

### 3.1 Surrogate — Gaussian process (GP)

A GP is a probabilistic interpolator: for any unmeasured input it returns a
**predicted mean μ(x)** and an **uncertainty σ(x)** (→ 95% credible band).
Uncertainty is modeled explicitly — this is what distinguishes BO from plain
regression.

- Kernel: **Matérn 5/2 + ARD** (Garnett §3.3 default; robust to noise).
- **ARD bonus**: each variable learns its own length scale; irrelevant
  variables are automatically down-weighted → free descriptor screening.
- Hyperparameters: marginal-likelihood MAP via `skopt` (fit_gp).

### 3.2 Acquisition — Expected Improvement (EI)

Given the current best observed value τ*, EI balances *exploit* (high μ) and
*explore* (high σ):

```
EI(x) = (μ(x) − τ*) · Φ(z) + σ(x) · φ(z),    z = (μ(x) − τ*) / σ(x)
```

- first term rewards beating τ* (exploitation),
- second term rewards uncertainty (exploration).

`PI` (conservative) and `UCB` (regret-bounded) are alternative policies.
The inner candidate-pool search concentrates near the data to avoid the
vanishing-gradient flat regions away from observations (Garnett §9.2).

### 3.3 Constraints, batches, multi-objective

- **Constraints** (Garnett §11.2): inequality constraints (e.g. stability
  loss ≤ 10 %, particle size ≤ 8 nm) penalize infeasible points.
- **Batches** (§11.3): `--batch q` recommends q parallel points for
  multi-channel electrochemical stations (penalize selected regions to avoid
  duplicates).
- **Multi-objective** (§11.7): weighted scalarization (activity 0.4, ECSA 0.3,
  ordering 0.2, size 0.1 in the PtCo example), priorities set by engineering
  judgment.

### 3.4 Convergence

`assess_convergence` reports: incumbent trajectory, relative improvement over
the last k rounds, EI magnitude (below `ei_tol` ⇒ converged), and the terminal
recommendation argmax μ with its credible band.

## 4 · Run it

```bash
python -m venv .venv && .venv/Scripts/activate   # or .venv/bin/activate
pip install -r requirements.txt

# one BO round on the PtCo L1₀ synthetic example
python scripts/bo_pipeline.py \
    --data assets/experiment_ptco_example.csv \
    --config assets/config_ptco_example.yaml \
    --acq EI --batch 1 --out next_experiment.json
```

Append the measured outcome to the CSV and re-run — the posterior updates and
the loop closes. Each run also emits `trace.json` (TRACE call-chain) for
full reproducibility.

## 5 · Mapping to electrocatalyst campaigns

| Campaign | Objective | Variables | Notes |
|---|---|---|---|
| PtCo L1₀ ordering | mass activity / ECSA / ordering S / size | T_red, t_red, T_alloy, t_alloy, cooling rate, Pt wt%, loading | 7-D, literature-range synthetic seed |
| IrOₓ OER | mass activity / −η@10 mA | potential window, pH, loading, composition | seed with 2⁵⁻¹ fractional-factorial DoE first |
| Descriptor screening | any activity metric | composition + condition descriptors | ARD length scales rank weak descriptors |

## 6 · Key references

- R. Garnett, *Bayesian Optimization*, Cambridge University Press, 2023 —
  formula numbers cited throughout `references/bo_theory.md`.
- Implementation: GP fitting via `skopt`; loop design, constraints, batch,
  multi-objective scalarization, TRACE observability and CLI are original
  work in `scripts/`.

---

*MIT © 2026 Yu Peng — no credentials, internal endpoints, or personal data are
bundled; `assets/` contain only synthetic seed data.*
