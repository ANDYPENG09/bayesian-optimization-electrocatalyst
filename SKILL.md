---
name: bayesian-optimization-electrocatalyst
description: 'Bayesian optimization (BO) skill for electrocatalyst development. This skill should be used when the user wants to optimize catalyst composition, reaction conditions (potential, pH, temperature), activity descriptors, or to perform multi-objective optimization of catalytic activity/selectivity/stability/cost using a closed-loop Gaussian-process surrogate model. Grounded in Roman Garnett''s "Bayesian Optimization" (Cambridge Univ. Press): GP surrogate with Matern 5/2 kernel, EI/PI/UCB acquisition functions, marginal-likelihood hyperparameter optimization, constrained/batch/multiobjective extensions. Outputs next-round experiment recommendations with confidence intervals and convergence analysis. Triggers: phrases such as 贝叶斯优化/BO, 下一轮实验/推荐实验, 催化剂优化, 高斯过程/GP, 采集函数/EI/UCB, 过电位/Tafel/ECSA/法拉第效率 优化, 多目标 催化.'
agent_created: true
author: "Yu Peng"
license: MIT
version: 1.0.0
metadata:
  openclaw:
    requires:
      env: []
      bins: [python]
    emoji: "🧪"
    homepage: "https://github.com/ANDYPENG09/bayesian-optimization-electrocatalyst"
---

# Bayesian Optimization for Electrocatalyst Development

## Overview

Closed-loop Bayesian optimization for electrocatalyst R&D. Maintains a Gaussian-process surrogate of the catalyst structure–activity landscape, selects the next experiment via an acquisition function (EI/PI/UCB), and iterates until convergence — each round emitting a recommended experiment with predicted value, 95% confidence band, and a convergence verdict.

**Theory base:** Roman Garnett, *Bayesian Optimization* (Cambridge University Press). The closed-loop procedure follows Garnett's general approach (Ch.1 §1.2): surrogate → acquisition → observation → update → terminal recommendation. Full derivations, formula numbers, and citations live in `references/bo_theory.md`; domain metrics and constraints in `references/electrocatalyst_metrics.md`.

## When to use

Trigger when the user wants any of:
- Optimize catalyst **composition** (element fractions, loading, dopant ratio).
- Tune **reaction conditions** (potential, pH, temperature, concentration).
- Screen / rank **activity descriptors** (d-band center, adsorption energies) via ARD length scales.
- **Multi-objective** trade-offs: activity (overpotential, Tafel, mass activity) vs selectivity (Faradaic efficiency) vs stability vs cost.
- Recommend the **next experiment** given a table of prior results.
- Convergence / confidence assessment of an ongoing optimization campaign.

Do NOT use for: pure kinetic modeling without optimization intent, single-shot DoE only (use a DOE skill), or when fewer than ~5–6 observations exist (start with LHS/fractional factorial DoE first, then switch to BO — see Step 0).

## Closed-loop workflow

### Step 0 — Data ingestion & preprocessing
1. Gather historical experiment rows into the standard schema (see `assets/experiment_template.csv`). Sources, merged by `scripts/data_io.py`:
   - local CSV / Excel of bench results;
   - **WorkBuddy task history & `.workbuddy/memory/*.md`** logs (parse `bo-table` fenced blocks);
   - **Notion** experiment logs — at runtime the agent calls `mcp__notion__notion-search` / `notion-fetch`, passes page text to `data_io.from_notion_text`;
   - **ima knowledge base** — `mcp__ima-mcp__search_knowledge` for literature/catalyst databases (provides prior descriptor ranges / seed points).
2. Normalize objective direction to maximization (negate min-direction metrics: overpotential, Tafel, R_ct) — Garnett maximizes utility.
3. Encode inequality constraints (stability ≤10% loss, cost cap) per `references/electrocatalyst_metrics.md` §3.
4. If fewer than n_0 points (typically 2·d + 2), run LHS/fractional-factorial initial DoE (`config.yaml` → `initial_doe`) before invoking the GP posterior.

### Step 1 — Surrogate model (GP)
- Kernel: **Matern 5/2 with ARD** (Garnett §3.3 off-the-shelf choice); per-dimension length scales auto-screen weak descriptors.
- Hyperparameters: **marginal-likelihood MAP** (Garnett §4.3, eq. 4.8) via skopt; for higher rigor, sample hyperparameters and average predictions (§4.4) to dodge the Bull (2011) failure mode (§10.7).
- Observation noise: supply per-point variance from replicates; if absent, enable skopt's noise estimate.

### Step 2 — Acquisition function
Choose per campaign goal (`config.yaml` → `acquisition`):

| Function | Use when | Formula (Garnett) |
|----------|----------|-------------------|
| **EI** (default) | balanced explore/exploit, robust to noise | (8.9) (μ−τ*)Φ(z) + σφ(z) |
| **PI** | conservative, probability of beating target ξ | (8.22) Φ(z_τ) |
| **UCB** | theoretical regret guarantees, tunable exploration β | (8.24) μ + βσ |
| KG / EI-with-noise | noisy objective, multi-step lookahead | §8.2, §8.6 |

Tune ξ (EI/PI margin) and β (UCB exploration) in config. UCB↔PI duality (§8.4): a fixed-β UCB equals a data-dependent-target PI — keep β modest to avoid over-exploration.

### Step 3 — Acquire next experiment
- Inner optimization (Garnett §9.2, eq. 9.8): `scripts/bo_pipeline.py` samples a candidate pool and evaluates the GP posterior — guards against the vanishing-gradient flat region away from observations (§9.2) by concentrating effort near the data.
- **Batch** (§11.3): set `--batch q` to recommend q parallel points for high-throughput electrochemistry (multi-channel stations / array electrodes).
- **Constrained** (§11.2): infeasible points (stability/cost violated) are penalized; the GP can jointly model uncertain constraint functions if needed.

### Step 4 — Experiment & feedback
1. Synthesize/characterize the recommended sample.
2. Measure target metrics (≥3 replicates → mean + noise variance). For PtCo ordering, reuse the **`xrd-rasx-ptco-analysis`** skill to get ordering degree S / Scherrer size; for OER/HER use standard electrochemistry (ECSA, η@10mA, Tafel, FE).
3. Append the row to the CSV; re-run `bo_pipeline.py` — the GP posterior updates and the loop closes.

### Step 5 — Convergence & recommendation
`scripts/bo_pipeline.py::assess_convergence` reports:
- incumbent trajectory (max-so-far);
- improvement over last k rounds (relative tol);
- EI magnitude (below `ei_tol` ⇒ converged);
- final terminal recommendation = argmax μ_D(x), with 95% credible band.

Stop when converged or budget exhausted; deliver the recommended catalyst + uncertainty.

## Quick start

```
# 1. activate an isolated venv, install deps
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
# 2. run one BO round on the template data
cd /path/to/skill-dir
python scripts/bo_pipeline.py \
	--data assets/experiment_template.csv \
	--config assets/config.yaml \
	--acq EI --batch 1 --out next_experiment.json
```

`next_experiment.json` contains the recommended conditions, acquisition value, GP prediction ± std, 95% confidence interval, and a convergence object. Append the executed result to the CSV and re-run to iterate.

## Resources

### scripts/
- `bo_pipeline.py` — main closed loop: GP fit (skopt) → EI/PI/UCB → next point → convergence. CLI-driven.
- `data_io.py` — merge CSV / WorkBuddy memory / Notion / ima sources into the standard schema.
- `trace_utils.py` — TRACE call-chain recorder (span tree → `trace.json`).

### references/
- `bo_theory.md` — Garnett-derived theory manual (GP, posterior moments, Matern, marginal likelihood, EI/PI/UCB closed forms, acquisition optimization, constrained/batch/multiobjective, convergence) with formula numbers.
- `electrocatalyst_metrics.md` — electrochemistry objectives & constraints (ECSA, overpotential, Tafel, mass/specific activity, FE, TOF, stability, cost) with direction and BO encoding; ties into `xrd-rasx-ptco-analysis`.

### assets/
- `experiment_template.csv` — standard schema with 9 seed runs (PtCo-style heat-treat grid).
- `config.yaml` — variables, objective weights, constraints, acquisition params, convergence thresholds, kernel choice.
- `config_ptco_v2.yaml` — real PtCo L1₀ ordering campaign config (9-D, corrected priorities).
- `experiment_ptco_v2.csv` — 7 real PtCo runs (alloying grid, MA/ECSA/ordering targets).

## Integration notes
- **PtCo L1₀ ordering campaign** (existing PtCo ordering workflow): set objective to ordering degree S (from `xrd-rasx-ptco-analysis`); variables = heat-treat T, hold time, ramp/cool rate, atmosphere ratio. The template CSV mirrors this (best historical 700°C×2h).
- **IrOₓ OER** (HZB-style): objective = mass_activity / −η@10mA multiobjective; constraints = stability, cost; DoE = 2^(5−1) fractional factorial seed before BO.
- Token/credit awareness: prefer running `bo_pipeline.py` directly (deterministic, no LLM calls) and only loading `references/*` when the agent needs to reason about theory or debug the model.

## TRACE — call-chain observability

Every `bo_pipeline.py` run emits a machine-readable `trace.json` recording the full optimization chain. This satisfies the **TRACE** contract used for reproducibility grading and AI-skill evaluation:

| Letter | Meaning |
|--------|---------|
| **T** race id | run-level UUID linking all spans of one BO round |
| **R** ecord | every stage emits a span (load → preprocess → surrogate → acquisition → recommend → assess → persist) |
| **A** ttributes | per span: `parent_id`, `duration_ms`, `input_summary`, `output_summary`, `status` |
| **C** hain | spans form a linked list via `parent_id` — the call order is fully reconstructable |
| **E** mit | written to `trace.json` alongside `next_experiment.json` |

Enable / disable with `--trace PATH` (default `trace.json`) / `--no-trace`. The `trace_id`, span count, and per-step timings are also embedded in `next_experiment.json` under `"trace"` for one-file auditing. The recorder lives in `scripts/trace_utils.py` and is imported by `bo_pipeline.py::run`.

## License

Released under the **MIT License** — see `LICENSE`. Free to use, modify, and redistribute with attribution. Author: `Yu Peng` (set in `SKILL.md` frontmatter, `skill-info.md`, and `LICENSE`). No credentials, internal endpoints, or personal data are bundled; the shipped `assets/` contain only synthetic seed data.
