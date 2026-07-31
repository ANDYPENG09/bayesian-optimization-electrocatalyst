# bayesian-optimization-electrocatalyst

Closed-loop **Bayesian Optimization** for electrocatalyst R&D — a skill (agent capability) that recommends the next experiment from a Gaussian-process surrogate of the catalyst structure–activity landscape.

> Theory base: Roman Garnett, *Bayesian Optimization* (Cambridge University Press, 2023).

## What it does

Given a table of prior experiments (composition, reaction conditions, measured metrics), the skill:

1. **Fits a GP surrogate** — Matérn 5/2 + ARD kernel, marginal-likelihood MAP hyperparameters (skopt).
2. **Selects the next experiment** via an acquisition function: **EI** (default), **PI**, or **UCB**.
3. **Handles real-world constraints** — stability/cost/particle-size inequality constraints (Garnett §11.2), batch recommendations for parallel stations (§11.3), multi-objective scalarization (§11.7).
4. **Reports uncertainty** — predicted mean, 95% credible band, convergence verdict (incumbent trajectory + EI magnitude).

Each run emits `next_experiment.json` (recommendation) and `trace.json` (auditable call chain, TRACE contract).

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate   # or: .venv/bin/activate (macOS/Linux)
pip install -r requirements.txt

# one BO round on the bundled template data
python scripts/bo_pipeline.py \
    --data assets/experiment_template.csv \
    --config assets/config.yaml \
    --acq EI --batch 1 --out next_experiment.json
```

Append the measured result of the recommended experiment to the CSV and re-run to close the loop.

## CLI reference

| Flag | Default | Meaning |
|------|---------|---------|
| `--data` | `assets/experiment_template.csv` | historical experiments CSV |
| `--config` | `assets/config.yaml` | campaign config (variables/objective/constraints/acquisition) |
| `--acq` | `EI` | `EI` \| `PI` \| `UCB` |
| `--beta` | `2.0` | UCB exploration parameter |
| `--xi` | `0.01` | EI/PI improvement margin |
| `--batch` | `1` | parallel batch size q (§11.3) |
| `--seed` | `0` | RNG seed (reproducibility) |
| `--out` | `next_experiment.json` | output recommendation path |
| `--trace` | `trace.json` | TRACE json output path |
| `--no-trace` | — | disable TRACE emission |

## Repository layout

```
├── SKILL.md                     skill definition (closed-loop workflow + TRACE spec)
├── LICENSE                      MIT
├── skill-info.md                author / license / metadata (human-readable)
├── _meta.json                   platform metadata (SkillHub / ClawHub import)
├── CHANGELOG.md
├── requirements.txt
├── references/
│   ├── bo_theory.md             theory manual with Garnett formula numbers
│   └── electrocatalyst_metrics.md  metrics & constraint encoding
├── scripts/
│   ├── bo_pipeline.py           main closed loop (GP → EI/PI/UCB → recommend → converge)
│   ├── trace_utils.py           TRACE call-chain recorder
│   └── data_io.py               CSV / WorkBuddy / Notion / ima multi-source merge
└── assets/
    ├── config.yaml              generic campaign config
    ├── experiment_template.csv  9-point synthetic seed data
    ├── config_ptco_v2.yaml      real PtCo L1₀ campaign config (9-D)
    └── experiment_ptco_v2.csv   real PtCo runs (synthetic-safe seed rows)
```

## Use cases

- **PtCo L1₀ ordering**: objective = ordering degree S (from `xrd-rasx-ptco-analysis`), variables = heat-treat T / hold / ramp-cool rate / atmosphere; see `assets/config_ptco_v2.yaml`.
- **IrOₓ OER**: multi-objective mass-activity / −η@10mA, constraints stability & cost; seed with a 2⁵⁻¹ fractional-factorial DoE before BO.
- **Descriptor screening**: ARD length scales auto-rank weak descriptors.

## License

MIT — see [LICENSE](LICENSE). No credentials, internal endpoints, or personal data are bundled; `assets/` contain only synthetic seed data.
