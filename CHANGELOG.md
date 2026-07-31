# Changelog

All notable changes to this project are documented in this file.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), versions follow [SemVer](https://semver.org/).

## [1.0.0] — 2026-08-01

### Added
- Closed-loop Bayesian optimization pipeline for electrocatalyst development.
- GP surrogate (Matérn 5/2 + ARD) with marginal-likelihood MAP hyperparameters (skopt).
- Acquisition functions: EI (default), PI, UCB — per Garnett §7–8 closed forms.
- Constrained (§11.2), batch (§11.3), and multiobjective scalarization (§11.7) support.
- TRACE call-chain observability (`trace_utils.py`) emitted as `trace.json`.
- Multi-source data ingestion (`data_io.py`): CSV, WorkBuddy memory logs, Notion pages, ima knowledge base.
- Assets: synthetic template config/data + literature-based PtCo L1₀ example config (7-D) & synthetic seed data (no proprietary data).
- Theory reference manual (`references/bo_theory.md`) with Garnett formula numbers.
- Electrocatalyst metrics & constraint encoding (`references/electrocatalyst_metrics.md`).
- CLI: `--data --config --acq --beta --xi --batch --seed --out --trace [--no-trace]`.

### Changed
- Fixed `trace_utils.py` span timing bug: span now appended on entry so `record_output`
  inside a `with tr.span(...)` block writes to the *current* span (index-based) instead
  of the previously closed span.
- `bo_pipeline.py`: `yaml` import moved to module top; added `--seed` for reproducibility;
  batch acquisition uses deterministic per-point seeds.
- Frontmatter of `SKILL.md` extended with `metadata.openclaw` for ClawHub compatibility.
- All `{{AUTHOR_NAME}}` placeholders replaced with `Yu Peng` (LICENSE, SKILL.md,
  skill-info.md, _meta.json, script headers).
- Added `requirements.txt`, `.gitignore`, and this changelog.

### Fixed
- GP robustness guard now caps non-finite / diverged posterior std (Garnett §10.7).
