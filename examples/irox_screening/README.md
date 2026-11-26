# IrOx OER small-sample screening

This AI for Materials Science case asks whether a surrogate can screen acid-OER recipes when the number of independent syntheses is small. The workflow starts with the data needed to make that decision. It estimates repeatability, fits a regularized linear baseline and a Gaussian process, and validates both at the synthesis-batch level.

[Notebook](irox_screening.ipynb) · [Benchmark CLI](benchmark.py) · [Bayesian optimization planner](../../README.md)

## Measured-data case

The historical dataset contains 20 synthesis batches, 6 commercial controls and 65 LSV scans in 32 inferred sessions. Commercial controls contribute to repeatability and benchmarking. Model fitting uses the 20 synthesis batches alone. The target is the iR-compensated overpotential at 10 mA/cm² geometric current density, measured in mV on the RHE scale.

| Evaluation set | Target | Batches | Ridge RMSE (mV) | Matérn GP RMSE (mV) |
|---|---|---:|---:|---:|
| All historical syntheses | First scan | 20 | 25.0 | 31.5 |
| All historical syntheses | Within-batch mean | 20 | 26.5 | 33.4 |
| 350–500 °C historical subset | First scan | 17 | 16.8 | 24.3 |
| 350–500 °C historical subset | Within-batch mean | 17 | 20.7 | 25.9 |

The same-session repeatability SD for the synthesis batches was about 8.3 mV. Four synthesis batches had only one scan. Cross-session variability was sensitive to one batch with three sessions. The scan sequence included both increasing and decreasing overpotential, so the first scan was used as the primary response, with the batch mean as a sensitivity analysis.

Use the model comparison to support coarse retrospective screening. For recipe decisions, check the measured neighbors and their repeatability. The 350–500 °C evaluation subset is a temperature filter. The observed recipe shortlist also filters the permitted pore-agent families. The planned DOE has no measured runs in this dataset. Quantitative predictions for new recipes and closed-loop experimental improvement would require separate prospective validation.

The measured-data case reports aggregate results. The repository's runnable example generates a separate, seeded synthetic dataset. Its metrics illustrate the workflow.

## Run it

From the repository root, with the repository dependencies installed:

```bash
python examples/irox_screening/benchmark.py --output irox_output/demo
```

To open the notebook, install JupyterLab and Matplotlib and launch JupyterLab from the repository root. Run all cells. The notebook uses the same benchmark code and plots held-out predictions rather than fitted values.

For local measured inputs:

```bash
python examples/irox_screening/benchmark.py \
  --batches irox_private/batches.csv \
  --scans irox_private/scans.csv \
  --output irox_output/local
```

### Input schema

| File | Required columns | Grain and units |
|---|---|---|
| Batches | `batch_id`, `role`, `precursor`, `temperature_c`, `hold_min`, `pore_family`, `pore_ratio`, `naoh_g`, `ir_mg` | One row per independent synthesis or control. Use `synthesis` or `control` for the role. Temperature in °C, time in min, NaOH in g and Ir metal in mg. |
| Scans | `batch_id`, `scan_order`, `session_id`, `eta10_mv` | One row per annotated LSV. Order starts at 1 within each batch. A session ID can repeat across different batches. Supply the finalized η₁₀ in mV. |

The batch and scan files must cover the same IDs. Synthesis rows require the design factors apart from `pore_ratio`, which can be blank when the solid mass basis is undefined. Use a dimensionless ratio of solid pore-agent mass to Ir metal mass. The model uses an observed ratio only within its pore-agent family. Control design fields can be blank. Session annotations are supplied by the analyst. In the measured case, the session proxy used the batch, instrument file, scan date and adjacent time gaps. Its electrode identity was inferred.

## How the validation works

- The scan audit summarizes the first scan, mean, sample SD, scan count, direction and rank-test result. Whole-batch tests for cross-session sequences are descriptive.
- The repeatability model uses REML with batch and nested session intercepts. It reports the between-batch, between-session and same-session variances in mV² and SD in mV. Batch variance includes synthesis and recipe differences. The model also reports how many degrees of freedom support each level.
- Each validation fold holds out a complete synthesis batch. Numeric scaling, family-specific ratio terms, weights and GP hyperparameters are estimated in the training fold. The inverse variance weights use a pooled training-fold SD as a floor, including for one-scan batches. The GP uses a Matérn 5/2 covariance, those per-batch noise estimates and an additional white-noise term.
- `metrics.csv`, `loo_predictions.csv` and `summary.json` capture validation and model diagnostics. The other tables preserve scan summaries, observed recipe rankings and temperature pairs matched on the recorded design factors. Historical matched pairs are observational comparisons.
