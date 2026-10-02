# Executed synthetic closed-loop example

**This is a mathematical algorithm demonstration, not measured catalyst performance.**

The reward is `1 − (x − 0.72)²`, with the measured-outcome constraint `1 + x ≤ 1.85`, x in [0, 1]. The known constrained optimum is x=0.72. Six initial synthetic observations include two infeasible points; those points remain available to the outcome-constraint model.

Run from the repository:

```bash
pip install -r requirements.txt
python scripts/run_synthetic_closed_loop.py --output demo_output
```

For each of three rounds, fit the posterior, recommend an untested point, evaluate the **synthetic** reward/cost, append that observation, and refit for the next round. `summary.json`, `history.csv`, per-round recommendations and traces are written to the output directory.

Executed on 2026-10-02 with Python 3.12, NumPy 2.3.5, scikit-learn 1.8.0 and scikit-optimize 0.10.2:

| Round | Proposed x | Predicted reward | Evaluated synthetic reward | Evaluated synthetic cost |
|---|---|---|---|---|
| 1 | 0.719426 | 1.001693 | 1.000000 | 1.719426 |
| 2 | 0.680848 | 0.998430 | 0.998467 | 1.680848 |
| 3 | 0.762032 | 0.998264 | 0.998233 | 1.762032 |

Initial best feasible reward: **0.951600**. Final best feasible reward: **1.000000**. All three evaluated proposed points meet the synthetic constraint. Numerical details can vary with package versions/platforms; fixed-seed repeats in one environment are regression-tested.

The first GP prediction slightly exceeds the synthetic reward's mathematical maximum. It is retained rather than clipped: model mean/uncertainty must be checked against observed data and physical bounds. This example does not calibrate high-dimensional catalyst models or guarantee real experimental improvement.

[Machine-readable executed summary](verified-example-summary.json) · [Tests](../tests/test_pipeline.py)
