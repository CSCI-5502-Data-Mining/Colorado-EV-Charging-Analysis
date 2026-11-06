# Experiment results (Milestone 3)

Reproducible record from the bounded regression experiment harness
(`src/experiment_harness.py`). Regenerate with:

```bash
python src/prepare_modeling.py        # builds data/processed/modeling_colorado.csv
python src/experiment_harness.py      # writes everything in this folder
```

All runs use seed = 42 and record the git commit, so a result ties back to the
exact code that produced it.

## Files

| File | What it is |
| --- | --- |
| `experiments.csv` | One row per experiment: name, hypothesis, features, model, param grid, preprocessing choices, seed, git commit, **nested** outer-fold MAE/RMSE/R² with fold std and per-fold MAEs, the biased single-level grouped-CV MAE (comparison only), subgroup MAE + n for all six equity subgroups, and the `selected` flag. |
| `best_config.json` | Full run summary: fixed rules (target, ZIP3 groups, leakage exclusions, data-quality rules, metric), every experiment, the selection rule, the selected candidate's tuned hyperparameters, and the **one-time** held-out test metrics + test subgroup MAE. |
| `figures/experiments_mae.png` | Nested-CV MAE per experiment with fold-std error bars; selected model highlighted. |
| `figures/experiments_subgroup_mae.png` | Subgroup (equity) MAE: selected model vs. the income-only baseline. |

## How a candidate is evaluated (fixed protocol)

Defined once in `src/experiment_eval.py` and never varied by an experiment:

- **Target** `ev_per_1000_households`; **groups** = ZIP3 (first 3 digits of ZCTA).
- **Held-out test set** of whole ZIP3 groups is reserved first (seed 42) and is
  evaluated exactly once, for the single selected procedure, at the end.
- **Nested ZIP3-GroupKFold** (outer = 5, inner = 4): the inner loop tunes
  hyperparameters, the outer loop scores the tuned procedure. The outer-fold
  means are the unbiased estimate reported for Milestone 3. The single-level
  grouped-CV number is kept only as the labelled (optimistic) comparison.
- **Preprocessing** (imputation, scaling, log transforms, feature selection) is
  fit inside folds, never on validation/test data.
- **Metric**: MAE primary; RMSE, R², fold-MAE std (geographic consistency), and
  subgroup MAE (equity) as supporting evidence.

## Selection rule

Lowest nested outer-fold mean MAE; candidates within 1.0 MAE of the best are
treated as not meaningfully different and decided by lower fold-MAE std, then
lower subgroup-MAE spread. Baselines (mean, income-only) are references and are
never selectable winners.

## Governance (what experiments may NOT do)

- Change the target definition, the ZIP3 groups, the held-out test set, the
  data-quality rules (verified-zero supply recoding, households ≥ 100), or the
  evaluation metric.
- Use a target-derived column as a feature (`ev_count` / `bev_count` /
  `phev_count`); every candidate passes the leakage guard before it runs.
- Silently drop a hard geographic area or redraw the subgroup boundaries — the
  six equity subgroups are fixed in `modeling_common.SUBGROUP_SPECS`.
