"""Milestone 3 - bounded, seeded experiment harness for regression.

A declarative, reproducible search over a PREDECLARED space of feature subsets,
preprocessing strategies, and model hyperparameters. Every candidate is scored
by the fixed nested ZIP3-grouped CV protocol in experiment_eval.py. The held-out
test set is reserved first and never touched during the search; it is evaluated
exactly ONCE, for the single selected procedure, at the end.

Governance (enforced, not just documented):
  * target, ZIP3 groups, held-out test set, data-quality rules, and the metric
    are fixed in experiment_eval.py / modeling_common.py; the harness only picks
    among candidates, it cannot redefine them.
  * every candidate's features pass the leakage guard before it runs.
  * preprocessing is fit inside folds (ColumnTransformer/Pipeline).
  * selection is MAE-first with fold-variability and subgroup consistency as
    tie-breakers, so a model that is good on average but uneven across Colorado
    communities does not automatically win.
  * every experiment is logged (config, features, params, seed, git commit,
    nested metrics, subgroup MAE, selected/rejected) to results/experiments.csv.

    python src/experiment_harness.py
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge

from modeling_common import (MODELING_CSV, ROOT, TARGET, assert_no_leakage,
                             git_commit, repo_relative)
from experiment_eval import (Candidate, SEED, grouped_test_split, nested_cv,
                             score_all, subgroup_mae)

RESULTS = ROOT / 'results'
ALL_FEATURES = ['median_income', 'renter_share', 'population', 'households']


def candidate_space():
    """The PREDECLARED candidate set. Each entry is a justified hypothesis, not
    a blind sweep. Baselines are included so the comparison shares one protocol.
    """
    cands = []

    # --- Required baselines (same protocol) --------------------------------
    cands.append(Candidate(
        name='baseline_mean', features=ALL_FEATURES,
        model_factory=lambda: DummyRegressor(strategy='mean'),
        hypothesis='Reference: predict the training mean for every area.'))
    cands.append(Candidate(
        name='baseline_income_only', features=['median_income'],
        model_factory=LinearRegression,
        hypothesis='Reference: income alone as a single-feature linear model.'))

    # --- Ridge: feature-set and alpha variations ---------------------------
    ridge_grid = {'model__alpha': [0.1, 1.0, 10.0, 50.0, 100.0, 300.0]}
    cands.append(Candidate(
        name='ridge_all_features', features=ALL_FEATURES,
        model_factory=lambda: Ridge(random_state=SEED), param_grid=ridge_grid,
        hypothesis='Linear signal from all four socioeconomic features.'))
    cands.append(Candidate(
        name='ridge_income_renter', features=['median_income', 'renter_share'],
        model_factory=lambda: Ridge(random_state=SEED), param_grid=ridge_grid,
        hypothesis='Does dropping population/households (weak in EDA) hurt?'))
    cands.append(Candidate(
        name='ridge_all_log_skewed', features=ALL_FEATURES,
        model_factory=lambda: Ridge(random_state=SEED), param_grid=ridge_grid,
        log_skewed=['population', 'households'],
        hypothesis='log1p the right-skewed size features before linear fit.'))

    # --- Random Forest: depth / leaf / trees + feature-set variation -------
    rf_grid = {'model__n_estimators': [200, 400],
               'model__max_depth': [4, 6, None],
               'model__min_samples_leaf': [1, 3, 5]}
    cands.append(Candidate(
        name='rf_all_features', features=ALL_FEATURES,
        model_factory=lambda: RandomForestRegressor(random_state=SEED, n_jobs=-1),
        param_grid=rf_grid, scale=False,
        hypothesis='Nonlinear interactions among all four features (trees need no scaling).'))
    cands.append(Candidate(
        name='rf_income_renter', features=['median_income', 'renter_share'],
        model_factory=lambda: RandomForestRegressor(random_state=SEED, n_jobs=-1),
        param_grid=rf_grid, scale=False,
        hypothesis='Can a two-feature forest match the four-feature one?'))
    cands.append(Candidate(
        name='rf_mean_impute', features=ALL_FEATURES,
        model_factory=lambda: RandomForestRegressor(random_state=SEED, n_jobs=-1),
        param_grid=rf_grid, scale=False, impute='mean',
        hypothesis='Does mean vs median imputation change the forest?'))

    return cands


def select_best(logged):
    """MAE-first selection with equity/stability tie-breakers.

    Primary: lowest nested outer-fold mean MAE. Tie-breakers (within 1 MAE unit
    of the best, i.e. not meaningfully different): prefer lower fold-MAE std
    (more consistent across geography) and lower spread of subgroup MAE (more
    even across income/renter/supply groups). Baselines are references, not
    eligible winners.
    """
    eligible = [r for r in logged if not r['name'].startswith('baseline')]
    if not eligible:
        return None
    best_mae = min(r['nested_outer_MAE_mean'] for r in eligible)

    def subgroup_spread(r):
        vals = [v['MAE'] for v in r['subgroup_oof_MAE'].values() if v['MAE'] is not None]
        return max(vals) - min(vals) if vals else np.inf

    # Candidates within 1.0 MAE of the best are treated as not meaningfully
    # different on average, so the geography/equity tie-breakers decide among
    # them: first lower fold-MAE std (more consistent across ZIP3 groups), then
    # lower subgroup-MAE spread (more even across income/renter/supply groups).
    # MAE is used only as the final tie-breaker here, so a 0.3-unit average edge
    # does not override a much steadier model.
    near = [r for r in eligible if r['nested_outer_MAE_mean'] <= best_mae + 1.0]
    near.sort(key=lambda r: (round(r['nested_outer_MAE_std'], 3),
                             round(subgroup_spread(r), 3),
                             r['nested_outer_MAE_mean']))
    return near[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default=str(MODELING_CSV))
    p.add_argument('--results-dir', default=str(RESULTS))
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    (results_dir / 'figures').mkdir(parents=True, exist_ok=True)
    commit = git_commit()

    df = pd.read_csv(args.input, dtype={'zcta': 'string', 'zip3': 'string'})
    cohort = df.loc[df['regression_cohort'] == True].copy()  # noqa: E712
    cohort = cohort.dropna(subset=[TARGET])

    # Reserve the held-out test set ONCE, before any search.
    train, test, test_groups = grouped_test_split(cohort, 'zip3')

    cands = candidate_space()
    for c in cands:
        assert_no_leakage(c.features)  # governance: no candidate may leak

    logged = []
    for c in cands:
        res = nested_cv(c, train)
        row = {
            'name': c.name,
            'hypothesis': c.hypothesis,
            'features': ','.join(c.features),
            'model': c.model_factory().__class__.__name__,
            'param_grid': json.dumps(c.param_grid),
            'impute': c.impute, 'scale': c.scale,
            'log_skewed': ','.join(c.log_skewed) if c.log_skewed else '',
            'seed': SEED, 'git_commit': commit,
            'nested_outer_MAE_mean': res['nested_outer_MAE_mean'],
            'nested_outer_MAE_std': res['nested_outer_MAE_std'],
            'nested_outer_RMSE_mean': res['nested_outer_RMSE_mean'],
            'nested_outer_R2_mean': res['nested_outer_R2_mean'],
            'nested_fold_MAEs': json.dumps(res['nested_fold_MAEs']),
            'biased_single_level_gridsearch_MAE': res['biased_single_level_gridsearch_MAE'],
            'subgroup_oof_MAE': res['subgroup_oof_MAE'],
            'selected': False,
        }
        logged.append(row)
        print(f"{c.name:24s} nestedMAE={row['nested_outer_MAE_mean']:6.2f} "
              f"+/-{row['nested_outer_MAE_std']:4.2f}  R2={row['nested_outer_R2_mean']:5.2f}")

    best = select_best(logged)

    # Final, ONE-TIME held-out test evaluation of the selected procedure only.
    final = None
    if best is not None:
        for r in logged:
            r['selected'] = (r['name'] == best['name'])
        chosen = next(c for c in cands if c.name == best['name'])
        pipe = chosen.build_pipeline()
        if chosen.param_grid:
            from sklearn.model_selection import GridSearchCV, GroupKFold
            inner = GroupKFold(n_splits=4)
            s = GridSearchCV(pipe, chosen.param_grid, cv=inner,
                             scoring='neg_mean_absolute_error', n_jobs=-1)
            s.fit(train[chosen.features], train[TARGET], groups=train['zip3'].values)
            fitted, best_params = s.best_estimator_, s.best_params_
        else:
            fitted, best_params = pipe.fit(train[chosen.features], train[TARGET]), {}
        test_pred = fitted.predict(test[chosen.features])
        final = {
            'selected_candidate': best['name'],
            'selection_rule': ('lowest nested outer-fold mean MAE; ties (within '
                               '1.0 MAE) broken by lower fold-MAE std then lower '
                               'subgroup-MAE spread (equity/stability).'),
            'best_params': {k: v for k, v in best_params.items()},
            'held_out_test': score_all(test[TARGET], test_pred),
            'held_out_test_subgroup_MAE': subgroup_mae(
                test, test[TARGET].to_numpy(), test_pred, train),
            'note': ('One-time test evaluation of the single selected procedure. '
                     'The test set was untouched during the search.'),
        }

    # --- Persist the experiment log and the run summary --------------------
    flat = []
    for r in logged:
        f = {k: v for k, v in r.items() if k != 'subgroup_oof_MAE'}
        for sg, d in r['subgroup_oof_MAE'].items():
            f[f'subMAE_{sg}'] = d['MAE']
            f[f'subN_{sg}'] = d['n']
        flat.append(f)
    pd.DataFrame(flat).to_csv(results_dir / 'experiments.csv', index=False)

    summary = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'input': repo_relative(args.input), 'seed': SEED, 'git_commit': commit,
        'protocol': ('nested ZIP3 GroupKFold (outer=5, inner=4); held-out test '
                     'of whole ZIP3 groups reserved first and scored once.'),
        'n_cohort': int(len(cohort)), 'n_train': int(len(train)),
        'n_test': int(len(test)), 'n_test_groups': len(test_groups),
        'fixed_rules': {
            'target': TARGET,
            'geographic_groups': 'zip3 (first 3 digits of ZCTA)',
            'excluded_leakage_columns': ['ev_count', 'bev_count', 'phev_count'],
            'data_quality': 'verified-zero supply recoding + households>=100 (from prepare_modeling.py)',
            'metric': 'MAE primary; RMSE, R2, fold variability, subgroup MAE supporting',
        },
        'experiments': logged,
        'selection': final,
    }
    with open(results_dir / 'best_config.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    _figures(logged, best, results_dir / 'figures')

    if final:
        print(f"\nSelected: {final['selected_candidate']}  "
              f"params={final['best_params']}")
        print(f"One-time held-out test: {final['held_out_test']}")
        print("Held-out test subgroup MAE:")
        for sg, d in final['held_out_test_subgroup_MAE'].items():
            print(f"  {sg:20s} n={d['n']:3d} MAE={d['MAE']}")
    print(f"\nWrote {repo_relative(results_dir / 'experiments.csv')} and "
          f"{repo_relative(results_dir / 'best_config.json')}.")


def _figures(logged, best, figdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    names = [r['name'] for r in logged]
    maes = [r['nested_outer_MAE_mean'] for r in logged]
    stds = [r['nested_outer_MAE_std'] for r in logged]
    colors = ['#c63' if best and r['name'] == best['name']
              else ('#999' if r['name'].startswith('baseline') else '#48c')
              for r in logged]

    # Fig: nested-CV MAE with fold-variability error bars.
    fig, ax = plt.subplots(figsize=(9, 5))
    order = np.argsort(maes)
    ax.barh([names[i] for i in order], [maes[i] for i in order],
            xerr=[stds[i] for i in order], color=[colors[i] for i in order],
            error_kw={'ecolor': '#333', 'capsize': 3})
    ax.set(xlabel='Nested outer-fold MAE (EVs per 1,000 hh), error bar = fold std',
           title='Experiment comparison (nested ZIP3-grouped CV)')
    ax.invert_yaxis()
    fig.tight_layout(); fig.savefig(figdir / 'experiments_mae.png', dpi=160); plt.close(fig)

    # Fig: subgroup MAE for the selected model vs the best baseline.
    if best:
        sgs = list(best['subgroup_oof_MAE'].keys())
        sel_vals = [best['subgroup_oof_MAE'][s]['MAE'] for s in sgs]
        base = next(r for r in logged if r['name'] == 'baseline_income_only')
        base_vals = [base['subgroup_oof_MAE'][s]['MAE'] for s in sgs]
        x = np.arange(len(sgs)); w = 0.38
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x - w/2, base_vals, w, label='income-only baseline', color='#6a9')
        ax.bar(x + w/2, sel_vals, w, label=f"selected ({best['name']})", color='#c63')
        ax.set_xticks(x); ax.set_xticklabels(sgs, rotation=30, ha='right')
        ax.set(ylabel='Out-of-fold MAE', title='Subgroup (equity) performance')
        ax.legend()
        fig.tight_layout(); fig.savefig(figdir / 'experiments_subgroup_mae.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    main()
