"""Milestone 3 - Step 2: regression on EV adoption rate (leakage-safe).

Target: ev_per_1000_households (registered EVs per 1,000 households).
Predictors: median_income, renter_share, population, households.
ev_count/bev_count/phev_count are EXCLUDED (the target is derived from them).

Four approaches compared on one protocol (plan Tasks 4-6):
  * Baseline 1  - mean predictor (DummyRegressor, strategy='mean')
  * Baseline 2  - income-only linear regression (single feature)
  * Model 1     - Ridge regression (tune alpha)
  * Model 2     - Random Forest regressor (tune depth / leaf / trees)

Leakage controls:
  1. A held-out TEST set is carved off FIRST (grouped by ZIP3 so whole
     geographic groups are held out), and never touched during tuning.
  2. All preprocessing (median imputation + scaling) lives INSIDE an sklearn
     Pipeline, so fold statistics are fit on training rows only.
  3. Hyperparameter search runs with GridSearchCV over GroupKFold on the
     TRAINING portion only.
  4. Primary validation = ZIP3 GroupKFold (neighbouring codes stay together);
     a plain random KFold is reported alongside purely for contrast.
  5. Winsorization sensitivity: the target cap is learned from TRAINING rows
     only and predictions are scored against the ORIGINAL uncapped test target,
     so capping cannot make the evaluation artificially easier.

    python src/train_regression.py
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modeling_common import (
    FIGURES, MODELING_CSV, PROCESSED, REGRESSION_FEATURES, TARGET,
    assert_no_leakage, repo_relative,
)
from experiment_eval import Candidate, nested_cv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from plot_style import apply_web_style  # noqa: E402
apply_web_style()

SEED = 42
N_SPLITS = 5
TEST_FRACTION = 0.20  # held-out ZIP3 groups ~ 20% of groups
MIN_TEST_GROUPS = 4   # a single grouped test slice is high-variance; keep >=4


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def score_all(y_true, y_pred):
    return {'MAE': round(float(mean_absolute_error(y_true, y_pred)), 3),
            'RMSE': round(rmse(y_true, y_pred), 3),
            'R2': round(float(r2_score(y_true, y_pred)), 3)}


def make_preprocessor(features):
    """Median imputation + standard scaling, fit inside each training fold."""
    return ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')),
                          ('scale', StandardScaler())]), features)])


def grouped_test_split(df, group_col, seed=SEED, test_fraction=TEST_FRACTION):
    """Hold out whole ZIP3 groups so no geographic group spans train and test."""
    rng = np.random.default_rng(seed)
    # Convert the pandas StringArray to a plain numpy object array before
    # shuffling (rng.shuffle warns and may misbehave on a StringArray).
    groups = np.array(df[group_col].dropna().unique(), dtype=object)
    rng.shuffle(groups)
    n_test = max(MIN_TEST_GROUPS, int(round(len(groups) * test_fraction)))
    n_test = min(n_test, len(groups) - 1)  # always leave training groups
    test_groups = set(groups[:n_test])
    is_test = df[group_col].isin(test_groups)
    return df.loc[~is_test].copy(), df.loc[is_test].copy(), sorted(test_groups)


def cv_predictions(estimator, X, y, splitter, groups=None):
    """Out-of-fold predictions so every training row is scored once while held
    out of its own fold. Preprocessing is refit per fold (no leakage)."""
    oof = np.full(len(y), np.nan)
    split_iter = (splitter.split(X, y, groups) if groups is not None
                  else splitter.split(X, y))
    for train_idx, val_idx in split_iter:
        est = estimator
        est.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[val_idx] = est.predict(X.iloc[val_idx])
    return oof


def evaluate_cv(build_estimator, X, y, groups):
    """Return grouped-CV and random-CV out-of-fold metrics for one approach."""
    gkf = GroupKFold(n_splits=N_SPLITS)
    grouped = cv_predictions(build_estimator(), X, y, gkf, groups=groups)
    rkf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    random = cv_predictions(build_estimator(), X, y, rkf)
    return {'grouped_cv': score_all(y, grouped),
            'random_cv': score_all(y, random)}


def tune(pipeline, param_grid, X, y, groups):
    """GridSearchCV over ZIP3 GroupKFold on the TRAINING portion only."""
    gkf = GroupKFold(n_splits=N_SPLITS)
    search = GridSearchCV(pipeline, param_grid, scoring='neg_mean_absolute_error',
                          cv=gkf, n_jobs=-1)
    search.fit(X, y, groups=groups)
    return search


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default=str(MODELING_CSV))
    p.add_argument('--metrics', default=str(PROCESSED / 'regression_metrics.json'))
    p.add_argument('--figures-only', action='store_true',
                   help='Redraw figures 06 and 09 from the committed --metrics '
                        'JSON without refitting. Figures 07-08 plot held-out '
                        'predictions, which are not stored, so they still need '
                        'a full run.')
    args = p.parse_args()

    if args.figures_only:
        with open(args.metrics, encoding='utf-8') as fh:
            saved = json.load(fh)
        FIGURES.mkdir(parents=True, exist_ok=True)
        _fig_model_comparison(saved['nested_cv']['results'])
        _fig_feature_importance(saved['rf_feature_importance'])
        print('Redrew figures 06 and 09 from', repo_relative(args.metrics))
        return

    assert_no_leakage(REGRESSION_FEATURES)
    FIGURES.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, dtype={'zcta': 'string', 'zip3': 'string'})
    cohort = df.loc[df['regression_cohort'] == True].copy()  # noqa: E712
    cohort = cohort.dropna(subset=[TARGET])

    report = {'generated_utc': datetime.now(timezone.utc).isoformat(),
              'input': repo_relative(args.input),
              'target': TARGET,
              'features': list(REGRESSION_FEATURES),
              'excluded_leakage_columns': ['ev_count', 'bev_count', 'phev_count'],
              'seed': SEED, 'cv_splits': N_SPLITS,
              'validation': ('Held-out ZIP3 groups reserved first; GroupKFold by '
                             'ZIP3 is primary, random KFold reported for contrast; '
                             'preprocessing + tuning fit on training folds only.'),
              'validation_note': ('Colorado has only ~17 ZIP3 groups, so a single '
                                  'held-out grouped test slice is high-variance; the '
                                  'grouped-CV out-of-fold metrics (every training row '
                                  'scored once while held out) are the primary, more '
                                  'stable estimate. The test split is a confirmatory '
                                  'check that tuning did not overfit.'),
              'n_cohort': int(len(cohort))}

    # --- 1. Held-out test split FIRST (whole ZIP3 groups) ------------------
    train, test, test_groups = grouped_test_split(cohort, 'zip3')
    report['n_train'] = int(len(train))
    report['n_test'] = int(len(test))
    report['n_test_groups'] = len(test_groups)
    report['n_train_groups'] = int(train['zip3'].nunique())

    X_tr, y_tr = train[REGRESSION_FEATURES], train[TARGET]
    X_te, y_te = test[REGRESSION_FEATURES], test[TARGET]
    g_tr = train['zip3'].values

    results = {}

    # --- Baseline 1: mean predictor ---------------------------------------
    def build_mean():
        return Pipeline([('pre', make_preprocessor(REGRESSION_FEATURES)),
                         ('model', DummyRegressor(strategy='mean'))])
    mean_cv = evaluate_cv(build_mean, X_tr, y_tr, g_tr)
    mean_fit = build_mean().fit(X_tr, y_tr)
    results['baseline_mean'] = {**mean_cv, 'test': score_all(y_te, mean_fit.predict(X_te))}

    # --- Baseline 2: income-only linear regression ------------------------
    income = ['median_income']
    def build_income():
        return Pipeline([('pre', make_preprocessor(income)),
                         ('model', LinearRegression())])
    inc_cv = evaluate_cv(lambda: build_income(), X_tr[income], y_tr, g_tr)
    inc_fit = build_income().fit(X_tr[income], y_tr)
    results['baseline_income_only'] = {**inc_cv, 'test': score_all(y_te, inc_fit.predict(X_te[income]))}

    # --- Model 1: Ridge (tune alpha) --------------------------------------
    ridge_pipe = Pipeline([('pre', make_preprocessor(REGRESSION_FEATURES)),
                           ('model', Ridge(random_state=SEED))])
    ridge_grid = {'model__alpha': [0.1, 1.0, 10.0, 50.0, 100.0, 300.0]}
    ridge_search = tune(ridge_pipe, ridge_grid, X_tr, y_tr, g_tr)
    def build_ridge():
        return Pipeline([('pre', make_preprocessor(REGRESSION_FEATURES)),
                         ('model', Ridge(random_state=SEED, **{k.split('__')[1]: v
                          for k, v in ridge_search.best_params_.items()}))])
    ridge_cv = evaluate_cv(build_ridge, X_tr, y_tr, g_tr)
    ridge_fit = build_ridge().fit(X_tr, y_tr)
    results['ridge'] = {**ridge_cv, 'test': score_all(y_te, ridge_fit.predict(X_te)),
                        'best_params': ridge_search.best_params_,
                        'search': 'GridSearchCV over ZIP3 GroupKFold (neg MAE)'}

    # --- Model 2: Random Forest (tune depth / leaf / trees) ---------------
    rf_pipe = Pipeline([('pre', make_preprocessor(REGRESSION_FEATURES)),
                        ('model', RandomForestRegressor(random_state=SEED, n_jobs=-1))])
    rf_grid = {'model__n_estimators': [200, 400],
               'model__max_depth': [4, 6, None],
               'model__min_samples_leaf': [1, 3, 5]}
    rf_search = tune(rf_pipe, rf_grid, X_tr, y_tr, g_tr)
    rf_best = {k.split('__')[1]: v for k, v in rf_search.best_params_.items()}
    def build_rf():
        return Pipeline([('pre', make_preprocessor(REGRESSION_FEATURES)),
                         ('model', RandomForestRegressor(random_state=SEED, n_jobs=-1, **rf_best))])
    rf_cv = evaluate_cv(build_rf, X_tr, y_tr, g_tr)
    rf_fit = build_rf().fit(X_tr, y_tr)
    results['random_forest'] = {**rf_cv, 'test': score_all(y_te, rf_fit.predict(X_te)),
                                'best_params': rf_search.best_params_,
                                'search': 'GridSearchCV over ZIP3 GroupKFold (neg MAE)'}

    report['results'] = results

    # --- Nested ZIP3-grouped CV (UNBIASED headline for Milestone 3) --------
    # The single-level grouped_cv above reuses the same folds for tuning and
    # reporting, so for the tuned models it is slightly optimistic. Nested CV
    # scores the whole select-and-fit PROCEDURE: an outer GroupKFold holds out
    # folds while an inner GroupKFold tunes inside each outer-train block. These
    # outer-fold means are the numbers we report as unbiased; the single-level
    # grouped_cv is kept as a labelled comparison. The held-out TEST set is not
    # touched here. (Shared protocol: experiment_eval.nested_cv.)
    nested_specs = [
        ('baseline_mean', REGRESSION_FEATURES,
         lambda: DummyRegressor(strategy='mean'), {}, True),
        ('baseline_income_only', ['median_income'], LinearRegression, {}, True),
        ('ridge', REGRESSION_FEATURES, lambda: Ridge(random_state=SEED),
         {'model__alpha': [0.1, 1.0, 10.0, 50.0, 100.0, 300.0]}, True),
        ('random_forest', REGRESSION_FEATURES,
         lambda: RandomForestRegressor(random_state=SEED, n_jobs=-1),
         {'model__n_estimators': [200, 400], 'model__max_depth': [4, 6, None],
          'model__min_samples_leaf': [1, 3, 5]}, False),
    ]
    nested = {}
    for name, feats, factory, grid, scale in nested_specs:
        cand = Candidate(name=name, features=feats, model_factory=factory,
                         param_grid=grid, scale=scale)
        res = nested_cv(cand, train)
        nested[name] = {
            'nested_outer_MAE_mean': res['nested_outer_MAE_mean'],
            'nested_outer_MAE_std': res['nested_outer_MAE_std'],
            'nested_outer_RMSE_mean': res['nested_outer_RMSE_mean'],
            'nested_outer_R2_mean': res['nested_outer_R2_mean'],
            'nested_fold_MAEs': res['nested_fold_MAEs'],
            'biased_single_level_grouped_cv': results[name]['grouped_cv'],
            'subgroup_oof_MAE': res['subgroup_oof_MAE'],
        }
    report['nested_cv'] = {
        'protocol': ('nested ZIP3 GroupKFold (outer=5, inner=4) on the training '
                     'set; outer-fold means are the unbiased estimate of the '
                     'tune-and-fit procedure. The held-out test set is untouched.'),
        'headline': 'Use nested_outer_* for Milestone 3 reporting.',
        'comparison_note': ('biased_single_level_grouped_cv is the earlier '
                            'single-level number (same folds tuned and scored); '
                            'it is optimistic for the tuned models and kept only '
                            'for comparison.'),
        'results': nested}

    # --- Winsorization sensitivity (caps from TRAIN only; score UNCAPPED) --
    cap = float(np.quantile(y_tr, 0.99))  # learned on training target only
    y_tr_capped = y_tr.clip(upper=cap)
    wins = {}
    for name, builder, cols in [('ridge', build_ridge, REGRESSION_FEATURES),
                                ('random_forest', build_rf, REGRESSION_FEATURES)]:
        est = builder()
        est.fit(X_tr[cols], y_tr_capped)                 # fit on capped train
        wins[name] = score_all(y_te, est.predict(X_te[cols]))  # score uncapped test
    report['winsorization_sensitivity'] = {
        'train_cap_value_p99': round(cap, 3),
        'note': ('Cap learned from TRAIN target 99th pct only; models fit on '
                 'capped train but scored against ORIGINAL uncapped test target.'),
        'uncapped_baseline_test': {'ridge': results['ridge']['test'],
                                   'random_forest': results['random_forest']['test']},
        'capped_train_test': wins}

    # --- Interpretation: Ridge coefficients + RF importances ---------------
    ridge_model = ridge_fit.named_steps['model']
    report['ridge_coefficients'] = {f: round(float(c), 3)
                                    for f, c in zip(REGRESSION_FEATURES, ridge_model.coef_)}
    rf_model = rf_fit.named_steps['model']
    report['rf_feature_importance'] = {f: round(float(i), 3)
                                       for f, i in zip(REGRESSION_FEATURES, rf_model.feature_importances_)}
    perm = permutation_importance(rf_fit, X_te, y_te, n_repeats=20,
                                  random_state=SEED, scoring='neg_mean_absolute_error')
    report['rf_permutation_importance_test'] = {f: round(float(m), 3)
                                                for f, m in zip(REGRESSION_FEATURES, perm.importances_mean)}

    # --- Subgroup error analysis (RF out-of-sample on the test set) --------
    test = test.copy()
    test['pred_rf'] = rf_fit.predict(X_te)
    test['abs_err'] = (test['pred_rf'] - y_te).abs()
    med_income = train['median_income'].median()
    med_renter = train['renter_share'].median()
    med_hh = train['households'].median()
    subgroups = {
        'lower_income': test['median_income'] < med_income,
        'higher_income': test['median_income'] >= med_income,
        'renter_heavy': test['renter_share'] >= med_renter,
        'owner_heavy': test['renter_share'] < med_renter,
        'smaller_zcta': test['households'] < med_hh,
        'larger_zcta': test['households'] >= med_hh,
    }
    report['subgroup_mae_rf_test'] = {
        name: {'n': int(mask.sum()),
               'MAE': round(float(test.loc[mask, 'abs_err'].mean()), 3) if mask.any() else None}
        for name, mask in subgroups.items()}

    with open(args.metrics, 'w') as f:
        json.dump(report, f, indent=2)

    _figures(results, nested, y_te, {'ridge': ridge_fit, 'random_forest': rf_fit},
             X_te, REGRESSION_FEATURES, report['rf_feature_importance'])

    # --- Console comparison table (nested = headline) ----------------------
    print(f"n_cohort={len(cohort)}  train={len(train)}  test={len(test)} "
          f"({report['n_test_groups']} held-out ZIP3 groups)")
    print(f"{'approach':22s} {'nMAE':>7s} {'+/-std':>7s} {'nR2':>6s} "
          f"| {'biasedMAE_grp':>13s} | {'MAE_test':>9s}")
    for key in ['baseline_mean', 'baseline_income_only', 'ridge', 'random_forest']:
        n = nested[key]
        r = results[key]
        print(f"{key:22s} {n['nested_outer_MAE_mean']:7.2f} "
              f"{n['nested_outer_MAE_std']:7.2f} {n['nested_outer_R2_mean']:6.2f} "
              f"| {r['grouped_cv']['MAE']:13.2f} | {r['test']['MAE']:9.2f}")
    print("nMAE/nR2 = unbiased nested ZIP3-grouped CV; biasedMAE_grp = "
          "single-level grouped CV (comparison only).")
    print(f"Wrote {repo_relative(args.metrics)} and regression figures.")


def _fig_model_comparison(nested):
    # Fig: model comparison (nested-CV MAE with fold-std error bars = headline)
    keys = ['baseline_mean', 'baseline_income_only', 'ridge', 'random_forest']
    labels = ['Mean', 'Income-only', 'Ridge', 'Random Forest']
    maes = [nested[k]['nested_outer_MAE_mean'] for k in keys]
    stds = [nested[k]['nested_outer_MAE_std'] for k in keys]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.bar(labels, maes, yerr=stds, color=['#999', '#6a9', '#48c', '#c63'],
           error_kw={'ecolor': '#333', 'capsize': 4})
    ax.set(ylabel='Nested-CV MAE (EVs per 1,000 households)',
           title='Regression comparison\nnested ZIP3-grouped CV (lower is better)')
    for i, v in enumerate(maes):
        ax.text(i, v + stds[i], f'{v:.1f}', ha='center', va='bottom')
    fig.tight_layout(); fig.savefig(FIGURES / '06_model_comparison.png', dpi=160); plt.close(fig)


def _fig_feature_importance(rf_importance):
    # Fig: RF feature importance
    items = sorted(rf_importance.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.barh([k for k, _ in items], [v for _, v in items], color='#c63')
    ax.set(xlabel='Random Forest feature importance',
           title='What drives predicted EV adoption')
    fig.tight_layout(); fig.savefig(FIGURES / '09_feature_importance.png', dpi=160); plt.close(fig)


def _figures(results, nested, y_te, fitted, X_te, features, rf_importance):
    _fig_model_comparison(nested)

    # Fig: predicted vs actual (both candidate models on the test set)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, (name, est) in zip(axes, fitted.items()):
        pred = est.predict(X_te)
        ax.scatter(y_te, pred, alpha=.6, s=18)
        lim = [0, max(float(y_te.max()), float(np.max(pred))) * 1.05]
        ax.plot(lim, lim, 'k--', lw=1)
        ax.set(xlabel='Actual EVs per 1,000 hh', ylabel='Predicted',
               title=f'{name} (held-out test)')
    fig.tight_layout(); fig.savefig(FIGURES / '07_predicted_vs_actual.png', dpi=160); plt.close(fig)

    # Fig: residuals (Random Forest on the test set)
    rf = fitted['random_forest']; pred = rf.predict(X_te); resid = pred - y_te
    fig, ax = plt.subplots(figsize=(7.2, 5))
    ax.scatter(pred, resid, alpha=.6, s=18); ax.axhline(0, color='k', lw=1)
    ax.set(xlabel='Predicted EVs per 1,000 hh', ylabel='Residual (pred - actual)',
           title='Random Forest residuals (held-out test)')
    fig.tight_layout(); fig.savefig(FIGURES / '08_residuals.png', dpi=160); plt.close(fig)

    _fig_feature_importance(rf_importance)


if __name__ == '__main__':
    main()
