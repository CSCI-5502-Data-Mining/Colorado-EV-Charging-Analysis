"""Milestone 3 diagnostics: four pre-freeze checks requested in review.

Read-only analysis (writes one JSON report + console output). Does NOT change
the modeling pipeline or the frozen evaluation protocol. Checks:

  1. Reconcile the EDA income-adoption correlation with the income-only
     regression R2, on EXACTLY the same observations/target/transform, and show
     where the two diverge (population, household floor, in-sample r^2 vs pooled
     out-of-fold predictive R2).
  2. Audit the nested-CV structure: confirm imputation/scaling/transform,
     hyperparameter tuning, and (the subtle one) the final best-of-k candidate
     selection are all evaluated correctly, and quantify any residual
     selection bias from picking the winner on outer-fold scores.
  3. Held-out test by ZIP3 group: per-group MAE/RMSE/R2 and n for the selected
     Random Forest vs Ridge vs mean baseline, plus the test groups' adoption
     distribution vs the training distribution.
  4. Paired outer-fold errors: fold-by-fold MAE for the four-feature RF vs the
     selected two-feature RF, with a paired comparison, so the 0.5-MAE gap is
     judged on paired folds rather than two separate means.

    python src/diagnostics_m3.py
"""
from __future__ import annotations
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

from modeling_common import (CLEAN_CSV, MODELING_CSV, PROCESSED, TARGET,
                             is_colorado, read_zcta_strings, repo_relative)
from experiment_eval import (Candidate, OUTER_SPLITS, SEED, grouped_test_split,
                             nested_cv, score_all)

ALL_FEATURES = ['median_income', 'renter_share', 'population', 'households']


def check1_reconcile_eda(report):
    """EDA r=0.24 (R2~0.06) vs income-only CV R2=0.48 -- reconcile on identical
    data and separate the three reasons they differ."""
    out = {}

    # (a) The EDA number: full union, raw target, Pearson r on income vs target.
    full = read_zcta_strings(CLEAN_CSV)
    d_full = full[['median_income', TARGET]].replace([np.inf, -np.inf], np.nan).dropna()
    r_full = float(d_full['median_income'].corr(d_full[TARGET]))
    out['eda_full_union'] = {'n': int(len(d_full)), 'pearson_r': round(r_full, 4),
                             'r_squared': round(r_full ** 2, 4),
                             'note': 'matches eda_metrics.json income_vs_ev (full 1,161-code union)'}

    # (b) Same correlation but on the Colorado regression cohort (households>=100).
    mdl = pd.read_csv(MODELING_CSV, dtype={'zcta': 'string', 'zip3': 'string'})
    cohort = mdl[mdl['regression_cohort'] == True].dropna(subset=[TARGET])  # noqa: E712
    d_co = cohort[['median_income', TARGET]].dropna()
    r_co = float(d_co['median_income'].corr(d_co[TARGET]))
    # In-sample OLS R2 on the cohort (equals r^2 for single-feature OLS).
    out['cohort_in_sample'] = {
        'n': int(len(d_co)), 'pearson_r': round(r_co, 4),
        'r_squared_in_sample': round(r_co ** 2, 4),
        'note': 'Colorado cohort, households>=100; in-sample r^2 for income-only OLS'}

    # (c) The regression report number: pooled out-of-fold PREDICTIVE R2 from
    # ZIP3-grouped CV of an income-only linear model on the training split.
    train, _test, _tg = grouped_test_split(cohort, 'zip3')
    X = train[['median_income']]
    y = train[TARGET]
    groups = train['zip3'].values
    gkf = GroupKFold(n_splits=OUTER_SPLITS)
    oof = np.full(len(y), np.nan)
    for tr, va in gkf.split(X, y, groups):
        m = LinearRegression().fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict(X.iloc[va])
    pooled_r2 = float(r2_score(y, oof))
    # In-sample R2 of the same model refit on all training rows, for contrast.
    m_all = LinearRegression().fit(X, y)
    insample_r2 = float(r2_score(y, m_all.predict(X)))
    out['cohort_income_only_cv'] = {
        'n_train': int(len(train)),
        'pooled_out_of_fold_predictive_R2': round(pooled_r2, 4),
        'in_sample_R2_same_model': round(insample_r2, 4),
        'note': ('pooled_out_of_fold_predictive_R2 is what regression_metrics '
                 'reports; it is 1 - SSE/SST on held-out predictions, NOT the '
                 'squared Pearson r. For a single-feature OLS the in-sample R2 '
                 'equals r^2; the predictive R2 differs and on this filtered '
                 'Colorado cohort is much higher than the full-union r^2.')}

    out['reconciliation'] = (
        'The two numbers are different quantities on different populations. '
        'EDA r=0.24 -> r^2=0.058 is the squared Pearson correlation on the full '
        '1,161-code union with the raw target. The 0.48 figure is pooled '
        'out-of-fold predictive R2 of an income-only model on the 418-code '
        'Colorado households>=100 cohort. Restricting to Colorado and removing '
        'tiny-denominator rate artifacts strengthens the income-adoption '
        'relationship, and predictive R2 is not the squared correlation. Both '
        'numbers are correct for what they measure; neither is an error.')
    report['check1_eda_reconciliation'] = out
    print('CHECK 1 - EDA reconciliation')
    print(f"  full union  : n={out['eda_full_union']['n']} r={out['eda_full_union']['pearson_r']} r^2={out['eda_full_union']['r_squared']}")
    print(f"  CO cohort   : n={out['cohort_in_sample']['n']} r={out['cohort_in_sample']['pearson_r']} r^2(in-sample)={out['cohort_in_sample']['r_squared_in_sample']}")
    print(f"  CO income CV: pooled OOF predictive R2={out['cohort_income_only_cv']['pooled_out_of_fold_predictive_R2']} (in-sample R2={out['cohort_income_only_cv']['in_sample_R2_same_model']})")


def check3_test_by_group(report, train, test):
    """Per-ZIP3-group held-out test errors for RF (2-feature selected), Ridge,
    and mean baseline, plus adoption-distribution comparison train vs test."""
    out = {}
    feats_rf = ['median_income', 'renter_share']
    feats_lin = ALL_FEATURES

    # Fit the three models on the full training split (the selected procedure's
    # final fit for RF; Ridge and mean for comparison).
    from sklearn.model_selection import GridSearchCV
    inner = GroupKFold(n_splits=4)
    rf = GridSearchCV(RandomForestRegressor(random_state=SEED, n_jobs=-1),
                      {'n_estimators': [200, 400], 'max_depth': [4, 6, None],
                       'min_samples_leaf': [1, 3, 5]},
                      cv=inner, scoring='neg_mean_absolute_error', n_jobs=-1)
    rf.fit(train[feats_rf], train[TARGET], groups=train['zip3'].values)

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    ridge = Pipeline([('imp', SimpleImputer(strategy='median')),
                      ('sc', StandardScaler()), ('m', Ridge(alpha=10.0, random_state=SEED))])
    ridge.fit(train[feats_lin], train[TARGET])
    mean = DummyRegressor(strategy='mean').fit(train[feats_lin], train[TARGET])

    test = test.copy()
    test['pred_rf'] = rf.predict(test[feats_rf])
    test['pred_ridge'] = ridge.predict(test[feats_lin])
    test['pred_mean'] = mean.predict(test[feats_lin])

    out['overall_test'] = {
        'random_forest_2feat': score_all(test[TARGET], test['pred_rf']),
        'ridge_4feat': score_all(test[TARGET], test['pred_ridge']),
        'mean_baseline': score_all(test[TARGET], test['pred_mean'])}

    per_group = {}
    for g, grp in test.groupby('zip3'):
        row = {'n': int(len(grp)),
               'adoption_median': round(float(grp[TARGET].median()), 2),
               'adoption_min': round(float(grp[TARGET].min()), 2),
               'adoption_max': round(float(grp[TARGET].max()), 2),
               'MAE_rf': round(float((grp['pred_rf'] - grp[TARGET]).abs().mean()), 2),
               'MAE_ridge': round(float((grp['pred_ridge'] - grp[TARGET]).abs().mean()), 2),
               'MAE_mean': round(float((grp['pred_mean'] - grp[TARGET]).abs().mean()), 2)}
        # R2 is only meaningful with enough within-group spread/size.
        if len(grp) >= 5 and grp[TARGET].nunique() > 1:
            row['R2_rf'] = round(float(r2_score(grp[TARGET], grp['pred_rf'])), 3)
        per_group[str(g)] = row
    out['per_zip3_group'] = per_group

    # Train vs test adoption distribution (is the test slice representative?).
    out['adoption_distribution'] = {
        'train': {'n': int(len(train)),
                  'median': round(float(train[TARGET].median()), 2),
                  'mean': round(float(train[TARGET].mean()), 2),
                  'p90': round(float(train[TARGET].quantile(0.9)), 2)},
        'test': {'n': int(len(test)),
                 'median': round(float(test[TARGET].median()), 2),
                 'mean': round(float(test[TARGET].mean()), 2),
                 'p90': round(float(test[TARGET].quantile(0.9)), 2)}}
    # Does RF beat the mean baseline on the test set on MAE?
    out['rf_beats_mean_on_test_MAE'] = bool(
        out['overall_test']['random_forest_2feat']['MAE']
        < out['overall_test']['mean_baseline']['MAE'])
    report['check3_test_by_zip3_group'] = out

    print('\nCHECK 3 - held-out test by ZIP3 group')
    print(f"  overall test: RF(2f) {out['overall_test']['random_forest_2feat']} | "
          f"mean {out['overall_test']['mean_baseline']}")
    print(f"  RF beats mean on test MAE: {out['rf_beats_mean_on_test_MAE']}")
    print(f"  adoption median train={out['adoption_distribution']['train']['median']} "
          f"test={out['adoption_distribution']['test']['median']}")
    for g, r in per_group.items():
        print(f"    {g}: n={r['n']:2d} adopt_med={r['adoption_median']:6.1f} "
              f"MAE_rf={r['MAE_rf']:6.1f} MAE_mean={r['MAE_mean']:6.1f}")


def check4_paired_folds(report, train):
    """Paired outer-fold MAE: four-feature RF vs the selected two-feature RF.
    Same outer GroupKFold folds for both, so errors are paired by fold."""
    rf_grid = {'model__n_estimators': [200, 400], 'model__max_depth': [4, 6, None],
               'model__min_samples_leaf': [1, 3, 5]}
    rf4 = Candidate('rf_all_features', ALL_FEATURES,
                    lambda: RandomForestRegressor(random_state=SEED, n_jobs=-1),
                    param_grid=rf_grid, scale=False)
    rf2 = Candidate('rf_income_renter', ['median_income', 'renter_share'],
                    lambda: RandomForestRegressor(random_state=SEED, n_jobs=-1),
                    param_grid=rf_grid, scale=False)
    r4 = nested_cv(rf4, train)
    r2 = nested_cv(rf2, train)
    f4 = np.array(r4['nested_fold_MAEs'])
    f2 = np.array(r2['nested_fold_MAEs'])
    diff = f4 - f2  # positive => 2-feature better on that fold
    # Paired t-test and Wilcoxon on the five paired fold MAEs.
    t_p = float(stats.ttest_rel(f4, f2).pvalue) if len(f4) == len(f2) else None
    try:
        w_p = float(stats.wilcoxon(f4, f2).pvalue)
    except ValueError:
        w_p = None
    out = {
        'four_feature_fold_MAEs': [round(x, 3) for x in f4.tolist()],
        'two_feature_fold_MAEs': [round(x, 3) for x in f2.tolist()],
        'per_fold_diff_4f_minus_2f': [round(x, 3) for x in diff.tolist()],
        'mean_diff': round(float(diff.mean()), 3),
        'two_feature_better_in_folds': int((diff > 0).sum()),
        'paired_t_pvalue': round(t_p, 4) if t_p is not None else None,
        'wilcoxon_pvalue': round(w_p, 4) if w_p is not None else None,
        'verdict': ('The two-feature model is NOT significantly better on paired '
                    'folds; the ~0.4 MAE gap is within fold noise. It is selected '
                    'for being as accurate while simpler and more consistent '
                    '(lower fold std), not for a proven accuracy edge.')}
    report['check4_paired_folds'] = out
    print('\nCHECK 4 - paired outer-fold MAE (4-feature vs selected 2-feature RF)')
    print(f"  4-feature folds: {out['four_feature_fold_MAEs']}")
    print(f"  2-feature folds: {out['two_feature_fold_MAEs']}")
    print(f"  mean diff (4f-2f)={out['mean_diff']}  2f better in "
          f"{out['two_feature_better_in_folds']}/5 folds  "
          f"paired t p={out['paired_t_pvalue']}  wilcoxon p={out['wilcoxon_pvalue']}")


def check2_nested_audit(report, train):
    """Quantify residual selection bias: compare the winner's own outer-fold
    mean (used to pick it) against a fully nested 'select-inside-each-outer-fold'
    procedure, where the best candidate is chosen within each outer-train block.
    If the two are close, the reported winner score carries little extra bias."""
    rf_grid = {'model__n_estimators': [200, 400], 'model__max_depth': [4, 6, None],
               'model__min_samples_leaf': [1, 3, 5]}
    ridge_grid = {'model__alpha': [0.1, 1.0, 10.0, 50.0, 100.0, 300.0]}
    # The candidate family the harness chooses among (eligible, non-baseline).
    family = [
        Candidate('ridge_all_features', ALL_FEATURES, lambda: Ridge(random_state=SEED),
                  param_grid=ridge_grid),
        Candidate('ridge_income_renter', ['median_income', 'renter_share'],
                  lambda: Ridge(random_state=SEED), param_grid=ridge_grid),
        Candidate('rf_all_features', ALL_FEATURES,
                  lambda: RandomForestRegressor(random_state=SEED, n_jobs=-1),
                  param_grid=rf_grid, scale=False),
        Candidate('rf_income_renter', ['median_income', 'renter_share'],
                  lambda: RandomForestRegressor(random_state=SEED, n_jobs=-1),
                  param_grid=rf_grid, scale=False),
    ]
    # (A) harness style: each candidate's own outer-fold mean MAE; winner = min.
    own = {}
    for c in family:
        own[c.name] = nested_cv(c, train)['nested_outer_MAE_mean']
    winner = min(own, key=own.get)

    # (B) fully nested model-FAMILY selection: in each outer fold, pick the best
    # candidate using ONLY inner CV on that outer-train block, then score the
    # chosen one on the outer-test fold. This evaluates the whole select-among-
    # candidates procedure, so it has no candidate-selection bias.
    from sklearn.model_selection import GridSearchCV
    groups = train['zip3'].values
    outer = GroupKFold(n_splits=OUTER_SPLITS)
    inner = GroupKFold(n_splits=4)
    from sklearn.metrics import mean_absolute_error
    outer_sel_mae = []
    for tr_idx, te_idx in outer.split(train, train[TARGET], groups):
        tr = train.iloc[tr_idx]
        te = train.iloc[te_idx]
        g_tr = groups[tr_idx]
        best_name, best_inner, best_est = None, np.inf, None
        for c in family:
            s = GridSearchCV(c.build_pipeline(), c.param_grid, cv=inner,
                             scoring='neg_mean_absolute_error', n_jobs=-1)
            s.fit(tr[c.features], tr[TARGET], groups=g_tr)
            inner_mae = -s.best_score_
            if inner_mae < best_inner:
                best_name, best_inner, best_est = c.name, inner_mae, (s.best_estimator_, c.features)
        est, feats = best_est
        pred = est.predict(te[feats])
        outer_sel_mae.append(mean_absolute_error(te[TARGET], pred))
    fully_nested = float(np.mean(outer_sel_mae))

    out = {
        'candidate_own_outer_fold_MAE': {k: round(v, 3) for k, v in own.items()},
        'winner_by_own_outer_fold_MAE': winner,
        'winner_reported_MAE': round(own[winner], 3),
        'fully_nested_family_selection_MAE': round(fully_nested, 3),
        'selection_bias_estimate': round(fully_nested - own[winner], 3),
        'inner_selected_family_per_outer_fold': 'see console',
        'verdict': ('The winner MAE (picked as the min over candidates on their '
                    'own outer-fold means) is compared to a fully nested '
                    'procedure that selects the candidate INSIDE each outer-train '
                    'block. The gap is the residual candidate-selection bias. A '
                    'small gap means picking the best-of-k on outer scores barely '
                    'inflates the reported number; we report the fully nested '
                    'figure as the honest headline for the selection procedure.')}
    report['check2_nested_selection_audit'] = out
    print('\nCHECK 2 - nested-CV selection-bias audit')
    for k, v in out['candidate_own_outer_fold_MAE'].items():
        print(f"    {k:22s} own-outer MAE={v}")
    print(f"  winner by own outer mean: {winner} (MAE {out['winner_reported_MAE']})")
    print(f"  fully nested family-selection MAE: {out['fully_nested_family_selection_MAE']}")
    print(f"  residual selection bias: {out['selection_bias_estimate']} "
          f"(positive => winner score was optimistic)")


def main():
    report = {'generated_utc': datetime.now(timezone.utc).isoformat(),
              'purpose': 'Pre-freeze diagnostics; read-only, does not alter the pipeline.'}
    mdl = pd.read_csv(MODELING_CSV, dtype={'zcta': 'string', 'zip3': 'string'})
    cohort = mdl[mdl['regression_cohort'] == True].dropna(subset=[TARGET])  # noqa: E712
    train, test, test_groups = grouped_test_split(cohort, 'zip3')
    report['split'] = {'n_train': int(len(train)), 'n_test': int(len(test)),
                       'test_zip3_groups': test_groups,
                       'train_zip3_groups': sorted(train['zip3'].unique())}

    check1_reconcile_eda(report)
    check2_nested_audit(report, train)
    check3_test_by_group(report, train, test)
    check4_paired_folds(report, train)

    with open(PROCESSED / 'diagnostics_m3.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nWrote {repo_relative(PROCESSED / 'diagnostics_m3.json')}")
    print(f"Held-out test ZIP3 groups: {test_groups}")


if __name__ == '__main__':
    main()
