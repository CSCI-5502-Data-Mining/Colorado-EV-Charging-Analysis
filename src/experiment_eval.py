"""Fixed evaluation protocol for the Milestone 3 regression experiments.

This module is the SINGLE source of truth for how a candidate is scored. It is
deliberately separate from the search/proposal code so that no experiment can
silently change the evaluation rules. What is FIXED here and never varied by an
experiment:

  * the target (modeling_common.TARGET)
  * the ZIP3 geographic grouping (modeling_common.zip3)
  * the held-out test split (reserved first, by whole ZIP3 groups)
  * the metrics (MAE primary; RMSE, R2 supporting)
  * the subgroup/equity definitions (modeling_common.SUBGROUP_SPECS)
  * the leakage guard (preprocessing fit inside folds; counts excluded)

Nested cross-validation
-----------------------
A candidate is a *procedure* (a preprocessing + model Pipeline plus an inner
hyperparameter grid), not a single fitted model. We score the PROCEDURE with
nested ZIP3-GroupKFold:

  * OUTER GroupKFold: splits the training rows into outer folds by ZIP3.
  * INNER GroupKFold: within each outer-train block, GridSearchCV tunes the
    hyperparameters.
  * The tuned pipeline is refit on the outer-train block and scored on the
    held-out outer-test fold.

The outer-fold scores are therefore an UNBIASED estimate of the whole
select-and-fit procedure, which is the number safe to report. A plain
(single-level) GridSearchCV that reports its own best CV score reuses the same
folds for tuning and reporting and is optimistic; we keep that number too, but
only labelled as the biased comparison.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from modeling_common import SUBGROUP_SPECS, TARGET, zip3

SEED = 42
OUTER_SPLITS = 5
INNER_SPLITS = 4
TEST_FRACTION = 0.20
MIN_TEST_GROUPS = 4


# --- Metrics (fixed) -------------------------------------------------------
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def score_all(y_true, y_pred) -> Dict[str, float]:
    return {'MAE': round(float(mean_absolute_error(y_true, y_pred)), 3),
            'RMSE': round(rmse(y_true, y_pred), 3),
            'R2': round(float(r2_score(y_true, y_pred)), 3)}


# --- Fixed held-out test split by whole ZIP3 groups ------------------------
def grouped_test_split(df: pd.DataFrame, group_col: str = 'zip3',
                       seed: int = SEED, test_fraction: float = TEST_FRACTION):
    """Hold out whole ZIP3 groups so no geographic group spans train and test.

    FIXED procedure: the same seed always yields the same held-out groups, so
    the test set is reserved once and is identical across every experiment.
    """
    rng = np.random.default_rng(seed)
    groups = np.array(df[group_col].dropna().unique(), dtype=object)
    rng.shuffle(groups)
    n_test = max(MIN_TEST_GROUPS, int(round(len(groups) * test_fraction)))
    n_test = min(n_test, len(groups) - 1)
    test_groups = set(groups[:n_test])
    is_test = df[group_col].isin(test_groups)
    return df.loc[~is_test].copy(), df.loc[is_test].copy(), sorted(test_groups)


# --- Preprocessing builders (fit INSIDE folds; selected per experiment) ----
def _log1p_skewed(features: List[str], skewed: List[str]):
    """A column transformer that log1p-transforms the named skewed columns and
    passes the rest through. Fit happens inside the pipeline, so inside folds."""
    cols_log = [f for f in features if f in skewed]
    cols_id = [f for f in features if f not in skewed]
    transformers = []
    if cols_log:
        transformers.append(('log', FunctionTransformer(np.log1p, feature_names_out='one-to-one'), cols_log))
    if cols_id:
        transformers.append(('id', 'passthrough', cols_id))
    return ColumnTransformer(transformers)


def make_preprocessor(features: List[str], impute: str = 'median',
                      scale: bool = True, log_skewed: Optional[List[str]] = None):
    """Build a preprocessing Pipeline for the given feature list.

    impute: 'median' or 'mean' (fit on training rows of each fold).
    scale:  StandardScaler on/off.
    log_skewed: optional list of columns to log1p before imputation/scaling.
    Every step is fit inside the CV fold, never on validation/test data.
    """
    steps = []
    if log_skewed:
        steps.append(('log', _log1p_skewed(features, log_skewed)))
    steps.append(('impute', SimpleImputer(strategy=impute)))
    if scale:
        steps.append(('scale', StandardScaler()))
    return Pipeline(steps)


# --- Candidate specification ----------------------------------------------
@dataclass
class Candidate:
    """One declarative experiment: a named procedure over a fixed protocol.

    name:        human-readable id, logged.
    features:    the predictor columns (subset of the allowed feature set).
    model_factory: zero-arg callable returning an unfitted sklearn estimator.
    param_grid:  inner-CV hyperparameter grid (keys prefixed 'model__').
    impute/scale/log_skewed: preprocessing choices (varied, but fold-safe).
    hypothesis:  short note on what this experiment tests, logged.
    """
    name: str
    features: List[str]
    model_factory: Callable
    param_grid: Dict[str, list] = field(default_factory=dict)
    impute: str = 'median'
    scale: bool = True
    log_skewed: Optional[List[str]] = None
    hypothesis: str = ''

    def build_pipeline(self) -> Pipeline:
        pre = make_preprocessor(self.features, impute=self.impute,
                                scale=self.scale, log_skewed=self.log_skewed)
        return Pipeline([('pre', pre), ('model', self.model_factory())])


# --- Nested CV scoring of a candidate PROCEDURE ----------------------------
def nested_cv(cand: Candidate, train: pd.DataFrame):
    """Score the candidate procedure with nested ZIP3-GroupKFold on TRAIN only.

    Returns per-outer-fold metrics, their mean/std (fold variability), the
    pooled out-of-fold subgroup MAE, and the biased single-level GridSearch CV
    score for contrast. The held-out TEST set is never touched here.
    """
    X = train[cand.features]
    y = train[TARGET]
    groups = train['zip3'].values

    outer = GroupKFold(n_splits=OUTER_SPLITS)
    inner = GroupKFold(n_splits=INNER_SPLITS)

    fold_scores = []
    oof_pred = np.full(len(train), np.nan)

    for tr_idx, te_idx in outer.split(X, y, groups):
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
        g_tr = groups[tr_idx]

        pipe = cand.build_pipeline()
        if cand.param_grid:
            search = GridSearchCV(pipe, cand.param_grid, cv=inner,
                                  scoring='neg_mean_absolute_error', n_jobs=-1)
            search.fit(X_tr, y_tr, groups=g_tr)
            best = search.best_estimator_
        else:
            best = pipe.fit(X_tr, y_tr)
        pred = best.predict(X_te)
        oof_pred[te_idx] = pred
        fold_scores.append(score_all(y_te, pred))

    maes = [f['MAE'] for f in fold_scores]
    rmses = [f['RMSE'] for f in fold_scores]
    r2s = [f['R2'] for f in fold_scores]

    # Pooled out-of-fold subgroup MAE: every training row has exactly one
    # out-of-fold prediction, so the subgroup error uses genuinely held-out
    # predictions. Thresholds come from the TRAINING rows (never test).
    subgroup = subgroup_mae(train, y.to_numpy(), oof_pred, train)

    # Biased single-level reference: one GridSearchCV over all training folds.
    biased = None
    if cand.param_grid:
        s = GridSearchCV(cand.build_pipeline(), cand.param_grid, cv=inner,
                         scoring='neg_mean_absolute_error', n_jobs=-1)
        s.fit(X, y, groups=groups)
        biased = round(float(-s.best_score_), 3)

    return {
        'nested_outer_MAE_mean': round(float(np.mean(maes)), 3),
        'nested_outer_MAE_std': round(float(np.std(maes)), 3),
        'nested_outer_RMSE_mean': round(float(np.mean(rmses)), 3),
        'nested_outer_R2_mean': round(float(np.mean(r2s)), 3),
        'nested_fold_MAEs': [round(m, 3) for m in maes],
        'biased_single_level_gridsearch_MAE': biased,
        'subgroup_oof_MAE': subgroup,
        'oof_pred': oof_pred,
    }


# --- Subgroup / equity scoring (fixed definitions) -------------------------
def _subgroup_mask(frame: pd.DataFrame, col: str, comp: str, source: str,
                   ref_frame: pd.DataFrame) -> np.ndarray:
    """Boolean mask for a subgroup. Median thresholds come from ref_frame
    (training rows), never from the rows being scored if those are test rows."""
    if source == 'median':
        thr = float(ref_frame[col].median())
    elif source == 'zero':
        thr = 0.0
    else:
        raise ValueError(f'unknown threshold source {source!r}')
    values = frame[col].to_numpy()
    if comp == '<':
        return values < thr
    if comp == '>=':
        return values >= thr
    if comp == '>':
        return values > thr
    if comp == '==':
        return values == thr
    raise ValueError(f'unknown comparison {comp!r}')


def subgroup_mae(frame: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
                 ref_frame: pd.DataFrame) -> Dict[str, Dict]:
    """MAE within each FIXED subgroup. ref_frame supplies median thresholds."""
    out = {}
    abs_err = np.abs(y_pred - y_true)
    valid = ~np.isnan(y_pred)
    for name, (col, comp, source) in SUBGROUP_SPECS.items():
        mask = _subgroup_mask(frame, col, comp, source, ref_frame) & valid
        n = int(mask.sum())
        out[name] = {'n': n,
                     'MAE': round(float(abs_err[mask].mean()), 3) if n else None}
    return out
