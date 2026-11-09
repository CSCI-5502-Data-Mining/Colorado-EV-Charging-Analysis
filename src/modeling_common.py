"""Shared helpers for the Milestone 3 modeling scripts.

Centralizes repo-root resolution, the Colorado-area working definition, the
feature/target column contract, and leakage guards so prepare_modeling.py,
train_regression.py, and cluster_communities.py stay consistent.

Nothing here fits a model or touches a test/validation split; it only loads and
labels data. See docs/limitations.md for the ZIP vs ZCTA caveat that makes the
Colorado prefix filter a *working* definition, not a verified boundary.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

# Repository root = parent of the src/ directory that holds this file.
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / 'data' / 'processed'
FIGURES = ROOT / 'figures'

CLEAN_CSV = PROCESSED / 'colorado_ev_zcta_clean.csv'
MODELING_CSV = PROCESSED / 'modeling_colorado.csv'

# Colorado ZCTA prefixes. USPS ZIPs and Census ZCTAs are not identical, so this
# is a working definition for Milestone 3 and documented as a limitation; a
# ZIP-to-county crosswalk or ZCTA polygons would be the accurate upgrade.
CO_PREFIXES = ('80', '81')

# --- Column contract -------------------------------------------------------
# Target for the regression task.
TARGET = 'ev_per_1000_households'

# Features allowed as regression predictors. Deliberately EXCLUDES ev_count,
# bev_count, phev_count because the target is derived directly from ev_count
# (ev_count / households * 1000) -- including any of them is target leakage.
REGRESSION_FEATURES = ['median_income', 'renter_share', 'population', 'households']

# Columns that must never be used as predictors for the adoption-rate target.
LEAKAGE_COLUMNS = ['ev_count', 'bev_count', 'phev_count', TARGET]

# Features for clustering: adoption + socioeconomic + verified public supply.
CLUSTER_FEATURES = ['ev_per_1000_households', 'median_income', 'renter_share',
                    'ports_per_1000_households']


def repo_relative(path) -> str:
    """Repo-root-relative POSIX path so logged outputs carry no machine paths."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).name


def git_commit() -> str:
    """Short git commit of the repo for experiment provenance, or a sentinel.

    Experiment provenance must be reproducible; recording the commit lets a
    reader tie a logged result back to the exact code that produced it.
    """
    import subprocess
    try:
        out = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'],
                             cwd=str(ROOT), capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else 'no-git-commit'
    except Exception:
        return 'no-git-commit'


# --- Fixed subgroup definitions (equity track) -----------------------------
# These are FIXED analysis rules, not experiment-tunable. Experiments report
# subgroup performance against these exact definitions so an agent cannot
# silently drop a hard geographic area or redraw the subgroup boundaries.
# Each maps a subgroup name to (column, comparison, threshold_source):
#   threshold_source 'median' => threshold is the TRAINING-fold median of the
#   column (computed at evaluation time, never from test data); 'zero' => a
#   fixed literal 0 boundary (e.g. verified-zero public charging supply).
SUBGROUP_SPECS = {
    'lower_income':   ('median_income', '<',  'median'),
    'higher_income':  ('median_income', '>=', 'median'),
    'renter_heavy':   ('renter_share',  '>=', 'median'),
    'owner_heavy':    ('renter_share',  '<',  'median'),
    'zero_supply':    ('ports_per_1000_households', '==', 'zero'),
    'has_infrastructure': ('ports_per_1000_households', '>', 'zero'),
}


def read_zcta_strings(path, usecols=None) -> pd.DataFrame:
    """Read the processed CSV keeping zcta as a zero-padded 5-digit string.

    zcta codes such as 01985 or 80301 must not be coerced to integers, or the
    leading zero (and therefore the geographic meaning) is lost.
    """
    df = pd.read_csv(path, dtype={'zcta': 'string'}, usecols=usecols, low_memory=False)
    df['zcta'] = df['zcta'].astype('string').str.strip()
    return df


def is_colorado(zcta: pd.Series) -> pd.Series:
    """Boolean mask of rows whose 5-digit code starts with a Colorado prefix."""
    code = zcta.astype('string').str.strip()
    valid = code.str.match(r'^\d{5}$').fillna(False)
    co = code.str.startswith(CO_PREFIXES).fillna(False)
    return valid & co


def zip3(zcta: pd.Series) -> pd.Series:
    """First three digits of the code, used as an approximate spatial group for
    grouped cross-validation. Not a true spatial boundary (documented)."""
    return zcta.astype('string').str.strip().str[:3]


def assert_no_leakage(feature_columns) -> None:
    """Fail loudly if any known target-derived column slips into the features."""
    leaked = sorted(set(feature_columns) & set(LEAKAGE_COLUMNS))
    if leaked:
        raise ValueError(
            f'Target-leaking columns present in features: {leaked}. '
            f'The target {TARGET!r} is derived from ev_count, so ev/bev/phev '
            f'counts and the target itself must not be predictors.')
