"""Milestone 3 - Step 1: build the Colorado modeling table from the cleaned CSV.

Three transparent, logged stages (plan Tasks 1-3):

  1. Filter to Colorado areas (80/81 prefix working definition) and drop the
     out-of-state registration noise. The prefix filter is a WORKING definition
     only; ZIP != ZCTA (docs/limitations.md). A ZIP-to-county crosswalk or ZCTA
     polygons would be the accurate upgrade and is logged as future work.

  2. Audit AFDC station matching, THEN encode verified zero supply. A Colorado
     ZCTA that appears in the registration data but has no AFDC row means AFDC
     lists no public/existing/electric station for that 5-digit code -- a
     genuine absence, because the join is a direct same-normalization 5-digit
     match and AFDC is a directory of existing stations. Those confirmed
     absences become station_count = 0 and known_ports = 0 /
     ports_per_1000_households = 0. A station that DID match but reported blank
     port counts (the *_partial flags) stays MISSING and is handled separately
     -- that is a real unknown, not a verified zero.

  3. Build feature/target columns, attach ZIP3 group labels, and tag cohorts:
       - regression_cohort : households >= 100 and target + features usable
       - cluster_cohort    : usable adoption, income, renter_share, supply
       - sensitivity_cohort: regression rows WITHOUT outlier capping (capping is
                             deferred to training folds in train_regression.py)

Run from anywhere; paths resolve relative to the repository root:

    python src/prepare_modeling.py
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from modeling_common import (
    CLEAN_CSV, MODELING_CSV, PROCESSED, REGRESSION_FEATURES, TARGET,
    assert_no_leakage, is_colorado, read_zcta_strings, repo_relative, zip3,
)

MIN_HOUSEHOLDS = 100  # households floor for the primary regression cohort
PARTIAL_FLAGS = ['l1_ports_partial', 'l2_ports_partial', 'dcfc_ports_partial']


def _as_bool(series: pd.Series) -> pd.Series:
    """Coerce a possibly-string/NaN flag column to a clean boolean series."""
    return (series.astype('string').str.strip().str.lower()
            .isin(['true', '1', 'yes', 't']))


def stage1_colorado(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    """Keep Colorado codes, drop out-of-state noise, record the before/after."""
    log['total_rows'] = int(len(df))
    co_mask = is_colorado(df['zcta'])
    co = df.loc[co_mask].copy()
    log['colorado_rows'] = int(len(co))
    log['out_of_state_rows_dropped'] = int((~co_mask).sum())
    log['colorado_definition'] = (
        'zcta is a valid 5-digit code beginning with 80 or 81 (USPS/ZCTA '
        'proxy). WORKING definition only; see docs/limitations.md.')
    log['colorado_definition_followup'] = (
        'Cross-check against a HUD/Census ZIP-to-county crosswalk or ZCTA '
        'polygons is future work; prefix reliance is a documented limitation.')
    return co


def stage2_supply_audit(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    """Audit AFDC matching for Colorado codes, then encode verified zero supply.

    A row is a *confirmed absence* when it has registration data but no AFDC
    match (ev_data_present True, afdc_data_present False, station_count NaN).
    Those get zero supply. A matched station with blank ports (any *_partial
    flag True, or known_ports NaN while a station exists) stays missing.
    """
    df = df.copy()
    ev_present = _as_bool(df['ev_data_present'])
    afdc_present = _as_bool(df['afdc_data_present'])
    station_missing = df['station_count'].isna()

    # Audit: the "no station" set should be exactly {afdc absent AND no count}.
    confirmed_absence = (~afdc_present) & station_missing
    # Sanity cross-check: every confirmed absence also has registration data
    # (otherwise it would not be in the union key at all for a CO code).
    audit_absence_with_ev = int((confirmed_absence & ev_present).sum())

    log['audit_colorado_with_station'] = int(afdc_present.sum())
    log['audit_colorado_without_station'] = int(confirmed_absence.sum())
    log['audit_absence_rows_that_have_ev_data'] = audit_absence_with_ev
    log['audit_absence_equals_no_ev_data_present'] = int(
        (confirmed_absence & ~ev_present).sum())
    log['audit_note'] = (
        'No AFDC match for a CO registration code = AFDC lists no '
        'public/existing/electric station there. Join is a direct '
        '5-digit match with identical normalization on both sides, so this is '
        'a genuine absence rather than a failed join.')

    # Encode verified zero supply for confirmed absences only.
    df['supply_is_verified_zero'] = confirmed_absence
    df.loc[confirmed_absence, 'station_count'] = 0.0
    df.loc[confirmed_absence, ['l1_ports', 'l2_ports', 'dcfc_ports']] = 0.0
    df.loc[confirmed_absence, 'known_ports'] = 0.0

    # ports_per_1000_households = 0 for a confirmed absence that has a usable
    # household denominator; leave NaN where households are unknown.
    hh = pd.to_numeric(df['households'], errors='coerce')
    zero_rate = confirmed_absence & hh.gt(0)
    df.loc[zero_rate, 'ports_per_1000_households'] = 0.0

    # Matched-but-blank ports remain MISSING. A station matched (afdc_present)
    # but at least one port type is partial OR known_ports is still NaN.
    partial_any = pd.concat([_as_bool(df[c]) for c in PARTIAL_FLAGS], axis=1).any(axis=1)
    matched_blank = afdc_present & (partial_any | df['known_ports'].isna())
    df['supply_is_unknown_at_station'] = matched_blank
    log['audit_matched_station_blank_ports'] = int(matched_blank.sum())
    log['recoded_rows_to_zero_supply'] = int(confirmed_absence.sum())
    log['preserved_missing_port_rows'] = int(matched_blank.sum())
    return df


def stage3_features(df: pd.DataFrame, log: dict) -> pd.DataFrame:
    """Build numeric features/target, ZIP3 groups, and cohort flags."""
    df = df.copy()

    numeric = REGRESSION_FEATURES + [TARGET, 'ev_count', 'station_count',
                                     'known_ports', 'ports_per_1000_households',
                                     'households']
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['zip3'] = zip3(df['zcta'])

    target_ok = df[TARGET].notna()
    features_ok = df[REGRESSION_FEATURES].notna().all(axis=1)
    households_ok = df['households'].ge(MIN_HOUSEHOLDS)

    # Primary regression cohort: usable target + features AND households >= 100.
    df['regression_cohort'] = target_ok & features_ok & households_ok
    # Sensitivity cohort: same usable rows but WITHOUT the household floor, so
    # the outlier/winsorization sensitivity is computed on the fuller set. No
    # capping is applied here (deferred to training folds to avoid leakage).
    df['sensitivity_cohort'] = target_ok & features_ok
    # Clustering cohort: usable adoption, income, renter_share, and a supply
    # rate (zero where verified absent). households floor also applied so the
    # unstable tiny-denominator adoption spikes do not distort the clusters.
    supply_ok = df['ports_per_1000_households'].notna()
    cluster_core = df[['ev_per_1000_households', 'median_income',
                       'renter_share']].notna().all(axis=1)
    df['cluster_cohort'] = cluster_core & supply_ok & households_ok

    assert_no_leakage(REGRESSION_FEATURES)

    log['min_households_filter'] = MIN_HOUSEHOLDS
    log['regression_cohort_rows'] = int(df['regression_cohort'].sum())
    log['sensitivity_cohort_rows'] = int(df['sensitivity_cohort'].sum())
    log['cluster_cohort_rows'] = int(df['cluster_cohort'].sum())
    log['regression_features'] = list(REGRESSION_FEATURES)
    log['excluded_leakage_columns'] = ['ev_count', 'bev_count', 'phev_count']
    log['target'] = TARGET
    # Target spread on the regression cohort (context for the winsorization step).
    reg_target = df.loc[df['regression_cohort'], TARGET]
    if len(reg_target):
        log['regression_target_summary'] = {
            'n': int(len(reg_target)),
            'min': round(float(reg_target.min()), 3),
            'median': round(float(reg_target.median()), 3),
            'mean': round(float(reg_target.mean()), 3),
            'max': round(float(reg_target.max()), 3),
            'p99': round(float(reg_target.quantile(0.99)), 3),
        }
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default=str(CLEAN_CSV))
    p.add_argument('--output', default=str(MODELING_CSV))
    p.add_argument('--report', default=str(PROCESSED / 'modeling_prep_report.json'))
    args = p.parse_args()

    log = {'generated_utc': datetime.now(timezone.utc).isoformat(),
           'input': repo_relative(args.input)}

    df = read_zcta_strings(args.input)
    df = stage1_colorado(df, log)
    df = stage2_supply_audit(df, log)
    df = stage3_features(df, log)

    out = df.copy()
    out.to_csv(args.output, index=False)
    log['output'] = repo_relative(args.output)
    with open(args.report, 'w') as f:
        json.dump(log, f, indent=2)

    print(f"Colorado modeling table: {log['colorado_rows']} CO rows "
          f"(dropped {log['out_of_state_rows_dropped']} out-of-state).")
    print(f"Supply audit: {log['audit_colorado_with_station']} with a station, "
          f"{log['recoded_rows_to_zero_supply']} recoded to verified zero, "
          f"{log['preserved_missing_port_rows']} matched-but-blank kept missing.")
    print(f"Cohorts -> regression: {log['regression_cohort_rows']}, "
          f"sensitivity: {log['sensitivity_cohort_rows']}, "
          f"cluster: {log['cluster_cohort_rows']}.")
    print(f"Wrote {repo_relative(args.output)} and {repo_relative(args.report)}.")


if __name__ == '__main__':
    main()
