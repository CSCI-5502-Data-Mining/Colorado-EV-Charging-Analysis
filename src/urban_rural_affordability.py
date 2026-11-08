"""Milestone 3 supporting analysis: urban/rural charging + renter affordability.

A SEPARATE descriptive analysis answering two original supporting questions. It
does NOT touch the frozen regression, clustering, or priority-ranking models --
it only joins two official Census tables onto the existing Colorado modeling
table and summarizes them.

RQ (urban vs rural): classify each Colorado ZCTA by its official Census-2020
urban population share (DHC P2), then compare public charging ports per 1,000
households -- total, Level 2, and DC fast -- across urban-dominant /
rural-dominant / mixed classes.

  LIMITATION (documented): this is a ZCTA-LEVEL comparison of urban-dominant vs
  rural-dominant areas, NOT a point-in-polygon classification of each charger's
  own location. A ZCTA can straddle the urban boundary; the "mixed" class and
  the reported urban-share distribution keep that visible. Full geospatial
  classification (station coordinates in Census urban polygons, block-level
  denominators) is the later H3 stage.

RQ (housing affordability): from ACS B25070, the share of renter households
paying >=30% of income on gross rent (cost-burdened), excluding the
"not computed" category, as supplementary descriptive evidence.

    python src/urban_rural_affordability.py
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from modeling_common import (FIGURES, MODELING_CSV, PROCESSED, ROOT, git_commit,
                             is_colorado, read_zcta_strings, repo_relative)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

RAW = ROOT / 'data' / 'raw'
DHC_CSV = RAW / 'dhc_p2_zcta_2020.csv'
ACS_B25070_CSV = RAW / 'acs_b25070_zcta_2024.csv'

# Documented urban-share thresholds. Census classifies individual blocks as
# urban/rural; at ZCTA level we use the urban POPULATION SHARE. A ZCTA that is
# >=80% urban is urban-dominant, <=20% urban is rural-dominant, and the band in
# between is genuinely "mixed" and reported separately rather than forced.
URBAN_DOMINANT = 0.80
RURAL_DOMINANT = 0.20


def _zip5(s: pd.Series) -> pd.Series:
    out = s.astype('string').str.strip().str.extract(r'(\d{5})')[0]
    return out


def load_dhc() -> pd.DataFrame:
    df = pd.read_csv(DHC_CSV, dtype='string')
    df['zcta'] = _zip5(df['zip code tabulation area'])
    for c in ['P2_001N', 'P2_002N', 'P2_003N']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['zcta']).drop_duplicates('zcta')
    # Urban share of the 2020 population; undefined where total population is 0.
    df['urban_share_2020'] = np.where(df['P2_001N'] > 0,
                                      df['P2_002N'] / df['P2_001N'], np.nan)
    return df[['zcta', 'P2_001N', 'P2_002N', 'P2_003N', 'urban_share_2020']].rename(
        columns={'P2_001N': 'pop2020_total', 'P2_002N': 'pop2020_urban',
                 'P2_003N': 'pop2020_rural'})


def load_b25070() -> pd.DataFrame:
    df = pd.read_csv(ACS_B25070_CSV, dtype='string')
    df['zcta'] = _zip5(df['zip code tabulation area'])
    cols = ['B25070_001E', 'B25070_007E', 'B25070_008E', 'B25070_009E',
            'B25070_010E', 'B25070_011E']
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=['zcta']).drop_duplicates('zcta')
    # Cost-burdened renters = share paying >=30% of income on rent, EXCLUDING
    # the "not computed" category (B25070_011E) from the denominator.
    burdened = df[['B25070_007E', 'B25070_008E', 'B25070_009E', 'B25070_010E']].sum(axis=1)
    denom = df['B25070_001E'] - df['B25070_011E']
    df['renter_cost_burden_share'] = np.where(denom > 0, burdened / denom, np.nan)
    df['renter_households_rent_computed'] = denom
    return df[['zcta', 'renter_cost_burden_share',
               'renter_households_rent_computed']]


def classify(urban_share: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=urban_share.index, dtype='string')
    out[urban_share >= URBAN_DOMINANT] = 'urban_dominant'
    out[urban_share <= RURAL_DOMINANT] = 'rural_dominant'
    out[(urban_share > RURAL_DOMINANT) & (urban_share < URBAN_DOMINANT)] = 'mixed'
    return out


def _ports_per_1000(df, port_col):
    """Row-level ports-per-1,000-households for a given port column, only where
    households are usable; relies on the frozen verified-zero supply recoding."""
    hh = pd.to_numeric(df['households'], errors='coerce')
    ports = pd.to_numeric(df[port_col], errors='coerce')
    return np.where(hh > 0, ports / hh * 1000, np.nan)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default=str(MODELING_CSV))
    p.add_argument('--out', default=str(PROCESSED / 'urban_rural_affordability.csv'))
    p.add_argument('--report', default=str(PROCESSED / 'urban_rural_affordability_report.json'))
    args = p.parse_args()

    mdl = pd.read_csv(args.input, dtype={'zcta': 'string', 'zip3': 'string'})
    co = mdl.loc[is_colorado(mdl['zcta'])].copy()

    dhc = load_dhc()
    aff = load_b25070()
    merged = co.merge(dhc, on='zcta', how='left').merge(aff, on='zcta', how='left')
    merged['urban_class'] = classify(merged['urban_share_2020'])

    # Aggregate port density by urban class, using aggregate totals (sum of
    # ports over sum of households) so large and small ZCTAs are weighted by
    # their household counts rather than averaging unstable per-ZCTA rates.
    report = {'generated_utc': datetime.now(timezone.utc).isoformat(),
              'git_commit': git_commit(), 'input': repo_relative(args.input),
              'n_colorado': int(len(co)),
              'thresholds': {'urban_dominant_min_urban_share': URBAN_DOMINANT,
                             'rural_dominant_max_urban_share': RURAL_DOMINANT},
              'join': {'colorado_zctas': int(len(co)),
                       'dhc_matched': int(merged['urban_share_2020'].notna().sum()),
                       'dhc_unmatched': int(merged['urban_share_2020'].isna().sum()),
                       'b25070_matched': int(merged['renter_cost_burden_share'].notna().sum())},
              'classification_counts': {
                  'urban_dominant': int((merged['urban_class'] == 'urban_dominant').sum()),
                  'mixed': int((merged['urban_class'] == 'mixed').sum()),
                  'rural_dominant': int((merged['urban_class'] == 'rural_dominant').sum()),
                  'unclassified_dhc_unmatched': int(merged['urban_class'].isna().sum()),
                  'reconciliation': ('urban_dominant + mixed + rural_dominant + '
                                     'unclassified = colorado_zctas (500); the 23 '
                                     'unclassified are exactly the DHC-unmatched '
                                     'ZCTAs, mostly small/rural codes also flagged '
                                     'in the ranking coverage audit.')},
              'limitation': ('ZCTA-level comparison of urban-dominant vs '
                             'rural-dominant areas by Census-2020 urban '
                             'population share -- NOT point-in-polygon '
                             'classification of each charger. Mixed ZCTAs are '
                             'reported as their own class. Full geospatial '
                             'classification is the later H3 stage.'),
              'vintages': {
                  'urban_rural_classification': '2020 Decennial Census DHC (P2)',
                  'renter_cost_burden': '2024 ACS 5-year (B25070, covers 2020-2024)',
                  'households_denominator': '2024 ACS 5-year B25003 (from the frozen modeling table)',
                  'ev_registrations': 'Atlas latest DMV snapshot 2026-07-07',
                  'charging_stations': 'AFDC 2026-10-15 download',
                  'vintage_caveat': ('These sources span different years and '
                                     'collection methods. The urban/rural split '
                                     'is from the 2020 Census point-in-time, '
                                     'while households, income, and renter cost '
                                     'burden are 2020-2024 ACS estimates, and '
                                     'EV/station data are 2026 snapshots. '
                                     'Comparisons are descriptive; a ZCTA\'s '
                                     'urban share could have shifted since 2020.')}}

    def agg_density(d):
        hh = pd.to_numeric(d['households'], errors='coerce')
        # Only ZCTAs with a usable household denominator AND known (verified or
        # reported) supply contribute; verified-zero rows count as 0 ports.
        usable = d.loc[hh.gt(0) & d['ports_per_1000_households'].notna()].copy()
        hh_u = pd.to_numeric(usable['households'], errors='coerce')
        out = {'n_zctas': int(len(d)),
               'n_zctas_usable_supply': int(len(usable)),
               'households_total': int(hh_u.sum()) if len(usable) else 0,
               'density_method': ('AGGREGATE: sum(ports over the class) / '
                                  'sum(households over the class) * 1000, so '
                                  'large and small ZCTAs are weighted by their '
                                  'household counts (NOT an unweighted mean of '
                                  'per-ZCTA ratios).')}
        tot_hh = hh_u.sum()
        for label, col in [('all', 'known_ports'), ('l2', 'l2_ports'),
                           ('dcfc', 'dcfc_ports')]:
            ports = pd.to_numeric(usable[col], errors='coerce')
            out[f'ports_per_1000_hh_{label}'] = (round(float(ports.sum() / tot_hh * 1000), 3)
                                                 if tot_hh > 0 else None)
        # Unweighted mean of per-ZCTA ratios, reported ONLY for contrast so a
        # reader can see the aggregate is not a mean of ratios.
        out['ports_per_1000_hh_all_unweighted_mean_for_contrast'] = (
            round(float(usable['ports_per_1000_households'].mean()), 3) if len(usable) else None)
        # Share of ZCTAs in this class with zero verified public supply.
        out['share_zero_supply'] = (round(float((usable['ports_per_1000_households'] == 0).mean()), 3)
                                    if len(usable) else None)
        # Small-household sensitivity: the zero-supply share at household floors,
        # because the zero-supply contrast is partly driven by very small ZCTAs.
        out['share_zero_supply_by_household_floor'] = {
            str(fl): (round(float((usable.loc[hh_u >= fl, 'ports_per_1000_households'] == 0).mean()), 3)
                      if (hh_u >= fl).any() else None)
            for fl in (0, 100, 500)}
        out['median_urban_share'] = (round(float(d['urban_share_2020'].median()), 3)
                                      if d['urban_share_2020'].notna().any() else None)
        return out

    by_class = {}
    for cls in ['urban_dominant', 'mixed', 'rural_dominant']:
        sub = merged.loc[merged['urban_class'] == cls]
        if len(sub):
            by_class[cls] = agg_density(sub)
    report['charging_by_urban_class'] = by_class
    report['interpretation_notes'] = {
        'dcfc_corridor_hypothesis': (
            'Higher DC-fast density toward rural ZCTAs is a POSSIBLE '
            'interpretation (highway-corridor fast charging rather than '
            'neighborhood Level 2), NOT a verified geographic finding: this '
            'analysis does not locate chargers relative to highways or confirm '
            'corridor placement. It would need station coordinates and road '
            'data (the H3 stage) to verify.'),
        'zero_supply_sensitivity': (
            'The rural zero-supply share softens with a minimum-household floor '
            '(see share_zero_supply_by_household_floor): the DIRECTION of the '
            'finding is robust (rural ZCTAs are far more likely to lack public '
            'charging at every floor), but the MAGNITUDE of the headline 64% is '
            'partly driven by very small rural ZCTAs (rural median ~442 '
            'households). Report it as a coverage gap, not a precise rate.')}

    # Affordability: cost-burden summary overall and by urban class. We report
    # BOTH the unweighted median across ZCTAs AND the household-weighted
    # statewide estimate, each labelled, because they answer different questions:
    # the median describes a typical ZCTA, the weighted figure describes a
    # typical renter household statewide.
    def burden_summary(d):
        s = d['renter_cost_burden_share'].dropna()
        w = d.dropna(subset=['renter_cost_burden_share', 'renter_households_rent_computed'])
        wsum = float(w['renter_households_rent_computed'].sum())
        weighted = (float((w['renter_cost_burden_share']
                           * w['renter_households_rent_computed']).sum() / wsum)
                    if wsum > 0 else None)
        return {'n_zctas': int(len(s)),
                'median_across_zctas': round(float(s.median()), 3) if len(s) else None,
                'mean_across_zctas': round(float(s.mean()), 3) if len(s) else None,
                'household_weighted_estimate': round(weighted, 3) if weighted is not None else None}
    report['renter_cost_burden'] = {
        'definition': ('share of renter households paying >=30% of income on '
                       'gross rent (ACS B25070 buckets 007-010), excluding the '
                       '"not computed" category (011) from the denominator'),
        'statistic_labels': ('median_across_zctas = unweighted median of per-ZCTA '
                             'shares (a typical ZCTA); household_weighted_estimate '
                             '= sum(burdened renters)/sum(rent-computed renters) '
                             '(a typical renter household statewide). The "51%" '
                             'headline is the median_across_zctas; the '
                             'household-weighted statewide estimate is ~53%.'),
        'colorado_overall': burden_summary(merged),
        'by_urban_class': {cls: burden_summary(merged.loc[merged['urban_class'] == cls])
                           for cls in ['urban_dominant', 'mixed', 'rural_dominant']
                           if (merged['urban_class'] == cls).any()},
        'moe_note': ('ACS 5-year estimates carry margins of error that are '
                     'proportionally larger for small renter populations; '
                     'per-ZCTA shares for tiny areas are noisy and should be '
                     'read as indicative, not precise.')}

    # Persist the per-ZCTA joined table (the columns this analysis adds).
    keep = ['zcta', 'zip3', 'households', 'known_ports', 'l2_ports', 'dcfc_ports',
            'ports_per_1000_households', 'supply_is_verified_zero',
            'pop2020_total', 'urban_share_2020', 'urban_class',
            'renter_cost_burden_share', 'renter_households_rent_computed']
    keep = [c for c in keep if c in merged.columns]
    merged[keep].to_csv(args.out, index=False)
    with open(args.report, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    _figures(merged, report)

    # Console summary.
    cc = report['classification_counts']
    print(f"Colorado ZCTAs: {len(co)} = {cc['urban_dominant']} urban + {cc['mixed']} mixed "
          f"+ {cc['rural_dominant']} rural + {cc['unclassified_dhc_unmatched']} unclassified "
          f"(DHC-unmatched) | B25070 matched: {report['join']['b25070_matched']}")
    print("Public charging ports per 1,000 households by urban class (aggregate):")
    for cls, d in by_class.items():
        print(f"  {cls:15s} n={d['n_zctas']:3d} all={d['ports_per_1000_hh_all']} "
              f"L2={d['ports_per_1000_hh_l2']} DCFC={d['ports_per_1000_hh_dcfc']} "
              f"zero_supply={d['share_zero_supply']}")
    cb = report['renter_cost_burden']
    o = cb['colorado_overall']
    print(f"Renter cost burden (>=30% income on rent): median across ZCTAs="
          f"{o['median_across_zctas']}, household-weighted statewide="
          f"{o['household_weighted_estimate']}")
    for cls, d in cb['by_urban_class'].items():
        print(f"  {cls:15s} median_across_zctas={d['median_across_zctas']} (n={d['n_zctas']})")
    print(f"Wrote {repo_relative(args.out)} and {repo_relative(args.report)}.")


def _figures(merged, report):
    FIGURES.mkdir(parents=True, exist_ok=True)
    classes = ['urban_dominant', 'mixed', 'rural_dominant']
    labels = ['Urban-\ndominant', 'Mixed', 'Rural-\ndominant']
    bc = report['charging_by_urban_class']
    present = [(c, l) for c, l in zip(classes, labels) if c in bc]

    # Fig 14: ports per 1,000 hh (all / L2 / DCFC) by urban class.
    x = np.arange(len(present)); w = 0.26
    allv = [bc[c]['ports_per_1000_hh_all'] or 0 for c, _ in present]
    l2v = [bc[c]['ports_per_1000_hh_l2'] or 0 for c, _ in present]
    dcv = [bc[c]['ports_per_1000_hh_dcfc'] or 0 for c, _ in present]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w, allv, w, label='all public ports', color='#48c')
    ax.bar(x, l2v, w, label='Level 2', color='#6a9')
    ax.bar(x + w, dcv, w, label='DC fast', color='#c63')
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in present])
    ax.set(ylabel='Public ports per 1,000 households (aggregate)',
           title='Charging-port density by ZCTA urban class (Census 2020)')
    for i, v in enumerate(allv):
        ax.text(x[i] - w, v, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / '14_urban_rural_ports.png', dpi=160); plt.close(fig)

    # Fig 15: renter cost burden distribution by urban class.
    data, ticks = [], []
    for c, l in present:
        s = merged.loc[merged['urban_class'] == c, 'renter_cost_burden_share'].dropna()
        if len(s):
            data.append(s.values); ticks.append(l)
    if data:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.boxplot(data, tick_labels=ticks, showmeans=True)
        ax.set(ylabel='Share of renters paying \u2265 30% of income on rent',
               title='Renter housing-cost burden by ZCTA urban class (ACS 2024 B25070)')
        ax.set_ylim(0, 1)
        fig.tight_layout(); fig.savefig(FIGURES / '15_renter_cost_burden.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    main()
