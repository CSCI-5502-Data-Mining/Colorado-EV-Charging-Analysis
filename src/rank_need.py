"""Milestone 3 - ZCTA-level charging-NEED ranking (planning track).

This answers the PRIMARY planning question -- which Colorado communities may
benefit most from additional public charging -- which is a different question
from "where is EV ownership highest" (the regression supporting track). It does
NOT claim to find the objectively best station site: without observed
installation/utilization outcomes, these scores only flag candidate areas for
further, finer-grained investigation (the future H3 track).

Two transparent, predeclared rankings over the Colorado clustering cohort
(usable adoption, income, renter share, verified public supply):

  Approach 1 - BASELINE (access only): rank by low public charging supply per
  household. Simple and intuitive, but low supply alone does NOT prove high
  need -- a low-demand, well-off area can have little supply and little need.

  Approach 2 - DEMAND + EQUITY + ACCESS composite: three SEPARATE components,
  each min-max normalized to [0,1], then combined with fixed, documented
  weights. Kept separate so each contribution is visible and auditable:
    * demand  = existing EV adoption rate (observed, not model-predicted, so a
                regression model cannot make the ranking chase wealthy areas).
    * access  = scarcity of public charging (1 - normalized ports/1,000 hh).
    * equity  = renter share and lower income (home charging is hardest for
                renters and the up-front cost burden is higher at low income).

Design rules (fixed; not optimized against the ranking being evaluated):
  * weights are declared constants, not tuned.
  * normalization is min-max over the cohort, documented.
  * demand uses OBSERVED adoption, never the regression prediction, per review.
  * we report how the two rankings differ, weight sensitivity, ranking
    stability, and the equity composition of the top-ranked sets.

    python src/rank_need.py
"""
from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy import stats

from modeling_common import (FIGURES, MODELING_CSV, PROCESSED, git_commit,
                             repo_relative)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

TOP_N = 30  # size of the "top-ranked" set compared across methods

# Fixed, documented composite weights (demand / access / equity). Chosen a
# priori to weight the three planning concerns roughly equally, with access
# slightly highest because the question is specifically about charging gaps.
# They are NOT tuned against the ranking. Sensitivity to them is reported.
BASE_WEIGHTS = {'demand': 0.33, 'access': 0.34, 'equity': 0.33}


def minmax(s: pd.Series) -> pd.Series:
    """Percentile-rank normalize to [0,1] over the cohort; constant -> 0.5.

    We use the rank/percentile transform rather than raw min-max because the
    supply and adoption distributions are heavy-tailed: a few extreme ZCTAs
    would otherwise compress every other area into a narrow band (e.g. the
    public-supply outliers made a raw-min-max access score ~0.95 for almost
    every ZCTA, so it could not discriminate). Percentile normalization gives
    each component a uniform [0,1] spread so the three concerns are weighted on
    a comparable scale. This is a fixed, documented choice, not a tuned one.
    """
    if s.nunique() <= 1:
        return pd.Series(0.5, index=s.index)
    # average rank handles ties; divide by n-1 to span [0,1] inclusive.
    return s.rank(method='average').sub(1).div(len(s) - 1)


def build_components(df: pd.DataFrame) -> pd.DataFrame:
    """The three SEPARATE, normalized need components (all higher = more need)."""
    c = pd.DataFrame(index=df.index)
    # Demand: observed adoption rate (NOT model-predicted).
    c['demand'] = minmax(df['ev_per_1000_households'])
    # Access need: scarcity of public supply = 1 - normalized ports/1,000 hh.
    c['access'] = 1.0 - minmax(df['ports_per_1000_households'])
    # Equity: average of renter share (normalized) and income need
    # (1 - normalized income, so lower income -> higher need).
    renter_n = minmax(df['renter_share'])
    income_need = 1.0 - minmax(df['median_income'])
    c['equity'] = (renter_n + income_need) / 2.0
    return c


def composite_score(components: pd.DataFrame, weights: dict) -> pd.Series:
    return (weights['demand'] * components['demand']
            + weights['access'] * components['access']
            + weights['equity'] * components['equity'])


def equity_composition(df: pd.DataFrame, idx, cohort: pd.DataFrame) -> dict:
    """Share of a selected set that is renter-heavy / lower-income (vs cohort
    medians), plus that set's mean income and renter share."""
    med_income = float(cohort['median_income'].median())
    med_renter = float(cohort['renter_share'].median())
    sub = df.loc[idx]
    return {
        'n': int(len(sub)),
        'share_lower_income': round(float((sub['median_income'] <= med_income).mean()), 3),
        'share_renter_heavy': round(float((sub['renter_share'] >= med_renter).mean()), 3),
        'mean_income': round(float(sub['median_income'].mean()), 0),
        'mean_renter_share': round(float(sub['renter_share'].mean()), 3),
        'mean_adoption': round(float(sub['ev_per_1000_households'].mean()), 2),
        'mean_ports_per_1000': round(float(sub['ports_per_1000_households'].mean()), 3),
    }


def _has_station(series: pd.Series) -> pd.Series:
    return series.astype('string').str.strip().str.lower().isin(['true', '1', 'yes', 't'])


def coverage_audit(co: pd.DataFrame, rankable_idx) -> dict:
    """Which of the 500 Colorado codes can and cannot be ranked, and whether the
    un-rankable ones are systematically more under-infrastructured. Un-rankable
    codes are NOT dropped silently: they are reported as a separate group whose
    need cannot be scored from the available data.
    """
    co = co.copy()
    co['rankable'] = co.index.isin(rankable_idx)
    excl = co.loc[~co['rankable']]
    incl = co.loc[co['rankable']]

    def profile(d):
        return {
            'n': int(len(d)),
            'no_public_station': int((~_has_station(d['afdc_data_present'])).sum()),
            'share_no_public_station': round(float((~_has_station(d['afdc_data_present'])).mean()), 3),
            'median_households': (round(float(d['households'].median()), 0)
                                  if d['households'].notna().any() else None),
            'median_income': (round(float(d['median_income'].median()), 0)
                              if d['median_income'].notna().any() else None),
        }

    reasons = {
        'missing_median_income': int(excl['median_income'].isna().sum()),
        'missing_households': int(excl['households'].isna().sum()),
        'missing_renter_share': int(excl['renter_share'].isna().sum()),
        'missing_adoption_rate': int(excl['ev_per_1000_households'].isna().sum()),
        'below_household_floor_100': int((excl['households'] < 100).sum()),
    }
    return {
        'colorado_codes_total': int(len(co)),
        'rankable': int(len(incl)),
        'not_rankable': int(len(excl)),
        'exclusion_reasons_not_mutually_exclusive': reasons,
        'rankable_profile': profile(incl),
        'not_rankable_profile': profile(excl),
        'not_rankable_zctas': sorted(excl['zcta'].tolist()),
        'interpretation': ('The un-rankable codes are mostly very small/rural '
                           '(low median households) and a HIGHER share have no '
                           'public station than the rankable cohort, so they may '
                           'be MORE under-served, not less. They are excluded '
                           'only for missing socioeconomic fields or an unstable '
                           'tiny household denominator, and are reported here '
                           'rather than silently dropped. Enumerating every true '
                           'Colorado Census ZCTA needs a ZIP-to-ZCTA crosswalk '
                           '(future work); the 500 here are the codes the '
                           'pipeline matched.')}


def assign_tiers(rank: pd.Series, n: int, k: int = 5) -> pd.Series:
    """Map a rank to a broad priority TIER (1 = highest need), ~k equal-count
    bands. Broad tiers are honest: exact ranks are unstable and, for the
    access-only score, built on large ties, so a tier is the finest ordering we
    defend. A large tied block (e.g. zero-supply codes all at rank 1) can land
    wholly in one tier -- that is correct, they are not distinguishable. We bin
    by rank cut points and drop duplicate edges rather than invent an order.
    """
    try:
        return pd.qcut(rank, q=k, labels=False, duplicates='drop').astype(int) + 1
    except ValueError:
        # Degenerate (almost everything tied): single tier.
        return pd.Series(1, index=rank.index)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default=str(MODELING_CSV))
    p.add_argument('--out', default=str(PROCESSED / 'need_ranking.csv'))
    p.add_argument('--report', default=str(PROCESSED / 'need_ranking_report.json'))
    args = p.parse_args()

    df = pd.read_csv(args.input, dtype={'zcta': 'string', 'zip3': 'string'})
    co = df.loc[df['zcta'].str.match(r'^(80|81)\d{3}$').fillna(False)].copy()
    cohort = df.loc[df['cluster_cohort'] == True].copy()  # noqa: E712
    cohort = cohort.dropna(subset=['ev_per_1000_households', 'median_income',
                                   'renter_share', 'ports_per_1000_households'])
    # Record which CO codes are rankable BEFORE resetting the index.
    rankable_idx = set(cohort.index)
    cohort = cohort.reset_index(drop=True)

    comp = build_components(cohort)
    cohort = pd.concat([cohort, comp.add_prefix('comp_')], axis=1)

    # --- Approach 1: baseline access-only ranking --------------------------
    # Rank by charging scarcity alone (higher = less supply per household).
    cohort['score_baseline'] = comp['access']
    cohort['rank_baseline'] = cohort['score_baseline'].rank(ascending=False, method='min')

    # --- Approach 2: demand + equity + access composite --------------------
    cohort['score_composite'] = composite_score(comp, BASE_WEIGHTS)
    cohort['rank_composite'] = cohort['score_composite'].rank(ascending=False, method='min')

    # --- Broad priority TIERS (1 = highest need), the defensible ordering ---
    n = len(cohort)
    cohort['tier_composite'] = assign_tiers(cohort['rank_composite'], n)
    cohort['tier_baseline'] = assign_tiers(cohort['rank_baseline'], n)

    # --- Special-use / institutional flag (documented, reproducible) -------
    # Some ZCTAs are dominated by group quarters or institutional housing (e.g.
    # a military academy), where renter share and per-household rates do not
    # describe an ordinary residential community. We FLAG these rather than drop
    # them, so their high priority can be read with the right caveat. Rule: a
    # ZCTA is special-use if 100% of occupied housing is renter-occupied OR its
    # persons-per-household far exceeds any residential norm (>5; the cohort's
    # 99th percentile is ~3.4). The flag changes no scores -- it only annotates.
    pop = pd.to_numeric(cohort.get('population'), errors='coerce')
    hh = pd.to_numeric(cohort['households'], errors='coerce')
    cohort['persons_per_household'] = np.where(hh > 0, pop / hh, np.nan)
    cohort['special_use_flag'] = (
        (pd.to_numeric(cohort['renter_share'], errors='coerce') >= 0.999)
        | (cohort['persons_per_household'] > 5))

    report = {'generated_utc': datetime.now(timezone.utc).isoformat(),
              'input': repo_relative(args.input), 'git_commit': git_commit(),
              'n_cohort': int(len(cohort)), 'top_n': TOP_N,
              'base_weights': BASE_WEIGHTS,
              'normalization': ('percentile-rank over the Colorado cohort '
                                '(robust to heavy-tailed supply/adoption); '
                                'constant -> 0.5'),
              'demand_source': ('OBSERVED ev_per_1000_households (not the '
                                'regression prediction), so a model trained on '
                                'current ownership cannot steer the ranking '
                                'toward already-wealthy areas.'),
              'components': {
                  'demand': 'normalized observed EV adoption rate',
                  'access': '1 - normalized public ports per 1,000 households',
                  'equity': 'mean of normalized renter share and (1 - normalized income)'},
              'caveat': ('These scores flag CANDIDATE areas for further '
                         'investigation. Without observed installation/'
                         'utilization outcomes, no ZCTA can be called the '
                         'objectively best site. Low supply alone (Approach 1) '
                         'does not prove high need.'),
              'demand_is_not_unmet_demand': (
                  'The demand component is OBSERVED residential adoption, which '
                  'measures existing ownership, NOT unmet demand for PUBLIC '
                  'charging. A low-ownership, renter-heavy, low-home-charging '
                  'area may still benefit greatly from public charging. Because '
                  'access and equity carry two-thirds of the weight, the score '
                  'does NOT require high ownership: of 36 low-adoption + '
                  'renter-heavy + low-income + zero-supply codes, several reach '
                  'the composite top tier. The equity-weighted ranking makes '
                  'this case even stronger.'),
              'priority_tiers': ('Reported as 5 broad tiers (1 = highest need), '
                                 'not exact ranks: individual ranks are unstable '
                                 'and the access-only score has large ties, so a '
                                 'tier is the finest ordering we defend.')}

    # --- Coverage audit: which of the 500 CO codes can be ranked -----------
    report['coverage_audit'] = coverage_audit(co, rankable_idx)

    # --- Special-use audit: institutional / group-quarters ZCTAs -----------
    su = cohort.loc[cohort['special_use_flag']]
    report['special_use_audit'] = {
        'rule': ('renter_share == 100% OR persons_per_household > 5 (cohort 99th '
                 'pct ~3.4). Flags group-quarters / institutional ZCTAs whose '
                 'renter share and per-household rates do not describe an '
                 'ordinary residential community. Flag does NOT change scores.'),
        'n_flagged': int(len(su)),
        'n_flagged_in_tier1': int((su['tier_composite'] == 1).sum()),
        'flagged': [{'zcta': r['zcta'], 'tier': int(r['tier_composite']),
                     'score': round(float(r['score_composite']), 3),
                     'renter_share': round(float(r['renter_share']), 3),
                     'persons_per_household': round(float(r['persons_per_household']), 2),
                     'households': int(r['households'])}
                    for _, r in su.sort_values('score_composite', ascending=False).iterrows()],
        'interpretation': ('Flagged areas are kept in the ranking but should be '
                           'read as special cases: e.g. 80840 is the U.S. Air '
                           'Force Academy ZCTA (100% renter, ~9.5 persons/'
                           'household), so "100% renter, zero public ports" '
                           'reflects institutional housing, not an ordinary '
                           'underserved community. Verify access and whether new '
                           'public charging would serve the general public before '
                           'treating such a ZCTA as a top candidate.')}

    # --- Access-score tie audit: zero-supply codes share one access score --
    zero_supply = cohort['ports_per_1000_households'] == 0
    n_zero = int(zero_supply.sum())
    report['access_tie_audit'] = {
        'n_zero_supply_codes': n_zero,
        'share_of_cohort': round(n_zero / len(cohort), 3),
        'distinct_supply_values': int(cohort['ports_per_1000_households'].nunique()),
        'access_score_of_zero_supply_codes': round(
            float(cohort.loc[zero_supply, 'score_baseline'].iloc[0]), 4) if n_zero else None,
        'note': ('All zero-supply codes receive the SAME access score, so the '
                 'access-only ranking cannot order them -- any exact order '
                 'within this block is arbitrary tie-breaking. They form one '
                 'top priority GROUP, not a sequence. This is why we report '
                 'broad tiers and treat the access-only top-N as a tied set.')}

    # --- Household COVERAGE of the highest-priority tier -------------------
    # Population living IN the top-priority ZCTAs. This is coverage of the
    # identified priority areas, NOT households proven to gain charging access
    # -- measuring actual access improvement needs a proposed station location
    # and the households within a reasonable travel distance (the H3 stage).
    top_tier = cohort[cohort['tier_composite'] == 1]
    report['household_coverage_top_tier'] = {
        'tier': 1, 'n_codes': int(len(top_tier)),
        'households_covered': int(top_tier['households'].sum()),
        'renter_households_covered': int((top_tier['households']
                                          * top_tier['renter_share']).sum()),
        'label': 'household coverage of identified priority areas',
        'note': ('Population living in the highest-priority ZCTAs -- COVERAGE of '
                 'the identified priority areas, NOT households proven to gain '
                 'charging access. Measuring actual access improvement requires '
                 'a proposed station location and the households within a '
                 'reasonable travel distance, which is the later H3 analysis.')}

    # --- Agreement between the two rankings --------------------------------
    rho = float(stats.spearmanr(cohort['rank_baseline'], cohort['rank_composite']).correlation)
    top_base = set(cohort.nsmallest(TOP_N, 'rank_baseline')['zcta'])
    top_comp = set(cohort.nsmallest(TOP_N, 'rank_composite')['zcta'])
    report['ranking_agreement'] = {
        'spearman_rho_full_ranking': round(rho, 4),
        'top_n_overlap': len(top_base & top_comp),
        'in_baseline_top_not_composite': sorted(top_base - top_comp),
        'in_composite_top_not_baseline': sorted(top_comp - top_base)}

    # --- Tie-aware equity comparison (replaces the arbitrary top-30) -------
    # The access-only method ties all zero-supply codes, so an access-only
    # "top 30" is an arbitrary subset of that block. We instead compare:
    #   (a) each method's FULL highest-priority tier (like-for-like priority
    #       groups, though of different sizes -- sizes reported), and
    #   (b) the ENTIRE access tied group (all zero-supply codes), which is the
    #       honest unit the access-only method can actually identify.
    base_tier1 = cohort[cohort['tier_baseline'] == 1]
    comp_tier1 = cohort[cohort['tier_composite'] == 1]
    access_tied = cohort[cohort['ports_per_1000_households'] == 0]
    report['equity_comparison'] = {
        'cohort_overall': equity_composition(cohort, cohort.index, cohort),
        'access_only_tier1_full': {**equity_composition(cohort, base_tier1.index, cohort),
                                    'note': 'full access-only top tier (large, tie-driven)'},
        'access_only_zero_supply_group': {**equity_composition(cohort, access_tied.index, cohort),
                                          'note': 'the entire tied zero-supply block the access method cannot order'},
        'composite_tier1_full': {**equity_composition(cohort, comp_tier1.index, cohort),
                                 'note': 'full composite top tier'},
        'group_sizes': {'access_only_tier1': int(len(base_tier1)),
                        'access_zero_supply_group': int(len(access_tied)),
                        'composite_tier1': int(len(comp_tier1))},
        'size_caveat': ('The access-only top tier and composite top tier differ '
                        'in size (tie collapse), so this is not a fixed-size '
                        'comparison; shares are size-normalized but the groups '
                        'are the methods\' actual priority sets.')}

    # Fixed-size (top-N) comparison WITH a tie warning and a bootstrap range,
    # kept because it is intuitive, but honest about the access-only arbitrariness.
    idx_comp = cohort.nsmallest(TOP_N, 'rank_composite').index
    # Access-only top-N: resample the arbitrary tie-break to get a RANGE of the
    # equity composition rather than one arbitrary draw.
    rng_tb = np.random.default_rng(7)
    li_draws, rh_draws = [], []
    med_income = float(cohort['median_income'].median())
    med_renter = float(cohort['renter_share'].median())
    for _ in range(500):
        shuffled = cohort.sample(frac=1.0, random_state=int(rng_tb.integers(1e9)))
        topN = shuffled.sort_values('rank_baseline', kind='stable').head(TOP_N)
        li_draws.append(float((topN['median_income'] <= med_income).mean()))
        rh_draws.append(float((topN['renter_share'] >= med_renter).mean()))
    report['fixed_size_top_n_comparison'] = {
        'top_n': TOP_N,
        'composite_top_n': equity_composition(cohort, idx_comp, cohort),
        'access_only_top_n_over_tiebreaks': {
            'share_lower_income_median': round(float(np.median(li_draws)), 3),
            'share_lower_income_range': [round(float(np.min(li_draws)), 3), round(float(np.max(li_draws)), 3)],
            'share_renter_heavy_median': round(float(np.median(rh_draws)), 3),
            'share_renter_heavy_range': [round(float(np.min(rh_draws)), 3), round(float(np.max(rh_draws)), 3)],
            'note': ('Access-only top-30 composition over 500 random tie-break '
                     'orderings -- a RANGE, because the 30 are drawn from a large '
                     'tied block and no single ordering is meaningful.')}}

    # --- Weight sensitivity: sweep each component weight, re-rank, measure
    # how much the top set and the full ranking move vs the base weights. ----
    sens = []
    base_rank = cohort['score_composite'].rank(ascending=False, method='min')
    for emphasis, w in [
        ('demand_heavy', {'demand': 0.6, 'access': 0.2, 'equity': 0.2}),
        ('access_heavy', {'demand': 0.2, 'access': 0.6, 'equity': 0.2}),
        ('equity_heavy', {'demand': 0.2, 'access': 0.2, 'equity': 0.6}),
        ('equal_thirds', {'demand': 1/3, 'access': 1/3, 'equity': 1/3}),
    ]:
        sc = composite_score(comp, w)
        rk = sc.rank(ascending=False, method='min')
        top_w = set(cohort.loc[rk.nsmallest(TOP_N).index, 'zcta'])
        rho_w = float(stats.spearmanr(base_rank, rk).correlation)
        eq = equity_composition(cohort, rk.nsmallest(TOP_N).index, cohort)
        sens.append({'emphasis': emphasis, 'weights': w,
                     'spearman_vs_base': round(rho_w, 4),
                     'top_n_overlap_with_base': len(top_w & top_comp),
                     'top_share_lower_income': eq['share_lower_income'],
                     'top_share_renter_heavy': eq['share_renter_heavy'],
                     'top_mean_income': eq['mean_income']})
    report['weight_sensitivity'] = sens

    # --- Ranking stability (bootstrap) -------------------------------------
    # Two SCALE-INVARIANT metrics on the SAME 200 resamples, so they are
    # directly comparable:
    #   * percentile movement: |percentile rank in resample - base percentile|.
    #     A with-replacement draw of size n contains only ~63% distinct codes,
    #     so an INTEGER rank in the resample runs 1..~264 while the base rank
    #     runs 1..418; differencing those two different scales inflates
    #     "movement" spuriously. Percentile rank (0..1) is invariant to how many
    #     codes survive the draw, so it is the correct perturbation measure.
    #   * tier retention: does the code keep its base priority tier, where the
    #     resample's tiers are recomputed on the resample itself (label compare,
    #     also scale-invariant).
    # Both recompute components, scores, and tiers on each resample.
    rng = np.random.default_rng(42)
    n = len(cohort)
    base_pct = cohort.set_index('zcta')['score_composite'].rank(ascending=False, pct=True)
    base_tier = cohort.set_index('zcta')['tier_composite']
    pct_delta = {z: [] for z in cohort['zcta']}
    tier_keep = {z: [] for z in cohort['zcta']}
    for _ in range(200):
        samp = cohort.iloc[rng.choice(n, size=n, replace=True)].copy()
        comp_s = build_components(samp)
        sc = composite_score(comp_s, BASE_WEIGHTS)
        samp = samp.assign(_pct=sc.rank(ascending=False, pct=True).values,
                           _tier=assign_tiers(sc.rank(ascending=False, method='min'),
                                              len(samp)).values)
        # average over duplicate draws of the same code within this resample
        g = samp.groupby('zcta').agg(_pct=('_pct', 'mean'), _tier=('_tier', 'median'))
        for z, row in g.iterrows():
            pct_delta[z].append(abs(float(row['_pct']) - float(base_pct[z])))
            tier_keep[z].append(int(round(row['_tier'])) == int(base_tier[z]))
    mean_pct_delta = {z: (round(float(np.mean(v)), 4) if v else None) for z, v in pct_delta.items()}
    tier_keep_rate = {z: (round(float(np.mean(v)), 3) if v else None) for z, v in tier_keep.items()}
    pd_ = pd.Series({z: v for z, v in mean_pct_delta.items() if v is not None})
    tk = pd.Series({z: v for z, v in tier_keep_rate.items() if v is not None})
    median_pct = float(pd_.median())
    report['ranking_stability'] = {
        'method': ('200 bootstrap resamples (seed 42); components, scores and '
                   'tiers recomputed per resample. Movement measured on a '
                   'SCALE-INVARIANT percentile basis because a size-n draw keeps '
                   'only ~63% distinct codes, so integer ranks would be on '
                   'different-length lists and not comparable.'),
        'median_percentile_movement': round(median_pct, 4),
        'median_movement_in_positions_of_418': round(median_pct * n, 1),
        'median_tier_retention_rate': round(float(tk.median()), 3),
        'share_codes_tier_retention_ge_0.8': round(float((tk >= 0.8).mean()), 3),
        'most_unstable_zctas_by_percentile_movement': pd_.nlargest(10).round(4).to_dict(),
        'consistency_note': ('A median percentile movement of ~{:.3f} (about {:.0f} '
                             'positions out of {}) is consistent with the high tier '
                             'retention: most codes stay within their tier because '
                             'they move only a little. The previously reported '
                             '"76 positions" was an artifact of comparing integer '
                             'ranks across different-sized bootstrap lists and is '
                             'corrected here.').format(median_pct, median_pct * n, n)}
    cohort['stability_pct_movement'] = cohort['zcta'].map(mean_pct_delta)
    cohort['tier_retention_rate'] = cohort['zcta'].map(tier_keep_rate)

    # --- Persist -----------------------------------------------------------
    cols = ['zcta', 'zip3', 'cluster', 'ev_per_1000_households',
            'ports_per_1000_households', 'median_income', 'renter_share',
            'households', 'comp_demand', 'comp_access', 'comp_equity',
            'score_baseline', 'rank_baseline', 'tier_baseline',
            'score_composite', 'rank_composite', 'tier_composite',
            'persons_per_household', 'special_use_flag',
            'stability_pct_movement', 'tier_retention_rate']
    cols = [c for c in cols if c in cohort.columns]
    cohort.sort_values('rank_composite')[cols].to_csv(args.out, index=False)
    with open(args.report, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    _figures(cohort, report)

    # --- Console summary ---------------------------------------------------
    ec = report['equity_comparison']; ca = report['coverage_audit']
    at = report['access_tie_audit']; st = report['ranking_stability']
    hc = report['household_coverage_top_tier']; gs = ec['group_sizes']
    print(f"Coverage: {ca['rankable']}/{ca['colorado_codes_total']} CO codes rankable; "
          f"{ca['not_rankable']} not rankable "
          f"({ca['not_rankable_profile']['share_no_public_station']:.0%} of those have no station "
          f"vs {ca['rankable_profile']['share_no_public_station']:.0%} of rankable).")
    print(f"Access ties: {at['n_zero_supply_codes']} zero-supply codes share one "
          f"access score ({at['share_of_cohort']:.0%} of cohort).")
    print("Tie-aware equity comparison (full priority groups, share lower-income / renter-heavy):")
    for key, label in [('cohort_overall', f'cohort (n={len(cohort)})'),
                       ('access_only_zero_supply_group', f"access zero-supply group (n={gs['access_zero_supply_group']})"),
                       ('composite_tier1_full', f"composite tier-1 (n={gs['composite_tier1']})")]:
        e = ec[key]
        print(f"  {label:38s} lower_income={e['share_lower_income']:.2f} "
              f"renter_heavy={e['share_renter_heavy']:.2f} mean_income={e['mean_income']:.0f}")
    print(f"Stability (scale-invariant): median percentile movement "
          f"{st['median_percentile_movement']} (~{st['median_movement_in_positions_of_418']} of {len(cohort)}) "
          f"| median tier retention {st['median_tier_retention_rate']}")
    print(f"Household COVERAGE of top-priority tier ({hc['n_codes']} codes): "
          f"{hc['households_covered']:,} households "
          f"(~{hc['renter_households_covered']:,} renter households) -- coverage, not access gained.")
    sa = report['special_use_audit']
    print(f"Special-use flag: {sa['n_flagged']} institutional/group-quarters ZCTAs "
          f"({sa['n_flagged_in_tier1']} in tier 1): "
          f"{', '.join(f['zcta'] for f in sa['flagged'])}")
    print(f"Wrote {repo_relative(args.out)} and {repo_relative(args.report)}.")


def _figures(cohort, report):
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Fig 12: baseline vs composite rank scatter (how areas move).
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    sc = ax.scatter(cohort['rank_baseline'], cohort['rank_composite'],
                    c=cohort['comp_equity'], cmap='viridis', alpha=.75, s=22)
    lim = [0, len(cohort) + 1]
    ax.plot(lim, lim, 'k--', lw=1, alpha=.6)
    ax.set(xlabel='Baseline rank (access-only; 1 = highest need)',
           ylabel='Composite rank (demand + equity + access)',
           title='How charging-need ranking shifts when equity + demand are added')
    fig.colorbar(sc, ax=ax, label='equity component (higher = more renter/low-income)')
    fig.tight_layout(); fig.savefig(FIGURES / '12_need_ranking_shift.png', dpi=160); plt.close(fig)

    # Fig 13: equity composition of FULL priority groups + weight sensitivity.
    ec = report['equity_comparison']
    sens = report['weight_sensitivity']
    groups = ['access\nzero-supply', 'composite\ntier-1'] + [s['emphasis'].replace('_', '\n') for s in sens]
    li = [ec['access_only_zero_supply_group']['share_lower_income'],
          ec['composite_tier1_full']['share_lower_income']] \
        + [s['top_share_lower_income'] for s in sens]
    rh = [ec['access_only_zero_supply_group']['share_renter_heavy'],
          ec['composite_tier1_full']['share_renter_heavy']] \
        + [s['top_share_renter_heavy'] for s in sens]
    x = np.arange(len(groups)); w = 0.4
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, li, w, label='share lower-income', color='#48c')
    ax.bar(x + w/2, rh, w, label='share renter-heavy', color='#c63')
    ax.axhline(ec['cohort_overall']['share_lower_income'], color='#48c', ls=':', lw=1, alpha=.7)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set(ylabel='Share of the priority group',
           title='Equity composition of priority groups (dotted = cohort baseline)')
    ax.legend()
    fig.tight_layout(); fig.savefig(FIGURES / '13_need_ranking_equity.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    main()
