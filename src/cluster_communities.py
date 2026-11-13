"""Milestone 3 - Step 3: K-Means community profiles (supporting analysis).

Groups Colorado ZCTAs by adoption, income, renter share, and VERIFIED public
charging supply to surface *candidate* underserved communities for RQ4. This is
a descriptive, exploratory method, not a prediction: a cluster with low supply
is only ONE signal. An underserved verdict also weighs demand (adoption) and
equity (income, renter share) -- low supply in a wealthy, low-demand area is not
the same as low supply in a high-demand, renter-heavy, lower-income area.

Model selection and honesty:
  * features standardized (K-Means uses Euclidean distance).
  * k swept 2..8, chosen by the best mean silhouette with an inertia/elbow
    reference recorded alongside.
  * stability assessed by re-running K-Means on bootstrap resamples and
    measuring label agreement (adjusted Rand index) against the full-data fit.

    python src/cluster_communities.py
"""
from __future__ import annotations
import argparse
import json
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from modeling_common import (
    CLUSTER_FEATURES, FIGURES, MODELING_CSV, PROCESSED, repo_relative,
)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from plot_style import apply_web_style  # noqa: E402
apply_web_style()

# Standardized ZCTA rows include many exact duplicates (e.g. every verified
# zero-supply area), so distance matmuls hit zero-norm pairs and numpy emits
# divide/overflow floating-point warnings deep inside KMeans/silhouette. These
# are expected and harmless for clustering (they do not affect labels); quiet
# just that numeric noise so the run output stays readable. np.seterr applies
# process-wide, including sklearn's worker threads, unlike warnings filters.
np.seterr(divide='ignore', over='ignore', invalid='ignore')

SEED = 42
K_RANGE = range(2, 9)
N_BOOTSTRAP = 25


def sweep_k(X, log):
    """Fit K-Means for each k; record silhouette and inertia (elbow ref)."""
    rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        labels = km.fit_predict(X)
        sil = float(silhouette_score(X, labels))
        rows.append({'k': k, 'silhouette': round(sil, 4),
                     'inertia': round(float(km.inertia_), 2)})
    log['k_sweep'] = rows
    best = max(rows, key=lambda r: r['silhouette'])
    log['chosen_k'] = best['k']
    log['chosen_k_rule'] = 'highest mean silhouette over k=2..8 (inertia/elbow recorded for reference)'
    return best['k']


def stability(X, k, log):
    """Bootstrap-resample stability: ARI of resample labels vs. full-fit labels
    on the shared rows. Higher mean ARI = more stable cluster structure."""
    base = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X)
    base_labels = base.labels_
    rng = np.random.default_rng(SEED)
    n = len(X)
    aris = []
    # Bootstrap resampling draws duplicate rows; the resulting zero-distance
    # pairs make numpy's matmul emit harmless divide/overflow warnings inside
    # KMeans. Silence just this expected numeric noise around the loop.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for _ in range(N_BOOTSTRAP):
            idx = rng.choice(n, size=n, replace=True)
            uniq = np.unique(idx)
            km = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X[idx])
            # Predict on the unique original rows and compare to the base fit.
            resample_labels = km.predict(X[uniq])
            aris.append(adjusted_rand_score(base_labels[uniq], resample_labels))
    log['stability'] = {'method': f'{N_BOOTSTRAP} bootstrap resamples; ARI vs full-data fit',
                        'mean_ari': round(float(np.mean(aris)), 4),
                        'min_ari': round(float(np.min(aris)), 4),
                        'max_ari': round(float(np.max(aris)), 4)}
    return base_labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default=str(MODELING_CSV))
    p.add_argument('--assignments', default=str(PROCESSED / 'cluster_assignments.csv'))
    p.add_argument('--profiles', default=str(PROCESSED / 'cluster_profiles.json'))
    args = p.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input, dtype={'zcta': 'string', 'zip3': 'string'})
    cohort = df.loc[df['cluster_cohort'] == True].copy()  # noqa: E712
    cohort = cohort.dropna(subset=CLUSTER_FEATURES)

    log = {'generated_utc': datetime.now(timezone.utc).isoformat(),
           'input': repo_relative(args.input),
           'features': list(CLUSTER_FEATURES), 'seed': SEED,
           'n_cohort': int(len(cohort)),
           'note': ('Descriptive clustering. Low public supply is one signal; an '
                    'underserved verdict also weighs demand (adoption) and equity '
                    '(income, renter share). Not a construction-suitability claim.')}

    scaler = StandardScaler()
    X = scaler.fit_transform(cohort[CLUSTER_FEATURES].to_numpy())

    k = sweep_k(X, log)
    labels = stability(X, k, log)
    cohort['cluster'] = labels

    # --- Cluster profiles (original units, not standardized) ---------------
    profiles = {}
    for c in sorted(cohort['cluster'].unique()):
        grp = cohort.loc[cohort['cluster'] == c]
        profiles[int(c)] = {
            'n': int(len(grp)),
            'mean_ev_per_1000_households': round(float(grp['ev_per_1000_households'].mean()), 2),
            'mean_median_income': round(float(grp['median_income'].mean()), 0),
            'mean_renter_share': round(float(grp['renter_share'].mean()), 3),
            'mean_ports_per_1000_households': round(float(grp['ports_per_1000_households'].mean()), 3),
            'share_zero_supply': round(float((grp['ports_per_1000_households'] == 0).mean()), 3),
        }
    log['cluster_profiles'] = profiles

    # --- Candidate underserved areas: a ZCTA-level screen, described BY cluster
    # Low public supply alone is not an underserved verdict. A candidate
    # underserved ZCTA must show DEMONSTRATED DEMAND (adoption at/above the
    # cohort median) together with THIN PUBLIC SUPPLY (ports/1,000 hh at/below
    # the cohort median). We compute this per ZCTA -- cluster means wash out
    # within-cluster mismatches -- and then report how the flagged ZCTAs
    # distribute across the discovered clusters. An equity lens (lower income
    # OR renter-heavy, each relative to the cohort median) is attached as a
    # separate, visible signal, not folded silently into the verdict.
    med_adopt = float(cohort['ev_per_1000_households'].median())
    med_supply = float(cohort['ports_per_1000_households'].median())
    med_income = float(cohort['median_income'].median())
    med_renter = float(cohort['renter_share'].median())

    high_demand = cohort['ev_per_1000_households'] >= med_adopt
    low_supply = cohort['ports_per_1000_households'] <= med_supply
    equity = (cohort['median_income'] <= med_income) | (cohort['renter_share'] >= med_renter)
    cohort['candidate_underserved'] = high_demand & low_supply
    cohort['equity_flag'] = equity

    # Cluster-level descriptive summary of the flagged ZCTAs.
    underserved_by_cluster = {}
    for c in sorted(cohort['cluster'].unique()):
        grp = cohort.loc[cohort['cluster'] == c]
        flagged = grp.loc[grp['candidate_underserved']]
        underserved_by_cluster[int(c)] = {
            'n_in_cluster': int(len(grp)),
            'n_candidate_underserved': int(len(flagged)),
            'n_also_equity_flagged': int((flagged['equity_flag']).sum()),
        }

    shortlist = cohort.loc[cohort['candidate_underserved'],
        ['zcta', 'cluster', 'ev_per_1000_households', 'ports_per_1000_households',
         'median_income', 'renter_share', 'households', 'equity_flag']
    ].sort_values(['equity_flag', 'ev_per_1000_households'], ascending=[False, False])

    log['underserved_assessment'] = {
        'reference_medians': {
            'adoption': round(med_adopt, 2), 'supply': round(med_supply, 3),
            'income': round(med_income, 0), 'renter_share': round(med_renter, 3)},
        'rule': ('candidate_underserved (per ZCTA) = adoption >= cohort median '
                 'AND public ports/1,000 hh <= cohort median. Equity flag '
                 '(income <= median OR renter_share >= median) reported '
                 'separately. Low supply alone is NOT an underserved verdict; '
                 'demand and equity must be weighed.'),
        'by_cluster': underserved_by_cluster}
    log['candidate_underserved_zcta_count'] = int(len(shortlist))
    log['candidate_underserved_equity_flagged_count'] = int(shortlist['equity_flag'].sum())
    log['candidate_underserved_zctas_top'] = shortlist.head(20)['zcta'].tolist()
    shortlist.to_csv(PROCESSED / 'candidate_underserved_zctas.csv', index=False)

    cohort[['zcta', 'zip3', 'cluster', *CLUSTER_FEATURES, 'median_income',
            'households', 'candidate_underserved', 'equity_flag']
           ].to_csv(args.assignments, index=False)
    with open(args.profiles, 'w') as f:
        json.dump(log, f, indent=2)

    _figures(log['k_sweep'], k, cohort, CLUSTER_FEATURES)

    print(f"Clustering cohort={len(cohort)}  chosen k={k} "
          f"(silhouette {dict((r['k'], r['silhouette']) for r in log['k_sweep'])[k]})")
    print(f"Stability mean ARI={log['stability']['mean_ari']}")
    ubc = log['underserved_assessment']['by_cluster']
    for c, prof in profiles.items():
        n_us = ubc[c]['n_candidate_underserved']
        print(f"  cluster {c}: n={prof['n']:3d} adoption={prof['mean_ev_per_1000_households']:7.1f} "
              f"income={prof['mean_median_income']:>8.0f} renter={prof['mean_renter_share']:.2f} "
              f"ports/1k={prof['mean_ports_per_1000_households']:.2f}  "
              f"candidate_underserved={n_us}")
    print(f"Candidate underserved ZCTAs: {log['candidate_underserved_zcta_count']} "
          f"({log['candidate_underserved_equity_flagged_count']} also equity-flagged)")
    print(f"Wrote {repo_relative(args.assignments)} and {repo_relative(args.profiles)}.")


def _figures(k_sweep, chosen_k, cohort, features):
    ks = [r['k'] for r in k_sweep]
    sil = [r['silhouette'] for r in k_sweep]
    inertia = [r['inertia'] for r in k_sweep]

    # Fig: silhouette + elbow vs k
    fig, ax1 = plt.subplots(figsize=(7.2, 4.6))
    ax1.plot(ks, sil, 'o-', color='#48c', label='silhouette')
    ax1.axvline(chosen_k, color='#c63', ls='--', lw=1, label=f'chosen k={chosen_k}')
    ax1.set(xlabel='k (number of clusters)', ylabel='Mean silhouette')
    ax2 = ax1.twinx()
    ax2.plot(ks, inertia, 's--', color='#999', alpha=.6, label='inertia')
    ax2.set_ylabel('Inertia (elbow reference)')
    ax1.set_title('K-Means model selection')
    ax1.legend(loc='upper right')
    fig.tight_layout(); fig.savefig(FIGURES / '10_cluster_selection.png', dpi=160); plt.close(fig)

    # Fig: cluster scatter (adoption vs public supply, colored by cluster)
    fig, ax = plt.subplots(figsize=(7.6, 5.4))
    sc = ax.scatter(cohort['ev_per_1000_households'], cohort['ports_per_1000_households'],
                    c=cohort['cluster'], cmap='tab10', alpha=.75, s=22)
    ax.set(xlabel='EVs per 1,000 households (demand)',
           ylabel='Public ports per 1,000 households (supply)',
           title='Colorado ZCTA community clusters')
    legend = ax.legend(*sc.legend_elements(), title='cluster', loc='upper right')
    ax.add_artist(legend)
    fig.tight_layout(); fig.savefig(FIGURES / '11_cluster_map.png', dpi=160); plt.close(fig)


if __name__ == '__main__':
    main()
