# Milestone 3 - Models & Evaluation

Markdown source of the website `#models` section. All numbers are produced by
`src/prepare_modeling.py`, `src/train_regression.py`, and
`src/cluster_communities.py` (seed = 42), not entered by hand.

## Modeling plan and why these methods

Two complementary data-mining tasks on the Milestone 2 analysis table, scoped to
Colorado ZCTA codes:

- **Regression (primary)** — how far do community socioeconomics explain EV
  adoption? *(RQ1)*
- **K-Means clustering (supporting)** — group communities into profiles and
  screen for demonstrated demand with thin public supply. *(RQ4)*

Target: **EV adoption rate** = registered EVs per 1,000 households
(`ev_per_1000_households`). Predictors: median household income, renter share,
population, household count. `ev_count` / `bev_count` / `phev_count` are
**excluded** — the target is derived from them, so using any as a feature is
target leakage.

Regression fits a continuous per-household rate with a few plausible
socioeconomic drivers; clustering fits the descriptive "underserved" question,
which benefits from grouping similar communities rather than predicting a label.

## Modeling data preparation

| Step | Count | Note |
|---|---:|---|
| Union codes (Milestone 2) | 1,161 | starting table |
| Colorado codes kept | 500 | 80/81 prefix — working definition (ZIP ≠ ZCTA) |
| Out-of-state codes dropped | 661 | registration-ZIP artifacts (mostly ev_count = 1) |
| CO codes with a matched AFDC station | 293 | supply observed |
| CO codes recoded to **verified zero** supply | 207 | no AFDC match → station_count = 0, ports = 0 |
| Matched-but-blank port codes kept missing | 0 | would stay missing if present |
| Primary regression cohort (`households ≥ 100`) | 418 | removes 4 tiny-denominator rate artifacts |
| No-floor sensitivity cohort | 444 | retains all usable rows for the winsorization check |
| Clustering cohort | 418 | usable adoption, income, renter share, verified supply |

The **audit before recoding** is the key data-quality decision: a Colorado code
that appears in the registrations but not in the AFDC station file means AFDC
lists no public/existing/electric station there. The join is a direct
same-normalization 5-digit match and AFDC is a directory of existing stations,
so those 207 codes are genuine absences → **verified zero supply**, not missing.
Treating them as missing (and dropping them) would discard the underserved
communities the project exists to find.

## Validation strategy (leakage-safe, nested)

1. Reserve a held-out **test set of whole ZIP3 groups** first (no group spans
   train and test); score it **once**, for the selected model, at the end.
2. On the training portion: **nested ZIP3-GroupKFold** (outer = 5, inner = 4).
   The inner loop tunes hyperparameters; the outer loop scores the tuned
   procedure. Outer-fold means are the **unbiased headline**. Single-level
   grouped CV and a random split are kept only as labelled (optimistic)
   comparisons.
3. Imputation, scaling, log transforms, feature selection, and tuning happen
   **inside the training folds only**.

Metrics: **MAE** (primary), with **RMSE**, **R²**, **fold-to-fold variability**
(geographic consistency), and **subgroup MAE** (equity) as supporting evidence.

## Comparison (nested ZIP3-grouped CV on the training set)

"± std" is the spread across the five outer geographic folds.

| Approach | Role | Nested MAE (±std) | RMSE | R² |
|---|---|---:|---:|---:|
| Mean predictor | Baseline 1 | 49.0 (±7.6) | 62.9 | −0.12 |
| Income-only linear | Baseline 2 | 27.8 (±2.2) | 41.5 | 0.48 |
| Ridge regression | Candidate 1 | 27.6 (±2.1) | 40.1 | 0.52 |
| Random Forest | Candidate 2 | **26.7 (±2.0)** | **38.8** | **0.55** |

The mean predictor confirms an honest baseline (R² ≈ 0). **Income alone already
explains most of what the models capture** (R² ≈ 0.48). Ridge adds little;
Random Forest is best (R² ≈ 0.55) — some nonlinearity, but a modest gain over a
one-feature linear model, with overlapping fold error bars. Following "prefer
the simpler model when performance is similar," income-only is a strong
interpretable reference and the forest is the best performer.

**Honesty check (nested vs. single-level).** Single-level grouped CV — which
reuses the tuning folds for reporting — put the Random Forest at MAE 25.8 /
R² 0.64. Nested CV puts it at MAE 26.7 / R² 0.55. The single-level number was
optimistic for the tuned model; the nested number is what we report.

## Automated experiment search and model selection

A bounded, seeded harness (`src/experiment_harness.py`) scores a predeclared set
of candidates — feature subsets, preprocessing strategies (log-transform skewed
size features, mean vs. median imputation), and Ridge/Random Forest
hyperparameter grids — through the same nested protocol, logging every run
(config, features, params, seed, git commit, nested + subgroup metrics,
selected/rejected) to `results/experiments.csv`.

- **Fixed and not experiment-tunable:** target, ZIP3 groups, held-out test set,
  data-quality rules, metric, and the six equity subgroups. Every candidate
  passes the leakage guard first.
- **Selection:** MAE-first; among candidates within 1 MAE unit of the best,
  prefer lower fold-std (geographic consistency), then lower subgroup-MAE spread
  (equity). Baselines are references, never winners.
- **Selected model:** a **two-feature (income + renter share) Random Forest** —
  it ties the four-feature forest on MAE (26.2) while being the most consistent
  across folds (±1.4, the lowest of any candidate). The held-out test set, scored
  once after selection, gives MAE 23.0, confirming the nested estimate held.

## Equity: performance across Colorado communities

Out-of-fold MAE within fixed subgroups for the selected model:

| Subgroup | Codes | MAE |
|---|---:|---:|
| Lower income | 164 | 16.3 |
| Higher income | 165 | 36.3 |
| Renter-heavy | 165 | 22.9 |
| Owner-heavy | 164 | 29.8 |
| Zero public supply | 107 | 26.5 |
| Has infrastructure | 222 | 26.2 |

The project-relevant finding: the model is **roughly even across zero-supply vs.
has-infrastructure areas** (≈ 26 either way), so it does not systematically fail
in the under-built communities the underserved screen targets. Its weakness is
**high-income, high-adoption** areas, where adoption varies most — a caution for
prediction, not for the equity screen.

## What the models learned, and where they fail

- **Feature importance.** Median household income dominates both the Ridge
  coefficients and the Random Forest importance (~0.81), confirmed by
  permutation importance on the test set. Renter share, population, and
  household count contribute little.
- **Error analysis.** Out-of-sample absolute error is larger for higher-income
  codes (MAE ≈ 32 vs. ≈ 14 for lower-income) and for larger ZCTAs — affluent
  high-adoption areas have the widest spread and are hardest to predict.
- **Outlier sensitivity.** Capping the target at the training 99th percentile
  (learned from training rows only, scored against the original uncapped test
  values) barely moves anything (RF test R² 0.33 → 0.34). Conclusions are not an
  artifact of a few extreme codes.

Figures: `06_model_comparison.png`, `07_predicted_vs_actual.png`,
`08_residuals.png`, `09_feature_importance.png`.

## Pre-freeze diagnostics (`src/diagnostics_m3.py` → `diagnostics_m3.json`)

1. **Fully-nested selection MAE = 26.84 (headline).** Picking the winner as the
   min over candidates' outer-fold scores is itself a selection step, so the
   selected candidate's own nested MAE (26.25) is mildly optimistic. A fully
   nested procedure (re-select the best candidate inside each outer fold) gives
   26.84 — a residual selection bias of ~+0.6. It picked the four-feature forest
   in 4 of 5 folds and the two-feature forest in 1.
2. **Two-feature vs four-feature RF: not significantly different.** Paired over
   the five outer folds, mean MAE gap 0.41 (two-feature better in 3/5), paired-t
   p = 0.68, Wilcoxon p = 1.0. The two-feature model is chosen on parsimony and
   lower fold variance, not a proven accuracy edge.
3. **EDA r = 0.24 vs. model R² reconciled.** Different populations: full
   1,161-code union r² = 0.06; Colorado households-≥-100 cohort r = 0.724
   (r² = 0.52); income-only pooled out-of-fold predictive R² = 0.56. Out-of-state
   single-EV ZIPs wash out the relationship in the full union. Predictive R² is
   also not the squared Pearson r (they coincide only for in-sample single-
   feature OLS). Neither figure is an error.
4. **Held-out test by ZIP3 group.** Test MAE 23.0 / RMSE 34.6 / R² 0.13 over 4
   held-out groups. The low R² reflects a low-variance test slice (test adoption
   median ≈ 30 vs. training ≈ 51), not poor prediction: the model beats the mean
   baseline on every held-out group (e.g. group 811 MAE 12.8 vs. 50.0). The
   nested outer-fold mean is the headline; the single grouped test is a one-time
   confirmation.

## Clustering: community profiles and candidate underserved areas

K-Means over 418 Colorado codes (adoption, income, renter share, verified public
ports per 1,000 households, standardized). **k = 3** by silhouette (0.40);
bootstrap stability mean ARI = 0.74.

| Cluster | Codes | Adoption /1k | Income | Renter | Ports /1k | Profile |
|---:|---:|---:|---:|---:|---:|---|
| 0 | 42 | 95.7 | $89k | 0.41 | 18.3 | Well-supplied, renter-heavy, moderate adoption |
| 1 | 81 | 149.6 | $140k | 0.17 | 2.0 | High-adoption affluent suburbs, thin public supply |
| 2 | 295 | 37.7 | $76k | 0.28 | 1.9 | Mainstream mid-income, low adoption and supply |

**Candidate underserved areas (RQ4).** A code is a *candidate* when it shows
**demonstrated demand** (adoption ≥ cohort median) together with **thin public
supply** (ports/1,000 hh ≤ cohort median). *Low supply alone is not an
underserved verdict.* This screens **76 Colorado codes**, of which **25 also
carry an equity flag** (below-median income or above-median renter share),
concentrated in cluster 1 (affluent suburbs) and cluster 2 (mid-income
mainstream). Full shortlist: `data/processed/candidate_underserved_zctas.csv`.

Figures: `10_cluster_selection.png`, `11_cluster_map.png`.

## ZCTA charging-need ranking (`src/rank_need.py`) — the planning track

Two predefined rankings over the 418 Colorado codes, percentile-normalized
(robust to heavy-tailed supply) with fixed, un-tuned weights:

- **Approach 1 — access-only baseline:** rank by public-charging scarcity per
  household. Low supply alone does not prove high need.
- **Approach 2 — demand + equity + access composite:** three separate
  components. Demand is **observed** adoption (never the regression prediction),
  access is charging scarcity, equity blends renter share and lower income.
  Base weights demand 0.33 / access 0.34 / equity 0.33.

The rankings disagree (Spearman ρ = 0.56; 4 of top-30 shared), and the composite
shifts toward the equity-intended communities:

| Top-30 set | Share lower-income | Share renter-heavy | Mean income | Mean adoption |
|---|---:|---:|---:|---:|
| Cohort overall | 0.50 | 0.50 | $90k | 65 |
| Access-only baseline | 0.27 | 0.10 | $108k | 67 |
| Demand + equity + access | **0.70** | **0.77** | **$75k** | 68 |

**Weight sensitivity.** Demand-heavy weighting pushes the top set to mean income
$134k (the failure mode of an adoption-only ranking); equity-heavy pushes it to
97% lower-income / 100% renter-heavy ($58k); equal-thirds ≈ base (ρ = 0.999).

**Pre-freeze checks (`need_ranking_report.json`):**

- **Broad tiers, not exact ranks (stability verified).** Bootstrap resampling
  (scores and tiers recomputed per resample) shows a code's priority moves only
  a little: median **percentile movement 0.024 (~10 of 418 positions)**, and
  **tier retention 0.97**. The two agree — codes stay in their tier because they
  move little. An earlier draft's "76 positions" was an artifact of comparing
  integer ranks across different-sized bootstrap lists (a size-n draw keeps only
  ~63% of codes); it is corrected to the scale-invariant percentile measure.
  - *What 0.97 retention means:* stability **under the chosen bootstrap
    procedure** — how often a ZCTA keeps its tier when the cohort is resampled —
    **not** a 97% probability that the areas are correct station locations.
  - *Reproducibility (fixed in `need_ranking_report.json`):* 200 bootstrap
    resamples of the 418-code cohort (with replacement, seed 42); components,
    composite score, and the five tiers recomputed from scratch each resample;
    resampled codes matched to the original by 5-digit code with duplicate draws
    averaged; movement = change in percentile rank (scale-invariant), retention
    = share of resamples keeping the base tier.
- **Coverage audit.** 418 of 500 Colorado codes are rankable; 82 are not
  (missing income/household data or households < 100). The un-rankable codes are
  mostly tiny/rural and 74% have no public station (vs 35% of the rankable
  cohort) — potentially *more* under-served. Reported as a separate group, not
  dropped. A full Colorado Census-ZCTA enumeration needs a ZIP-to-ZCTA crosswalk
  (future work).
- **Access ties, handled tie-aware.** 146 of 418 codes (35%) have exactly zero
  public supply and share one access score. Selecting a top-30 from that block
  is arbitrary — across 500 tie-break orderings its lower-income share ranges
  0.37–0.87 — so we report the *entire* tied group and compare full priority
  groups (sizes stated), not an arbitrary top-30. Tie-aware comparison:

  | Priority group (full) | n | Lower-income | Renter-heavy | Mean income |
  |---|---:|---:|---:|---:|
  | Cohort overall | 418 | 0.50 | 0.50 | $90k |
  | Access-only tied zero-supply | 146 | 0.64 | 0.31 | $81k |
  | Composite top tier | 84 | 0.49 | 0.64 | $93k |

  This is a **trade-off between equity definitions**, not a uniform improvement:
  the composite is more renter-heavy (0.64 vs 0.31) but slightly *less*
  lower-income (0.49 vs 0.64) than the zero-supply group. Which ranking is "more
  equitable" depends on the definition of need a planner adopts; the project
  reports both rather than claiming one dominates.
- **Demand ≠ unmet demand.** Observed adoption measures existing ownership, not
  unmet public-charging need. Access + equity carry two-thirds of the weight, so
  a low-ownership, renter-heavy, low-income, zero-supply area can still reach the
  top tier.
- **Household coverage (renamed).** The top tier (84 codes) contains ~546k
  households (~228k renter households) — the population **living in** the
  identified priority areas (coverage), NOT households proven to gain access.
  Actual access improvement needs a proposed station location and travel-distance
  analysis (the H3 stage).

Figures: `12_need_ranking_shift.png`, `13_need_ranking_equity.png`. Data:
`need_ranking.csv` (incl. `tier_composite`, `tier_baseline`, stability columns),
`need_ranking_report.json`.

## Interpretation, limitations, and what this does not prove

- **Two objectives, kept separate.** *Primary (planning):* which communities may
  benefit most from new charging — clustering + need ranking. *Supporting:* how
  socioeconomics relate to existing adoption — regression, as demand evidence.
- **Descriptive, not causal.** Income predicts adoption well, but this is
  association from a ZIP↔ZCTA proxy join across mixed time periods.
- **Equity caveat, handled explicitly.** A model of current ownership encodes
  income inequality — a demand-only ranking reaches mean income $134k. The need
  ranking therefore uses observed demand as only one of three components and
  reports the equity-weighted ranking alongside.
- **Ranks are candidate flags, not verdicts.** Individual ranks move under
  resampling and no station-outcome data exists, so no code is "the best site."
- **Geography is a proxy.** The 80/81 prefix and ZIP3 grouping approximate
  Colorado membership and spatial structure; ZCTA polygons or a ZIP-to-county
  crosswalk would be the accurate upgrade.
- **Next stage (future work, not Milestone 3 results).** H3 resolution 10
  candidate cells, road accessibility, proximity to existing chargers,
  substation proximity, and land-use suitability need spatial data the repo does
  not yet hold. The regression, clustering, and need ranking are the
  community-need inputs to that later site-level layer.
