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

## Validation strategy (leakage-safe)

1. Reserve a held-out **test set of whole ZIP3 groups** first (no group spans
   train and test).
2. On the training portion: **GroupKFold by ZIP3** (primary) + random KFold
   (comparison).
3. Imputation, scaling, and hyperparameter tuning happen **inside the training
   folds only**.
4. Colorado has ~17 ZIP3 groups, so a single grouped test slice is
   high-variance; **grouped CV is the headline**, the test split is a
   confirmatory check.

Metrics: **MAE** (typical error in EVs per 1,000 households), **RMSE**
(penalizes large misses), **R²** (variation explained).

## Comparison (grouped CV on the training set)

| Approach | Role | MAE | RMSE | R² |
|---|---|---:|---:|---:|
| Mean predictor | Baseline 1 | 49.1 | 64.3 | −0.05 |
| Income-only linear | Baseline 2 | 27.9 | 41.7 | 0.56 |
| Ridge regression | Candidate 1 | 27.7 | 40.3 | 0.59 |
| Random Forest | Candidate 2 | **25.8** | **37.8** | **0.64** |

The mean predictor confirms an honest baseline (R² ≈ 0). **Income alone already
explains most of what the models capture** (R² = 0.56). Ridge adds little;
Random Forest is best (R² = 0.64) — some nonlinearity, but a modest gain over a
one-feature linear model. Following "prefer the simpler model when performance
is similar," income-only is a strong interpretable reference and Random Forest
is the best performer.

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

## Interpretation, limitations, and what this does not prove

- **Descriptive, not causal.** Income predicts adoption well, but this is
  association from a ZIP↔ZCTA proxy join across mixed time periods.
- **Equity caveat.** A model trained on *current* ownership partly encodes
  existing income inequality; chasing predicted demand alone would reinforce it,
  so the underserved screen weighs demand and equity together.
- **Geography is a proxy.** The 80/81 prefix and ZIP3 grouping approximate
  Colorado membership and spatial structure; ZCTA polygons or a ZIP-to-county
  crosswalk would be the accurate upgrade.
- **Feeds, does not settle, the H3 map.** Regression and clustering are
  supporting evidence for the eventual site-priority layer. Road access, land
  use, charger coverage, and electrical-infrastructure capacity still need
  separate assessment before any construction claim.
