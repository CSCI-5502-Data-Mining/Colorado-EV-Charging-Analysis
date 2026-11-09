# Colorado EV Charging Access & Adoption

CSCI 5502 Data Mining semester group project.

**Team:** Eddie Strand, Snehal Bhatnagar, Abdullah Bharde

**Repository:** https://github.com/CSCI-5502-Data-Mining/Colorado-EV-Charging-Analysis

**Website:** https://colorado-ev-charging-analysis.vercel.app

## Project goal

Identify the socioeconomic and geographic factors associated with EV adoption in Colorado and determine where additional public charging infrastructure could best balance demand and equity.

Approved Milestone 0 question: *What factors are associated with EV adoption in Colorado, and which areas appear underserved by the existing charging network?*

## Current research questions

1. How do income and housing characteristics (including affordability and renter share, as data allow) relate to EV adoption and public charger availability across Colorado? *(relationship / pattern)*
2. Which Colorado areas have high EV adoption relative to the number of available public charging ports? *(comparative)*
3. How does charging-port density differ between urban and rural areas of Colorado? *(comparative)*
4. Which Colorado areas show the greatest combination of expected charging demand and limited existing charging access? *(descriptive / identification)*

## Data-source references

- Atlas EV Hub / Colorado Energy Office — [State EV registration data](https://www.atlasevhub.com/market-data/state-ev-registration-data/)
- U.S. DOE Alternative Fuels Data Center — [Data downloads](https://afdc.energy.gov/data_download)
- Atlas Public Policy / Colorado Energy Office — [EValuateCO](https://atlaspolicy.com/evaluateco/)
- Colorado Information Marketplace — [Charging station map](https://data.colorado.gov/Energy/Map-of-Alternative-Fuels-and-Electric-Vehicle-Char/fwyp-a5y7)
- U.S. Census Bureau ACS 5-year estimates — added after Milestone 0 for income/housing context; tables used: `B01003_001E` (population), `B19013_001E` (median household income), `B25003_001E` (occupied households), `B25003_003E` (renter-occupied) — [ACS 5-year](https://www.census.gov/data/developers/data-sets/acs-5year.html)

The four sources above are the **original Milestone 2 source list**. The Milestone 3 supporting analysis (urban/rural + affordability) added two more official Census tables, fetched from the Census APIs on **2026-11-08** at national ZCTA geography (filtered to Colorado):

- U.S. Census Bureau **2020 Decennial DHC, Table P2 (urban/rural population)** — variables `P2_001N`/`P2_002N`/`P2_003N`; endpoint `https://api.census.gov/data/2020/dec/dhc`; 33,774 ZCTA rows (477 of 500 Colorado codes matched)
- U.S. Census Bureau **ACS 2024 5-year, Table B25070 (gross rent as % of household income)** — variables `B25070_001E` + `007E`–`010E` (with `011E` "not computed" excluded from the denominator); endpoint `https://api.census.gov/data/2024/acs/acs5`; 33,772 ZCTA rows (437 of 500 Colorado codes matched)

**Join limitation:** both match on the 5-digit ZCTA code (ZIP↔ZCTA proxy), and the vintages differ (2020 Census vs 2020–2024 ACS vs 2026 EV/station snapshots), so comparisons are descriptive. Full provenance in [`docs/provenance.csv`](docs/provenance.csv); raw-fetch record in `data/raw/urban_affordability_provenance.json`.

Acquisition instructions: [`data/README.md`](data/README.md). Do not commit secrets, credentials, or restricted raw extracts.

## Reproduce the Milestone 2 analysis

Raw data files are **not** in the repository (large / licensed). Follow [`data/README.md`](data/README.md) to place them in `data/raw/`, then:

Run these from the repository root, one at a time:

```bash
# 1. Create and activate a repo-local virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1

# 2. Install pinned dependencies
pip install -r requirements.txt

# 3. Add your Census API key (free: https://api.census.gov/data/key_signup.html)
cp .env.example .env             # Windows PowerShell: copy .env.example .env
#    then edit .env and set CENSUS_API_KEY=your_key  (.env is git-ignored)

# 4. (optional) Fetch the national Census ACS ZCTA estimates.
#    python-dotenv auto-loads CENSUS_API_KEY from .env, so no manual export is needed.
#    Skip this if data/raw/acs_zcta_2024.csv is already present — the pipeline runs without it.
python src/fetch_acs.py
#    (An exported env var still works and overrides .env:
#     export CENSUS_API_KEY="YOUR_KEY"  / PowerShell: $env:CENSUS_API_KEY="YOUR_KEY")

# 5. Clean, join, and generate the processed dataset + five EDA charts
python src/prepare_and_eda.py

# 6. (optional) Run the offline cleaning-rule tests
python -m unittest discover -s tests   # or: pytest tests
```

View the results after step 5:
- Processed tables and reports are in `data/processed/` (open the `.csv`/`.json` files).
- The five EDA charts are PNGs in `figures/` — open `figures/01_ev_rate_distribution.png` through `figures/05_missingness.png`.

Outputs (regenerated, not committed as raw data):
- `data/processed/colorado_ev_zcta_clean.csv` — analysis-ready, one row per matched code
- `data/processed/{quality_report,eda_metrics}.json`, `join_audit.csv`, `outlier_review.csv`
- `figures/01_ev_rate_distribution.png` … `figures/05_missingness.png`

The narrative walkthrough (acquisition → cleaning → join → EDA, with inline charts and interpretations) is in [`notebooks/cleaning_and_preprocessing.ipynb`](notebooks/cleaning_and_preprocessing.ipynb). Documentation of provenance, the data dictionary, and limitations is in [`docs/`](docs/). All paths resolve relative to the repository root, so no machine-specific absolute paths are needed.

## Reproduce the Milestone 3 modeling & evaluation

Milestone 3 builds on the Milestone 2 output (`data/processed/colorado_ev_zcta_clean.csv`). After running the Milestone 2 pipeline above, run these from the repository root, one at a time (all use a fixed seed = 42, so results are deterministic):

```bash
# 0. (one-time) scikit-learn is pinned in requirements.txt; reinstall if needed
pip install -r requirements.txt

# 1. Build the Colorado modeling table: filter to CO (80/81), audit AFDC
#    station matching and recode verified zero supply, build features/target
#    and cohorts. Writes data/processed/modeling_colorado.csv + prep report.
python src/prepare_modeling.py

# 2. Regression: mean baseline + income-only baseline + Ridge + Random Forest.
#    Headline evaluation is NESTED ZIP3-grouped CV (unbiased); single-level
#    grouped CV and a random split are kept as labelled comparisons. Includes
#    training-only tuning and winsorization sensitivity. Writes
#    regression_metrics.json and figures 06-09.
python src/train_regression.py

# 2b. Bounded, seeded experiment harness: a predeclared search over feature
#     subsets, preprocessing strategies, and hyperparameters, each scored by the
#     same nested protocol, with MAE-first + equity/stability model selection
#     and a one-time held-out test evaluation of the selected model. Logs every
#     experiment (config, seed, git commit, nested + subgroup metrics,
#     selected/rejected) to results/. The held-out test set is untouched during
#     the search.
python src/experiment_harness.py

# 3. K-Means community clustering: k sweep (silhouette + elbow), bootstrap
#    stability, cluster profiles, and the candidate-underserved screen. Writes
#    cluster_assignments.csv, cluster_profiles.json, candidate_underserved_zctas.csv,
#    and figures 10-11.
python src/cluster_communities.py

# 3b. ZCTA charging-NEED ranking (planning track): access-only baseline vs a
#     demand + equity + access composite (demand = OBSERVED adoption, not the
#     model prediction). Reports 5 broad priority TIERS (exact ranks are
#     unstable), a 500->418 coverage audit of who cannot be ranked, an
#     access-score tie audit (zero-supply codes tie), weight sensitivity, rank +
#     tier stability, and a top-tier household-reach measure. Writes
#     need_ranking.csv, need_ranking_report.json, and figures 12-13.
python src/rank_need.py

# 3d. Supporting descriptive analysis (urban/rural + renter affordability).
#     Separate from the frozen models. First fetch two official Census tables
#     (needs a free CENSUS_API_KEY in .env; never printed/committed):
#       - 2020 DHC P2 urban/rural population, ACS 2024 B25070 renter cost burden
python src/fetch_urban_affordability.py
#     Then classify CO ZCTAs (urban-dominant/mixed/rural-dominant by Census 2020
#     urban share) and compare ports/1,000 hh (all/L2/DCFC) + renter cost
#     burden. Writes urban_rural_affordability.csv + report and figures 14-15.
python src/urban_rural_affordability.py

# 3c. (optional) Pre-freeze diagnostics: reconcile the EDA correlation with the
#     regression R2, audit nested-CV selection bias, held-out test by ZIP3
#     group, and paired-fold comparison of the two Random Forests. Read-only;
#     writes data/processed/diagnostics_m3.json.
python src/diagnostics_m3.py

# 4. (optional) Offline checks for leakage guards, CO filter, supply recoding,
#    cohort logic, grouped-split holdout, nested-CV + selection-bias logic, the
#    fixed equity subgroups, and the need-ranking components
python -m unittest discover -s tests   # or: pytest tests
```

Milestone 3 outputs (regenerated, not committed as raw data):
- `data/processed/modeling_colorado.csv` — Colorado modeling table with cohorts and ZIP3 groups
- `data/processed/modeling_prep_report.json` — CO filter + supply-audit + cohort counts
- `data/processed/regression_metrics.json` — nested + single-level comparison, tuning, winsorization sensitivity, feature importance, subgroup errors
- `data/processed/cluster_assignments.csv`, `cluster_profiles.json`, `candidate_underserved_zctas.csv`
- `data/processed/need_ranking.csv`, `need_ranking_report.json` — ZCTA need ranking (both approaches), priority tiers, weights, sensitivity, scale-invariant percentile + tier stability, the 500→418 coverage audit, the access-tie audit, a tie-aware full-priority-group comparison, and household coverage of the top tier
- `data/processed/diagnostics_m3.json` — the four pre-freeze diagnostic checks
- `data/processed/urban_rural_affordability.csv`, `…_report.json` — supporting analysis: Census-2020 urban/rural classification, ports/1,000 hh (all/L2/DCFC) by class, and ACS B25070 renter cost burden (raw Census extracts are gitignored under `data/raw/`; provenance in `data/raw/urban_affordability_provenance.json`)
- `figures/14_urban_rural_ports.png`, `figures/15_renter_cost_burden.png`
- `results/experiments.csv`, `results/best_config.json`, `results/figures/*.png` — the experiment log, run summary (incl. the fully-nested procedure MAE), and harness figures (see [`results/README.md`](results/README.md))
- `figures/06_model_comparison.png` … `figures/13_need_ranking_equity.png`

The modeling scripts share `src/modeling_common.py` (repo-root paths, the Colorado working definition, the feature/target contract, the leakage guard, and the fixed equity-subgroup definitions). The fixed evaluation protocol — nested ZIP3-grouped CV, the held-out test split, metrics, subgroup scoring — lives in `src/experiment_eval.py` and is shared by both `train_regression.py` and `experiment_harness.py`, so no experiment can change the evaluation rules. The Models & Evaluation write-up is the `#models` section of `index.html` and [`docs/website_models_section.md`](docs/website_models_section.md).

**Milestone 3 — two objectives.** *Primary (planning):* which Colorado communities may benefit most from new public charging — answered by K-Means profiles + the ZCTA need ranking. *Supporting:* how socioeconomic factors relate to *existing* adoption — answered by regression, treated as demand evidence, not the objective.

**Design choices (short):** target `ev_per_1000_households`; predictors exclude `ev_count`/`bev_count`/`phev_count` (leakage); Colorado scoped by 80/81 prefix (working definition, ZIP ≠ ZCTA); no-AFDC-match codes recoded to *verified zero* supply after an audit, not treated as missing; `households ≥ 100` primary cohort with a training-only winsorized sensitivity run; held-out ZIP3 test set reserved first and scored once, then **nested** ZIP3-grouped CV (outer=5, inner=4) with all preprocessing/feature-selection/tuning inside folds. The **fully-nested procedure MAE (26.84)** — which also accounts for candidate selection — is the reported headline; the selected candidate's own nested MAE (26.25) and single-level grouped CV are kept only as labelled comparisons. The two Random Forests are statistically indistinguishable (paired-fold t p = 0.68), so the two-feature model is chosen on parsimony, not a proven edge. Model selection is MAE-first with geographic-consistency and equity-subgroup tie-breakers; the six equity subgroups (income, renter, zero-supply vs. infrastructure) are fixed. The **need ranking** keeps demand (observed, not predicted), access, and equity as separate percentile-normalized components with fixed, un-tuned weights; the equity-weighted ranking shifts the top set toward renter-heavy, lower-income communities. H3, roads, substations, and land-use are documented as future site-level work, not Milestone 3 results.

**Scope:** the analysis counts all vehicle classes (215,105 EVs in the 2026-07-07 snapshot), summing `Vehicle Count`. The light-duty subtotal (202,639) is reported as a diagnostic only.

## Repository structure

```text
.
├── index.html          # project website
├── styles.css
├── script.js
├── requirements.txt    # Python dependencies
├── data/               # acquisition instructions + processed outputs (raw gitignored)
├── notebooks/          # cleaning_and_preprocessing.ipynb (narrative)
├── src/                # fetch_acs.py, prepare_and_eda.py (reproducible pipeline)
├── tests/              # offline cleaning-rule checks
├── figures/            # five EDA charts
├── docs/               # provenance, data dictionary, limitations, website section
├── references/
└── team/               # team photos for the website
```

## Milestone roadmap

- Milestone 0 — Idea approval and initial feasibility
- Milestone 1 — Project framing, literature/context, repository, and website
- Later milestones — Data preparation, exploratory analysis, modeling (only if justified), evaluation, and final communication

