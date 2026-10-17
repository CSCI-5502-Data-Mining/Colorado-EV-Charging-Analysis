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

