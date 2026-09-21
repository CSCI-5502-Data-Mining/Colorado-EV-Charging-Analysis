# Colorado EV Charging Access & Adoption

CSCI 5502 Data Mining semester group project.

**Team:** Eddie Strand, Snehal Bhatnagar, Abdullah Bharde

**Repository:** [https://github.com/eddies410/Colorado-EV-Charging-Analysis](https://github.com/CSCI-5502-Data-Mining/Colorado-EV-Charging-Analysis)

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
- U.S. Census Bureau ACS 5-year estimates — added after Milestone 0 for income/housing context; exact tables still TBD — [ACS 5-year](https://www.census.gov/data/developers/data-sets/acs-5year.html)

Acquisition instructions: [`data/README.md`](data/README.md). Do not commit secrets, credentials, or restricted raw extracts.

## Repository structure

```text
.
├── index.html          # project website
├── styles.css
├── script.js
├── data/               # instructions + optional non-sensitive sample
├── notebooks/
├── src/
├── figures/
├── docs/
├── references/
└── team/               # team photos for the website
```

## Milestone roadmap

- Milestone 0 — Idea approval and initial feasibility
- Milestone 1 — Project framing, literature/context, repository, and website
- Later milestones — Data preparation, exploratory analysis, modeling (only if justified), evaluation, and final communication

