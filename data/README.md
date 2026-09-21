# Data

Raw datasets are **not** stored in this repository. Several sources are public, but files can be large, licensed, or updated frequently. Do not commit API keys, credentials, or restricted extracts.

## Planned sources (from Milestone 0)

| Dataset | Provider | Access | Geographic grain | Notes |
| --- | --- | --- | --- | --- |
| State EV registration data | Atlas EV Hub / Colorado Energy Office | Public CSV, ODbL | ZIP code | Registration ZIP is a proxy for where vehicles are driven or charged. Filter historical snapshots. [Download page](https://www.atlasevhub.com/market-data/state-ev-registration-data/) |
| Alternative Fueling Station Locator | U.S. DOE AFDC | Public download / API | Station coordinates; ZIP | Update cadence varies; listed stations may not be operational; no utilization. [Data download](https://afdc.energy.gov/data_download) |
| EValuateCO charging usage | Atlas Public Policy / Colorado Energy Office | Public dashboard; some exports may require request | Selected funded stations | Covers a subset of Colorado stations (selection bias). Use as supplementary validation. [EValuateCO](https://atlaspolicy.com/evaluateco/) |
| Map of alternative fuels and EV charging | Colorado Information Marketplace | Public open-data portal | Station-level | May overlap AFDC; check duplicates. [Portal](https://data.colorado.gov/Energy/Map-of-Alternative-Fuels-and-Electric-Vehicle-Char/fwyp-a5y7) |

## Added after Milestone 0 (to be confirmed in analysis)

| Dataset | Provider | Why added | Status |
| --- | --- | --- | --- |
| American Community Survey 5-year estimates | U.S. Census Bureau | Milestone 0 questions require income and housing context that were not in the original data table | Acquisition method and exact tables still to be documented. [ACS 5-year](https://www.census.gov/data/developers/data-sets/acs-5year.html) |

## Acquisition checklist (next iteration)

- [ ] Record the exact snapshot date for Atlas Colorado EV registrations.
- [ ] Filter AFDC stations to Colorado and document charger-type / access-type fields used.
- [ ] Decide whether EValuateCO can be exported at station level for this project.
- [ ] Compare Colorado Information Marketplace records to AFDC and note overlap.
- [ ] Choose a common geography (ZIP vs ZCTA vs county) and document the join rules.
- [ ] If a dataset cannot be redistributed, keep only a small non-sensitive sample in `data/sample/`.

## Sample folder

`data/sample/` is reserved for a tiny, non-sensitive excerpt **after** license and privacy review. Empty until that review is complete.
