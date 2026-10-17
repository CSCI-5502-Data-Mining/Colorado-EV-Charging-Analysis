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

## Acquisition for Milestone 2

Raw files go in `data/raw/` (gitignored). Place them with these names (the pipeline and the notebook pick them up by glob):

| File | Source | How to obtain |
| --- | --- | --- |
| `CO_EV_Registrations_*.csv` | Atlas EV Hub / Colorado Energy Office | Download the **Colorado** CSV from the [state EV registration data page](https://www.atlasevhub.com/market-data/state-ev-registration-data/). Keep the full file with all historical snapshots so the latest-snapshot flag can be applied reproducibly. |
| `alt_fuel_stations*.csv` | U.S. DOE AFDC | Download from [afdc.energy.gov/data_download](https://afdc.energy.gov/data_download) with filters **Electric / Public / Existing / Colorado**. |
| `acs_zcta_2024.csv` | U.S. Census Bureau | Generate with the script: `export CENSUS_API_KEY=...` then `python src/fetch_acs.py`. Pulls **national** ZCTAs (not Colorado-only) so the ZIP/ZCTA code match can be audited. Free key: https://api.census.gov/data/key_signup.html |

### Acquisition checklist (Milestone 2)

- [x] Record the exact snapshot date for Atlas Colorado EV registrations — **latest DMV snapshot 2026-07-07** (file stacks monthly snapshots back to 2010; only the latest flag is kept).
- [x] Filter AFDC stations to Colorado and document charger-type / access-type fields — **State = CO, Fuel Type Code = ELEC, Access Code = PUBLIC, Status Code = E (existing)**, deduplicated by station `ID`.
- [x] Choose a common geography and document the join rules — **5-digit USPS ZIP matched to Census ZCTA code (proxy, not point-in-polygon)**; EV and station ZIPs form the union key, ACS joined left, unmatched codes preserved, absent rows not treated as zeros. See `docs/limitations.md`.
- [ ] Decide whether EValuateCO can be exported at station level for this project.
- [ ] Compare Colorado Information Marketplace records to AFDC and note overlap.
- [ ] If a dataset cannot be redistributed, keep only a small non-sensitive sample in `data/sample/`.

### Scope decision

The analysis counts **all vehicle classes** (215,105 EVs in the latest snapshot), summing `Vehicle Count` rather than counting rows. The light-duty subtotal (202,639) is reported in the notebook as a diagnostic only.

### Vehicle totals (latest snapshot, 2026-07-07)

| Metric | Value |
| --- | ---: |
| Atlas raw rows (all snapshots) | 5,970,397 |
| Latest-snapshot rows kept | 161,053 |
| Vehicles, all classes | 215,105 |
| AFDC open + public + electric CO stations | 2,910 |
| ACS national ZCTA rows | 33,772 |
| Analysis codes after join (union) | 1,161 |

## Sample folder

`data/sample/` is reserved for a tiny, non-sensitive excerpt **after** license and privacy review. Empty until that review is complete.
