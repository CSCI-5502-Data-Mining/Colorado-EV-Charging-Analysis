# Milestone 2 - Data & Data Exploration

## Data sources and provenance

We collected Colorado EV registration data from Atlas/Colorado Energy Office (snapshot: **October 2026 DMV snapshot**, accessed: **2026-10-15**), public charging-station data from the DOE AFDC (snapshot: **2026-10-15 download**, accessed: **2026-10-15**), and ZCTA demographics from the U.S. Census Bureau's 2024 ACS 5-year release (accessed: **2026-10-15**). A provenance table records the original data sources, licensing or terms, geography, unit of observation, and download details.

## Dataset and cleaning decisions

The main table is joined by matching five-digit ZIP/ZCTA codes, with **one row per matched geographic code**. Since USPS ZIPs and Census ZCTAs are not identical, this is an exploratory approximation. We used Atlas's latest DMV snapshot flag and summed `Vehicle Count` rather than counting registration rows. We retained audit files for unmatched geographic codes, invalid IDs, missing fields, excluded station statuses, and reported port counts.

**Insert observed before/after audit:**

| Step | Before | After | Reason |
|---|---:|---:|---|
| Atlas rows | 5,970,397 | 161,053 | The file stacks ~90 complete monthly DMV snapshots back to 2010, so the same vehicles repeat once per snapshot; summing them would massively double-count. Keeping only the latest snapshot (2026-07-07) removes that 5.8M-row redundancy, not bad data (0 invalid ZIP, 0 invalid/negative count). The 161,053 rows are ZIP × make/model/year/drivetrain/class groups whose Vehicle Count sums to 215,105 vehicles across 1,150 ZIP codes. |
| AFDC rows | 3,169 | 2,910 | Public, existing, electric, Colorado, unique ID; filtered to valid 5-digit ZIP (0 duplicate IDs, 0 invalid ZIPs) |
| Geographic join | 1,150 EV + 2,910 station ZIPs | 1,161 union codes | 5-digit ZIP matched to ZCTA; 20 codes unmatched to ACS (12 EV ZIPs, 11 station ZIPs without ACS match), all preserved, none dropped |

## Five EDA charts and interpretations

1. **EV adoption distribution.** [Insert `figures/01_ev_rate_distribution.png`] Observed n: **1,129 ZCTAs**. Finding: the distribution is strongly right-skewed. Median adoption is **0.61 registered EVs per 1,000 households** while the mean is roughly 30 per 1,000, so a small number of high-adoption ZIP codes pull the average far above the typical area; most ZIP codes have only one or two registered EVs. Caveat: areas with missing households are excluded.
2. **Income vs EV adoption.** [Insert `figures/02_income_vs_ev.png`] Observed n: **1,097 ZCTAs**. Finding: a weak positive association between median household income and EVs per 1,000 households (**Pearson r = 0.24**); higher-income ZIP codes tend to show somewhat higher adoption, but income explains only a small share of the variation. Caveat: association is not causal.
3. **Supply vs adoption.** [Insert `figures/03_supply_vs_adoption.png`] Observed n: **280 ZCTAs**. Finding: essentially no linear relationship between public charging ports per 1,000 households and EV adoption per 1,000 households (**Pearson r = 0.01**). Caveat: port counts do not measure actual usage, and only ~25% of codes have an AFDC station match.
4. **Renter share vs supply.** [Insert `figures/04_renter_vs_ports.png`] Observed n: **283 ZCTAs**. Finding: a weak positive association between renter share and public charging ports per 1,000 households (**Pearson r = 0.16**); areas with more renters show slightly more public port supply, consistent with public charging partially substituting for home charging. Caveat: renter share is a community aggregate.
5. **Join and missingness quality.** [Insert `figures/05_missingness.png`] Finding: EV and demographic coverage is high (EV count missing for ~1% of codes; households/population ~2%), but charging-supply coverage is sparse — **about 75% of analysis ZIP codes have no matched AFDC station** (station_count, port fields, and derived ports-per-household all ~75% missing). Caveat: unmatched ZIPs and absent stations are not demonstrated true zeros and need independent interpretation.

## Updated research questions

RQ1: How is household income related to observed ZIP-level EV registrations per household? [Keep/revise after results]

RQ2: How is renter share related to observed charging-port availability per household? [Keep/revise after results]

RQ3: How does charger availability differ between formally classified urban and rural Colorado areas? [Pending classification data; do not answer prematurely]

RQ4: Which Colorado areas show a potential mismatch between registrations and public charging supply, and what additional spatial data would support H3 site selection? [Keep/revise after results]

## Limitations and ethics

Report temporal mismatch, missingness, ZIP/ZCTA mismatch, charging-station status limitations, source licenses, and the difference between actual EV registrations and estimated H3-level potential demand. Do not publish VINs, residential addresses, or private utility capacity data.
