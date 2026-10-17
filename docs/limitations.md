# Quality, ethics, and methodological limitations

1. **Observed registrations are a residential location proxy**, not charger use or traffic flows. Low current EV ownership may indicate barriers to adoption instead of a lack of future demand.
2. **ZIP ≠ ZCTA.** Atlas registration ZIP, AFDC station mailing ZIP, and ACS Census ZCTA represent different geographic entities. The first exploratory join on matching strings is imperfect and does not measure exact geometric containment.
3. **Mixed time periods.** 2024 ACS 5-year data covers 2020-2024, while EV registrations and stations can be later point-in-time snapshots. Correlations are descriptive, not causal.
4. **Missingness != no infrastructure.** If a ZIP is not present in the station file, it should not be reported as zero public charging until complete geographic coverage has been confirmed.
5. **Port counts differ in usability.** Level 2 and DC fast ports serve different needs. Ports are not guaranteed operational or available at all times; station status is only a source snapshot.
6. **Urban-rural comparison requires formal classification**, not arbitrary ZIP prefixes or population thresholds. A spatial intersection with Census urban areas (or a documented official definition) is the next step.
7. **H3 at resolution 10 is not observed EV count.** Allocating a ZIP-level count into hexagons produces estimated spatial demand and must be labeled as a model. Geolocated household and building data can be more useful than uniform allocations.
8. **Substation locations are screening evidence**, not evidence of available transformer/feeder capacity; later siting work may need utility engagement and capacity data.
9. **Sensitive data and licensing.** Share aggregates and permitted source data only. Avoid VINs and addresses; cite Atlas reuse terms and AFDC terms when distributing files.
10. **Chart denominators and exclusions.** Do not compare raw EV counts across areas of very different size; show per-household rates, nonmatching geography counts, and samples used in scatterplots.
