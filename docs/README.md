# Docs

Place milestone notes, meeting logs, and longer write-ups here.

- Milestone 0 proposal: problem, goal, data sources, and constraints (approved idea).
- Milestone 1: project framing on the website (`index.html`) plus this repository setup.

## Milestone 2 artifacts

- [`provenance.csv`](provenance.csv) — source, provider, URL, access date, snapshot, geography, time coverage, raw row/column counts, original unit, license/terms, and limitations for all three data sources (Atlas EV, AFDC, Census ACS).
- [`data_dictionary.csv`](data_dictionary.csv) — every column in `data/processed/colorado_ev_zcta_clean.csv`: name, type, units/categories, meaning, source, and notes.
- [`limitations.md`](limitations.md) — quality, bias, ethics, and methodological limitations (ZIP≠ZCTA proxy join, temporal mismatch, missingness is not zero, urban/rural needs formal classification, etc.).
- [`website_section_template.md`](website_section_template.md) — ready-to-publish Data / Data Exploration section with the before/after cleaning audit table and the five chart findings, for the project website.

The reproducible pipeline that generates the processed dataset and the five figures lives in `src/` (`fetch_acs.py`, `prepare_and_eda.py`); the narrative walkthrough is in `notebooks/cleaning_and_preprocessing.ipynb`. See the repository root `README.md` for how to reproduce.
