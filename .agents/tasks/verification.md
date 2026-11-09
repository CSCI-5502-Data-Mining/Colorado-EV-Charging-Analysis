# Verification — Reproducible Colorado EV pipeline

All commands were run from the repo root through the repo-local `.venv`
(`./.venv/bin/python`), Python 3.9.6 on macOS. The `CENSUS_API_KEY` was never
printed, echoed, or written anywhere; it is referenced only by name.

## 1. Environment and dependencies

- Recreated `.venv` fresh: `python3 -m venv .venv`, upgraded pip, then
  `pip install -r requirements.txt` — exit 0.
- Resolved versions (Python 3.9.6): pandas 2.3.3, numpy 2.0.2, matplotlib 3.9.4,
  requests 2.32.5, python-dotenv 1.2.1, pytest 8.4.2. All import cleanly.
- `requirements.txt` additions (pinned to match existing style):
  - `python-dotenv>=1.0,<2` — auto-load `CENSUS_API_KEY` from `.env`.
  - `pytest>=8,<9` — optional offline test runner (tests also run under unittest).
  - Existing four pins unchanged.
- `.venv/` is git-ignored (`git check-ignore .venv` → `.venv`). `.env` is also
  git-ignored and NOT tracked (`git ls-files .env` → empty).

## 2. `.env` auto-load

- `src/fetch_acs.py` and `src/prepare_and_eda.py` both call
  `load_dotenv(ROOT / '.env', override=False)` at import time, guarded by
  `try/except ImportError`. `override=False` keeps a real exported env var
  authoritative. Existing `os.environ.get('CENSUS_API_KEY')` / `--key` behavior
  is intact (default is read inside `main()`, after import-time load).
- `python src/fetch_acs.py --help` exits 0 and shows `--output`/`--key`.

## 3. Full pipeline run from a clean state

Deleted the 5 regenerable processed outputs and 5 figures, then ran with
FutureWarnings elevated to errors:

```
./.venv/bin/python -W error::FutureWarning src/prepare_and_eda.py
```

Result: exit 0, **no errors, no FutureWarnings**, ~6.9s wall. Printed
"Prepared 1161 5-digit matched-code records."

All outputs regenerated:
- `data/processed/`: colorado_ev_zcta_clean.csv, join_audit.csv,
  outlier_review.csv, quality_report.json, eda_metrics.json
- `figures/`: 01_ev_rate_distribution.png, 02_income_vs_ev.png,
  03_supply_vs_adoption.png, 04_renter_vs_ports.png, 05_missingness.png

### Headline metrics (reproduced, match baseline)

From `eda_metrics.json`:
- n_zcta_codes: 1161
- median_ev_per_1000_households: 0.61
- income_vs_ev: r = 0.244, n = 1097
- supply_vs_adoption: r = 0.013, n = 280
- renter_vs_ports: r = 0.157, n = 283

From `quality_report.json`:
- ev_source_rows: 5,970,397
- ev_snapshot_rows_after_validation: 161,053
- ev_total_vehicles_all_classes: 215,105
- afdc_filtered_stations_with_valid_zip: 2,910
- acs_source_rows: 33,772
- analysis_zip_union_rows: 1,161

### No absolute-path leak

`quality_report.json` `source_files` now records repo-root-relative paths:
`data/raw/CO_EV_Registrations_10_2026.csv`,
`data/raw/alt_fuel_stations (Oct 15 2026).csv`, `data/raw/acs_zcta_2024.csv`.
`grep -c "/Users/" data/processed/quality_report.json` → 0.

## 4. ACS fetch path (live, to a temp output)

Recorded the committed CSV's SHA-256 and size, ran the fetch to a temp path
with the key auto-loaded from `.env` (no `--key`, key not echoed), then deleted
the temp file:

```
./.venv/bin/python src/fetch_acs.py --output .agents/tasks/_acs_tmp.csv
```

Result: exit 0. **Live authentication succeeded** — wrote 33,772 national ZCTA
rows (33,773 lines incl. header). Temp header:
`NAME,B01003_001E,B19013_001E,B25003_001E,B25003_003E,zip code tabulation area`.

Committed `data/raw/acs_zcta_2024.csv` was **unchanged**: identical size
(1,310,006 bytes), identical mtime, identical SHA-256
(`18eba0ed274f72d29e92172ccccf55b5b966d21c8cdac821fc3c27c806068a7a`) before and
after. Temp file deleted. The key value never appeared in stdout.

(The only stdout noise was an unrelated `NotOpenSSLWarning` from urllib3 about
macOS LibreSSL — not from pipeline code, not a FutureWarning.)

## 5. Test suite (offline)

- `./.venv/bin/python -m unittest discover -s tests -v` → OK, 1 test passed.
- `./.venv/bin/python -m pytest -q tests` → 1 passed. (14 warnings are
  `PyparsingDeprecationWarning` emitted from inside matplotlib's own modules —
  third-party library code, not our pipeline, not FutureWarnings.)

Both runs are fully offline.

## 6. Docs

- `README.md` "Reproduce" section rewritten as a one-command-at-a-time path:
  create venv → install → copy `.env.example` to `.env` and set the key →
  optional `python src/fetch_acs.py` (dotenv auto-load, export still works and
  wins) → `python src/prepare_and_eda.py` → where to view the 5 figures and
  processed outputs → how to run tests (unittest or pytest).
- `data/README.md` ACS acquisition row updated to the `.env` copy-and-fill
  workflow (export still supported). Raw filenames/globs confirmed to match the
  real files: `CO_EV_Registrations_*.csv`, `alt_fuel_stations*.csv`,
  `acs_zcta_2024.csv`.
- No machine-specific absolute paths in tracked files:
  `grep -rn "/Users/abdullahbharde" README.md data/README.md quality_report.json
  eda_metrics.json src/*.py requirements.txt` → no matches.

## 7. Cleanup

Removed `src/__pycache__`, `tests/__pycache__`, `.pytest_cache`, and the temp
ACS output. `.venv` and all committed `data/processed/*` and `figures/*`
deliverables were left in place. No tracked deliverable was deleted.

## 8. Final `git status --short`

```
 M README.md
 M data/README.md
 M data/processed/quality_report.json
 M figures/01_ev_rate_distribution.png
 M figures/02_income_vs_ev.png
 M figures/03_supply_vs_adoption.png
 M figures/04_renter_vs_ports.png
 M figures/05_missingness.png
 M requirements.txt
 M src/fetch_acs.py
 M src/prepare_and_eda.py
?? .agents/
?? plan.md
```

Only intended edits appear. The processed `.csv`/`.json` other than
`quality_report.json` do not show because `data/*.csv` is git-ignored by design
(only `quality_report.json` is tracked). `.env` is not tracked. The root
`plan.md` is the pre-existing Milestone 2 brief (not modified). Nothing was
committed or pushed — all changes are left staged/unstaged for the user.
