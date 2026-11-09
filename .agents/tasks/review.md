# Reproducible Colorado EV pipeline: `.env` auto-load, pinned deps, path hygiene

The change makes the Colorado EV charging repo reproducible end to end: it pins the two missing dependencies (`python-dotenv`, `pytest`), wires both `src` scripts to auto-load `CENSUS_API_KEY` from a git-ignored `.env` without disturbing existing `os.environ`/`--key` behavior, strips machine-specific absolute paths out of the tracked `quality_report.json`, and rewrites `README.md` / `data/README.md` into a one-command-at-a-time reproduction path. The coder recreated a clean `.venv`, ran the full pipeline with FutureWarnings elevated to errors, verified a live ACS fetch to a throwaway temp path, and ran the test suite offline — all recorded in the verification note. The surface area is small and surgical; no cleaning/EDA logic changed.

Watch for: nothing blocking. The secret-handling, path-hygiene, and no-deletion criteria all hold under spot-check (confirmed). The verification note's live-ACS and full-pipeline runs are taken from the coder's evidence per the task constraint, not re-run (confirmed not re-run by design).

**Verdict**: APPROVED

## High-level view

Dependencies are pinned in the project's existing `>=,<` style — `python-dotenv>=1.0,<2` and `pytest>=8,<9` appended, the original four pins untouched. The note records a clean resolve on Python 3.9.6, so the pins are not just plausible but verified to install.

The `.env` auto-load is implemented identically in both scripts: a `try/except ImportError` guard around `load_dotenv(ROOT / '.env', override=False)` at import time. `override=False` keeps an exported env var authoritative, and the key is still read inside `main()` via the pre-existing `os.environ.get`/`--key` path, so nothing about the old interface breaks and the module still runs if dotenv is absent.

Secret hygiene is clean. `.env` is git-ignored and untracked; `.env.example` carries only a `your_census_api_key_here` placeholder and is trackable. The actual key value appears in no edited file, doc, log, or the verification note — the only long alphanumeric token flagged in a scan was the word "PyparsingDeprecationWarning".

Path hygiene is fixed at the source: `prepare_and_eda.py` now routes the three `source_files` entries through a `_rel_to_root` helper, so the regenerated `quality_report.json` records `data/raw/...` relative paths instead of the previous `/Users/abdullahbharde/...` absolutes. No tracked file carries a machine-specific absolute path.

Deliverables are intact. All five figures and five processed outputs are present and show as modifications, not deletions; `data/raw/` is untracked; the committed `acs_zcta_2024.csv` was left byte-identical (same SHA-256) because the live fetch went to a temp path. Nothing was committed.

<details>
<summary>Issues (0)</summary>

No blocking or non-blocking action items. All success criteria are satisfied.

</details>

<details>
<summary>Details</summary>

### Dependencies pinned without regression

`requirements.txt` gains exactly two lines in the existing `>=,<` convention, each with a trailing rationale comment:

```
python-dotenv>=1.0,<2  # auto-load CENSUS_API_KEY from .env in src scripts
pytest>=8,<9  # optional offline test runner (tests also run under unittest)
```

The original `pandas>=2.2,<3`, `numpy>=2.0,<3`, `matplotlib>=3.8,<4`, `requests>=2.31,<3` are unchanged (confirmed in diff). The note records a fresh `.venv` install resolving to pandas 2.3.3 / numpy 2.0.2 / matplotlib 3.9.4 / requests 2.32.5 / python-dotenv 1.2.1 / pytest 8.4.2 on Python 3.9.6 — so the pins resolve on the stated interpreter, not just in theory (confirmed via diff; install result taken from note).

### `.env` auto-load leaves the old interface intact

Both scripts get the same import-time block:

```python
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
except ImportError:
    pass
```

`override=False` means a real exported `CENSUS_API_KEY` still wins, matching what both READMEs now promise. The key continues to be read in `main()` through the existing `os.environ.get`/`--key` default, so dropping dotenv does not change the resolution order and an absent dotenv install does not crash the module. `prepare_and_eda.py` does not itself need the key; adding the same block there is a consistency choice, not a functional dependency (confirmed in diff).

### `quality_report.json` no longer leaks the author's home path

The pre-existing committed report recorded absolute `source_files` paths under `/Users/abdullahbharde/...`. The fix adds a small `_rel_to_root` helper and routes the three entries through it:

```python
def _rel_to_root(path):
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).name
```

The regenerated report now shows `data/raw/CO_EV_Registrations_10_2026.csv`, `data/raw/alt_fuel_stations (Oct 15 2026).csv`, `data/raw/acs_zcta_2024.csv` (confirmed in diff). A workspace scan for `/Users/abdullahbharde` across `README.md`, `data/README.md`, `quality_report.json`, `src/*.py`, and `requirements.txt` returned no matches (confirmed). The `ValueError` fallback to basename covers a raw file supplied from outside the repo root.

### ACS fetch verified without clobbering the committed CSV

The note records the committed `data/raw/acs_zcta_2024.csv` SHA-256 (`18eba0ed…68a7a`), size (1,310,006 bytes), and mtime before and after a live fetch that was pointed at a temp path via `--output`, with the key auto-loaded from `.env` (no `--key`, not echoed). The fetch wrote 33,772 national ZCTA rows and the committed CSV was byte-identical afterward (same SHA). `data/raw/` is untracked (confirmed via `git ls-files`), so even a stray write there could not become a tracked change. Live-auth result is taken from the note per the no-re-run constraint.

### Deliverables preserved, nothing committed

`git diff HEAD --diff-filter=D` returns no deletions (confirmed). All five figures (`01`–`05`) and the five processed outputs (`colorado_ev_zcta_clean.csv`, `join_audit.csv`, `outlier_review.csv`, `quality_report.json`, `eda_metrics.json`) are present and appear as modifications. The pre-existing extra artifacts (`afdc_zip.csv`, `atlas_zip.csv`, `cleaning_log.json`, `co-ev-charging-map.png`) were left untouched, matching the plan's decision to leave notebook artifacts alone. `git status` shows only the intended edits plus the untracked `.agents/` and the pre-existing root `plan.md`; nothing is staged or committed (confirmed).

### Headline metrics reproduced

The note's regenerated `eda_metrics.json` / `quality_report.json` match the plan's baseline: n_zcta_codes 1161, median EV/1000 households 0.61, income_vs_ev r=0.244 (n=1097), supply_vs_adoption r=0.013 (n=280), renter_vs_ports r=0.157 (n=283), EV total 215,105, snapshot rows 161,053, AFDC stations 2,910, ACS source rows 33,772. Taken from the note; the full pipeline was not re-run per the task constraint.

### Tests pass offline

The note records both `unittest discover` (OK, 1 test) and `pytest -q` (1 passed) running fully offline; the 14 pytest warnings are third-party `PyparsingDeprecationWarning` from inside matplotlib, not pipeline FutureWarnings. Taken from the note.

</details>

<details>
<summary>File map</summary>

- `requirements.txt` — added `python-dotenv>=1.0,<2` and `pytest>=8,<9`; original four pins unchanged.
- `src/fetch_acs.py` — guarded import-time `load_dotenv(ROOT/'.env', override=False)`; `--key`/`os.environ` path intact.
- `src/prepare_and_eda.py` — same `.env` auto-load; new `_rel_to_root` helper; `source_files` now repo-root-relative.
- `data/processed/quality_report.json` — regenerated; `source_files` now relative paths, no `/Users/` absolutes.
- `figures/01`–`05*.png` — regenerated by the pipeline run (modified, not deleted).
- `README.md` — reproduce section rewritten: venv → install → copy `.env.example` → optional fetch → pipeline → view outputs → tests.
- `data/README.md` — ACS acquisition row updated to the `.env` workflow; real raw filenames confirmed.
- `.env.example` — placeholder key template (trackable); `.env` itself git-ignored and untracked.

Full diff: `git diff HEAD` from the repo root.

</details>
