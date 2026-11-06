# Implementation Plan — Reproducible Colorado EV pipeline

Goal: make the repo reproducible end-to-end (venv → deps → `.env` auto-load → `prepare_and_eda.py` regenerates all processed outputs + 5 figures → ACS fetch verified without clobbering the committed CSV → tests pass offline → docs updated), and verify it by actually running it. All paths are absolute under the repo root `/Users/abdullahbharde/CU Boulder/Data Mining/Group Project/Colorado-EV-Charging-Analysis` or repo-root-relative. Do NOT create a git worktree, do NOT commit or push, do NOT print the Census key.

## Decisions (made during exploration, with rationale)

- **Interpreter: `python3` (3.9.6), driven through a repo-local `.venv`.** The machine has no bare `python` on PATH; only `python3`. Creating `.venv` and activating it gives a `python`/`pip` that match the README's documented commands, so the README stays correct as written. `python3 -m venv` is available (verified). `.venv/` is already git-ignored.
- **Dependency pins are Python-3.9-compatible but must be verified, not assumed.** On Python 3.9, `numpy>=2.0,<3` resolves to the 2.0.x line (numpy 2.1 dropped 3.9; 2.0.x keeps it), and pandas 2.2 / matplotlib 3.8–3.9 / requests 2.31+ all support 3.9. The plan installs and then runs the real pipeline to confirm the resolved set works, rather than trusting the pins. (Content rephrased for compliance with licensing restrictions; sources: [numpy 1.26 notes list 3.9–3.12](https://numpy.org/doc/stable/release/1.26.0-notes.html), [numpy 2.0 ecosystem tracking](https://github.com/numpy/numpy/issues/26191).)
- **`.env` auto-load via `python-dotenv` (preferred over documenting manual export).** It is minimal, must-not-break existing `os.environ` usage, and makes `python src/fetch_acs.py` work with zero shell ceremony. Pin `python-dotenv>=1.0,<2`. Add `pytest>=8,<9` as the optional test runner the task permits (tests are stdlib `unittest`, so pytest is additive, not required).
- **ACS fetch is verified to a TEMP output path**, never the committed `data/raw/acs_zcta_2024.csv`, so a successful live fetch cannot overwrite the committed deliverable. If the network/key is unavailable, record that and confirm the pipeline still runs from the present `acs_zcta_2024.csv` (the ACS network fetch is not required for the pipeline).
- **`quality_report.json` must stop recording machine-specific absolute paths.** The current committed `data/processed/quality_report.json` records absolute `source_files` paths (e.g. `/Users/abdullahbharde/...`). The task forbids machine-specific absolute paths in tracked files, so `prepare_and_eda.py` must record repo-root-relative source paths instead. This is a small, surgical change in `main()`.
- **Leave the extra `data/processed/` files alone** (`afdc_zip.csv`, `atlas_zip.csv`, `cleaning_log.json`). They are pre-existing notebook artifacts, not written by `prepare_and_eda.py`, and are not in the task's deliverable list. Do not delete them (not temp verification files) and do not depend on them.
- **Do NOT touch the repo-root `plan.md`.** That file is the Milestone 2 assignment brief, not this plan. This plan lives only at `.agents/tasks/plan.md`.

## Expected headline metrics (baseline to reproduce; from current committed outputs)

Snapshot rows kept 161,053; EVs all classes 215,105; AFDC filtered stations 2,910; ACS source rows 33,772; analysis union codes 1,161; `income_vs_ev` r=0.244 (n=1097); `supply_vs_adoption` r=0.013 (n=280); `renter_vs_ports` r=0.157 (n=283); median EV per 1,000 households 0.61. A correct re-run should reproduce these (small float rounding aside).

---

- [ ] 1. Add missing pinned dependencies to `requirements.txt`.
      Append `python-dotenv>=1.0,<2` (auto-load `.env`) and `pytest>=8,<9` (optional runner the task permits), keeping the existing four pins unchanged. Add a short trailing comment on each new line stating why.
      Files: `requirements.txt`
      Verify: `cat requirements.txt` shows six pins; no range opened on the existing four. (Install is verified in step 2.)

- [ ] 2. Create a fresh repo-local virtual environment and install dependencies.
      Remove any stale `.venv`, then `python3 -m venv .venv`, upgrade pip, and install from `requirements.txt`. This leaves a `.venv/python` that matches the README commands.
      Files: `.venv/` (git-ignored; not tracked)
      Verify: run
      `rm -rf ".venv" && python3 -m venv .venv && ./.venv/bin/python -m pip install --upgrade pip && ./.venv/bin/python -m pip install -r requirements.txt`
      then `./.venv/bin/python -c "import pandas,numpy,matplotlib,requests,dotenv,pytest;print(pandas.__version__,numpy.__version__,matplotlib.__version__)"` — all import without error and prints resolved versions (expect numpy 2.0.x on Python 3.9). If pip cannot resolve a pin on 3.9, record the actual resolution error in `.agents/tasks/plan.md` findings and relax only the offending pin minimally (document why) before continuing.

- [ ] 3. Make `src/fetch_acs.py` auto-load `.env` without breaking existing `os.environ` usage.
      At import time, before reading `os.environ.get('CENSUS_API_KEY')`, call `python-dotenv`'s `load_dotenv()` guarded by `try/except ImportError` (so the module still runs if dotenv is absent) and loading from the repo-root `.env` (`ROOT / '.env'`, `override=False` so a real exported env var still wins). Do not print the key. Keep the `--key` arg and the existing default `os.environ.get('CENSUS_API_KEY')` behavior intact.
      Files: `src/fetch_acs.py`
      Verify: `./.venv/bin/python -c "import ast;ast.parse(open('src/fetch_acs.py').read())"` parses, and `./.venv/bin/python src/fetch_acs.py --help` exits 0 showing `--output`/`--key`. (Live auth is verified in step 6.)

- [ ] 4. Make `src/prepare_and_eda.py` auto-load `.env` and record repo-root-relative source paths.
      (a) Add the same guarded `load_dotenv(ROOT / '.env', override=False)` near the top (the pipeline itself does not need the key, but keeping both scripts consistent means any future env-driven option works the same way; harmless if dotenv missing). (b) In `main()`, change the `log['source_files']` entries so they store paths relative to `ROOT` (e.g. `os.path.relpath(ev_path, ROOT)` or `Path(ev_path).relative_to(ROOT)` with a fallback to the name) instead of the absolute `ev_path`/`st_path`/`acs_path`, so `quality_report.json` contains no machine-specific absolute path. Change nothing else about the cleaning/EDA logic or the five output names.
      Files: `src/prepare_and_eda.py`
      Verify: `./.venv/bin/python -c "import ast;ast.parse(open('src/prepare_and_eda.py').read())"` parses. (Output correctness and the no-absolute-path check are verified in step 5.)

- [ ] 5. Run the full pipeline from a clean state and confirm it regenerates every output with no errors and no FutureWarnings.
      Delete only the five regenerable processed outputs and five figures first (not raw data, not the extra notebook artifacts), then run the pipeline with warnings elevated to errors to catch FutureWarnings.
      Files: regenerates `data/processed/{colorado_ev_zcta_clean.csv,join_audit.csv,outlier_review.csv,quality_report.json,eda_metrics.json}` and `figures/{01_ev_rate_distribution,02_income_vs_ev,03_supply_vs_adoption,04_renter_vs_ports,05_missingness}.png`
      Verify: run
      `rm -f data/processed/colorado_ev_zcta_clean.csv data/processed/join_audit.csv data/processed/outlier_review.csv data/processed/quality_report.json data/processed/eda_metrics.json figures/0?_*.png && ./.venv/bin/python -W error::FutureWarning src/prepare_and_eda.py`
      exits 0 (no traceback, no FutureWarning). Then confirm all five processed files and all five figures exist (`ls data/processed figures`), and confirm headline metrics match the baseline by reading `data/processed/eda_metrics.json` (income_vs_ev r≈0.244, supply_vs_adoption r≈0.013, renter_vs_ports r≈0.157, n_zcta_codes 1161) and `data/processed/quality_report.json` (ev_total_vehicles_all_classes 215105, ev_snapshot_rows_after_validation 161053, afdc_filtered_stations_with_valid_zip 2910, analysis_zip_union_rows 1161). Finally confirm no absolute path leaked: `grep -c "/Users/" data/processed/quality_report.json` returns 0. Record the observed headline metrics in `.agents/tasks/plan.md` findings / the step report.

- [ ] 6. Verify `fetch_acs.py` authenticates with the key loaded from `.env`, writing to a TEMP path, without printing the key.
      Run the fetch with `--output` pointed at a throwaway temp file so the committed `data/raw/acs_zcta_2024.csv` is never touched. The key must come from `.env` auto-load (do not pass `--key`, do not echo the key).
      Files: temp file only (e.g. `$TMPDIR/acs_verify.csv`); no tracked files change
      Verify: run
      `TMP_ACS="$(mktemp -t acs_verify).csv" && ./.venv/bin/python src/fetch_acs.py --output "$TMP_ACS" && head -1 "$TMP_ACS" | grep -q "zip code tabulation area" && wc -l "$TMP_ACS" && rm -f "$TMP_ACS"`
      — on success the temp file has a header row containing `zip code tabulation area` and many data rows, and the committed `data/raw/acs_zcta_2024.csv` is unchanged (confirm its size/mtime did not change). If the network or key is unavailable, DO NOT treat it as failure: record "ACS live fetch skipped — network/key unavailable" in the step report and re-confirm step 5 already ran the pipeline successfully from the present `data/raw/acs_zcta_2024.csv`. Never print the key.

- [ ] 7. Run the test suite offline and confirm all tests pass.
      The tests are stdlib `unittest`; also exercise the pytest path since it is now installed.
      Files: none changed (reads `tests/test_pipeline.py`)
      Verify: `./.venv/bin/python -m unittest discover -s tests -v` reports OK (all pass), and `./.venv/bin/python -m pytest -q tests` also passes. Both run fully offline.

- [ ] 8. Update `README.md` "Reproduce the Milestone 2 analysis" section for the real workflow.
      Reflect: `python-dotenv` auto-loads `.env` (copy `.env.example` to `.env`, fill `CENSUS_API_KEY`, then just `python src/fetch_acs.py` — no manual `export` needed, though `export`/PowerShell `$env:` still works and takes precedence); note the ACS step is optional because `data/raw/acs_zcta_2024.csv` may already be present and the pipeline runs without a live fetch; keep both `python -m unittest discover -s tests` and add `pytest` as an alternative; keep all paths repo-root-relative with the real raw filenames. Do not reintroduce machine-specific absolute paths.
      Files: `README.md`
      Verify: `grep -n "python-dotenv\|\.env\|fetch_acs\|prepare_and_eda\|unittest\|pytest" README.md` shows the updated instructions, and `grep -c "/Users/" README.md` returns 0.

- [ ] 9. Update `data/README.md` acquisition instructions to match the dotenv workflow and real filenames.
      In the `acs_zcta_2024.csv` row and the surrounding text, replace the bare `export CENSUS_API_KEY=...; python src/fetch_acs.py` instruction with the `.env` copy-and-fill workflow (export still supported), and confirm the three raw filenames/globs shown match the actual files: `CO_EV_Registrations_10_2026.csv` (glob `CO_EV_Registrations_*.csv`), `alt_fuel_stations (Oct 15 2026).csv` (glob `alt_fuel_stations*.csv`), `acs_zcta_2024.csv`. Keep repo-root-relative paths.
      Files: `data/README.md`
      Verify: `grep -n "\.env\|CENSUS_API_KEY\|CO_EV_Registrations\|alt_fuel_stations\|acs_zcta_2024" data/README.md` shows consistent, correct names, and `grep -c "/Users/" data/README.md` returns 0.

- [ ] 10. Clean up temporary verification artifacts without deleting committed deliverables.
      Remove the temp ACS file (already deleted in step 6 if that path ran), any `__pycache__`, `.pytest_cache`, and `tests/__pycache__` created during verification. Do NOT delete `.venv` (it is the reproducible environment), the five `data/processed/*` deliverables, the five `figures/*.png` deliverables, raw data, or the pre-existing `afdc_zip.csv`/`atlas_zip.csv`/`cleaning_log.json`.
      Files: deletes only caches/temp
      Verify: `find . -path ./.venv -prune -o -name "__pycache__" -print` and `ls -d .pytest_cache 2>/dev/null` show nothing left; then re-confirm deliverables still present: `ls data/processed/colorado_ev_zcta_clean.csv data/processed/eda_metrics.json figures/01_ev_rate_distribution.png figures/05_missingness.png` all exist. `git status --short` shows no deletion of tracked deliverables.

- [ ] 11. Final reproducibility sanity check and no-leak confirmation.
      Confirm the whole chain is coherent: dependencies install, pipeline regenerates outputs, tests pass, docs are consistent, and no tracked file carries a machine-specific absolute path.
      Files: none changed
      Verify: `grep -rn "/Users/abdullahbharde" README.md data/README.md data/processed/quality_report.json src/*.py requirements.txt` returns no matches; `git status --short` shows only intended edits (requirements.txt, src/*.py, README.md, data/README.md, regenerated processed/figures) and `.env` is NOT listed as a new tracked file. Summarize the reproduced headline metrics in the step report.

## Gaps / assumptions

- The ACS live fetch (step 6) depends on network availability and a valid key in `.env`. The plan assumes it may be unavailable in the execution sandbox and explicitly treats a skip as acceptable, because the task states the ACS network fetch is not required for the pipeline to run (the committed `acs_zcta_2024.csv` is present).
- If pip cannot resolve the current pins on Python 3.9 (contrary to the compatibility check), step 2 relaxes only the offending pin minimally and documents why; the intent is a working, reproducible install, not preserving a broken pin.
- The extra `data/processed/` files (`afdc_zip.csv`, `atlas_zip.csv`, `cleaning_log.json`) are assumed to be intentional pre-existing artifacts and are left untouched.
