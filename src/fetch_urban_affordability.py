"""Fetch the two Census tables for the urban/rural + affordability analysis.

Two separate official Census APIs, both at ZCTA geography (national, so the
ZIP<->ZCTA match can be audited exactly like src/fetch_acs.py):

  1. 2020 Decennial DHC table P2 -- urban/rural population counts:
       P2_001N total, P2_002N urban, P2_003N rural.
     Used to classify each ZCTA as urban-dominant / rural-dominant / mixed by
     its official Census-2020 urban population share. (ZCTA-level: a ZCTA can
     itself straddle the urban boundary, which the "mixed" class makes visible.)

  2. 2024 ACS 5-year table B25070 -- gross rent as a share of household income
     (renter housing-cost burden):
       B25070_001N total renters paying rent; the >=30% buckets are
       B25070_007E (30-34.9), _008E (35-39.9), _009E (40-49.9), _010E (>=50);
       B25070_011E is "not computed" and is EXCLUDED from the denominator.

The API key is loaded from .env (CENSUS_API_KEY) and never printed, logged, or
committed. If a fetch fails the script raises -- it does NOT write placeholder
data. Raw outputs go to data/raw/ (gitignored); provenance is recorded.

Usage (from the repository root):

    python src/fetch_urban_affordability.py
"""
from __future__ import annotations
import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
except ImportError:
    pass

DHC_URL = 'https://api.census.gov/data/2020/dec/dhc'
DHC_VARS = ['P2_001N', 'P2_002N', 'P2_003N']
ACS_URL = 'https://api.census.gov/data/2024/acs/acs5'
# Renter cost-burden buckets: total + the four >=30% buckets + not-computed.
ACS_VARS = ['B25070_001E', 'B25070_007E', 'B25070_008E', 'B25070_009E',
            'B25070_010E', 'B25070_011E']


def _fetch(url, variables, key):
    """GET a Census table for all ZCTAs; return rows (header + data) or raise."""
    resp = requests.get(url, params={'get': ','.join(variables),
                                     'for': 'zip code tabulation area:*',
                                     'key': key}, timeout=180)
    resp.raise_for_status()
    rows = resp.json()
    if not rows or len(rows) < 2 or 'zip code tabulation area' not in rows[0]:
        raise RuntimeError(f'Unexpected Census response from {url}')
    return rows


def _write(rows, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerows(rows)
    return len(rows) - 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dhc-out', default=str(RAW / 'dhc_p2_zcta_2020.csv'))
    p.add_argument('--acs-out', default=str(RAW / 'acs_b25070_zcta_2024.csv'))
    p.add_argument('--provenance', default=str(RAW / 'urban_affordability_provenance.json'))
    p.add_argument('--key', default=os.environ.get('CENSUS_API_KEY'))
    args = p.parse_args()
    if not args.key:
        raise SystemExit('CENSUS_API_KEY required (free: '
                         'https://api.census.gov/data/key_signup.html). '
                         'Set it in .env; it is never printed or committed.')

    prov = {'retrieved_utc': datetime.now(timezone.utc).isoformat(), 'sources': {}}

    dhc_rows = _fetch(DHC_URL, DHC_VARS, args.key)
    n_dhc = _write(dhc_rows, Path(args.dhc_out))
    prov['sources']['dhc_p2_2020'] = {
        'table': 'P2 (urban/rural population)', 'vintage': '2020 Decennial DHC',
        'api': DHC_URL, 'geography': 'ZCTA (national)', 'variables': DHC_VARS,
        'rows': n_dhc, 'output': str(Path(args.dhc_out).relative_to(ROOT).as_posix())}

    acs_rows = _fetch(ACS_URL, ACS_VARS, args.key)
    n_acs = _write(acs_rows, Path(args.acs_out))
    prov['sources']['acs_b25070_2024'] = {
        'table': 'B25070 (gross rent as % of household income)',
        'vintage': '2024 ACS 5-year', 'api': ACS_URL,
        'geography': 'ZCTA (national)', 'variables': ACS_VARS,
        'rows': n_acs, 'output': str(Path(args.acs_out).relative_to(ROOT).as_posix())}

    with open(args.provenance, 'w') as f:
        json.dump(prov, f, indent=2)

    print(f'Wrote {n_dhc:,} DHC P2 ZCTA rows and {n_acs:,} ACS B25070 ZCTA rows '
          f'(national; Colorado matched during analysis).')
    print(f'Provenance: {Path(args.provenance).relative_to(ROOT).as_posix()}')


if __name__ == '__main__':
    main()
