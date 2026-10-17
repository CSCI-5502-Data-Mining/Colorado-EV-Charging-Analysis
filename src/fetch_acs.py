"""Download 2024 ACS 5-year ZCTA estimates (national ZCTAs); API key required.

Usage (from the repository root):

    export CENSUS_API_KEY="YOUR_KEY"   # free: https://api.census.gov/data/key_signup.html
    python src/fetch_acs.py

The default output path resolves relative to the repository root, so no
machine-specific absolute paths are needed.
"""
import argparse
import os
import csv
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://api.census.gov/data/2024/acs/acs5'
VARIABLES = ['B01003_001E','B19013_001E','B25003_001E','B25003_003E']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', default=str(ROOT / 'data' / 'raw' / 'acs_zcta_2024.csv'))
    p.add_argument('--key', default=os.environ.get('CENSUS_API_KEY'))
    args = p.parse_args()
    if not args.key:
        raise SystemExit('CENSUS_API_KEY required: register for free at https://api.census.gov/data/key_signup.html')
    resp = requests.get(BASE, params={
       'get': ','.join(['NAME']+VARIABLES), 'for':'zip code tabulation area:*', 'key':args.key
    }, timeout=120)
    resp.raise_for_status()
    rows = resp.json()
    if not rows or len(rows) < 2 or 'zip code tabulation area' not in rows[0]:
        raise RuntimeError('Unexpected Census API response')
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('w',newline='',encoding='utf-8') as f:
        wr=csv.writer(f);wr.writerows(rows)
    print(f'Wrote {len(rows)-1:,} national ZCTA rows to {dest}. These are not all Colorado. Geographic matching occurs during preparation.')

if __name__ == '__main__': main()
