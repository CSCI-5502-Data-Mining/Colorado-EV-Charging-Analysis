"""Colorado EV Milestone 2: transparent ZCTA proxy joins and descriptive EDA.

USPS ZIPs and Census ZCTAs are not identical. This first-stage pipeline joins
matching 5-digit codes for exploration; it DOES NOT claim point-in-polygon
spatial accuracy, H3 resolution, or station demand. See docs/limitations.md.

Scope: all vehicle classes (not light-duty only). This matches the team's
Milestone 2 decision; the cleaning notebook reports the light-duty subtotal as
a diagnostic only.

Run from anywhere; default paths resolve relative to the repository root so no
machine-specific absolute paths are needed:

    python src/prepare_and_eda.py
"""
from __future__ import annotations
import argparse, json, re
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Repository root = parent of the src/ directory that holds this file.
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'

# Auto-load .env (e.g. CENSUS_API_KEY) if python-dotenv is installed, so both
# pipeline scripts behave the same way. The pipeline itself does not need the
# key; this is harmless when dotenv is absent. override=False keeps a real
# exported env var authoritative.
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env', override=False)
except ImportError:
    pass


def _rel_to_root(path):
    """Repo-root-relative POSIX path so tracked outputs carry no machine-specific
    absolute path. Falls back to the basename if the path is outside the root."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).name

ALIAS = {
 'zip': ['ZIP Code','ZIP','Station ZIP','Postal Code','zip code tabulation area','zcta'],
 'snapshot_flag': ['Latest DMV Snapshot Flag'],
 'snapshot_date': ['DMV Snapshot (Date)','DMV Snapshot Date','Snapshot Date'],
 'registration_count': ['Vehicle Count'],
 'powertrain': ['Powertrain Type','Drivetrain Type'],
 'station_id': ['ID','Station ID'],
 'state': ['State'],
 'fuel': ['Fuel Type Code'],
 'access': ['Access Code'],
 'status': ['Status Code'],
 'l1': ['EV Level1 EVSE Num','EV Level1 EVSE Count','EV Level 1 Ports'],
 'l2': ['EV Level2 EVSE Num','EV Level2 EVSE Count','EV Level 2 Ports'],
 'dc': ['EV DC Fast Count','EV DC Fast EVSE Num','EV DC Fast Ports'],
}

def resolve_raw(explicit, *patterns):
    """Return an explicit path if given, else the first raw file matching any
    glob pattern (so the repo's downloaded filenames are picked up without
    hard-coding them). Falls back to the first pattern as a literal path."""
    if explicit:
        return explicit
    for pat in patterns:
        matches = sorted(RAW.glob(pat))
        if matches:
            if len(matches) > 1:
                print(f'WARNING: several files match {pat}; using {matches[0].name}: {[m.name for m in matches]}')
            return str(matches[0])
    return str(RAW / patterns[0])

def normalized(s): return re.sub(r'[^a-z0-9]', '', str(s).lower())
def pick(df, variants, mandatory=True):
    mapping={normalized(x):x for x in df.columns}
    for v in variants:
        if normalized(v) in mapping:return mapping[normalized(v)]
    if mandatory: raise ValueError(f'Missing expected field among {variants}; available: {list(df.columns)}')
    return None

def zips(s):
    raw=s.astype('string').str.strip().str.replace(r'\.0$','',regex=True)
    out=raw.str.extract(r'^(\d{5})(?:-\d{4})?$')[0]
    return out

def snap(df, flag, date, log):
    if flag:
        f=df[flag].astype('string').str.strip().str.lower()
        is_latest=f.isin(['true','1','yes','t'])
        if is_latest.any():
            log['latest_snapshot_method']='Latest DMV Snapshot Flag == TRUE'
            return df.loc[is_latest].copy()
        raise ValueError('Snapshot flag column exists but has no TRUE rows; check source file before proceeding')
    if date:
        d=pd.to_datetime(df[date],errors='coerce',utc=True)
        if d.notna().any():
            latest=d.max();log['latest_snapshot_method']=f'Max snapshot date ({latest.date()})'
            return df.loc[d.eq(latest)].copy()
    raise ValueError('No reliable latest-snapshot flag or DMV snapshot date; cannot safely aggregate registrations')

def clean_ev(path,log):
    df=pd.read_csv(path,dtype='string',low_memory=False)
    log['ev_source_rows']=len(df)
    state=pick(df,ALIAS['state'],False)
    if state:
        df=df[df[state].astype('string').str.upper().str.strip().isin(['COLORADO','CO'])].copy()
    flag=pick(df,ALIAS['snapshot_flag'],False); date=pick(df,ALIAS['snapshot_date'],False)
    df=snap(df,flag,date,log)
    zz=pick(df,ALIAS['zip']); count=pick(df,ALIAS['registration_count']); power=pick(df,ALIAS['powertrain'],False)
    df['zcta']=zips(df[zz]);df['vehicle_count']=pd.to_numeric(df[count],errors='coerce')
    log['ev_invalid_zip_rows']=int(df['zcta'].isna().sum())
    log['ev_invalid_or_negative_vehicle_count_rows']=int((df['vehicle_count'].isna() | (df['vehicle_count']<0)).sum())
    df=df.loc[df['zcta'].notna() & df['vehicle_count'].notna() & (df['vehicle_count']>=0)].copy()
    df['type']=df[power].astype('string').str.upper().str.strip() if power else 'UNKNOWN'
    df['type']=df['type'].where(df['type'].isin(['BEV','PHEV']),other='OTHER')
    totals=df.groupby('zcta')['vehicle_count'].sum().rename('ev_count')
    breakdown=df.pivot_table(index='zcta',columns='type',values='vehicle_count',aggfunc='sum',fill_value=0)
    breakdown.columns=[f'{x.lower()}_count' for x in breakdown.columns]
    result=pd.concat([totals,breakdown],axis=1).reset_index()
    for c in ['bev_count','phev_count']:
        if c not in result:result[c]=0
    log['ev_snapshot_rows_after_validation']=len(df)
    log['ev_total_vehicles_all_classes']=int(df['vehicle_count'].sum())
    log['ev_zips']=int(len(result))
    return result

def clean_stations(path,log):
    df=pd.read_csv(path,dtype='string',low_memory=False)
    log['afdc_source_rows']=len(df)
    for key, allow in [('state',['CO','COLORADO']),('fuel',['ELEC']),('access',['PUBLIC']),('status',['E'])]:
        field=pick(df,ALIAS[key],False)
        if not field: raise ValueError(f'AFDC required filter field {key} missing; do not silently count inaccessible or non-EV stations')
        df=df[df[field].astype('string').str.upper().str.strip().isin(allow)].copy()
    sid=pick(df,ALIAS['station_id'],False)
    if not sid: raise ValueError('Cannot deduplicate AFDC stations safely: station ID missing')
    log['afdc_duplicate_station_id_rows']=int(df[sid].duplicated(keep=False).sum())
    df=df.drop_duplicates(subset=[sid],keep='first').copy()
    df['zcta']=zips(df[pick(df,ALIAS['zip'])])
    log['afdc_invalid_zip_rows']=int(df['zcta'].isna().sum())
    df=df[df['zcta'].notna()].copy()
    for dest,key in [('l1_ports','l1'),('l2_ports','l2'),('dcfc_ports','dc')]:
        col=pick(df,ALIAS[key],False)
        if col:
            df[dest]=pd.to_numeric(df[col],errors='coerce').where(lambda s:s.ge(0))
            log[f'afdc_missing_{dest}_rows']=int(df[dest].isna().sum())
        else:
            log[f'afdc_missing_{dest}_field']=True
            df[dest]=np.nan
    # Documented assumption: when a station reports at least one charging
    # level, a blank at a different level means that level is not offered (0).
    # If ALL port fields are blank, capacity is unknown and remains missing.
    types=['l1_ports','l2_ports','dcfc_ports']
    has_known_type=df[types].notna().any(axis=1)
    for name in types:
        df.loc[has_known_type & df[name].isna(),name]=0
    log['afdc_stations_all_port_types_unknown']=int((~has_known_type).sum())
    log['afdc_port_blank_handling']='If at least one port type is reported, blank other types -> zero. If all port types absent, remain missing.'
    agg=df.groupby('zcta').agg(station_count=(sid,'size'),
        l1_ports=('l1_ports',lambda s:s.sum(min_count=1)),
        l2_ports=('l2_ports',lambda s:s.sum(min_count=1)),
        dcfc_ports=('dcfc_ports',lambda s:s.sum(min_count=1))).reset_index()
    # Even where a row has a missing port count, sums are incomplete; retain missing indicator.
    partial=df[['l1_ports','l2_ports','dcfc_ports']].isna().groupby(df['zcta']).any()
    partial.columns=[f'{c}_partial' for c in partial.columns]
    agg=agg.merge(partial.reset_index(),on='zcta',how='left')
    log['afdc_filtered_stations_with_valid_zip']=len(df)
    return agg

def clean_acs(path,log):
    df=pd.read_csv(path,dtype='string',low_memory=False)
    code=pick(df,['zip code tabulation area','zcta','ZIP Code'])
    needed={
       'population':'B01003_001E',
       'median_income':'B19013_001E',
       'households':'B25003_001E',
       'renter_households':'B25003_003E',
    }
    df['zcta']=zips(df[code])
    for dst,c in needed.items():
        field=pick(df,[c])
        vals=pd.to_numeric(df[field],errors='coerce')
        # Census negative coded values can indicate unavailable/suppressed estimates.
        df[dst]=vals.where(vals>=0)
    log['acs_source_rows']=len(df)
    log['acs_invalid_zcta_rows']=int(df['zcta'].isna().sum())
    df=df.dropna(subset=['zcta']).drop_duplicates(subset=['zcta']).copy()
    # Coerce nullable numeric dtypes (Int64/Float64) to plain float64 so that
    # missing values are np.nan. Downstream comparisons feed np.where, which
    # cannot evaluate a pandas <NA> in a boolean mask after the outer join.
    for c in needed.keys():
        df[c]=df[c].astype('float64')
    df['renter_share']=np.where(df['households']>0,df['renter_households']/df['households'],np.nan)
    return df[['zcta',*needed.keys(),'renter_share']]

def quality_plot(df,out):
    fields=['ev_count','households','median_income','renter_share','station_count','l2_ports','dcfc_ports']
    missing=df[fields].isna().mean().sort_values()
    fig,ax=plt.subplots(figsize=(8.5,4.6))
    ax.barh(missing.index,missing.values*100)
    ax.set(xlabel='Missing ZCTA rows (%)',title='Data completeness after joining sources')
    ax.set_xlim(0,100);fig.tight_layout();fig.savefig(out/'05_missingness.png',dpi=160);plt.close(fig)

def plot_x_y(df,x,y,out,title,xlabel,ylabel):
    d=df[[x,y]].replace([np.inf,-np.inf],np.nan).dropna()
    fig,ax=plt.subplots(figsize=(7.2,5.2))
    ax.scatter(d[x],d[y],alpha=.5,s=14)
    ax.set(xlabel=xlabel,ylabel=ylabel,title=f'{title} (n={len(d)})')
    fig.tight_layout();fig.savefig(out,dpi=160);plt.close(fig)
    return {'n':len(d),'pearson_r':round(float(d[x].corr(d[y])),3) if len(d)>2 and d[x].nunique()>1 and d[y].nunique()>1 else None}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--ev',default=None,help='Atlas EV registrations CSV (default: newest data/raw/CO_EV_Registrations*.csv)')
    p.add_argument('--stations',default=None,help='AFDC stations CSV (default: newest data/raw/alt_fuel_stations*.csv)')
    p.add_argument('--acs',default=None,help='ACS ZCTA CSV (default: data/raw/acs_zcta_2024.csv)')
    p.add_argument('--out',default=str(ROOT/'data'/'processed'))
    p.add_argument('--figures',default=str(ROOT/'figures'))
    args=p.parse_args()
    ev_path=resolve_raw(args.ev,'CO_EV_Registrations*.csv','colorado_ev_registrations.csv')
    st_path=resolve_raw(args.stations,'alt_fuel_stations*.csv','afdc_co_stations.csv')
    acs_path=resolve_raw(args.acs,'acs_zcta_2024.csv')
    out=Path(args.out);figs=Path(args.figures);out.mkdir(parents=True,exist_ok=True);figs.mkdir(parents=True,exist_ok=True)
    log={'generated_utc':datetime.now(timezone.utc).isoformat(), 'join_method':'5-digit USPS ZIP matched to 2020-based Census ZCTA code (proxy; not guaranteed spatial match)', 'vehicle_scope':'all vehicle classes', 'source_files':{'ev':_rel_to_root(ev_path),'stations':_rel_to_root(st_path),'acs':_rel_to_root(acs_path)}}
    ev=clean_ev(ev_path,log);st=clean_stations(st_path,log);acs=clean_acs(acs_path,log)
    # Preserve all EV ZIPs and public-charger ZIPs; ACS unmatched rows remain visible.
    key=pd.concat([ev[['zcta']],st[['zcta']]],ignore_index=True).drop_duplicates()
    result=key.merge(ev,on='zcta',how='left',validate='one_to_one').merge(st,on='zcta',how='left',validate='one_to_one').merge(acs,on='zcta',how='left',validate='one_to_one',indicator='acs_join')
    result['matched_acs']=result['acs_join'].eq('both')
    result['ev_data_present']=result['ev_count'].notna();result['afdc_data_present']=result['station_count'].notna()
    # "No matching row" is not automatically a demonstrated true zero. Do not fill with 0.
    result['ev_per_1000_households']=np.where(result['households']>0,result['ev_count']/result['households']*1000,np.nan)
    result['known_ports']=result[['l1_ports','l2_ports','dcfc_ports']].sum(axis=1,min_count=1)
    # Unmatched ZIPs after the outer join have NaN partial flags (object dtype);
    # treat unknown as potentially partial. Build the boolean mask without
    # fillna to avoid pandas' deprecated silent object-downcast: a value is
    # "partial" when it is explicitly True OR when it is missing (unknown).
    partial_cols=[f'{x}_partial' for x in ['l1_ports','l2_ports','dcfc_ports']]
    pf=result[partial_cols]
    has_partial=(pf.eq(True) | pf.isna()).any(axis=1)
    result['known_ports']=result['known_ports'].where(~has_partial)
    result['ports_per_1000_households']=np.where(result['households']>0,result['known_ports']/result['households']*1000,np.nan)
    log['analysis_zip_union_rows']=len(result)
    log['unmatched_acs_zcta_codes']=int((~result['matched_acs']).sum())
    log['ev_zips_without_acs_match']=int((result['ev_data_present'] & ~result['matched_acs']).sum())
    log['station_zips_without_acs_match']=int((result['afdc_data_present'] & ~result['matched_acs']).sum())
    log['percent_missing_by_field']={k:round(float(v),4) for k,v in result.isna().mean().items()}
    # Review extreme rates rather than automatically removing them. Small
    # household denominators can create legitimate but unstable extreme ratios.
    rate=result['ev_per_1000_households'].replace([np.inf,-np.inf],np.nan)
    p99=rate.quantile(0.99) if rate.notna().any() else np.nan
    flagged=result.loc[(rate>=p99) & rate.notna(),['zcta','ev_count','households','ev_per_1000_households']].copy()
    flagged['reason']='At/above observed 99th percentile; review small denominator and source records, do not drop automatically'
    flagged.to_csv(out/'outlier_review.csv',index=False)
    log['outlier_flagged_zcta_count']=int(len(flagged))
    result.to_csv(out/'colorado_ev_zcta_clean.csv',index=False)
    result[['zcta','ev_data_present','afdc_data_present','matched_acs']].to_csv(out/'join_audit.csv',index=False)
    with (out/'quality_report.json').open('w') as f:json.dump(log,f,indent=2)
    # Plot 1: adoption rate distribution
    d=result['ev_per_1000_households'].replace([np.inf,-np.inf],np.nan).dropna()
    fig,ax=plt.subplots(figsize=(7.2,5));ax.hist(d,bins=25);ax.set(xlabel='Registered EVs per 1,000 households',ylabel='Number of ZCTAs',title=f'Observed EV adoption rate by matched code (n={len(d)})');fig.tight_layout();fig.savefig(figs/'01_ev_rate_distribution.png',dpi=160);plt.close(fig)
    # Plot 2: income vs adoption
    c1=plot_x_y(result,'median_income','ev_per_1000_households',figs/'02_income_vs_ev.png','Income and EV adoption','Median household income (USD)','EVs per 1,000 households')
    # Plot 3: access vs adoption
    c2=plot_x_y(result,'ev_per_1000_households','ports_per_1000_households',figs/'03_supply_vs_adoption.png','Public charging supply and EV adoption','EVs per 1,000 households','Public charging ports per 1,000 households')
    # Plot 4: renters vs ports
    c3=plot_x_y(result,'renter_share','ports_per_1000_households',figs/'04_renter_vs_ports.png','Renter share and public charging supply','Renter occupied share (0 to 1)','Public ports per 1,000 households')
    # Plot 5: data quality
    quality_plot(result,figs)
    with (out/'eda_metrics.json').open('w') as f:json.dump({'n_zcta_codes':len(result),'median_ev_per_1000_households':None if d.empty else round(float(d.median()),2),'income_vs_ev':c1,'supply_vs_adoption':c2,'renter_vs_ports':c3},f,indent=2)
    print('Prepared',len(result),'5-digit matched-code records.');print('Outputs:',out,'and',figs)
    print('WARNING: ZIP and ZCTA mismatches are unresolved; no urban/rural or H3 claims are made. Consult join_audit.csv.')

if __name__=='__main__':main()
