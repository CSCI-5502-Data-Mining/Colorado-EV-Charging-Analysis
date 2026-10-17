"""Offline smoke checks for cleaning rules; test values are not Colorado findings."""
import tempfile
import unittest
from pathlib import Path
import pandas as pd
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from prepare_and_eda import clean_ev,clean_stations,clean_acs

class CleaningChecks(unittest.TestCase):
    def test_snapshot_dedup_and_missing(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            ev=root/'ev.csv'
            pd.DataFrame({'State':['CO','CO','CO'], 'ZIP Code':['80301','80301','80301'],'Vehicle Count':[2,8,3],'Powertrain Type':['BEV']*3,'Latest DMV Snapshot Flag':['TRUE','FALSE','TRUE']}).to_csv(ev,index=False)
            ev_result=clean_ev(ev,{})
            self.assertEqual(float(ev_result['ev_count'].iloc[0]),5)
            stations=root/'stations.csv'
            pd.DataFrame({'ID':['A','A','B'],'State':['CO']*3,'Fuel Type Code':['ELEC']*3,'Access Code':['public']*3,'Status Code':['E']*3,'ZIP':['80301']*3,'EV Level2 EVSE Num':[2,2,None],'EV DC Fast Count':[None,None,None]}).to_csv(stations,index=False)
            station_result=clean_stations(stations,{})
            self.assertEqual(int(station_result['station_count'].iloc[0]),2)
            self.assertTrue(bool(station_result['l2_ports_partial'].iloc[0]))
            acs=root/'acs.csv'
            pd.DataFrame({'zip code tabulation area':['80301'],'B01003_001E':[4000],'B19013_001E':[-666666666],'B25003_001E':[1600],'B25003_003E':[650]}).to_csv(acs,index=False)
            acs_result=clean_acs(acs,{})
            self.assertTrue(pd.isna(acs_result['median_income'].iloc[0]))
            self.assertAlmostEqual(float(acs_result['renter_share'].iloc[0]),650/1600)

if __name__=='__main__':unittest.main()
