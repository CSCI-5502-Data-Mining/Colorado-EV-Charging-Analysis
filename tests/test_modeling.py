"""Offline checks for the Milestone 3 modeling pipeline.

These test the leakage guards, the Colorado filter, the station-supply
recoding, the cohort logic, and the leakage-safe structure of the regression
harness on tiny synthetic frames. They do not assert Colorado findings.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))

import modeling_common as mc  # noqa: E402
import prepare_modeling as pm  # noqa: E402


def _synthetic_clean_frame():
    """A small stand-in for colorado_ev_zcta_clean.csv with the key cases:
      - out-of-state code (02056) that must be dropped
      - CO code with a matched station and complete ports
      - CO code with NO station (verified zero supply after recoding)
      - CO code with a tiny household denominator (dropped by households>=100)
    """
    return pd.DataFrame({
        'zcta': ['02056', '80301', '80024', '80939'],
        'ev_count': [1, 500, 120, 37],
        'bev_count': [0, 400, 90, 30],
        'phev_count': [1, 100, 30, 7],
        'station_count': [np.nan, 8.0, np.nan, np.nan],
        'l1_ports': [np.nan, 0.0, np.nan, np.nan],
        'l2_ports': [np.nan, 20.0, np.nan, np.nan],
        'dcfc_ports': [np.nan, 4.0, np.nan, np.nan],
        'l1_ports_partial': [np.nan, False, np.nan, np.nan],
        'l2_ports_partial': [np.nan, False, np.nan, np.nan],
        'dcfc_ports_partial': [np.nan, False, np.nan, np.nan],
        'population': [11673.0, 20000.0, 8000.0, 50.0],
        'median_income': [200720.0, 95000.0, 80000.0, 60000.0],
        'households': [3453.0, 9000.0, 3000.0, 17.0],
        'renter_households': [176.0, 3000.0, 900.0, 5.0],
        'renter_share': [0.05, 0.33, 0.30, 0.29],
        'acs_join': ['both', 'both', 'both', 'both'],
        'matched_acs': [True, True, True, True],
        'ev_data_present': [True, True, True, True],
        'afdc_data_present': [False, True, False, False],
        'ev_per_1000_households': [0.29, 55.5, 40.0, 2176.47],
        'known_ports': [np.nan, 24.0, np.nan, np.nan],
        'ports_per_1000_households': [np.nan, 2.67, np.nan, np.nan],
    })


class LeakageGuard(unittest.TestCase):
    def test_target_derived_columns_rejected(self):
        with self.assertRaises(ValueError):
            mc.assert_no_leakage(['median_income', 'ev_count'])
        with self.assertRaises(ValueError):
            mc.assert_no_leakage(['bev_count'])
        # The approved feature set must pass.
        mc.assert_no_leakage(mc.REGRESSION_FEATURES)

    def test_regression_features_exclude_counts(self):
        for banned in ['ev_count', 'bev_count', 'phev_count',
                       'ev_per_1000_households']:
            self.assertNotIn(banned, mc.REGRESSION_FEATURES)


class ColoradoFilter(unittest.TestCase):
    def test_prefix_and_validity(self):
        s = pd.Series(['80301', '81001', '02056', '1234', '80a01', None],
                      dtype='string')
        mask = mc.is_colorado(s)
        self.assertEqual(list(mask), [True, True, False, False, False, False])

    def test_zip3(self):
        self.assertEqual(list(mc.zip3(pd.Series(['80301', '81426']))),
                         ['803', '814'])


class SupplyRecoding(unittest.TestCase):
    def setUp(self):
        self.df = _synthetic_clean_frame()
        # Keep only Colorado rows the way the pipeline does, then recode.
        co = self.df.loc[mc.is_colorado(self.df['zcta'])].copy()
        self.recoded = pm.stage2_supply_audit(co, {})

    def test_no_station_becomes_verified_zero(self):
        row = self.recoded.loc[self.recoded['zcta'] == '80024'].iloc[0]
        self.assertEqual(float(row['station_count']), 0.0)
        self.assertEqual(float(row['known_ports']), 0.0)
        self.assertEqual(float(row['ports_per_1000_households']), 0.0)
        self.assertTrue(bool(row['supply_is_verified_zero']))

    def test_matched_station_keeps_its_ports(self):
        row = self.recoded.loc[self.recoded['zcta'] == '80301'].iloc[0]
        self.assertEqual(float(row['station_count']), 8.0)
        self.assertEqual(float(row['known_ports']), 24.0)
        self.assertFalse(bool(row['supply_is_verified_zero']))


class Cohorts(unittest.TestCase):
    def setUp(self):
        co = _synthetic_clean_frame()
        co = co.loc[mc.is_colorado(co['zcta'])].copy()
        co = pm.stage2_supply_audit(co, {})
        self.df = pm.stage3_features(co, {})

    def test_small_denominator_excluded_from_regression(self):
        # 80939 has 17 households -> excluded from the households>=100 cohort.
        row = self.df.loc[self.df['zcta'] == '80939'].iloc[0]
        self.assertFalse(bool(row['regression_cohort']))
        # ...but it is retained in the no-floor sensitivity cohort.
        self.assertTrue(bool(row['sensitivity_cohort']))

    def test_valid_rows_in_regression_cohort(self):
        for zcta in ['80301', '80024']:
            row = self.df.loc[self.df['zcta'] == zcta].iloc[0]
            self.assertTrue(bool(row['regression_cohort']), zcta)

    def test_zip3_labels_present(self):
        self.assertTrue((self.df['zip3'].str.len() == 3).all())


class RegressionHarness(unittest.TestCase):
    def test_grouped_test_split_holds_out_whole_groups(self):
        import train_regression as tr
        df = pd.DataFrame({
            'zip3': (['800'] * 10 + ['801'] * 10 + ['802'] * 10
                     + ['803'] * 10 + ['804'] * 10),
            'median_income': np.random.default_rng(0).normal(100, 10, 50),
            'renter_share': np.random.default_rng(1).uniform(0, 1, 50),
            'population': np.random.default_rng(2).uniform(1e3, 5e4, 50),
            'households': np.random.default_rng(3).uniform(500, 2e4, 50),
            'ev_per_1000_households': np.random.default_rng(4).uniform(1, 200, 50),
        })
        train, test, test_groups = tr.grouped_test_split(df, 'zip3')
        # No ZIP3 group may appear in both train and test (no spatial leak).
        self.assertEqual(set(train['zip3']) & set(test['zip3']), set())
        # Every row is accounted for exactly once.
        self.assertEqual(len(train) + len(test), len(df))
        self.assertGreaterEqual(len(test_groups), 1)


if __name__ == '__main__':
    unittest.main()
