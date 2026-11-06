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


def _synthetic_train(n_per_group=24, n_groups=6, seed=7):
    """A learnable synthetic training frame with ZIP3 groups for nested CV."""
    rng = np.random.default_rng(seed)
    zip3 = np.repeat([f'80{d}' for d in range(n_groups)], n_per_group)
    income = rng.normal(90000, 25000, n_per_group * n_groups)
    renter = rng.uniform(0, 1, n_per_group * n_groups)
    pop = rng.uniform(1e3, 5e4, n_per_group * n_groups)
    hh = rng.uniform(500, 2e4, n_per_group * n_groups)
    # Target genuinely depends on income so a model can beat the mean.
    target = 0.0008 * income + rng.normal(0, 8, n_per_group * n_groups)
    return pd.DataFrame({
        'zip3': zip3, 'median_income': income, 'renter_share': renter,
        'population': pop, 'households': hh,
        'ports_per_1000_households': rng.choice([0.0, 2.0, 5.0], n_per_group * n_groups),
        'ev_per_1000_households': target})


class EvaluationProtocol(unittest.TestCase):
    """The fixed evaluation module: nested CV, subgroup fixedness, test isolation."""

    def test_nested_cv_beats_mean_on_learnable_data(self):
        import experiment_eval as ee
        from sklearn.linear_model import LinearRegression
        from sklearn.dummy import DummyRegressor
        train = _synthetic_train()
        mean_c = ee.Candidate('mean', ['median_income'],
                              lambda: DummyRegressor(strategy='mean'))
        lin_c = ee.Candidate('lin', ['median_income'], LinearRegression)
        mean_mae = ee.nested_cv(mean_c, train)['nested_outer_MAE_mean']
        lin_mae = ee.nested_cv(lin_c, train)['nested_outer_MAE_mean']
        # A real signal means the linear model should beat the mean baseline.
        self.assertLess(lin_mae, mean_mae)

    def test_oof_predictions_cover_every_training_row(self):
        import experiment_eval as ee
        from sklearn.linear_model import LinearRegression
        train = _synthetic_train()
        res = ee.nested_cv(ee.Candidate('lin', ['median_income'], LinearRegression),
                           train)
        # Nested CV must score every training row exactly once out-of-fold.
        self.assertFalse(np.isnan(res['oof_pred']).any())

    def test_subgroup_definitions_are_fixed_and_cover_all(self):
        import experiment_eval as ee
        from modeling_common import SUBGROUP_SPECS
        train = _synthetic_train()
        y = train['ev_per_1000_households'].to_numpy()
        pred = y + 1.0
        sg = ee.subgroup_mae(train, y, pred, train)
        # Every declared subgroup is reported (none silently dropped).
        self.assertEqual(set(sg.keys()), set(SUBGROUP_SPECS.keys()))
        # Zero-supply + has-infrastructure partition the supply axis exactly.
        self.assertEqual(sg['zero_supply']['n'] + sg['has_infrastructure']['n'],
                         len(train))

    def test_zero_supply_threshold_is_literal_zero(self):
        # The zero-supply subgroup must use a fixed 0 boundary, never a median,
        # so verified-zero-charging areas are always the true-zero rows.
        from modeling_common import SUBGROUP_SPECS
        col, comp, source = SUBGROUP_SPECS['zero_supply']
        self.assertEqual((col, comp, source),
                         ('ports_per_1000_households', '==', 'zero'))


class HarnessGovernance(unittest.TestCase):
    def test_every_candidate_passes_the_leakage_guard(self):
        import experiment_harness as eh
        import modeling_common as mc
        for c in eh.candidate_space():
            # Must not raise: no candidate may use a target-derived column.
            mc.assert_no_leakage(c.features)

    def test_baselines_are_not_selectable_winners(self):
        import experiment_harness as eh
        logged = [
            {'name': 'baseline_mean', 'nested_outer_MAE_mean': 1.0,
             'nested_outer_MAE_std': 0.1, 'subgroup_oof_MAE': {}},
            {'name': 'rf_x', 'nested_outer_MAE_mean': 5.0,
             'nested_outer_MAE_std': 0.1, 'subgroup_oof_MAE': {}},
        ]
        best = eh.select_best(logged)
        # Even though the baseline has a far lower MAE here, it cannot be the
        # selected model; selection is among substantive candidates only.
        self.assertEqual(best['name'], 'rf_x')

    def test_selection_prefers_consistency_on_a_tie(self):
        import experiment_harness as eh
        logged = [
            {'name': 'rf_a', 'nested_outer_MAE_mean': 26.0,
             'nested_outer_MAE_std': 4.0,
             'subgroup_oof_MAE': {'g1': {'MAE': 10}, 'g2': {'MAE': 40}}},
            {'name': 'rf_b', 'nested_outer_MAE_mean': 26.3,
             'nested_outer_MAE_std': 1.0,
             'subgroup_oof_MAE': {'g1': {'MAE': 24}, 'g2': {'MAE': 28}}},
        ]
        # Within 1.0 MAE: the lower fold-std (more consistent) model wins.
        self.assertEqual(eh.select_best(logged)['name'], 'rf_b')


if __name__ == '__main__':
    unittest.main()
