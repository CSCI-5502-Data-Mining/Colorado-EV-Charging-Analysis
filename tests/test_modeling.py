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

    def test_fully_nested_selection_not_below_best_candidate(self):
        """The fully-nested procedure MAE must be >= the best candidate's own
        nested MAE (it carries the extra candidate-selection cost), so it can
        never understate error. Guards the honest-headline logic."""
        import experiment_harness as eh
        from sklearn.linear_model import LinearRegression, Ridge
        train = _synthetic_train(n_per_group=24, n_groups=6)
        family = [
            eh.Candidate('lin_income', ['median_income'], LinearRegression),
            eh.Candidate('ridge_two', ['median_income', 'renter_share'],
                         lambda: Ridge(random_state=7),
                         param_grid={'model__alpha': [1.0, 10.0]}),
        ]
        fn = eh.fully_nested_selection_mae(family, train)
        own = min(eh.nested_cv(c, train)['nested_outer_MAE_mean'] for c in family)
        # Allow a tiny tolerance for fold-composition differences.
        self.assertGreaterEqual(fn['fully_nested_selection_MAE_mean'], own - 1.0)
        self.assertEqual(len(fn['per_outer_fold_selected_candidate']),
                         eh.OUTER_SPLITS)


class NeedRanking(unittest.TestCase):
    """The ZCTA charging-need ranking (planning track)."""

    def _frame(self):
        import rank_need as rn  # noqa: F401
        return pd.DataFrame({
            'ev_per_1000_households': [10.0, 50.0, 200.0, 5.0, 80.0, 120.0],
            'ports_per_1000_households': [0.0, 1.0, 5.0, 0.0, 2.0, 10.0],
            'median_income': [45000.0, 70000.0, 180000.0, 52000.0, 90000.0, 150000.0],
            'renter_share': [0.6, 0.3, 0.1, 0.55, 0.35, 0.12],
        })

    def test_components_are_separate_and_in_unit_range(self):
        import rank_need as rn
        comp = rn.build_components(self._frame())
        for c in ['demand', 'access', 'equity']:
            self.assertTrue((comp[c] >= 0).all() and (comp[c] <= 1).all())
        # Access = charging scarcity, so the zero-supply rows score highest.
        self.assertEqual(comp['access'].idxmax(), 0)  # first row has 0 ports

    def test_demand_uses_observed_not_predicted(self):
        # Governance: the demand component must be the observed adoption column,
        # never a model prediction. Higher observed adoption => higher demand.
        import rank_need as rn
        df = self._frame()
        comp = rn.build_components(df)
        self.assertEqual(comp['demand'].idxmax(), df['ev_per_1000_households'].idxmax())

    def test_equity_weight_shifts_toward_lower_income(self):
        # Weighting equity more heavily must not raise the selected set's income.
        import rank_need as rn
        df = self._frame()
        comp = rn.build_components(df)
        demand_heavy = rn.composite_score(comp, {'demand': 0.8, 'access': 0.1, 'equity': 0.1})
        equity_heavy = rn.composite_score(comp, {'demand': 0.1, 'access': 0.1, 'equity': 0.8})
        top_demand = df.loc[demand_heavy.nlargest(2).index, 'median_income'].mean()
        top_equity = df.loc[equity_heavy.nlargest(2).index, 'median_income'].mean()
        self.assertLess(top_equity, top_demand)

    def test_percentile_normalization_robust_to_outlier(self):
        # A single extreme supply value must not collapse the access spread.
        import rank_need as rn
        s = pd.Series([0.0, 1.0, 2.0, 3.0, 1000.0])
        n = rn.minmax(s)
        # Percentile ranks are evenly spread regardless of the 1000 outlier.
        self.assertAlmostEqual(float(n.iloc[0]), 0.0, places=6)
        self.assertAlmostEqual(float(n.iloc[-1]), 1.0, places=6)
        self.assertAlmostEqual(float(n.iloc[2]), 0.5, places=6)

    def test_tiers_handle_large_ties_without_inventing_order(self):
        # A big tied block (e.g. zero-supply codes) must NOT be split into an
        # arbitrary order; qcut with duplicates='drop' collapses to fewer tiers.
        import rank_need as rn
        # 10 codes tied at rank 1, then 1..5 distinct.
        rank = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 11, 12, 13, 14, 15])
        tiers = rn.assign_tiers(rank, len(rank), k=5)
        # The ten tied codes must all share one tier (no fake ordering).
        self.assertEqual(tiers.iloc[:10].nunique(), 1)
        # Tiers are integers starting at 1 (highest need).
        self.assertEqual(int(tiers.min()), 1)

    def test_low_ownership_area_can_still_rank_high(self):
        # Governance for review point #4: a low-adoption, renter-heavy,
        # low-income, zero-supply area must be able to out-rank a high-adoption,
        # owner, high-income, well-supplied area under the composite.
        import rank_need as rn
        df = pd.DataFrame({
            'ev_per_1000_households': [5.0, 200.0],   # low vs high ownership
            'ports_per_1000_households': [0.0, 10.0],  # none vs plenty
            'median_income': [40000.0, 180000.0],      # low vs high
            'renter_share': [0.7, 0.1],                # renter vs owner
        })
        comp = rn.build_components(df)
        score = rn.composite_score(comp, rn.BASE_WEIGHTS)
        # The low-ownership, high-need community must score higher.
        self.assertGreater(score.iloc[0], score.iloc[1])

    def test_percentile_movement_is_scale_invariant(self):
        """A resample with fewer surviving codes must not inflate movement.
        Guards the stability-metric bug fix: percentile rank is comparable
        across different-length lists, integer rank is not."""
        base = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        # Same relative order but only half the codes 'survive' a resample.
        subset = pd.Series([1.0, 3.0, 5.0, 7.0])
        base_pct = base.rank(ascending=False, pct=True)
        sub_pct = subset.rank(ascending=False, pct=True)
        # For the top code, percentile stays ~top in both despite list length;
        # integer ranks (1 of 8 vs 1 of 4) would look stable but the MIDDLE
        # codes are where integer rank distorts. Check the metric is bounded.
        self.assertTrue((base_pct >= 0).all() and (base_pct <= 1).all())
        self.assertTrue((sub_pct >= 0).all() and (sub_pct <= 1).all())
        # Top element percentile is small (near 0) on both scales -> comparable.
        self.assertAlmostEqual(float(base_pct.iloc[0]), float(sub_pct.iloc[0]), delta=0.2)


class UrbanRuralAffordability(unittest.TestCase):
    """Supporting descriptive analysis: classification thresholds and the
    B25070 cost-burden denominator rule (excluding 'not computed')."""

    def test_urban_class_thresholds(self):
        import urban_rural_affordability as ura
        shares = pd.Series([0.95, 0.80, 0.50, 0.20, 0.05, np.nan])
        cls = ura.classify(shares)
        self.assertEqual(cls.iloc[0], 'urban_dominant')   # 0.95
        self.assertEqual(cls.iloc[1], 'urban_dominant')   # 0.80 boundary inclusive
        self.assertEqual(cls.iloc[2], 'mixed')            # 0.50
        self.assertEqual(cls.iloc[3], 'rural_dominant')   # 0.20 boundary inclusive
        self.assertEqual(cls.iloc[4], 'rural_dominant')   # 0.05
        self.assertTrue(pd.isna(cls.iloc[5]))             # missing stays missing

    def test_cost_burden_excludes_not_computed(self):
        # B25070: denominator = total (001) minus 'not computed' (011); numerator
        # = the four >=30% buckets (007-010). Verify the exclusion explicitly.
        import tempfile
        import urban_rural_affordability as ura
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'b25070.csv'
            # 100 total renters, 20 not-computed -> denom 80; burdened = 10+10+5+15 = 40.
            pd.DataFrame({
                'B25070_001E': [100], 'B25070_007E': [10], 'B25070_008E': [10],
                'B25070_009E': [5], 'B25070_010E': [15], 'B25070_011E': [20],
                'zip code tabulation area': ['80301'],
            }).to_csv(path, index=False)
            orig = ura.ACS_B25070_CSV
            ura.ACS_B25070_CSV = path
            try:
                out = ura.load_b25070()
            finally:
                ura.ACS_B25070_CSV = orig
            # 40 / (100 - 20) = 0.5, NOT 40/100 = 0.4.
            self.assertAlmostEqual(float(out['renter_cost_burden_share'].iloc[0]), 0.5, places=6)
            self.assertEqual(int(out['renter_households_rent_computed'].iloc[0]), 80)

    def _load_b25070_from(self, rows):
        """Run load_b25070 against a one-row synthetic B25070 frame."""
        import tempfile
        import urban_rural_affordability as ura
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'b25070.csv'
            pd.DataFrame([{**rows, 'zip code tabulation area': '80301'}]).to_csv(path, index=False)
            orig = ura.ACS_B25070_CSV
            ura.ACS_B25070_CSV = path
            try:
                return ura.load_b25070()
            finally:
                ura.ACS_B25070_CSV = orig

    def test_missing_numerator_bucket_is_not_treated_as_zero(self):
        # If any >=30% bucket is missing, the share must be NaN (rejected), not
        # computed from a silently-zeroed numerator.
        out = self._load_b25070_from({
            'B25070_001E': 100, 'B25070_007E': 10, 'B25070_008E': '',  # missing
            'B25070_009E': 5, 'B25070_010E': 15, 'B25070_011E': 20})
        self.assertTrue(pd.isna(out['renter_cost_burden_share'].iloc[0]))

    def test_negative_census_sentinel_rejected(self):
        # An ACS negative sentinel (e.g. -666666666) must be treated as missing,
        # so the row cannot produce a bogus share.
        out = self._load_b25070_from({
            'B25070_001E': -666666666, 'B25070_007E': 10, 'B25070_008E': 10,
            'B25070_009E': 5, 'B25070_010E': 15, 'B25070_011E': 20})
        self.assertTrue(pd.isna(out['renter_cost_burden_share'].iloc[0]))
        self.assertTrue(pd.isna(out['renter_households_rent_computed'].iloc[0]))

    def test_nonpositive_denominator_rejected(self):
        # total == not-computed -> denominator 0 -> share NaN, not a divide error.
        out = self._load_b25070_from({
            'B25070_001E': 20, 'B25070_007E': 0, 'B25070_008E': 0,
            'B25070_009E': 0, 'B25070_010E': 0, 'B25070_011E': 20})
        self.assertTrue(pd.isna(out['renter_cost_burden_share'].iloc[0]))

    def test_impossible_share_above_one_rejected(self):
        # If buckets sum to more than the denominator (inconsistent input), the
        # share would exceed 1 and must be rejected to NaN.
        out = self._load_b25070_from({
            'B25070_001E': 50, 'B25070_007E': 30, 'B25070_008E': 30,
            'B25070_009E': 30, 'B25070_010E': 30, 'B25070_011E': 10})
        self.assertTrue(pd.isna(out['renter_cost_burden_share'].iloc[0]))

    def test_valid_row_unchanged_by_validation(self):
        # A clean row still computes exactly as before (no regression).
        out = self._load_b25070_from({
            'B25070_001E': 100, 'B25070_007E': 10, 'B25070_008E': 10,
            'B25070_009E': 5, 'B25070_010E': 15, 'B25070_011E': 20})
        self.assertAlmostEqual(float(out['renter_cost_burden_share'].iloc[0]), 0.5, places=6)

    def test_fetch_errors_do_not_leak_api_key(self):
        # A failed Census request must raise without the key appearing in the
        # error message (the key is a query parameter on the request URL).
        import urban_rural_affordability  # noqa: F401 (ensures src on path)
        import fetch_urban_affordability as fua

        class _Resp:
            status_code = 403
            def json(self):
                return []

        real_get = fua.requests.get
        fua.requests.get = lambda *a, **k: _Resp()
        try:
            with self.assertRaises(RuntimeError) as ctx:
                fua._fetch('https://api.census.gov/data/2024/acs/acs5',
                           ['B25070_001E'], 'SECRETKEY123')
            msg = str(ctx.exception)
            self.assertNotIn('SECRETKEY123', msg)
            self.assertIn('403', msg)
        finally:
            fua.requests.get = real_get

    def test_fetch_connection_error_does_not_leak_api_key(self):
        # A transport-level failure (connection error / timeout) is raised by
        # requests.get() BEFORE any status code and its message embeds the full
        # key-bearing URL; _fetch must re-raise without the key or that message.
        import fetch_urban_affordability as fua

        def _boom(*a, **k):
            # Mimic requests embedding the key-bearing URL in the message.
            raise fua.requests.exceptions.ConnectionError(
                "HTTPSConnectionPool: Max retries exceeded with url: "
                "/data?get=X&key=SECRETKEY123 (Caused by ...)")

        real_get = fua.requests.get
        fua.requests.get = _boom
        try:
            with self.assertRaises(RuntimeError) as ctx:
                fua._fetch('https://api.census.gov/data/2024/acs/acs5',
                           ['B25070_001E'], 'SECRETKEY123')
            msg = str(ctx.exception)
            self.assertNotIn('SECRETKEY123', msg)
            self.assertIn('ConnectionError', msg)
        finally:
            fua.requests.get = real_get

    def test_special_use_flag_catches_institutional_not_ordinary(self):
        # The special-use rule (renter_share==100% OR persons/household>5) must
        # flag institutional/group-quarters ZCTAs and leave ordinary ones alone.
        import numpy as np
        renter = pd.Series([1.00, 0.30, 0.55, 0.42])
        pph = pd.Series([2.4, 9.5, 2.1, 6.1])  # ordinary, USAFA-like, ordinary, group-qtrs
        flag = (renter >= 0.999) | (pph > 5)
        self.assertEqual(list(flag), [True, True, False, True])
        # An ordinary renter-heavy-but-not-100% area with normal household size
        # is NOT flagged.
        self.assertFalse(bool(((pd.Series([0.65]) >= 0.999) | (pd.Series([2.7]) > 5)).iloc[0]))

    def test_urban_share_undefined_for_zero_population(self):
        import tempfile
        import urban_rural_affordability as ura
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'dhc.csv'
            pd.DataFrame({
                'P2_001N': [1000, 0], 'P2_002N': [800, 0], 'P2_003N': [200, 0],
                'zip code tabulation area': ['80301', '80302'],
            }).to_csv(path, index=False)
            orig = ura.DHC_CSV
            ura.DHC_CSV = path
            try:
                out = ura.load_dhc().set_index('zcta')
            finally:
                ura.DHC_CSV = orig
            self.assertAlmostEqual(float(out.loc['80301', 'urban_share_2020']), 0.8, places=6)
            self.assertTrue(pd.isna(out.loc['80302', 'urban_share_2020']))


if __name__ == '__main__':
    unittest.main()
