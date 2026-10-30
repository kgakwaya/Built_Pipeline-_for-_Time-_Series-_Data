"""Unit tests on small synthetic fixtures (run: ``python -m unittest discover -s tests -v`` or ``pytest``)."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data_loading as dl, feature_engineering as fe, modeling as md, preprocessing as pp  # noqa: E402

RAW_TEXT = (
    "Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3\n"
    "16/12/2006;17:25:00;5.360;0.436;233.630;23.000;0.000;1.000;16.000\n"
    "16/12/2006;17:24:00;4.216;0.418;234.840;18.400;0.000;1.000;17.000\n"
    "16/12/2006;17:26:00;?;?;?;?;?;?;?\n"
    "17/12/2006;00:00:00;1.000;0.1;240.0;4.0;0.0;0.0;0.0\n"
)


def write_tmp(text: str) -> Path:
    d = Path(tempfile.mkdtemp()) / dl.RAW_FILENAME
    d.write_text(text)
    return d


def hourly_minute_frame(n_hours=6, seed=0):
    idx = pd.date_range("2020-01-01", periods=n_hours * 60, freq="min", name="timestamp")
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.random(len(idx)) + 1 for c in dl.NUMERIC_COLUMNS}, index=idx)
    return df


class TestLoading(unittest.TestCase):
    def test_load_parse_and_missing(self):
        raw = dl.load_raw_data(write_tmp(RAW_TEXT))
        self.assertEqual(list(raw.columns), dl.EXPECTED_COLUMNS)
        self.assertEqual(int(raw["Global_active_power"].isna().sum()), 1)       # '?' -> NaN
        self.assertEqual(raw["Global_active_power"].dtype, "float64")

    def test_timestamp_parsing_and_order(self):
        frame = dl.build_minute_frame(dl.load_raw_data(write_tmp(RAW_TEXT)))
        self.assertTrue(frame.index.is_monotonic_increasing)                      # input was unsorted
        self.assertEqual(frame.index[0], pd.Timestamp("2006-12-16 17:24:00"))     # day-first parsing
        self.assertEqual(frame.index[-1], pd.Timestamp("2006-12-17 00:00:00"))

    def test_invalid_timestamp_raises(self):
        bad = RAW_TEXT.replace("17/12/2006", "31/02/2006")
        with self.assertRaises(ValueError):
            dl.build_minute_frame(dl.load_raw_data(write_tmp(bad)))

    def test_missing_file_message(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            dl.locate_raw_file(Path(tempfile.mkdtemp()))
        self.assertIn("Dataset not found", str(ctx.exception))

    def test_token_audit(self):
        audit = dl.audit_numeric_tokens(write_tmp(RAW_TEXT.replace("0.1;240.0", "abc;240.0")), chunksize=2)
        self.assertEqual(int(audit.loc["Global_reactive_power", "missing_token"]), 1)
        self.assertEqual(int(audit.loc["Global_reactive_power", "unexpected_nonnumeric"]), 1)


class TestPreprocessing(unittest.TestCase):
    def test_duplicate_policy(self):
        idx = pd.to_datetime(["2020-01-01 00:00", "2020-01-01 00:00", "2020-01-01 00:01", "2020-01-01 00:01"])
        df = pd.DataFrame({"a": [1.0, 1.0, 2.0, 4.0]}, index=idx)
        out, rep = pp.handle_duplicates(df)
        self.assertEqual(rep["duplicate_timestamps"], 2)
        self.assertEqual(rep["duplicate_rows"], 1)                  # only the first pair is fully identical
        self.assertEqual(rep["conflicting_timestamps_averaged"], 1)
        self.assertEqual(len(out), 2)
        self.assertEqual(out.iloc[1, 0], 3.0)                       # mean of 2 and 4

    def test_timestamp_inspection_detects_gap(self):
        idx = pd.DatetimeIndex(["2020-01-01 00:00", "2020-01-01 00:01", "2020-01-01 00:04"])
        info = pp.inspect_timestamps(idx)
        self.assertEqual(info["timestamps_missing_from_grid"], 2)
        self.assertEqual(info["modal_interval"], pd.Timedelta("1min"))

    def test_hourly_aggregation_rules(self):
        df = hourly_minute_frame(3)
        df["Global_active_power"] = 2.0
        df["Sub_metering_1"] = 1.0                                   # 1 Wh per minute
        df.iloc[60:120, :] = np.nan                                  # hour 2 fully missing
        df.iloc[120:160, :] = np.nan                                 # hour 3: 20/60 observed -> invalid with min_obs=45
        h = pp.aggregate_to_hourly(df, min_obs=45)
        self.assertAlmostEqual(h["Global_active_power"].iloc[0], 2.0)  # mean, not sum
        self.assertAlmostEqual(h["Sub_metering_1"].iloc[0], 60.0)      # 1 Wh/min * 60 min = 60 Wh
        self.assertTrue(h.iloc[1][["Global_active_power"]].isna().all())
        self.assertEqual(int(h["n_obs"].iloc[2]), 20)
        self.assertFalse(bool(h["is_valid_hour"].iloc[2]))
        self.assertTrue(np.isnan(h["Global_active_power"].iloc[2]))

    def test_causal_fill_limit_and_flags(self):
        s = pd.DataFrame({"x": [1.0, np.nan, np.nan, np.nan, np.nan, 6.0]})
        filled, flags = pp.causal_fill(s, ["x"], limit=2)
        self.assertEqual(filled["x"].tolist()[:3], [1.0, 1.0, 1.0])
        self.assertTrue(np.isnan(filled["x"].iloc[3]))                # beyond the limit stays missing
        self.assertEqual(int(flags["x"].sum()), 2)

    def test_find_runs(self):
        m = pd.Series([False, True, True, False, True], index=pd.date_range("2020", periods=5, freq="h"))
        runs = pp.find_runs(m)
        self.assertEqual(runs["length"].tolist(), [2, 1])


class TestFeatures(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2020-01-01", periods=400, freq="h")
        self.df = pd.DataFrame({c: np.arange(400, dtype=float) + i for i, c in enumerate(
            ["Global_active_power", *fe.DEFAULT_EXOG])}, index=idx)

    def test_lag_alignment_horizon1(self):
        lag = fe.make_lag_features(self.df["Global_active_power"], [1, 24], horizon=1)
        self.assertEqual(lag["gap_lag1"].iloc[30], self.df["Global_active_power"].iloc[29])
        self.assertEqual(lag["gap_lag24"].iloc[30], self.df["Global_active_power"].iloc[6])
        self.assertTrue(np.isnan(lag["gap_lag24"].iloc[23]))

    def test_lag_alignment_horizon3(self):
        lag = fe.make_lag_features(self.df["Global_active_power"], [1], horizon=3)
        self.assertEqual(lag["gap_lag1"].iloc[30], self.df["Global_active_power"].iloc[27])   # latest known = T-3

    def test_lag_zero_rejected(self):
        with self.assertRaises(ValueError):
            fe.make_lag_features(self.df["Global_active_power"], [0])

    def test_rolling_excludes_current_target(self):
        r = fe.make_rolling_features(self.df["Global_active_power"], [3], horizon=1)
        y = self.df["Global_active_power"].to_numpy()
        self.assertAlmostEqual(r["gap_roll_mean_3h"].iloc[10], y[7:10].mean())               # y[7..9], NOT y[10]
        self.assertAlmostEqual(r["gap_roll_std_3h"].iloc[10], y[7:10].std(ddof=1))

    def test_calendar_from_target_timestamp(self):
        cal = fe.add_calendar_features(pd.DatetimeIndex(["2020-01-04 05:00", "2020-01-06 23:00"]))  # Sat, Mon
        self.assertEqual(cal["is_weekend"].tolist(), [1, 0])
        self.assertEqual(cal["hour"].tolist(), [5, 23])

    def test_no_leakage(self):
        self.assertTrue(fe.leakage_check(self.df, self.df["Global_active_power"], horizon=1, n_trials=3))
        self.assertTrue(fe.leakage_check(self.df, self.df["Global_active_power"], horizon=6, n_trials=3))

    def test_pacf_of_ar1(self):
        phi = 0.6
        acf = np.array([phi ** k for k in range(6)])               # theoretical AR(1) ACF
        pacf = fe.partial_autocorrelation(acf)
        self.assertAlmostEqual(pacf[1], phi)
        self.assertTrue(np.allclose(pacf[2:], 0, atol=1e-9))       # AR(1): PACF cuts off after lag 1

    def test_supervised_frame_drops_missing_target(self):
        obs = self.df["Global_active_power"].copy()
        obs.iloc[300] = np.nan
        X, y, info = fe.build_supervised_frame(self.df, obs)
        self.assertNotIn(self.df.index[300], y.index)
        self.assertTrue(X.notna().all().all())
        self.assertEqual(X.index.min(), self.df.index[168])                                  # 168-h lag/window warm-up


class TestModeling(unittest.TestCase):
    def test_chronological_split(self):
        idx = pd.date_range("2020-01-01", periods=100, freq="h")
        X, y = pd.DataFrame({"a": range(100)}, index=idx), pd.Series(range(100), index=idx)
        s = md.chronological_split(X, y, 0.7, 0.15)
        self.assertEqual([len(s[k][0]) for k in ("train", "val", "test")], [70, 15, 15])
        self.assertLess(s["train"][0].index.max(), s["val"][0].index.min())
        self.assertLess(s["val"][0].index.max(), s["test"][0].index.min())
        with self.assertRaises(AssertionError):
            md.assert_chronological_separation(s["val"][0].index, s["train"][0].index, s["test"][0].index)

    def test_metrics(self):
        m = md.regression_metrics([1.0, 2.0, 3.0], [1.0, 3.0, 5.0])
        self.assertAlmostEqual(m["MAE"], 1.0)
        self.assertAlmostEqual(m["RMSE"], np.sqrt(5 / 3))
        self.assertAlmostEqual(m["R2"], 1 - 5 / 2)
        self.assertAlmostEqual(m["MAPE_pct"], (0 + 50 + 66.6666667) / 3, places=4)
        self.assertTrue(np.isnan(md.regression_metrics([0.0, 1.0], [1.0, 1.0])["MAPE_pct"]))

    def test_end_to_end_small_experiment(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2020-01-01", periods=300, freq="h")
        X = pd.DataFrame({"gap_lag1": rng.random(300), "x": rng.random(300)}, index=idx)
        y = X["gap_lag1"] * 0.5 + X["x"] + 0.01 * rng.standard_normal(300)
        s = md.chronological_split(X, y)
        row, _ = md.run_experiment("B", "persistence", s, baseline_column="gap_lag1")
        self.assertIn("Test_MAE", row)
        search, res, summary = md.tune_model(md.build_ridge_pipeline(), {"model__alpha": [0.1, 1.0]}, *s["train"], n_splits=3)
        self.assertEqual(summary["n_candidates"], 2)
        row, preds = md.run_experiment("R", "ridge", s, estimator=search.best_estimator_, features=list(X.columns), fitted=True)
        self.assertLess(row["Test_MAE"], 0.05)


if __name__ == "__main__":
    unittest.main()
