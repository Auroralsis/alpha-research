"""用带休市日的小日历验证 Qlib 滑动、时点约束和连续执行。"""

import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.evaluation.portfolio import QuoteGrid, backtest_long_only
from src.ml.models import fit_lightgbm
from src.ml.rolling import (check_prediction_windows, fit_window, init_rolling_qlib,
                            merge_rolling_predictions, rolling_windows, train_rolling)
from src.ml.rolling_report import hac_mean, window_portfolio_returns
from src.ml.strategy import dropout_targets


def fixture():
    """创建带休市日和多余未来日的日历，留出 11 个全局测试日。"""
    dates = pd.bdate_range("2020-01-02", periods=75).delete([5, 9, 31, 49, 53])
    segments = {name: [str(dates[start].date()), str(dates[end].date())]
                for name, start, end in (("train", 0, 26), ("validation", 27, 43), ("test", 44, 54))}
    config = dict(segments=segments, qlib_version="0.9.7", label_start_days=1, label_end_days=6, seed=158,
                  ridge_alpha=100., training_label="daily_sample_zscore", num_boost_round=8, early_stopping_rounds=3,
                  lightgbm=dict(objective="regression", metric="l2", num_threads=1, verbosity=-1, min_data_in_leaf=2))
    return dates, config


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("qlib", "mlflow", "pymongo", "fire")),
                     "Requires requirements.txt")
class RollingWindowTests(unittest.TestCase):
    def test_slide_and_truncate_on_independent_calendar(self):
        dates, config = fixture()
        with TemporaryDirectory() as temp:
            init_rolling_qlib(dates, "0.9.7", Path(temp))
            windows, step = rolling_windows(dates, config, dates[2])
        self.assertEqual(step, 4)
        self.assertEqual([w["test_sessions"] for w in windows], [4, 4, 3])
        for i, window in enumerate(windows):
            move = 4 * i
            self.assertEqual(window["raw"]["train"], (dates[2+move], dates[26+move]))
            self.assertEqual(window["raw"]["validation"], (dates[27+move], dates[43+move]))
            self.assertEqual(window["raw"]["test"], (dates[44+move], dates[47+move]))
            self.assertEqual(window["effective"]["train"], (dates[2+move], dates[20+move]))
            self.assertEqual(window["effective"]["validation"], (dates[27+move], dates[37+move]))
            self.assertEqual(window["effective"]["test"], (dates[44+move], dates[min(47+move, 54)]))
            self.assertLess(dates[20+move+6], dates[27+move])
            self.assertLess(dates[37+move+6], dates[44+move])

    def test_exact_calendar_end_and_unbounded_raw_last_endpoint(self):
        dates, config = fixture()
        with TemporaryDirectory() as temp:
            out = Path(temp)
            for size in (55, 56):
                limited = dates[:size]
                config["segments"]["test"][1] = str(limited[-1].date())
                init_rolling_qlib(limited, "0.9.7", out)
                windows, step = rolling_windows(limited, config, dates[2])
                self.assertEqual(step, 4)
                self.assertEqual(windows[-1]["raw"]["test"][1], None if size == 55 else limited[-1])
                self.assertEqual(windows[-1]["effective"]["test"][1], limited[-1])

    def test_native_recorders_and_fresh_models_preserve_each_windows_scores(self):
        from qlib.workflow.recorder import MLflowRecorder

        dates, config = fixture()
        names = [f"F{i}" for i in range(158)]
        day = np.repeat(np.arange(2, len(dates)), 5)
        ticker = np.tile(np.arange(5), len(dates)-2)
        series = pd.Series(dates)
        values = (ticker + .01 * day).astype(np.float32)
        rows = pd.DataFrame({name: values + i for i, name in enumerate(names)}).assign(
            ticker=ticker.astype(str), trade_date=dates[day], feature_date=dates[day], source_snapshot_date=dates[day],
            label_start_date=series.shift(-1).iloc[day].to_numpy(),
            label_end_date=series.shift(-6).iloc[day].to_numpy(), label=.01 * ticker,
        )
        rows["label"] = rows.label.where(rows.label_end_date.notna())
        with TemporaryDirectory() as temp:
            root = Path(temp)
            out = root / "run"
            out.mkdir()
            (out / "alpha158.json").write_text(json.dumps([dict(name=name) for name in names]))
            audit = json.dumps(dict(first_feature_date=str(dates[2].date())))
            (out / "data_audit.json").write_text(audit)
            rows.to_parquet(out / "features.parquet", index=False)
            (out / "predictions.parquet").write_bytes(b"fixed predictions")
            models, chunks = [], []

            def capture(*args, **kwargs):
                """记录每次实际拟合返回的模型实例。"""
                model, curve = fit_lightgbm(*args, **kwargs)
                models.append(model)
                return model, curve

            def capture_window(*args, **kwargs):
                predictions, curve = fit_window(*args, **kwargs)
                chunks.append(predictions)
                return predictions, curve

            with patch("src.ml.rolling.fit_lightgbm", side_effect=capture), patch(
                "src.ml.rolling.fit_window", side_effect=capture_window
            ), patch.object(MLflowRecorder, "_log_uncommitted_code", return_value=None):
                merged = train_rolling(config, dates, out)
            self.assertEqual(len(set(map(id, models))), 3)
            self.assertEqual((out / "data_audit.json").read_text(), audit)
            self.assertEqual((out / "predictions.parquet").read_bytes(), b"fixed predictions")
            specs = json.loads((out / "windows.json").read_text())
            windows = specs["windows"]
            expected = rows.loc[rows.trade_date.between(dates[44], dates[54])]
            check_prediction_windows(merged, windows, expected)
            curves = pd.read_csv(out / "training_mse.csv")
            self.assertEqual(set(curves.window), {1, 2, 3})
            for window in windows:
                folder = out / "models" / f'window_{window["window"]:02d}'
                self.assertFalse((folder / "predictions.parquet").exists())
                self.assertTrue((folder / "lightgbm.txt").exists())
                log = json.loads((folder / "training_log.json").read_text())
                for split, boundary in (("train", "validation"), ("validation", "test")):
                    self.assertLess(pd.Timestamp(log["fit_audit"][split]["latest_label_end"]),
                                    pd.Timestamp(window["raw"][boundary][0]))
                self.assertEqual(log["fitted_preprocessors"], [])
            pd.testing.assert_frame_equal(merged.reindex(columns=chunks[0].columns),
                                          pd.concat(chunks, ignore_index=True), check_exact=True)
            self.assertEqual(len(specs["recorders"]), 3)
            # 首窗末日的标签可以跨到下一窗，用于事后评价。
            self.assertTrue(merged.loc[merged.trade_date.eq(dates[47]), "evaluable"].all())
            with self.assertRaises(ValueError):
                check_prediction_windows(pd.concat([merged, merged.iloc[:1]]), windows, expected)
            corrupted = merged.assign(window=merged.window.mask(merged.index == 0, 9))
            with self.assertRaises(ValueError):
                check_prediction_windows(corrupted, windows, expected)
            with self.assertRaises(ValueError):
                check_prediction_windows(merged.iloc[1:], windows, expected)


class RollingEvaluationTests(unittest.TestCase):
    def test_window_return_includes_first_day_and_keeps_the_selected_offset(self):
        dates = pd.bdate_range("2024-01-02", periods=4)
        windows = [dict(window=i + 1, effective={"test": [dates[2*i], dates[2*i+1]]}, test_sessions=2)
                   for i in range(2)]
        ledger = pd.concat([
            pd.DataFrame(dict(trade_date=dates, strategy=strategy, offset=offset,
                              daily_return=[.1, .1, -.2, .2] if offset == 0 else 0.))
            for strategy in ("lightgbm", "rolling_lightgbm") for offset in (0, 1)
        ], ignore_index=True).iloc[::-1]
        results = window_portfolio_returns(ledger, windows, offset=0)
        np.testing.assert_allclose(results.annual_return, [1.21**126 - 1, 1.21**126 - 1,
                                                         .96**126 - 1, .96**126 - 1])
        self.assertTrue(results.portfolio_days.eq(2).all())
        missing = ledger.loc[~(ledger.trade_date.eq(dates[0]) & ledger.strategy.eq("lightgbm") & ledger.offset.eq(0))]
        with self.assertRaises(ValueError):
            window_portfolio_returns(missing, windows, offset=0)

    def test_combined_predictions_align_scores_and_share_the_label(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
        fixed = pd.DataFrame(dict(trade_date=dates, ticker="A", label=[.1, .2, .3],
                                  evaluable=True, split=["validation", "test", "test"], lightgbm=[1., 2., 3.]))
        rolling = fixed.iloc[1:].drop(columns="lightgbm").assign(window=1, rolling_lightgbm=[5., 7.])
        windows = [dict(window=1, effective={"test": [dates[1], dates[2]]}, test_sessions=2)]
        combined = merge_rolling_predictions(fixed, rolling.iloc[::-1], windows)
        self.assertEqual(len(combined), len(fixed))
        self.assertTrue(pd.isna(combined.loc[0, "rolling_lightgbm"]))
        self.assertEqual(combined.rolling_lightgbm.iloc[1:].to_list(), [5., 7.])
        pd.testing.assert_series_equal(combined.label, fixed.label)
        with self.assertRaises(ValueError):
            merge_rolling_predictions(fixed, rolling.iloc[:1], windows)
        with self.assertRaises(AssertionError):
            merge_rolling_predictions(fixed, rolling.assign(label=.5), windows)

    def test_model_switch_does_not_schedule_a_reset(self):
        dates = pd.bdate_range("2024-01-02", periods=11)
        prices = pd.DataFrame({"A": 10. + np.arange(11), "B": 20. + np.arange(11)}, index=dates)
        first, second, third = dates[:4], dates[4:8], dates[8:]
        ranked = {day: ["A", "B"] if day in first else ["B", "A"] for day in [*first, *second, *third]}
        inventory = {}

        def record(day, tickers, units, marks):
            """记录每个交易日实际持有的股票代码。"""
            inventory[day] = tickers[units > 0].to_list()

        result = backtest_long_only(
            QuoteGrid(prices, prices, prices, prices*2, prices/2), ranked,
            rebalance_every=5, fee_bps=5, sell_fee_bps=15,
            select_targets=lambda day, held, sellable: dropout_targets(ranked[day], held, sellable, 1, 1),
            target_count=1, retry_exits=False, record_holdings=record,
        )
        book = result.ledger
        self.assertEqual(book.loc[book.rebalance, "trade_date"].to_list(), [dates[1], dates[6]])
        for boundary, ticker in ((4, "A"), (8, "B")):
            self.assertEqual(inventory[dates[boundary]], [ticker])
            self.assertEqual(book.cash.iloc[boundary-1], book.cash.iloc[boundary])
            self.assertGreater(book.nav.iloc[boundary], book.nav.iloc[boundary-1])
        np.testing.assert_allclose(book.nav, (1 + book.daily_return).cumprod())

    def test_hac_preserves_calendar_gaps(self):
        values = pd.Series([1., np.nan, -1.])
        self.assertAlmostEqual(hac_mean(values, 1)["se"], 1.)
        self.assertAlmostEqual(hac_mean(values.dropna(), 1)["se"], np.sqrt(.5))


if __name__ == "__main__":
    unittest.main()
