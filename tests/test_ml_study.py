"""Checks for Alpha158 timing, LightGBM training, and weekly execution accounting."""

import importlib.util
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.data.research_panel import prepare_membership, prepare_panel
from src.evaluation.portfolio import QuoteGrid, backtest_long_only, performance
from src.ml.alpha158 import execution_labels, initialize_qlib, native_features, qlib_bars, segment_masks, write_provider
from src.ml.strategy import dropout_orders, dropout_targets
from src.ml.models import daily_zscore, quintile_assignments, train_and_predict


def synthetic_panel(dates: pd.DatetimeIndex, gap: int | None = None):
    price = 10. + np.arange(len(dates))
    shares = 200. + 100 * np.arange(len(dates))
    raw = pd.DataFrame(dict(ticker="A", trade_date=dates, open=price - .5, close=price, high=price + 1,
                            low=price - 1, vol=shares / 100, amount=(price + .25) * shares / 1000,
                            adj_factor=np.where(np.arange(len(dates)) < 7, 2., 4.)))
    return prepare_panel(raw.drop(index=gap) if gap is not None else raw, dates)


class TrainingTests(unittest.TestCase):
    def test_zscore_uses_each_days_sample_deviation_and_excludes_degenerate_days(self):
        dates = pd.Series(pd.to_datetime(["2020-01-02"] * 4 + ["2020-01-03", "2020-01-06", "2020-01-06"]))
        labels = pd.Series([.1, .2, .3, np.nan, .01, .02, .02])
        actual = daily_zscore(labels, dates)
        np.testing.assert_allclose(actual.iloc[:3], [-1., 0., 1.], atol=1e-12)
        self.assertTrue(actual.iloc[3:].isna().all())

    def test_directional_fees_affect_cash_nav_and_trade_rows(self):
        dates = pd.bdate_range("2023-01-03", periods=3)
        prices = pd.DataFrame({"A": [10., 10., 10.]}, index=dates)
        signals = {dates[0]: ["A"], dates[1]: []}
        result = backtest_long_only(
            QuoteGrid(prices, prices, prices, prices * 2, prices / 2), signals,
            fee_bps=5, sell_fee_bps=15, rebalance_every=1,
        )
        bought_notional = 1 / 1.0005
        np.testing.assert_allclose(result.trades.fee, [bought_notional * .0005, bought_notional * .0015])
        np.testing.assert_allclose(result.ledger.cost.iloc[1:], result.trades.fee)
        self.assertAlmostEqual(result.ledger.cash.iloc[-1], bought_notional * .9985)


class MLDataTests(unittest.TestCase):
    def test_qlib_price_volume_and_vwap_conversion_and_stocks_before_entry(self):
        dates = pd.bdate_range("2023-01-03", periods=10)
        panel = synthetic_panel(dates)
        bars = qlib_bars(panel).set_index("trade_date")
        self.assertEqual(bars.close.iloc[0], 1.)
        self.assertAlmostEqual(bars.vwap.iloc[0], 1.025)
        self.assertAlmostEqual(bars.factor.iloc[7], .2)
        self.assertAlmostEqual(bars.close.iloc[7], 3.4)
        self.assertAlmostEqual(bars.volume.iloc[7], 4500.)
        member = pd.DataFrame(dict(ticker=["A"], trade_date=[dates[8]], source_snapshot_date=[dates[7]]))
        self.assertEqual(len(prepare_membership(member)), 1)
        self.assertEqual(len(bars.loc[:dates[7]]), 8)
        member["source_snapshot_date"] = dates[9]
        with self.assertRaises(ValueError):
            prepare_membership(member)

    def test_labels_have_fixed_calendar_endpoints_and_purge(self):
        dates = pd.bdate_range("2023-01-03", periods=14)
        keys = pd.DataFrame(dict(ticker="A", trade_date=dates[:8]))
        panel = synthetic_panel(dates, gap=6)
        labeled = execution_labels(panel, keys, 1, 6)
        self.assertEqual(labeled.label_end_date.iloc[0], dates[6])
        self.assertTrue(np.isnan(labeled.label.iloc[0]))
        self.assertTrue(np.isnan(labeled.label.iloc[5]))  # Missing start, never advance to day 7.
        self.assertAlmostEqual(labeled.label.iloc[1], 2 * 17 / 12 - 1)
        self.assertEqual(labeled.label_start_date.iloc[1], dates[2])
        parts = {"train": [str(dates[0].date()), str(dates[7].date())],
                 "validation": [str(dates[8].date()), str(dates[-1].date())]}
        self.assertEqual(segment_masks(labeled, parts)["train"].to_list(), [False, True] + [False] * 6)

    @unittest.skipUnless(importlib.util.find_spec("qlib"), "Requires the pinned ML environment")
    def test_native_alpha158_matches_known_formulas_and_is_prefix_invariant(self):
        dates = pd.bdate_range("2023-01-03", periods=80)
        full = synthetic_panel(dates, gap=60)
        cut = 70
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for name, stop in (("full", len(dates)), ("prefix", cut + 1)):
                provider = root / name
                bars = qlib_bars(full)
                write_provider(bars.loc[bars.trade_date.le(dates[stop - 1])], dates[:stop], provider)
                spec = initialize_qlib(provider, "0.9.7")
                values = native_features("A", spec, dates[65], dates[cut])
                self.assertEqual(len(spec), 158)
                self.assertAlmostEqual(values.OPEN0.iloc[-1], 79.5 / 80, places=6)
                self.assertAlmostEqual(values.VWAP0.iloc[-1], 80.25 / 80, places=6)
                self.assertAlmostEqual(values.ROC5.iloc[-1], 75 / 80, places=6)
                self.assertTrue(np.isnan(values.ROC5.iloc[0]))
                if name == "full":
                    baseline = values
                else:
                    pd.testing.assert_frame_equal(values, baseline)

    def test_five_groups_use_score_universe_before_label_lookup(self):
        day = pd.Timestamp("2023-01-03")
        frame = pd.DataFrame(dict(trade_date=day, ticker=list("ABCDE"), lightgbm=[3., 1., 5., 2., 4.],
                                  label=[0., 1., np.nan, 2., 3.]))
        groups = quintile_assignments(frame).set_index("ticker").group.to_dict()
        self.assertEqual(groups, {"B": 1, "D": 2, "A": 3, "E": 4, "C": 5})
        frame["label"] = np.nan
        self.assertEqual(quintile_assignments(frame).set_index("ticker").group.to_dict(), groups)

    @unittest.skipUnless(importlib.util.find_spec("lightgbm") and importlib.util.find_spec("sklearn"),
                         "Requires the pinned ML environment")
    def test_repeated_fit_ignores_cross_boundary_and_holdout_labels(self):
        dates = pd.bdate_range("2023-01-03", periods=30)
        periods = {name: [str(dates[lo].date()), str(dates[lo + 9].date())]
                   for name, lo in (("train", 0), ("validation", 10), ("test", 20))}
        names = [f"F{i}" for i in range(158)]
        day = np.repeat(np.arange(len(dates)), 5)
        ticker = np.tile(np.arange(5), len(dates))
        raw = ticker.astype(np.float32)
        sample = pd.DataFrame({name: raw + i for i, name in enumerate(names)})
        sample = sample.assign(trade_date=dates[day], ticker=ticker.astype(str), label=.005 * ticker,
                               label_end_date=pd.Series(dates).shift(-6).iloc[day].to_numpy())
        sample.loc[sample.label_end_date.isna(), "label"] = np.nan
        config = dict(segments=periods, seed=158, training_label="daily_sample_zscore", min_ic_pairs=3, ridge_alpha=100.,
                      num_boost_round=8, early_stopping_rounds=3,
                      lightgbm=dict(objective="regression", metric="l2", min_data_in_leaf=2, num_threads=1,
                                    verbosity=-1, deterministic=True, force_col_wise=True))
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for name, relabel in (("normal", False), ("future_changed", True)):
                out = root / name
                out.mkdir()
                data = sample.copy()
                if relabel:
                    changed = (data.trade_date.le(dates[9]) & data.label_end_date.gt(dates[9])
                               | data.trade_date.ge(dates[20])) & data.label.notna()
                    data.loc[changed, "label"] *= -100.
                (out / "alpha158.json").write_text(json.dumps([{"name": feature} for feature in names]))
                data.to_parquet(out / "features.parquet", index=False)
                train_and_predict(config, out)
                log = json.loads((out / "training_log.json").read_text())
                self.assertEqual(log["split_audit"]["train"]["fitted_rows"], 20)
                self.assertEqual(log["split_audit"]["validation"]["fitted_rows"], 20)
                curve = pd.read_csv(out / "training_mse.csv")
                self.assertEqual(log["best_iteration"], curve.loc[curve.validation_mse.idxmin(), "iteration"])
                scores = pd.read_parquet(out / "predictions.parquet")[["trade_date", "ticker", "lightgbm", "ridge"]]
                self.assertGreater(scores.lightgbm.nunique(), 1)
                self.assertGreater(scores.ridge.nunique(), 1)
                if relabel:
                    pd.testing.assert_frame_equal(scores, expected)
                else:
                    expected = scores


class MLPortfolioTests(unittest.TestCase):
    def test_official_top_bottom_candidates_and_slot_safety(self):
        self.assertEqual(dropout_orders(list("ABCDEFG"), set(), 3, 1), (set(), list("ABC")))
        self.assertEqual(dropout_orders(list("ABCDEFG"), set("DEF"), 3, 1), ({"F"}, ["A"]))
        self.assertEqual(dropout_orders(list("ABCDEFG"), set("ABC"), 3, 1), (set(), []))
        self.assertEqual(dropout_orders(list("ABCDE"), {"B", "C", "Z"}, 3, 1), ({"Z"}, ["A"]))
        self.assertEqual(dropout_targets(list("ABCDE"), set("BCD"), {"B", "C"}, 3, 1), ["B", "C"])
        held = set(f"{i:02}" for i in range(50, 100))
        sells, buys = dropout_orders([f"{i:02}" for i in range(100)], held, 50, 5)
        self.assertEqual(sells, {"95", "96", "97", "98", "99"})
        self.assertEqual(buys, ["00", "01", "02", "03", "04"])

    def test_daily_predictions_five_day_execution_offsets_and_failed_exit(self):
        dates = pd.bdate_range("2023-01-03", periods=12)
        quotes = pd.DataFrame(10., index=dates, columns=list("ABC"))
        down = quotes / 2
        down.loc[dates[6], "A"] = 10.
        signals = {day: list("ABC" if i < 5 else "CBA") for i, day in enumerate(dates)}
        def select(day, held, sellable):
            return dropout_targets(signals[day], held, sellable, 2, 1)
        summaries = []
        for offset in range(5):
            orders = []
            ledger, trades, holdings = backtest_long_only(
                QuoteGrid(quotes, quotes, quotes, quotes * 2, down), signals, fee_bps=15, rebalance_every=5, offset=offset,
                select_targets=select, target_count=2, retry_exits=False, record_orders=orders.append,
            )
            np.testing.assert_array_equal(ledger.loc[ledger.rebalance, "trade_date"], dates[offset + 1::5])
            self.assertTrue(trades.trade_date.isin(dates[offset + 1::5]).all())
            self.assertTrue((dates.get_indexer(trades.trade_date) - dates.get_indexer(trades.signal_date) == 1).all())
            self.assertTrue(ledger.n_holdings.le(2).all())
            summaries.append((ledger, trades, holdings, pd.concat(orders, ignore_index=True)))
        ledger, trades, holdings, orders = summaries[0]
        self.assertEqual(ledger.n_blocked_sells.iloc[6], 1)
        self.assertEqual(set(holdings.ticker), {"B", "C"})
        self.assertEqual(orders.loc[orders.trade_date.eq(dates[1]), "status"].unique().tolist(), ["partial_cash"])
        self.assertEqual(orders.loc[orders.trade_date.eq(dates[6]), "status"].unique().tolist(), ["limit_or_hold_constraint"])
        first = trades.loc[trades.trade_date.eq(dates[1])]
        self.assertAlmostEqual(first.fee.sum(), .0015 / 1.0015)
        self.assertAlmostEqual(ledger.nav.iloc[1], 1. / 1.0015)

    def test_return_metrics_independent_example(self):
        ledger = pd.DataFrame(dict(trade_date=pd.bdate_range("2023-01-30", periods=3),
                                   nav=[1., 1.10, .99], daily_return=[0., .1, -.10],
                                   turnover_two_sided=[0., .5, 0.], cost=[0., .001, 0.], stale_value=[0., 0., 0.]))
        result = performance(ledger)
        self.assertAlmostEqual(result["total_return"], -.01)
        self.assertAlmostEqual(result["annual_return"], .99**84 - 1)
        self.assertAlmostEqual(result["annual_volatility"], .1 * np.sqrt(252))
        self.assertAlmostEqual(result["sharpe"], 0.)
        self.assertAlmostEqual(result["max_drawdown"], -.10)
        self.assertAlmostEqual(result["annual_turnover_one_way"], 21.)
        self.assertAlmostEqual(result["month_win_rate"], .5)
