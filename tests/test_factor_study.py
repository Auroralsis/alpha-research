import unittest
from typing import cast

import numpy as np
import pandas as pd

from src.evaluation.cross_section import weighted_slope
from src.evaluation.layer_backtest import build_targets, spread_track
from src.evaluation.portfolio import QuoteGrid, backtest_long_only, performance


class FactorStudyTests(unittest.TestCase):
    def test_weighted_slope_and_standard_error(self):
        design = np.column_stack([[0., 1., 2.], np.ones(3)])
        slope, se = weighted_slope(design, np.array([1., 2., 6.]), np.array([1., 2., 1.]))
        self.assertAlmostEqual(slope, 2.5)
        self.assertAlmostEqual(se, np.sqrt(1.125))

    def test_singular_regression_has_no_slope(self):
        slope, se = weighted_slope(np.ones((4, 2)), np.arange(4), np.ones(4))
        self.assertTrue(np.isnan(slope) and np.isnan(se))

    def test_layers_start_at_highest_score_with_deterministic_ties(self):
        day = cast(pd.Timestamp, pd.Timestamp("2023-01-03"))
        frame = pd.DataFrame(dict(trade_date=day, ticker=["C", "B", "A", "D"], score=[4., 9., 9., np.nan]))
        layers = build_targets(frame, "score", 2)
        self.assertEqual([targets[day] for targets in layers], [["A", "B"], ["C"]])

    def test_next_close_fill_charges_fee_and_limit_blocks_buy(self):
        dates = pd.bdate_range("2023-01-03", periods=3)
        close = pd.DataFrame({"A": [10., 20., 30.]}, index=dates)
        signals = {dates.to_list()[0]: ["A"]}
        quotes = QuoteGrid(close, close, close, close * 2, close / 2)
        book = backtest_long_only(quotes, signals).ledger
        np.testing.assert_allclose(book.nav, [1., 1. / 1.0015, 1.5 / 1.0015])
        blocked = backtest_long_only(quotes._replace(up_limit=close), signals).ledger
        np.testing.assert_array_equal(blocked.nav, [1., 1., 1.])
        self.assertEqual(blocked.n_blocked_buys.sum(), 1)

    def test_spread_compounds_daily_difference_not_terminal_difference(self):
        dates = pd.bdate_range("2023-01-03", periods=3)
        top = pd.DataFrame(dict(trade_date=dates, daily_return=[0., .1, 0.]))
        bottom = top.assign(daily_return=[0., .02, -.01])
        spread = spread_track(top, bottom)
        np.testing.assert_allclose(spread.nav, [1., 1.08, 1.08 * 1.01])
        stats = performance(spread)
        self.assertTrue(np.isnan(stats["cost_initial_nav"]))
        self.assertTrue(np.isnan(stats["annual_turnover_one_way"]))
