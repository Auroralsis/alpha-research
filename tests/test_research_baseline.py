"""P0 baselines for the existing research helpers; no downloads or file writes."""

import unittest

import numpy as np
import pandas as pd

from src.evaluation.factor_metrics import calc_ic_rank
from src.features.momentum import calc_momentum


class ResearchBaselineTests(unittest.TestCase):
    def test_momentum20_formula_and_warmup(self):
        prices = pd.Series(np.arange(10.0, 35.0))
        result = calc_momentum(prices, 20)
        self.assertTrue(result.iloc[:20].isna().all())
        np.testing.assert_allclose(
            result.iloc[20:], prices.iloc[20:].to_numpy() / prices.iloc[:5] - 1
        )

    def test_constant_adjustment_anchor_cancels_in_momentum(self):
        adjusted = pd.Series([10.0, 11.0, 12.0, 15.0])
        np.testing.assert_allclose(
            calc_momentum(adjusted / 3.5, 2),
            calc_momentum(adjusted / 7.0, 2),
            equal_nan=True,
        )

    def test_appending_prices_does_not_change_past_momentum(self):
        prices = pd.Series([10.0, 11.0, 12.0, 15.0])
        extended = pd.concat([prices, pd.Series([100.0, 1.0])], ignore_index=True)
        pd.testing.assert_series_equal(
            calc_momentum(prices, 2), calc_momentum(extended, 2).iloc[: len(prices)]
        )

    def test_rank_ic(self):
        values = pd.Series(np.arange(10.0))
        self.assertAlmostEqual(calc_ic_rank(values, -values), -1.0)


if __name__ == "__main__":
    unittest.main()
