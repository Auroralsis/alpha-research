"""P0 baselines for the existing research helpers; no downloads or file writes."""

import unittest

import numpy as np
import pandas as pd

from src.evaluation.factor_metrics import calc_ic, calc_ic_rank
from src.evaluation.quantile_analysis import assign_quantiles, calc_quantile_returns
from src.features.momentum import calc_momentum
from src.features.targets import calc_future_return


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

    def test_forward_returns_are_research_labels(self):
        prices = pd.Series([10.0, 11.0, 12.0, 15.0])
        result = calc_future_return(prices, 2)
        np.testing.assert_allclose(result.iloc[:2], [0.2, 15 / 11 - 1])
        self.assertTrue(result.iloc[-2:].isna().all())

    def test_ic_and_rank_ic(self):
        values = pd.Series(np.arange(10.0))
        self.assertAlmostEqual(calc_ic(values, values), 1.0)
        self.assertAlmostEqual(calc_ic_rank(values, -values), -1.0)

    def test_quantiles_partition_distinct_values(self):
        values = pd.Series(np.arange(10.0))
        np.testing.assert_array_equal(assign_quantiles(values), [1, 1, 2, 2, 3, 3, 4, 4, 5, 5])

    def test_quantiles_preserve_empty_and_all_missing_indices(self):
        for values in [pd.Series(dtype=float), pd.Series([np.nan, np.nan], index=[4, 9])]:
            result = assign_quantiles(values)
            self.assertTrue(result.isna().all())
            pd.testing.assert_index_equal(result.index, values.index)

    def test_quantile_returns_are_arithmetic_means(self):
        frame = pd.DataFrame({"q": [1, 1, 2, 2], "r": [0.1, 0.3, -0.1, 0.1]})
        np.testing.assert_allclose(calc_quantile_returns(frame, "q", "r"), [0.2, 0.0])


if __name__ == "__main__":
    unittest.main()
