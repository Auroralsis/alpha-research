"""检查 Alpha101 的窗口方向、缺值传播和信号日可知性。"""

import unittest

import numpy as np

from src.factors.alpha101 import formula_values
from src.factors.price_volume import (
    alpha13, alpha3, alpha44, choose, correlation, covariance, cross_sectional_rank, decay_linear,
    indneutralize, less, ts_argmax, ts_rank,
)


class Alpha101Test(unittest.TestCase):
    def test_windows_and_missing(self):
        values = np.array([[1., 4.], [4., np.nan], [4., 9.]])
        self.assertAlmostEqual(decay_linear(values, 3.9)[2, 0], 3.5)
        self.assertEqual(ts_argmax(values, 3)[2, 0], 2)
        self.assertAlmostEqual(ts_rank(values, 3)[2, 0], 2.5 / 3)
        self.assertTrue(np.isnan(ts_rank(values, 3)[2, 1]))
        self.assertTrue(np.isnan(choose(less(values, 3), 7, -1)[1, 1]))
        constant = np.full((10, 1), 438.21)
        varying = np.arange(10.)[:, None]
        self.assertTrue(np.isnan(correlation(constant, varying, 10)[-1, 0]))
        self.assertEqual(covariance(constant, varying, 10)[-1, 0], 0)

    def test_industry_and_formula_prefix(self):
        classes = np.array([[0, 0, 1, -1], [1, 0, 1, -1]])
        values = np.array([[1., 3., 9., 8.], [1., 3., 9., 8.]])
        np.testing.assert_allclose(indneutralize(values, classes),
                                   [[-1, 1, 0, np.nan], [-4, 0, 4, np.nan]], equal_nan=True)
        rng = np.random.default_rng(101)
        close = 50 + rng.normal(size=(270, 8)).cumsum(axis=0)
        open_price = close + rng.normal(size=close.shape)
        high = np.maximum(open_price, close) + 1
        low = np.minimum(open_price, close) - 1
        volume = rng.uniform(5, 50, close.shape)
        industry = np.broadcast_to(np.arange(8) % 3, close.shape)
        args = [open_price, high, low, close, volume, (high + low) / 2, volume * close, industry]
        with np.errstate(invalid='ignore', divide='ignore'):
            complete = dict(formula_values(*args))
            truncated = dict(formula_values(*(field[:-3] for field in args)))
        self.assertEqual(list(complete), [f'Alpha{i}' for i in range(1, 102)])
        for name in complete:
            np.testing.assert_allclose(complete[name][:-3], truncated[name], equal_nan=True, err_msg=name)
        ranks = cross_sectional_rank(volume)
        np.testing.assert_allclose(complete['Alpha3'][-1], alpha3(cross_sectional_rank(open_price[-10:]), ranks[-10:]))
        np.testing.assert_allclose(complete['Alpha13'][-1], alpha13(cross_sectional_rank(close[-5:]), ranks[-5:]))
        np.testing.assert_allclose(complete['Alpha44'][-1], alpha44(high[-5:], ranks[-5:]))
        np.testing.assert_allclose(complete['Alpha56'][-1],
            -cross_sectional_rank(np.sum((close[1:]/close[:-1]-1)[-10:],axis=0) /
                                  sum(np.sum((close[1:]/close[:-1]-1)[i-2:i],axis=0) for i in (267,268,269)))
            * cross_sectional_rank((close[-1]/close[-2]-1) * args[6][-1]))
        np.testing.assert_allclose(complete['Alpha101'], (close - open_price)/(high - low + .001))
