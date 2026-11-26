import unittest

import numpy as np
class MLPTests(unittest.TestCase):
    def test_mlp_refit_windows_are_adjacent_and_clip_partial_test_periods(self):
        from src.scripts.train_mlp import mlp_windows

        windows = list(mlp_windows(["2023-01-05", "2023-04-20"], train_months=36,
                                   stop_months=1, test_months=3))
        windows = [{k: [str(d.date()) for d in dates] for k, dates in w.items()} for w in windows]
        self.assertEqual(windows, [
            {"train": ["2019-12-01", "2022-11-30"], "early_stop": ["2022-12-01", "2022-12-31"],
             "test": ["2023-01-05", "2023-03-31"]},
            {"train": ["2020-03-01", "2023-02-28"], "early_stop": ["2023-03-01", "2023-03-31"],
             "test": ["2023-04-01", "2023-04-20"]},
        ])

    def test_ndcg_gradient_matches_lambdas_and_metric(self):
        import torch
        from src.ml.mlp import ndcg_day

        # 并列分数保持代码顺序，不同档收益允许并列；训练截断与评价截断分开。
        for values in ([.3, .3, -.2, .8, -.5], [0.] * 5):
            x = np.array(values)
            gains = np.array([3., 0., 9., 1., 3.])
            order = np.argsort(-x, kind="stable")
            discounts = 1/np.log2(np.arange(5)+2)
            ideal = np.sort(gains)[::-1]
            gradient = np.zeros(5)
            mass = 0.
            for i in range(3):
                for j in range(i+1, len(x)):
                    a, b = order[i], order[j]
                    if gains[a] == gains[b]:
                        continue
                    high, low = (a, b) if gains[a] > gains[b] else (b, a)
                    margin = x[high]-x[low]
                    weight = (gains[high]-gains[low])*(discounts[i]-discounts[j])/(ideal[:3]@discounts[:3])
                    if x.max() != x.min():
                        weight /= .01+abs(margin)
                    update = weight/(1+np.exp(margin))
                    gradient[high] -= update
                    gradient[low] += update
                    mass += 2*update
            scores = torch.tensor(x, requires_grad=True)
            target = torch.tensor(gains)
            ndcg_day(scores, target, 2, 3).backward()
            np.testing.assert_allclose(scores.grad, gradient*np.log2(1+mass)/mass, atol=1e-12)
            loss = ndcg_day(scores, target, 2, 3, evaluate=True)
            self.assertAlmostEqual(loss.item(), 1-(gains[order[:2]]@discounts[:2])/(ideal[:2]@discounts[:2]))
        zero = torch.zeros_like(target)
        self.assertEqual(ndcg_day(scores, zero, 2, 3, evaluate=True).item(), 0.)
        self.assertEqual(ndcg_day(scores, zero, 2, 3).item(), 0.)

    def test_mlp_reuses_training_scale_for_missing_and_out_of_range_values(self):
        from src.ml.mlp import fit_scale, transform

        train = np.array([[1, 3, np.nan], [3, 3, np.nan], [np.nan, 3, np.nan]])
        stats = fit_scale(train, ["f1", "f2", "f3"])
        held = np.array([[np.nan, 6, 9], [8, 3, np.nan], [-1, 3, 0]])
        np.testing.assert_allclose(transform(held, stats).numpy(), [[2, 6, 9], [8, 3, 0], [-1, 3, 0]])
        np.testing.assert_allclose(transform(held, stats, "minmax").numpy(), [[.5, 0, 0], [1, 0, 0], [0, 0, 0]])

    def test_daily_scaling_uses_only_same_day_observations(self):
        import pandas as pd
        from src.ml.dataset import standardize_daily_features

        rows = pd.DataFrame({"trade_date": [1, 2, 1, 2, 1], "factor": [1., 100., 3., 100., np.nan],
                             "missing": np.nan, "label": [1., 1., np.nan, 1., 1.]})
        standardize_daily_features(rows, ["factor", "missing"])
        np.testing.assert_allclose(rows.factor, [-1., 0., 1., 0., 0.])
        np.testing.assert_allclose(rows.missing, 0.)
        self.assertTrue(np.isnan(rows.label.iloc[2]))
