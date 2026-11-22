import unittest

import numpy as np
from scipy.special import logsumexp

from src.ml.listfold import ListFold, fold_day


class ListFoldTests(unittest.TestCase):
    def test_ndcg_gradient_matches_lambdas_and_metric(self):
        import torch
        from src.ml.listfold import ndcg_day

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

    def test_gpu_batch_keeps_daily_groups_and_gradients(self):
        import torch
        from src.ml.listfold import batch_loss, network

        if not torch.cuda.is_available():
            self.skipTest("需要 CUDA 显卡")
        days = [torch.tensor([[0.8], [0.2], [0.1]]), torch.tensor([[0.4], [0.3]])]
        cpu = network([1, 1])
        with torch.no_grad():
            cpu[0].weight.fill_(0.5)
        gpu = network([1, 1]).to("cuda")
        gpu.load_state_dict(cpu.state_dict())
        expected, actual = batch_loss(cpu, days), batch_loss(gpu, days)
        self.assertTrue(actual.is_cuda)
        expected.backward()
        actual.backward()
        torch.testing.assert_close(actual.cpu(), expected)
        torch.testing.assert_close(gpu[0].weight.grad.cpu(), cpu[0].weight.grad)

    def test_mlp_loss_matches_verified_listfold_gradient(self):
        import torch
        from src.ml.listfold import torch_fold_day

        for n in (2, 5, 8):
            x = torch.linspace(0.8, -0.4, n, dtype=torch.float64, requires_grad=True)
            expected, gradient, _ = fold_day(x.detach().numpy())
            loss = torch_fold_day(x)
            loss.backward()
            self.assertAlmostEqual(loss.item(), expected)
            np.testing.assert_allclose(x.grad.numpy(), gradient, atol=1e-12)

    def test_mlp_reuses_training_scale_for_missing_and_out_of_range_values(self):
        from src.ml.listfold import fit_scale, transform

        train = np.array([[1, 3, np.nan], [3, 3, np.nan], [np.nan, 3, np.nan]])
        stats = fit_scale(train, ["f1", "f2", "f3"])
        held = np.array([[np.nan, 6, 9], [8, 3, np.nan], [-1, 3, 0]])
        np.testing.assert_allclose(transform(held, stats).numpy(), [[.5, 0, 0], [1, 0, 0], [0, 0, 0]])

    def test_matches_ordered_pair_likelihood_and_its_derivatives(self):
        def direct(scores):
            total = 0.
            for j in range(len(scores) // 2):
                rest = scores[j:len(scores)-j]
                diffs = rest[:, None] - rest[None, :]
                total += logsumexp(diffs[~np.eye(len(rest), dtype=bool)]) - rest[0] + rest[-1]
            return total

        for n in (2, 3, 6, 7):
            x = np.random.default_rng(n).normal(size=n)
            loss, g, h = fold_day(x)
            self.assertAlmostEqual(loss, direct(x))
            eps = .0002
            unit = eps * np.eye(n)
            np.testing.assert_allclose(g, [(direct(x+d)-direct(x-d))/(2*eps) for d in unit], atol=1e-7)
            np.testing.assert_allclose(h, [(direct(x+d)+direct(x-d)-2*loss)/eps**2 for d in unit], atol=5e-7)
        labels = np.array([2., 3., 1., 5., 4.])
        x = labels / 10
        obj = ListFold(labels, np.array([3, 2]))
        g, h = obj.objective(x, None)
        idx = np.array([1, 0, 2, 3, 4])
        seg = [fold_day(x[idx[:3]]), fold_day(x[idx[3:]])]
        np.testing.assert_allclose(g[idx], np.r_[seg[0][1],seg[1][1]])
        np.testing.assert_allclose(h[idx], np.r_[seg[0][2],seg[1][2]])
