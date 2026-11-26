import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from qlib.contrib.model.pytorch_tra import RNN, TRA

from src.ml.tra import CausalTRA, KEYS, LENGTH, MODEL, Panel, ROUTER, TRAINING, replay
from src.scripts.run_tra import ARMS, diagnostics, fusion, history_keys


def fixture():
    dates = pd.bdate_range("2020-12-22", periods=12)
    hist = pd.DataFrame(product(dates, ["a", "b", "c"]), columns=KEYS)
    hist["x"] = np.arange(len(hist), dtype=np.float32)
    s = hist.loc[hist.trade_date.ge(dates[2]), KEYS].copy()
    day = pd.Series(dates, index=dates)
    s["label_start_date"] = s.trade_date.map(day.shift(-1))
    s["label_end_date"] = s.trade_date.map(day.shift(-2))
    s["source_snapshot_date"] = dates[0]
    s["label"] = s.ticker.map({"a": -1., "b": 0., "c": 1.}).where(s.label_end_date.notna())
    return dates, hist, s


def panel(dates, hist, samples):
    return Panel(hist, samples, ["x"], dates, [dates[2], dates[6]])


class MemoryRouter(torch.nn.Module):
    """仅根据传入的历史误差，选择此前预测最准的专家。"""
    def forward(self, x, state):
        heads = x.new_tensor([-1., 0., 1.]).expand(len(x), 3)
        return heads, None, (-state[:, -1]).softmax(1)


class TestTRA(unittest.TestCase):
    def test_market_sessions_padding_pre_entry_and_train_only_scaler(self):
        dates, hist, s = fixture()
        hist = hist.loc[~(hist.ticker.eq("b") & hist.trade_date.eq(dates[1]))].copy()
        p = panel(dates, hist, s)
        x, state = p.window([1])  # 股票 b 在第 2 号交易日首次作为成分股产生信号。
        self.assertEqual(tuple(x.shape), (1, LENGTH, 1))
        self.assertEqual(tuple(state.shape), (1, LENGTH - 2, 3))
        self.assertLess(x[0, -3, 0].item(), 0)
        self.assertEqual(x[0, -2, 0].item(), 0)
        np.testing.assert_array_equal(x[0, :-3, 0].cpu(), 0)
        kept = history_keys(hist, s.iloc[:3], dates)
        self.assertEqual(set(kept.trade_date), set(dates[:3]))
        hist.loc[hist.trade_date > dates[6], "x"] = 1e10
        p2 = panel(dates, hist, s)
        pd.testing.assert_frame_equal(p.scaler, p2.scaler)
        train = p.segment([dates[2], dates[6]], supervised=True)
        self.assertEqual(p.samples.iloc[train.rows].trade_date.max(), dates[4])
        self.assertTrue(p.samples.iloc[train.rows].label_end_date.le(dates[6]).all())

    def test_replay_cannot_use_current_or_unrealized_labels(self):
        dates, hist, s = fixture()
        model = SimpleNamespace(model=torch.nn.Identity(), tra=MemoryRouter())

        def predict(samples):
            p = panel(dates, hist, samples)
            return replay(model, p.segment([dates[2], dates[-1]], supervised=False), output_start=dates[2])

        before = predict(s)
        s.loc[s.trade_date.eq(dates[4]), "label"] *= -1
        after = predict(s)
        mask = before.trade_date < dates[6]
        pd.testing.assert_frame_equal(before.loc[mask], after.loc[mask])
        self.assertFalse(before.loc[before.trade_date.eq(dates[6]), "expert"].equals(
            after.loc[after.trade_date.eq(dates[6]), "expert"]))
        self.assertTrue(after.loc[after.trade_date.eq(dates[-1]), "tra_raw"].notna().all())
        s.loc[s.trade_date.eq(dates[4]), "label_end_date"] = dates[5]
        with self.assertRaisesRegex(ValueError, "label_end_date"):
            predict(s)

    def test_long_horizon_releases_only_mature_errors(self):
        dates, history, samples = fixture()
        for offset in (6, 11):
            samples["label_end_date"] = samples.trade_date.map(pd.Series(dates, index=dates).shift(-offset))
            p = Panel(history, samples, ["x"], dates, [dates[2], dates[6]], horizon=offset)
            p.memory[:-1] = 1
            for day, rows in p.samples.groupby("trade_date").groups.items():
                _, state = p.window(list(rows))
                self.assertEqual(state.shape[1], LENGTH - offset)
                self.assertEqual(bool(state.any()), dates.get_loc(day) >= 2 + offset)

    def test_qlib_architecture_and_fit_boundaries(self):
        model = CausalTRA()
        self.assertIsInstance(model.model, RNN)
        self.assertIsInstance(model.tra, TRA)
        self.assertEqual(MODEL["input_size"], 158)
        self.assertEqual(ROUTER["num_states"], 3)
        self.assertFalse(TRAINING["eval_test"])
        train, early = object(), object()
        def fit_stage(train_set, valid_set, test_set, logs, is_pretrain):
            logs["valid"].append({"IC": 0.1})
        with patch.object(model, "_fit", side_effect=fit_stage) as stage:
            model.fit_windows(train, early)
        self.assertEqual([c.kwargs["is_pretrain"] for c in stage.call_args_list], [True, False])
        for c in stage.call_args_list:
            self.assertEqual(c.args[:3], (train, early, None))
        model.model.eval()
        model.tra.eval()
        dev = next(model.model.parameters()).device
        with torch.no_grad():
            heads, _, prob = model.tra(model.model(torch.zeros(2, 60, 158, device=dev)),
                                       torch.zeros(2, 58, 3, device=dev))
        self.assertEqual(tuple(heads.shape), (2, 3))
        torch.testing.assert_close(prob.sum(1), torch.ones(2, device=dev))

    def test_official_two_stage_training_on_synthetic_samples(self):
        dates, hist, s = fixture()
        hist["x"] = hist.ticker.map({"a": -1., "b": .2, "c": .8})
        p = panel(dates, hist, s)
        tiny = dict(MODEL, input_size=1, hidden_size=4, dropout=0.0)
        with patch("src.ml.tra.MODEL", tiny), patch("src.ml.tra.BATCH", 4):
            model = CausalTRA()
            model.n_epochs = 1
            logs = model.fit_windows(p.segment([dates[2], dates[6]], supervised=True),
                                     p.segment([dates[7], dates[-1]], supervised=True))
        self.assertEqual(list(logs.phase), ["pretrain", "router"])
        self.assertTrue(np.isfinite(logs.rank_ic).all())

    def test_five_arms_preserve_pair_and_align_by_stock(self):
        _, hist, _ = fixture()
        tree = hist.iloc[:3].assign(equal=[.1, .2, .9], xgb_equal=[.3, .9, .7],
                                    label=[1, 2, 3], evaluable=True)
        tree["blend_equal"] = (tree.equal + tree.xgb_equal) / 2
        pred = tree[KEYS].assign(tra_raw=[4., 4., 2.]).iloc[::-1]
        scores = fusion(tree, pred)
        np.testing.assert_array_equal(scores.tree_pair, tree.blend_equal)
        expected = pd.concat([tree.equal, tree.xgb_equal, pred.tra_raw], axis=1).rank(pct=True)
        np.testing.assert_allclose(scores.tree_tra, expected.mean(axis=1))
        pairs, ic = diagnostics(scores, topk=2)
        self.assertEqual(len(pairs), 10)
        self.assertEqual(ic.iloc[0].lightgbm, 1)
        self.assertEqual(list(ic.columns), list(ARMS))
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            fusion(tree, pred.iloc[1:])
