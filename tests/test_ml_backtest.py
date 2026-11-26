"""验证固定与滚动训练、预测评价和共用的 Qlib 组合回测。"""

import importlib.util
import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtests
from src.backtest.factor_metrics import prediction_diagnostics, quintile_assignments
from src.backtest.portfolio import performance, qlib_excess_metrics, window_portfolio_returns
from src.config import CONFIG_FILE, load_config, load_tuning
from src.data.research_panel import prepare_panel
from src.ml.dataset import daily_zscore, read_feature_split
from src.ml.models import fit_lightgbm, load_predictions, train_and_predict, training_settings
from src.ml.rolling import (
    check_prediction_windows, fit_window, init_rolling_qlib, merge_rolling_predictions, rolling_windows, train_rolling,
)

FEATURE_NAMES = [f"F{i}" for i in range(158)]


def write_feature_spec(out: Path) -> None:
    """为合成训练样本写入与特征列顺序一致的清单。"""
    spec = [{"name": name} for name in FEATURE_NAMES]
    (out / "alpha158.json").write_text(json.dumps(spec))


class TrainingTests(unittest.TestCase):
    def test_label_horizons_replace_cached_returns_before_cutoff(self):
        from src.config import set_label_days
        from src.data.research_panel import segment_masks

        dates = pd.bdate_range("2020-12-10", periods=18)
        close = pd.DataFrame({"a": np.arange(18) + 10.}, index=dates)
        close.loc[dates[7], "a"] = np.nan
        cached = pd.DataFrame({"trade_date": dates, "ticker": "a", "label": -99.})
        bounds = [dates[0], dates[14]]
        with TemporaryDirectory() as tmp, patch("src.ml.dataset.label_prices", return_value=close):
            path = Path(tmp) / "features.parquet"
            cached.to_parquet(path)
            for days in (1, 5, 10):
                cfg = {}
                set_label_days(cfg, days)
                frame = read_feature_split(path, bounds, "train", cfg)
                self.assertEqual(frame.label_end_date.iloc[0], dates[1 + days])
                self.assertAlmostEqual(frame.label.iloc[0], close.a.iloc[1 + days] / close.a.iloc[1] - 1)
                self.assertTrue(pd.isna(frame.label.iloc[6]))
                valid = segment_masks(frame, {"train": bounds})["train"]
                self.assertEqual(frame.trade_date[valid].max(), dates[13 - days])
                self.assertTrue(frame.label_end_date[valid].le(bounds[1]).all())
            pd.testing.assert_frame_equal(pd.read_parquet(path), cached)

    def test_composite_objective_scales_and_thresholds(self):
        from src.ml.tree_experiment import composite_loss
        self.assertAlmostEqual(composite_loss(.20, 1.5, -.10)["composite_loss"], -.6)
        row = composite_loss(-.30, -1.5, -.30)
        np.testing.assert_allclose((row["P_r"], row["P_s"], row["P_d"], row["P_floor"]), (3.375, 1.5, 1., 10.))
        self.assertAlmostEqual(row["composite_loss"], 17.475)
        at_threshold = composite_loss(-.20, 0., -.15)
        self.assertEqual((at_threshold["P_d"], at_threshold["P_floor"]), (0., 0.))

    def test_regime_weights_use_only_current_members_and_realized_history(self):
        from src.scripts.regime_fusion import indicators

        dates = pd.bdate_range("2022-01-01", periods=80)
        close = pd.Series(100 + np.arange(80), index=dates)
        prices = pd.DataFrame({"x": close, "y": 200 - close}, index=dates)
        members = pd.DataFrame(True, index=dates, columns=prices.columns)
        members.loc[dates[59], "y"] = False
        state = indicators(close, prices, members)
        self.assertEqual(state.loc[dates[59], "breadth20"], 1)
        self.assertEqual(state.loc[dates[60], "breadth20"], .5)
        self.assertEqual(state.loc[dates[58], "D"], .5)
        self.assertEqual(state.loc[dates[59], "D"], .7)
        self.assertTrue(state.iloc[:20][["E","F"]].eq(.5).all().all())
        pd.testing.assert_frame_equal(state.iloc[:61], indicators(close.iloc[:61], prices.iloc[:61], members.iloc[:61]))
        partial = prices.copy()
        partial.loc[dates[57], "x"] = np.nan
        missing = indicators(close, partial, members)
        self.assertEqual(missing.loc[dates[59], "F"], .5)
        self.assertEqual(missing.loc[dates[60], "coverage"], .5)

    def test_walk_forward_windows_and_label_cutoffs(self):
        from src.ml.walk_forward import folds, period

        cfg = load_config()
        recent = folds(cfg)
        self.assertEqual(recent, [dict(train=["2014-01-02", "2020-12-31"], early_stop=["2021-01-01", "2021-12-31"],
                                       validation=["2022-01-01", "2022-12-31"])])
        cfg["walk_forward"].pop("selection_windows")
        windows = folds(cfg)
        self.assertEqual(len(windows), 5)
        for year, window in enumerate(windows, 2017):
            self.assertEqual(window["train"], ["2014-01-02", f"{year-1}-12-31"])
            self.assertEqual(window["early_stop"], [f"{year}-01-01", f"{year}-12-31"])
            self.assertEqual(window["validation"], [f"{year+1}-01-01", f"{year+1}-12-31"])
        frame = pd.DataFrame(dict(trade_date=pd.to_datetime(["2017-01-03", "2018-12-26", "2018-12-28"]),
                                  label_end_date=pd.to_datetime(["2017-01-05", "2018-12-28", "2019-01-02"]),
                                  ticker="stock", feature=[1., 2., 3.], label=[0.1, 0.2, 0.3]))
        with TemporaryDirectory() as tmp:
            frame.to_parquet(Path(tmp) / "features.parquet")
            _, labels, _ = period(Path(tmp), windows[0]["validation"], ["feature"], dict(cfg, training_label="raw"), False)
            _, stop_labels, _ = period(Path(tmp), windows[0]["early_stop"], ["feature"], dict(cfg, training_label="raw"), False)
        np.testing.assert_array_equal(labels, [0.2])
        np.testing.assert_array_equal(stop_labels, [0.1])

    def test_eight_rankers_fit_only_train_and_early_stop(self):
        from src.ml.walk_forward import BASES, fold_models, folds

        cfg = load_config()
        cfg["label_gains"] = cfg["four_model_targets"]
        fold = folds(cfg)[0]
        block = (pd.DataFrame({"feature": [1., 2.]}), np.array([0, 4]), np.array([2]))
        for family in ("xgboost", "lightgbm"):
            module = "xgb" if family == "xgboost" else "lgb"
            matrix = "QuantileDMatrix" if family == "xgboost" else "Dataset"
            with patch("src.ml.walk_forward.period", return_value=block) as reader, patch(
                f"src.ml.walk_forward.{module}.{matrix}"
            ), patch(f"src.ml.walk_forward.{module}.train") as fit:
                fit.return_value.best_iteration = 2
                models = list(fold_models(None, cfg, fold, family, {"nthread": 1}, ["feature"], True))
            self.assertEqual([call.args[1] for call in reader.call_args_list], [fold["train"], fold["early_stop"]])
            self.assertEqual([name for name, _, _ in models], list(BASES))
            self.assertEqual(fit.call_count, 4)
            self.assertEqual([n for _, _, n in models], [3 if family == "xgboost" else 2] * 4)

    def test_cli_selects_requested_model(self):
        from src.scripts.__main__ import main

        with patch("src.ml.tree_experiment.model_run") as train:
            main(["xgboost", "--loss", "rank:pairwise"])
            args, families = train.call_args.args
            self.assertEqual((args.xgboost_loss, families), ("rank:pairwise", ("xgboost",)))
            main(["eight-rankers"])
            self.assertEqual(train.call_args.args[1], ("lightgbm", "xgboost"))
            main(["lightgbm", "--tpe"])
            self.assertEqual(train.call_args.args[1], ("lightgbm",))
        with patch("src.scripts.run_all.main") as study:
            main(["lightgbm", "--loss", "mse", "--workers", "2"])
            self.assertEqual(study.call_args.args[0][-2:], ["--workers", "2"])
            self.assertTrue(study.call_args.kwargs["lightgbm_only"])

    def test_stability_objective_selects_consistent_candidate(self):
        from src.ml.tuning import rank_stage_trials

        table = pd.DataFrame({
            "lightgbm": ["volatile", "steady"] * 2, "xgboost": ["same"] * 4,
            "valid_start": ["2018", "2018", "2019", "2019"], "days": [200, 200, 230, 230],
            "blend_loss": [0.2, 0.45, 0.6, 0.45],
        })
        ordinary, best_mean = rank_stage_trials(table, "depth", "lambdarank")
        stable, best_stable = rank_stage_trials(table, "depth", "lambdarank_stable")
        self.assertEqual(best_mean["lightgbm"], "volatile")
        self.assertEqual(best_stable["lightgbm"], "steady")
        np.testing.assert_allclose(ordinary.loss, [0.4, 0.45])
        np.testing.assert_allclose(stable.std_loss, [0.2, 0.0])
        np.testing.assert_allclose(stable.L_opt, [0.5, 0.45])

    def test_standalone_lightgbm_equal_fixed_predictions(self):
        from src.ml.rank_validation import BASES, equal_ranks
        from src.scripts.__main__ import main

        settings = load_config()
        rng = np.random.default_rng(158)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_feature_spec(root)
            frames = []
            for year in range(2014, 2024):
                frame = pd.DataFrame(rng.normal(size=(210, 158)), columns=FEATURE_NAMES)
                for month in ([1, 4, 7, 10] if year == 2022 else [6]):
                    frames.append(frame.assign(ticker=[f"S{i:03}" for i in range(210)],
                                               trade_date=pd.Timestamp(year, month, 2), label=frame.F0,
                                               label_end_date=pd.Timestamp(year, month, 4)))
            pd.concat(frames).to_parquet(root / "features.parquet", index=False)
            cfg = dict(feature_source=tmp, seed=158, final_early_stop_year=2021, topk=50, n_drop=5,
                       segments=dict(train=["2014-01-01", "2020-12-31"], validation=["2022-01-01", "2022-12-31"],
                                     test=["2023-01-01", "2023-12-31"]), walk_forward=settings["walk_forward"],
                       num_boost_round=2, early_stopping_rounds=1, rebalance_days=1,
                       label_gains=settings["four_model_targets"], rank_pair_tuning=dict(stages=["depth"], depth=[2]),
                       lightgbm=dict(objective="lambdarank", metric="ndcg", eval_at=[50], max_depth=7,
                                     learning_rate=.025, num_leaves=117, lambda_l1=7, lambda_l2=3,
                                     colsample_bytree=.5, subsample=.8, bagging_freq=1, min_data_in_leaf=84,
                                     num_threads=1, verbosity=-1, seed=158))
            config = root / "config.json"
            config.write_text(json.dumps(cfg))
            with patch("src.scripts.train_lightgbm.evaluate_stack") as backtest, patch(
                    "src.ml.walk_forward.xgb.train", side_effect=AssertionError("unexpected XGBoost training")):
                out = root / "fixed"
                main(["lightgbm-equal", "--config", str(config), "--out", str(out)])
                scores, _, _, methods = backtest.call_args.args
                self.assertEqual(methods, ("equal",))
                np.testing.assert_allclose(scores.equal, equal_ranks(scores, list(BASES)))
                self.assertEqual(len(list((out / "final").glob("*/lambdarank.txt"))), 4)
                for mode, early_end, first in (("split", 2021, 2022), ("joint", 2022, 2021)):
                    with self.subTest(mode=mode):
                        folder = root / mode
                        main(["lightgbm-equal", "--config", str(config), "--out", str(folder),
                              "--validation-mode", mode])
                        periods = json.loads((folder / "final/periods.json").read_text())
                        self.assertEqual(periods[0]["end"], "2020-12-31")
                        self.assertEqual(periods[1]["end"], f"{early_end}-12-31")
                        self.assertLessEqual(periods[1]["last_label_end"], periods[1]["end"])
                        scores = backtest.call_args.args[0]
                        self.assertEqual(set(scores.loc[scores.split.eq("validation"), "trade_date"].dt.year),
                                         set(range(first, 2023)))
                        self.assertEqual(set(scores.loc[scores.split.eq("test"), "trade_date"].dt.year), {2023})

    def test_fixed_optuna_selection_and_label_maturity(self):
        from src.config import fixed_validation
        from src.ml import tree_experiment as pair

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_feature_spec(root)
            dates = ["2019-06-01", "2019-12-31", "2020-06-01", "2020-12-31", "2021-06-01", "2021-12-31", "2022-06-01", "2022-12-31"]
            panel = pd.DataFrame(np.random.default_rng(8).normal(size=(210, 158)), columns=FEATURE_NAMES)
            panel["ticker"] = [f"S{i}" for i in range(len(panel))]
            panel["label"] = panel.F0
            pd.concat([panel.assign(trade_date=pd.Timestamp(day), label_end_date=pd.Timestamp(day) +
                                    pd.Timedelta(days=2)) for day in dates]).to_parquet(root / "features.parquet", index=False)
            for mode in ("split", "joint"):
                cfg = load_config()
                fixed_validation(cfg, mode)
                folder = root / mode
                folder.mkdir()
                cfg.update(feature_source=tmp, reference=tmp, topk=50, seed=158,
                           num_boost_round=2, early_stopping_rounds=1, label_gains=cfg["four_model_targets"],
                           lightgbm=dict(cfg["lightgbm_four_equal"]["params"], num_threads=1),
                           xgboost=dict(cfg["xgboost"], nthread=1))
                cfg["rank_pair_tpe"] = dict(lightgbm={"max_depth": ["int", 1, 10]},
                                            xgboost={"max_depth": ["int", 1, 10]}, models="pair",
                                            joint_validation=mode != "split", trials=1, startup_trials=1)
                with self.subTest(mode=mode), patch.object(pair, "train_candidate", wraps=pair.train_candidate) as fit:
                    winner = pair.tune_tpe(folder, cfg)
                    self.assertEqual(set(winner), {"lightgbm", "xgboost"})
                    self.assertEqual(len(fit.call_args_list), 2)
                    for call in fit.call_args_list:
                        train, early = call.args[4]
                        self.assertEqual(len(train[0]), 630)
                        self.assertEqual(len(early[0]), 210 if mode == "split" else 630)
                    daily = pd.read_csv(folder / "validation_loss_daily.csv")
                    eligible = ["2021-06-01", "2021-12-31", "2022-06-01"] if mode == "joint" else ["2022-06-01"]
                    self.assertEqual(daily.trade_date.tolist(), eligible)

    def test_model_entries_train_only_the_selected_rankers(self):
        from src.ml.rank_validation import BASES, equal_ranks
        from src.scripts.__main__ import main

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_feature_spec(root)
            samples = pd.DataFrame(np.random.default_rng(8).normal(size=(211, 158)), columns=FEATURE_NAMES)
            samples = samples.assign(ticker=[f"S{i:03}" for i in samples.index], label=samples.F0 / 100)
            dates = pd.to_datetime(["2020-06-01", "2021-06-01", "2022-06-01", "2023-06-01"])
            pd.concat([samples.assign(trade_date=d, label_end_date=d + pd.Timedelta(days=2))
                       for d in dates], ignore_index=True).to_parquet(root / "features.parquet", index=False)
            settings = json.loads(CONFIG_FILE.read_text())
            settings.update(num_boost_round=2, early_stopping_rounds=1)
            settings["single_xgboost"]["params"]["nthread"] = 1
            settings["lightgbm_four_equal"]["params"]["num_threads"] = 1
            settings["mlp"].update(device="cpu", cpu_threads=1, hidden_sizes=[4], updates=1, evaluate_every=1)
            config = root / "config.json"
            config.write_text(json.dumps(settings))
            for command, loss in (("xgboost", "rank:pairwise"), ("eight-rankers", None), ("mlp", "lambdarank"), ("mlp", "mse")):
                with self.subTest(command=command, loss=loss):
                    folder = root / (command + str(loss).replace(":", "_"))
                    argv = [command, "--config", str(config), "--source", tmp, "--out", str(folder)]
                    if loss:
                        argv.extend(["--loss", loss])
                    with patch("src.ml.tree_experiment.evaluate_signals"), patch("src.scripts.train_mlp.evaluate_signals"):
                        main(argv)
                    frame = pd.read_parquet(folder / "predictions.parquet")
                    self.assertEqual(set(frame.split), {"validation", "test"})
                    if command == "mlp":
                        self.assertEqual(len(list(folder.rglob("mlp.pt"))), 4 if loss == "lambdarank" else 1)
                        if loss == "lambdarank":
                            np.testing.assert_allclose(frame.mlp, equal_ranks(frame, list(BASES)))
                        else:
                            self.assertTrue(np.isfinite(frame.mlp).all())
                    else:
                        eight = command == "eight-rankers"
                        self.assertEqual(len(list(folder.rglob("model.ubj"))), 4)
                        self.assertEqual(len(list(folder.rglob("lambdarank.txt"))), 4 if eight else 0)
                        columns = ["xgb_" + name for name in BASES] + (list(BASES) if eight else [])
                        signal = frame.blend_equal if eight else frame.xgb_equal
                        np.testing.assert_allclose(signal, equal_ranks(frame, columns))

    def test_pair_default_uses_fixed_validation(self):
        from src.ml import tree_experiment as pair

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ref = root / "daily_xgboost_tuned_20261121_172019_402981"
            ref.mkdir()
            cfg = load_config()
            cfg.update(label_gains=cfg["four_model_targets"], segments=dict(
                train=["2014-01-02", "2020-12-31"], validation=["2022-01-01", "2022-12-31"],
                test=["2023-01-01", "2025-12-31"]))
            (ref / "config.json").write_text(json.dumps(cfg))
            with patch.object(pair, "RUNS", root), patch.object(pair, "new_run", return_value=root / "out"), patch.object(
                    pair, "tune", return_value={}) as tune, patch.object(pair, "evaluate"):
                pair.main([])
            actual = tune.call_args.args[1]
            self.assertNotIn("walk_forward", actual)
            self.assertEqual(tuple(actual["rank_pair_tuning"]["stages"]), pair.STAGES)
            self.assertEqual(actual["segments"], cfg["segments"])

    def test_pair_tuning_keeps_incumbent_and_uses_effective_leaf_limits(self):
        from src.ml.tree_experiment import candidates

        grid = {"leaf_cap_fractions": [0.5, 1], "min_child_weight": [1, 10]}
        lightgbm = {"max_depth": 5, "num_leaves": 117}
        xgboost = {"max_depth": 3, "min_child_weight": 10}
        self.assertEqual(candidates("leaves", "lightgbm", lightgbm, grid),
                         [lightgbm, {**lightgbm, "num_leaves": 16}])
        self.assertEqual(candidates("leaves", "xgboost", xgboost, grid),
                         [xgboost, {**xgboost, "max_leaves": 4}])
        self.assertEqual(candidates("min_child", "xgboost", xgboost, grid),
                         [xgboost, {**xgboost, "min_child_weight": 1}])

    def test_xgb_grid_uses_only_training_early_stop_and_validation(self):
        from src.scripts.train_lightgbm import BASES
        from src.scripts.compare_tree_models import train_xgboost

        cfg = load_config()
        cfg.pop("walk_forward")  # 单验证期历史归档
        cfg.update(num_boost_round=4, early_stopping_rounds=2, label_gains=cfg["four_model_targets"])
        cfg["xgboost"].update(max_depth=2, nthread=1, reg_alpha=0, seed=cfg["seed"])
        periods, counts, frames = [], [], {}
        for name, year in (("train", 2020), ("early_stop", 2021), ("validation", 2022)):
            dates = pd.date_range(f"{year}-01-06", periods=3)
            grade = np.tile(np.arange(5), 30)
            frame = pd.DataFrame(dict(trade_date=dates.repeat(50), label_end_date=dates.repeat(50),
                                      grade=grade, ticker=np.tile(np.arange(50).astype(str), 3),
                                      signal=grade + np.sin(np.arange(len(grade))), evaluable=True))
            start, end = str(dates[0].date()), str(dates[-1].date())
            frames[start] = frame
            if name == "validation":
                cfg["segments"][name] = [start, end]
            else:
                periods.append(dict(period=name, start=start, end=end, stock_days=len(frame), last_label_end=end))
                counts.append(pd.crosstab(frame.trade_date, frame.grade).assign(period=name))
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "final").mkdir()
            frame.assign(split="validation", equal=1 / (frame.signal.abs() + 1)).to_parquet(out / "predictions.parquet")
            (out / "final/periods.json").write_text(json.dumps(periods))
            pd.concat(counts).to_csv(out / "final/grade_counts.csv")
            with patch("src.ml.rank_validation.read_period", side_effect=lambda _, bounds, **kwargs: frames[bounds[0]]) as fixed_read:
                fixed_models = train_xgboost(out, cfg, out, ["signal"], out)
            self.assertEqual(set(fixed_models), set(BASES))
            self.assertEqual([call.args[1][0] for call in fixed_read.call_args_list], list(frames)[:2])
            self.assertNotIn("selected_trial", cfg)
            self.assertFalse((out / "trials.csv").exists())
            cfg["xgboost_tuning"] = {"search_space": {"max_depth": [2, 3]}}
            with patch("src.ml.rank_validation.read_period", side_effect=lambda _, bounds, **kwargs: frames[bounds[0]]) as read:
                models = train_xgboost(out, cfg, out, ["signal"], out, tune=True)
            self.assertEqual([call.args[1][0] for call in read.call_args_list], list(frames))
            trials = pd.read_csv(out / "trials.csv")
            losses = pd.read_csv(out / "blend_trials.csv").set_index("trial").blend_loss
            self.assertEqual(len(trials), 2 * len(BASES))
            self.assertEqual(cfg["selected_trial"], losses.idxmin())
            winners = trials.loc[trials.trial.eq(losses.idxmin())].set_index("target")
            self.assertEqual(cfg["xgboost"]["max_depth"], winners.max_depth.iloc[0])
            for name in BASES:
                self.assertEqual(models[name].num_boosted_rounds(), winners.at[name, "rounds"])
                self.assertTrue((out / f"{name}.ubj").is_file())

    def test_zscore_fusion_keeps_daily_score_distances_and_equal_weights(self):
        from src.ml.rank_validation import fuse_rankers

        frame = pd.DataFrame({"trade_date": [1, 1, 1, 2], "a": [1., 2., 15., 200.],
                              "b": [3., 2., 1., 300.], "constant": [4.] * 4})
        score = fuse_rankers(frame, ["a", "b", "constant"], "zscore")
        first = frame.iloc[:3][["a", "b"]]
        expected = ((first - first.mean()) / first.std()).sum(axis=1) / 3
        np.testing.assert_allclose(score.iloc[:3], expected)
        self.assertEqual(score.iloc[3], 0.)

    def test_final_rank_loss_uses_one_list_for_all_targets(self):
        from src.ml.rank_validation import equal_ranks, validation_loss

        frame = pd.DataFrame(dict(trade_date=[1]*4+[2]*2, ticker=list("CABDAC"), grade=[1, 2, 0, 0, 0, 0],
                                  evaluable=[True, True, True, False, False, False],
                                  first=[4, 3, 1, 5, 4, 1], second=[5, 6, 1, 0, 4, 1]))
        ranks = equal_ranks(frame, ["first", "second"])
        self.assertEqual(ranks.iloc[0], .75)
        gains = dict(A=[0, 1, 5], B=[0, 4, 5])
        # 同分时按代码先选 A。强制 C 分数最高后，两种真实增益对应同一张 C 名单。
        self.assertEqual(validation_loss(frame, ranks, gains, 1).loss.iloc[0], 0)
        frame.loc[0, "first"] = 9
        loss = validation_loss(frame, frame["first"], gains, 1)
        self.assertEqual(loss.trade_date.tolist(), [1])
        self.assertAlmostEqual(loss.loss.iloc[0], 1-(1/5+4/5)/2)

    def test_paired_rankings_use_daily_scores_and_ticker_tiebreak(self):
        from src.scripts.compare_tree_models import paired_signals
        from src.scripts.train_lightgbm import BASES

        frame = pd.DataFrame(dict(trade_date=[1, 1, 1, 2, 2, 2], ticker=list("CABABC"), split="test"))
        for name in (*BASES, "equal"):
            frame[name] = [2, 2, 1, 1, 2, 3]
            frame[f"xgb_{name}"] = [3, 2, 1, 3, 2, 1]
        result = paired_signals(frame, 1).query("target == 'equal'").set_index("trade_date")
        # 第一天原模型 A/C 并列，按代码选 A，另一模型选 C。两天都无需未来收益。
        self.assertEqual(result.shared_picks.tolist(), [0, 0])
        np.testing.assert_allclose(result.spearman, [np.sqrt(3) / 2, -1])

    def test_ranking_targets_reset_daily_and_preserve_ties(self):
        from src.scripts.compare_rank_labels import relevance, ranking_diagnostics

        frame = pd.DataFrame(dict(trade_date=[1, 1, 1, 1, 2, 2, 2], ticker=list("ABCDABC"),
                                  label=[3., 2., 2., 0., .2, .1, np.nan], split="test",
                                  evaluable=[True] * 6 + [False]))
        frame["mse"] = [4, 3, 2, 1, 3, 2, 9]
        frame["lambdarank"] = -frame.mse
        self.assertEqual(relevance(frame.loc[frame.evaluable], 2).tolist(), [2, 1, 1, 0, 2, 1])
        daily = ranking_diagnostics(frame, 2).set_index(["trade_date", "strategy"])
        self.assertAlmostEqual(daily.at[(1, "mse"), "ndcg"], 1.)
        self.assertEqual(daily.at[(1, "lambdarank"), "hit_rate"], .5)
        self.assertEqual(daily.at[(2, "mse"), "n_scored"], 2)

    def test_graded_relevance_spreads_the_tail_without_changing_head_evaluation(self):
        from src.scripts.compare_rank_labels import relevance, ranking_diagnostics

        frame = pd.DataFrame(dict(trade_date=[1]*6 + [2]*2, ticker=list("ABCDEFAB"),
                                  label=[6., 5., 4., 3., 2., 1., .1, .1], evaluable=True, split="test"))
        self.assertEqual(relevance(frame, 2, graded=True).tolist(), [3, 2, 1, 1, 0, 0, 3, 3])
        frame["graded"] = -frame.label
        daily = ranking_diagnostics(frame, 2, ("graded",)).set_index("trade_date")
        self.assertEqual(daily.at[1, "hit_rate"], 0.)
        self.assertEqual(daily.at[1, "ndcg"], 0.)

    def test_quantile_target_is_daily_linear_and_keeps_tied_returns_together(self):
        from src.scripts.compare_rank_labels import quantile_relevance

        frame = pd.DataFrame({"trade_date": [1] * 10 + [2] * 5 + [3] * 3,
                              "label": list(range(10)) + [8., 1., 1., 5., 6.] + [.01] * 3})
        expected = {
            5: [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 1, 1, 2, 3, 2, 2, 2],
            10: list(range(10)) + [9, 2, 2, 5, 7, 5, 5, 5],
        }
        shuffled = frame.sample(frac=1, random_state=7)
        for grades, values in expected.items():
            with self.subTest(grades=grades):
                ranked = quantile_relevance(frame, grades)
                self.assertEqual(ranked.tolist(), values)
                pd.testing.assert_series_equal(quantile_relevance(shuffled, grades).sort_index(), ranked)

    def test_bottom_target_and_execution_day_holdings(self):
        from src.scripts.compare_rank_labels import bottom_relevance, holding_bottom_rates

        dates = pd.bdate_range("2024-01-04", periods=3)
        values = pd.DataFrame(dict(trade_date=[dates[0]] * 5 + [dates[1]] * 2,
                                   ticker=list("ABCDEAB"), label=[0., 1., 1., 2., 3., 3., 2.], evaluable=True))
        self.assertEqual(bottom_relevance(values, 2).tolist(), [0, 0, 0, 3, 7, 0, 0])
        held = pd.DataFrame(dict(trade_date=[dates[1]] * 3 + [dates[2]], ticker=list("ADXZ"),
                                 amount=1, strategy="test", offset=0))
        rates = holding_bottom_rates(held, values, dates, 2).set_index("trade_date")
        self.assertEqual(rates.loc[dates[1], ["held", "evaluable", "in_bottom"]].tolist(), [3, 2, 1])
        self.assertEqual(rates.at[dates[1], "held_bottom_rate"], .5)
        self.assertTrue(pd.isna(rates.at[dates[2], "held_bottom_rate"]))

    def test_fixed_grades_include_boundary_ties_and_reset_by_day(self):
        from src.ml.rank_validation import fixed_grades

        label = np.arange(300)
        label[50] = label[49]
        label[-51] = label[-50]
        label[-26] = label[-25]
        frame = pd.DataFrame(dict(trade_date=np.repeat([1, 2], 300), label=np.tile(label, 2)))
        grade = fixed_grades(frame.sample(frac=1, random_state=17)).sort_index()
        np.testing.assert_array_equal(grade[:300], grade[300:])
        self.assertEqual(grade[:300].value_counts().sort_index().tolist(), [51, 49, 100, 49, 51])
        with self.assertRaisesRegex(ValueError, "重叠"):
            fixed_grades(frame.iloc[:150])

    def test_rank_blend_rewards_agreement_and_limits_each_vote(self):
        from src.scripts.compare_tree_models import fit_rank_blend
        from src.scripts.train_lightgbm import BASES, rank_inputs

        oof = pd.DataFrame(dict(trade_date=np.repeat([1, 2], 12), label=np.tile(np.arange(12), 2), evaluable=True))
        for name in BASES:
            oof[name] = -oof.label
        oof["top50"] = oof.label
        w = fit_rank_blend(oof, .15, .35)
        self.assertAlmostEqual(w.sum(), 1)
        self.assertAlmostEqual(w["top50"], .35)
        self.assertTrue(w.between(.15 - 1e-9, .35 + 1e-9).all())
        ranks = rank_inputs(oof)
        self.assertTrue((ranks @ w).between(ranks.min(axis=1), ranks.max(axis=1)).all())
        pd.testing.assert_series_equal(w, fit_rank_blend(oof.assign(label=oof.label ** 3), .15, .35))

    def test_stack_ranks_include_stocks_with_no_future_label(self):
        from src.scripts.train_lightgbm import BASES, rank_inputs

        scores = pd.DataFrame(dict(trade_date=[1, 1, 1, 2], evaluable=[True, False, True, True]))
        for name in BASES:
            scores[name] = [2., 3., 2., -1.]
        ranks = rank_inputs(scores)
        self.assertEqual(list(ranks.columns), list(BASES))
        self.assertEqual(ranks.top50.tolist(), [.5, 1., .5, 1.])

    def test_experiment_sections_keep_single_and_ensemble_settings_independent(self):
        document = json.loads(CONFIG_FILE.read_text())
        document["single_lightgbm"]["params"]["max_depth"] = 2
        document["lightgbm_four_equal"]["params"]["max_depth"] = 4
        document["single_xgboost"]["params"]["max_depth"] = 6
        document["lightgbm_xgboost_blend"]["xgboost_weight"] = 0.25
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(document))
            config = load_config(path)
            self.assertEqual(training_settings(config)["lightgbm_params"]["max_depth"], 2)
            self.assertEqual(config["lightgbm_four_equal"]["params"]["max_depth"], 4)
            self.assertEqual(config["xgboost"]["max_depth"], 6)
            for section in ("lightgbm_tuning", "xgboost_tuning", "rank_pair_tuning",
                            "ridge_strong_alphas", "lambdarank_bottom_gaps"):
                self.assertNotIn(section, config)
                self.assertTrue(load_tuning(section, config))
            self.assertEqual(config["lightgbm_xgboost_blend"]["xgboost_weight"], 0.25)
            path.write_text(json.dumps(config))
            self.assertEqual(load_config(path), config)

    def test_tuning_override_and_fixed_export_of_archived_grid(self):
        from src.config import write_best_config

        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            path = out / "tuning.json"
            custom_grid = {"search_space": {"min_child_weight": [75]}}
            path.write_text(json.dumps({"xgboost_tuning": custom_grid}))
            config = load_config()
            old_grid = {"search_space": {"min_child_weight": [250]}}
            config["xgboost_tuning"] = old_grid
            self.assertEqual(load_tuning("xgboost_tuning", config), old_grid)
            self.assertEqual(load_tuning("xgboost_tuning", config, path), custom_grid)
            with self.assertRaisesRegex(ValueError, "lightgbm_tuning"):
                load_tuning("lightgbm_tuning", config, path)
            config["lightgbm_tuning"] = {"search_space": {"max_depth": [3]}}
            write_best_config(out, config)
            self.assertEqual(load_config(out / "best_config.json"), load_config())
            self.assertEqual(config["xgboost_tuning"], old_grid)

    def test_unified_configuration_keeps_training_controls_out_of_model_params(self):
        config = load_config()
        settings = training_settings(config)
        self.assertEqual(settings["lightgbm_params"]["seed"], config["seed"])
        self.assertEqual(settings["ridge_loss"], "mse")
        self.assertEqual(settings["ridge_solver"], config["ridge_solver"])
        for key in ("num_boost_round", "early_stopping_rounds", "log_evaluation_period"):
            self.assertEqual(settings["training_control"][key], config[key])
            self.assertNotIn(key, settings["lightgbm_params"])

    def test_ridge_loss_defaults_to_mse_and_rejects_other_losses(self):
        config = load_config()
        config.pop("ridge_loss")
        with TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            path.write_text(json.dumps(config))
            self.assertEqual(load_config(path)["ridge_loss"], "mse")

            config["ridge_loss"] = "mae"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "ridge_loss 必须为 mse"):
                load_config(path)

    def test_strong_ridge_selects_on_validation_and_scores_missing_test_labels(self):
        import joblib
        from src.ml.models import compare_ridge

        days = pd.bdate_range("2020-01-02", periods=18)
        config = dict(load_config(), ridge_alpha=1, ridge_strong_alphas=[10, 100],
                      segments={name: [str(days[i].date()), str(days[i+5].date())]
                                for name, i in (("train", 0), ("validation", 6), ("test", 12))})
        config.pop("walk_forward")  # 验证旧实验的单窗选参和缺失标签行为
        rng = np.random.default_rng(15)
        panel = pd.DataFrame(rng.normal(size=(180, 158)).astype(np.float32), columns=FEATURE_NAMES)
        panel = panel.assign(trade_date=days.repeat(10), ticker=np.tile(list("ABCDEFGHIJ"), 18),
                             label=panel.F0 * .01, label_end_date=pd.Series(days).shift(-2).repeat(10).to_numpy())
        panel.loc[panel.trade_date.ge(days[12]), "label"] = np.nan
        panel.loc[::4, "F1"] = np.nan
        with TemporaryDirectory() as temp:
            out = Path(temp)
            write_feature_spec(out)
            panel.to_parquet(out / "features.parquet", index=False)
            with redirect_stdout(StringIO()):
                scores = compare_ridge(config, out, out)
            trials = pd.read_csv(out / "ridge_trials.csv").set_index("alpha")
            winner = trials.validation_mse.idxmin()
            self.assertEqual(winner, json.loads((out / "best_config.json").read_text())["ridge_alpha"])
            self.assertTrue(trials.coefficient_norm.is_monotonic_decreasing)
            valid = scores.loc[scores.split.eq("validation") & scores.evaluable]
            targets = daily_zscore(valid.label, valid.trade_date)
            self.assertAlmostEqual(((valid.ridge_strong - targets) ** 2).mean(),
                                   trials.validation_mse.min(), places=6)
            test = scores.loc[scores.split.eq("test")]
            self.assertEqual(len(test), 60)
            self.assertFalse(test.evaluable.any())
            self.assertEqual(list(trials.index), [10, 100])
            self.assertNotIn("ridge_base", scores)
            model = joblib.load(out / "ridge.joblib")
            np.testing.assert_allclose(model.predict(panel.loc[panel.trade_date.ge(days[12]), FEATURE_NAMES]),
                                       test.ridge_strong, atol=1e-6)

    def test_zscore_uses_each_days_sample_deviation_and_excludes_degenerate_days(self):
        dates = pd.Series(pd.to_datetime(["2020-01-02"] * 4 + ["2020-01-03", "2020-01-06", "2020-01-06"]))
        labels = pd.Series([.1, .2, .3, np.nan, .01, .02, .02])
        actual = daily_zscore(labels, dates)
        np.testing.assert_allclose(actual.iloc[:3], [-1., 0., 1.], atol=1e-12)
        self.assertTrue(actual.iloc[3:].isna().all())

    def test_config_accepts_training_label_modes_and_legacy_name(self):
        config = load_config()
        with TemporaryDirectory() as temp:
            path = Path(temp) / "config.json"
            for mode, expected in (("raw", "raw"), ("zscore", "zscore"), ("daily_sample_zscore", "zscore")):
                path.write_text(json.dumps(dict(config, training_label=mode)))
                self.assertEqual(load_config(path)["training_label"], expected)
            path.write_text(json.dumps(dict(config, training_label="invalid")))
            with self.assertRaisesRegex(ValueError, "training_label"):
                load_config(path)

    def test_optuna_grid_uses_qlib_fit_and_selects_by_validation_mse(self):
        import optuna
        from qlib.contrib.model.gbdt import LGBModel
        from qlib.workflow.recorder import MLflowRecorder

        from src.ml.tuning import tune_lightgbm

        dates = pd.bdate_range("2024-01-02", periods=15)
        segments = {name: [str(dates[start].date()), str(dates[start + 4].date())]
                    for name, start in (("train", 0), ("validation", 5), ("test", 10))}
        config = dict(load_config(), segments=segments, training_label="zscore", num_boost_round=6,
                      early_stopping_rounds=2, log_evaluation_period=0,
                      lightgbm_tuning=dict(search_space={"lambda_l1": [0., 30.], "max_depth": [2, 3, 2],
                                                         "colsample_bytree": [0.5], "subsample": [0.8]}),
                      lightgbm=dict(objective="regression", metric="l2", lambda_l1=200., max_depth=8,
                                    colsample_bytree=1., subsample=1., bagging_freq=1,
                                    min_data_in_leaf=2, num_threads=1, verbosity=-1))
        config.pop("walk_forward")  # 复核旧版 Optuna 单窗流程
        ticker = np.tile(np.arange(5), len(dates))
        frame = pd.DataFrame({name: (ticker + i).astype(np.float32) for i, name in enumerate(FEATURE_NAMES)}).assign(
            trade_date=dates.repeat(5), ticker=ticker.astype(str), label=ticker * .01,
            label_end_date=pd.Series(dates).shift(-2).repeat(5).to_numpy(),
        )
        frame.loc[frame.trade_date.ge(dates[10]), FEATURE_NAMES[0]] = np.inf
        frame = frame.sort_values(["ticker", "trade_date"])
        original_fit = LGBModel.fit
        output = StringIO()
        with TemporaryDirectory() as temp:
            out = Path(temp)
            write_feature_spec(out)
            frame.to_parquet(out / "features.parquet", index=False)
            with patch("src.ml.tuning.read_feature_split", wraps=read_feature_split) as reader, patch.object(
                LGBModel, "fit", autospec=True, side_effect=original_fit,
            ) as fit, patch.object(MLflowRecorder, "_log_uncommitted_code", return_value=None), redirect_stdout(output):
                tune_lightgbm(config, out)
            self.assertEqual([call.args[2] for call in reader.call_args_list], ["train", "validation"])
            self.assertEqual(fit.call_count, 5)
            dataset = fit.call_args.args[1]
            self.assertEqual(set(dataset.segments), {"train", "valid"})
            for segment, begin in (("train", 0), ("valid", 5)):
                expected = pd.MultiIndex.from_frame(frame.loc[
                    frame.trade_date.between(dates[begin], dates[begin + 2]), ["trade_date", "ticker"]
                ], names=["datetime", "instrument"])
                pd.testing.assert_index_equal(dataset.prepare(segment).index, expected)
            trials = pd.read_csv(out / "trials.csv")
            lines = output.getvalue().splitlines()
            self.assertEqual(len(lines), 5)
            self.assertEqual(output.getvalue().count("Train Mean Rank IC="), 5)
            self.assertEqual(output.getvalue().count("Valid Mean Rank IC="), 5)
            for line, row in zip(lines, trials.itertuples(index=False), strict=True):
                self.assertIn(f"Train MSE={row.train_mse:.8g} | Valid MSE={row.validation_mse:.8g}", line)
            tested = list(zip(trials.lambda_l1, trials.max_depth))
            self.assertEqual(tested[0], (200., 8))
            self.assertCountEqual(tested[1:], product([0., 30.], [2, 3]))
            self.assertEqual(tested, [(call.args[0].params["lambda_l1"], call.args[0].params["max_depth"])
                                      for call in fit.call_args_list])
            best = trials.loc[trials.validation_mse.idxmin()]
            self.assertGreater(best.trial, 0)
            self.assertTrue(np.isfinite([best.train_rank_ic, best.validation_rank_ic]).all())
            best_params = load_config(out / "best_config.json")["lightgbm"]
            self.assertEqual(best_params["lambda_l1"], best.lambda_l1)
            self.assertEqual(best_params["max_depth"], best.max_depth)
            self.assertEqual(best_params["subsample"], best.subsample)
            self.assertEqual(best_params["colsample_bytree"], best.colsample_bytree)
            for call, row in zip(fit.call_args_list, trials.itertuples(), strict=True):
                model_text = call.args[0].model.model_to_string()
                self.assertIn(f"[feature_fraction: {row.colsample_bytree:g}]", model_text)
                self.assertIn(f"[bagging_fraction: {row.subsample:g}]", model_text)
                self.assertIn("[bagging_freq: 1]", model_text)
            curve = pd.read_csv(out / "best_training_mse.csv")
            self.assertAlmostEqual(curve.validation_mse.min(), best.validation_mse)
            study = optuna.load_study(study_name="alpha158_lightgbm", storage=f"sqlite:///{out / 'optuna.sqlite3'}")
            self.assertAlmostEqual(study.best_value, best.validation_mse)

    def test_sampling_grid_rejects_invalid_rates_and_disabled_bagging(self):
        from src.ml.tuning import tune_lightgbm

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            for key in ("colsample_bytree", "subsample"):
                for value in (0., 1.01, True):
                    config = load_config()
                    config["lightgbm_tuning"] = {"search_space": {key: [value]}}
                    path.write_text(json.dumps(config))
                    with self.assertRaisesRegex(ValueError, key):
                        load_config(path)
            config = load_config()
            config["lightgbm_tuning"] = load_tuning("lightgbm_tuning", config)
            config["lightgbm"]["bagging_freq"] = 0
            with self.assertRaisesRegex(ValueError, "bagging_freq"):
                tune_lightgbm(config, Path(tmp))

    @unittest.skipUnless(importlib.util.find_spec("lightgbm") and importlib.util.find_spec("sklearn"),
                         "Requires the pinned ML environment")
    def test_repeated_fit_ignores_cross_boundary_and_holdout_labels(self):
        dates = pd.bdate_range("2023-01-03", periods=30)
        periods = {name: [str(dates[lo].date()), str(dates[lo + 9].date())]
                   for name, lo in (("train", 0), ("validation", 10), ("test", 20))}
        day = np.repeat(np.arange(len(dates)), 5)
        ticker = np.tile(np.arange(5), len(dates))
        raw = ticker.astype(np.float32)
        sample = pd.DataFrame({name: raw + i for i, name in enumerate(FEATURE_NAMES)})
        sample = sample.assign(trade_date=dates[day], ticker=ticker.astype(str), label=.005 * ticker,
                               label_end_date=pd.Series(dates).shift(-6).iloc[day].to_numpy())
        sample.loc[sample.label_end_date.isna(), "label"] = np.nan
        sample.loc[sample.trade_date.eq(dates[10]), "label"] = .01
        config = dict(segments=periods, seed=158, ridge_alpha=100.,
                      num_boost_round=8, early_stopping_rounds=3,
                      lightgbm=dict(objective="regression", metric="l2", min_data_in_leaf=2, num_threads=1,
                                    verbosity=-1, deterministic=True, force_col_wise=True))
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for mode, relabel in product(("zscore", "raw"), (False, True)):
                config["training_label"] = mode
                use_raw = mode == "raw"
                out = root / f"{mode}_{relabel}"
                out.mkdir()
                data = sample.copy()
                if relabel:
                    changed = (data.trade_date.le(dates[9]) & data.label_end_date.gt(dates[9])
                               | data.trade_date.ge(dates[20])) & data.label.notna()
                    data.loc[changed, "label"] *= -100.
                write_feature_spec(out)
                data.to_parquet(out / "features.parquet", index=False)
                predictions = train_and_predict(config, out)
                log = json.loads((out / "training_log.json").read_text())
                self.assertEqual(log["split_audit"]["train"]["fitted_rows"], 20)
                self.assertEqual(log["split_audit"]["validation"]["fitted_rows"], 20 if use_raw else 15)
                self.assertEqual(log["training_label"], mode)
                self.assertEqual(log["ridge_training_label"], mode)
                curve = pd.read_csv(out / "training_mse.csv")
                best = curve.loc[curve.validation_mse.idxmin()]
                self.assertEqual(log["best_iteration"], best.iteration)
                valid = predictions.loc[predictions.trade_date.between(dates[10 if use_raw else 11], dates[13])]
                original = .005 * valid.ticker.astype(int)
                original = original.mask(valid.trade_date.eq(dates[10]), .01)
                target = original if use_raw else (valid.ticker.astype(int) - 2) / np.sqrt(2.5)
                self.assertAlmostEqual(log["ridge_mse"]["validation"], ((valid.ridge - target) ** 2).mean())
                self.assertAlmostEqual(best.validation_mse, ((valid.lightgbm - target) ** 2).mean())
                expected_rank_ic = np.mean([group.lightgbm.corr(group.label, method="spearman")
                                           for _, group in valid.groupby("trade_date") if group.label.nunique() > 1])
                self.assertAlmostEqual(log["validation_rank_ic"], expected_rank_ic)
                self.assertTrue(np.isfinite(log["train_rank_ic"]))
                np.testing.assert_allclose(valid.label, original)
                scores = predictions[["trade_date", "ticker", "lightgbm", "ridge"]]
                self.assertGreater(scores.lightgbm.nunique(), 1)
                self.assertGreater(scores.ridge.max(), .01 if use_raw else 1.)
                if use_raw:
                    self.assertGreater(scores.ridge.min(), 0.)
                else:
                    self.assertLess(scores.ridge.min(), -1.)
                if relabel:
                    pd.testing.assert_frame_equal(scores, expected)
                else:
                    expected = scores
            restored = load_predictions(dict(config, log_evaluation_period=50), out)
            pd.testing.assert_frame_equal(restored, predictions.reset_index(drop=True))
            with self.assertRaisesRegex(ValueError, "different training configuration"):
                load_predictions(dict(config, training_label="zscore"), out)


def rolling_fixture():
    """创建带休市日和多余未来日的日历，留出 11 个全局测试日。"""
    dates = pd.bdate_range("2020-01-02", periods=75).delete([5, 9, 31, 49, 53])
    segments = {name: [str(dates[start].date()), str(dates[end].date())]
                for name, start, end in (("train", 0, 26), ("validation", 27, 43), ("test", 44, 54))}
    config = dict(segments=segments, qlib_version="0.9.7", label_start_days=1, label_end_days=6, seed=158,
                  ridge_alpha=100., training_label="zscore", num_boost_round=8, early_stopping_rounds=3,
                  lightgbm=dict(objective="regression", metric="l2", num_threads=1, verbosity=-1, min_data_in_leaf=2))
    return dates, config


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ("qlib", "mlflow", "pymongo", "fire")),
                     "Requires requirements.txt")
class RollingWindowTests(unittest.TestCase):
    def test_rank_pair_expands_from_fixed_start_and_preserves_partial_test_years(self):
        from src.ml.tree_experiment import expanding_years

        windows = dict(expanding_years({"train": ["2014-01-02", "2020-12-31"],
                                        "test": ["2023-04-03", "2025-08-20"]}))
        self.assertEqual(list(windows), [2023, 2024, 2025])
        self.assertEqual(windows[2023]["test"][0], "2023-04-03")
        self.assertEqual(windows[2025]["test"][1], "2025-08-20")
        for year, window in windows.items():
            self.assertEqual(window["train"], ["2014-01-02", f"{year-2}-12-31"])
            self.assertEqual(window["early_stop"], [f"{year-1}-01-01", f"{year-1}-12-31"])
            self.assertLess(window["early_stop"][1], window["test"][0])
        with self.assertRaisesRegex(ValueError, "冻结树参数"):
            list(expanding_years({"train": ["2014-01-02", "2020-12-31"], "validation": ["2022-01-01", "2023-06-01"],
                                  "test": ["2023-04-03", "2025-08-20"]}))

    def test_sliding_pair_advances_both_training_boundaries(self):
        from src.ml.tree_experiment import expanding_years

        segments = {"train": ["2014-01-02", "2020-12-31"], "test": ["2023-01-01", "2025-12-31"]}
        expanding = dict(expanding_years(segments))
        sliding = dict(expanding_years(segments, sliding=True))
        self.assertEqual(sliding[2023], expanding[2023])
        for year in sliding:
            self.assertEqual(sliding[year]["train"], [f"{year-9}-01-02", f"{year-2}-12-31"])
            self.assertEqual(sliding[year]["early_stop"], expanding[year]["early_stop"])
            self.assertEqual(sliding[year]["test"], expanding[year]["test"])

    def test_expanding_pair_purges_late_training_labels_and_scores_unlabeled_stocks(self):
        from src.ml.rank_validation import BASES
        from src.ml.tree_experiment import expanding_window, expanding_years

        bounds = dict(expanding_years({"train": ["2014-01-02", "2020-12-31"],
                                       "test": ["2023-01-01", "2023-12-31"]}))[2023]
        days = pd.to_datetime(["2014-01-02", "2021-12-31", "2022-06-01", "2022-12-30", "2023-01-03"])
        ends = pd.to_datetime(["2014-01-06", "2022-01-05", "2022-06-06", "2023-01-04", None])
        frames = [pd.DataFrame(dict(trade_date=day, ticker=np.arange(220), label_end_date=end,
                                    feature=i*1000 + np.arange(220), label=np.arange(220, dtype=float)))
                  for i, (day, end) in enumerate(zip(days, ends))]
        frames[-1]["label"] = np.nan

        def fit(folder, family, params, cfg, blocks, matrices, features):
            self.assertEqual([int(x.feature.min()) for x, _, _ in blocks], [0, 2000])
            self.assertEqual([groups.tolist() for _, _, groups in blocks], [[220], [220]])
            self.assertEqual(features.columns.tolist(), ["feature"])
            self.assertEqual(features.feature.min(), 4000)
            return pd.DataFrame({name: features.feature for name in BASES})

        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            pd.concat(frames, ignore_index=True).to_parquet(out / "features.parquet")
            cfg = dict(feature_source=tmp, lightgbm={}, xgboost={})
            with (patch("src.ml.tree_experiment.train_candidate", side_effect=fit) as train,
                  patch("src.ml.tree_experiment.quantile_matrices", return_value=[])):
                scores = expanding_window(out, cfg, 2023, bounds, ["feature"])
            self.assertEqual(train.call_count, 2)
            self.assertTrue(scores.label.isna().all())
            np.testing.assert_allclose(scores["expanding"], np.arange(1, 221) / 220)

    def test_window_count_uses_configuration(self):
        dates, config = rolling_fixture()
        config["rolling_test_windows"] = 1
        with TemporaryDirectory() as temp:
            init_rolling_qlib(dates, "0.9.7", Path(temp))
            windows, step = rolling_windows(dates, config, dates[2])
        self.assertEqual(step, 11)
        self.assertEqual([window["test_sessions"] for window in windows], [11])

    def test_slide_and_truncate_on_independent_calendar(self):
        dates, config = rolling_fixture()
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
        dates, config = rolling_fixture()
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

        dates, config = rolling_fixture()
        config["training_label"] = "raw"
        day = np.repeat(np.arange(2, len(dates)), 5)
        ticker = np.tile(np.arange(5), len(dates)-2)
        series = pd.Series(dates)
        values = (ticker + .01 * day).astype(np.float32)
        rows = pd.DataFrame({name: values + i for i, name in enumerate(FEATURE_NAMES)}).assign(
            ticker=ticker.astype(str), trade_date=dates[day], feature_date=dates[day], source_snapshot_date=dates[day],
            label_start_date=series.shift(-1).iloc[day].to_numpy(),
            label_end_date=series.shift(-6).iloc[day].to_numpy(), label=.01 * ticker,
        )
        rows["label"] = rows.label.where(rows.label_end_date.notna())
        with TemporaryDirectory() as temp:
            root = Path(temp)
            out = root / "run"
            out.mkdir()
            write_feature_spec(out)
            audit = json.dumps(dict(first_feature_date=str(dates[2].date())))
            (out / "data_audit.json").write_text(audit)
            rows.to_parquet(out / "features.parquet", index=False)
            (out / "predictions.parquet").write_bytes(b"fixed predictions")
            models, chunks = [], []

            def capture(*args, **kwargs):
                """记录每次实际拟合返回的模型实例。"""
                model, curve, rank_metrics = fit_lightgbm(*args, **kwargs)
                np.testing.assert_allclose(np.unique(args[0].prepare("valid", col_set="label")), [0., .01, .02, .03, .04])
                models.append(model)
                return model, curve, rank_metrics

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
                self.assertEqual(log["training_label"], "raw")
                self.assertIn("train_rank_ic", log)
                self.assertIn("validation_rank_ic", log)
                for split, boundary in (("train", "validation"), ("validation", "test")):
                    self.assertLess(pd.Timestamp(log["fit_audit"][split]["latest_label_end"]),
                                    pd.Timestamp(window["raw"][boundary][0]))
                self.assertEqual(log["fitted_preprocessors"], [])
            pd.testing.assert_frame_equal(merged.reindex(columns=chunks[0].columns),
                                          pd.concat(chunks, ignore_index=True), check_exact=True)
            self.assertTrue(np.isfinite(merged.rolling_lightgbm).all())
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
    def test_window_excess_return_includes_first_day_and_averages_all_offsets(self):
        dates = pd.bdate_range("2024-01-02", periods=4)
        windows = [dict(window=i + 1, effective={"test": [dates[2*i], dates[2*i+1]]}, test_sessions=2)
                   for i in range(2)]
        ledger = pd.concat([
            pd.DataFrame(dict(trade_date=dates, strategy=strategy, offset=offset,
                              daily_return=np.array([.02, 0., -.02, .01] if offset == 0 else [0., .01, .02, 0.])
                              + (.01 if strategy == "rolling_lightgbm" else 0.),
                              benchmark_return=[.005, .005, .003, -.001]))
            for strategy in ("lightgbm", "rolling_lightgbm") for offset in (0, 1)
        ], ignore_index=True).iloc[::-1]
        results = window_portfolio_returns(ledger, windows, rebalance_days=2)
        np.testing.assert_allclose(results.annualized_excess_return, 238 * np.array([.0025, .0125, .0015, .0115]))
        self.assertTrue(results.portfolio_days.eq(2).all())
        self.assertTrue(results.n_tracks.eq(2).all())
        missing = ledger.loc[~(ledger.trade_date.eq(dates[0]) & ledger.strategy.eq("lightgbm") & ledger.offset.eq(0))]
        with self.assertRaises(ValueError):
            window_portfolio_returns(missing, windows, rebalance_days=2)
        with self.assertRaises(ValueError):
            window_portfolio_returns(ledger.loc[ledger.offset.eq(0)], windows, rebalance_days=2)

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


class PredictionMetricsTests(unittest.TestCase):
    def test_ic_metrics_follow_qlib_with_small_and_missing_label_groups(self):
        from qlib.contrib.eva.alpha import calc_ic

        dates = pd.bdate_range("2024-01-02", periods=3)
        keys = pd.MultiIndex.from_product([dates, ["A", "B", "C"]], names=["datetime", "instrument"])
        scores = pd.Series([1., 2., 3., 1., 4., 5., 1., 2., 3.], index=keys)
        labels = pd.Series([1., 10., 2., 3., np.nan, 1., np.nan, np.nan, np.nan], index=keys)
        expected_ic, expected_rank = calc_ic(scores, labels)
        alpha = scores.copy()
        alpha.iloc[[0, 3]] = [np.nan, np.inf]
        predictions = scores.rename("lightgbm").reset_index().rename(
            columns={"datetime": "trade_date", "instrument": "ticker"},
        ).assign(split="test", label=labels.fillna(100.).to_numpy(), evaluable=labels.notna().to_numpy(),
                 Alpha3=alpha.to_numpy())
        config = {"segments": {"test": ["2024-01-02", "2024-01-04"]}}
        with TemporaryDirectory() as temp:
            out = Path(temp)
            daily = prediction_diagnostics(predictions, config, dates, out,
                                           strategies_by_split={"test": ("lightgbm", "Alpha3")})
            summary = pd.read_csv(out / "rank_ic_summary.csv").query('year == "all" and strategy == "lightgbm"').iloc[0]
            groups = pd.read_csv(out / "quintile_daily.csv", parse_dates=["trade_date"])

        model = daily.loc[daily.strategy.eq("lightgbm")]
        np.testing.assert_allclose(model.ic, expected_ic, equal_nan=True)
        np.testing.assert_allclose(model.rank_ic, expected_rank, equal_nan=True)
        self.assertEqual(model.n_pairs.to_list(), [3, 2, 0])
        factor = daily.loc[daily.strategy.eq("Alpha3")]
        expected_alpha_ic, expected_alpha_rank = calc_ic(alpha.where(np.isfinite(alpha)), labels)
        np.testing.assert_allclose(factor.ic, expected_alpha_ic, equal_nan=True)
        np.testing.assert_allclose(factor.rank_ic, expected_alpha_rank, equal_nan=True)
        self.assertEqual(factor.n_pairs.to_list(), [2, 1, 0])
        first_day = groups.loc[groups.trade_date.eq(dates[0])]
        self.assertEqual(first_day.groupby("strategy").n_assigned.sum().to_dict(), {"Alpha3": 2, "lightgbm": 2})
        self.assertEqual(summary.n_pairs, 5)
        self.assertAlmostEqual(summary.mean_ic, expected_ic.mean())
        self.assertAlmostEqual(summary.icir, expected_ic.mean() / expected_ic.std())
        self.assertAlmostEqual(summary.mean_rank_ic, expected_rank.mean())
        self.assertAlmostEqual(summary.rank_icir, expected_rank.mean() / expected_rank.std())
        self.assertEqual(summary.positive_rate, expected_rank.dropna().gt(0).mean())

    def test_five_groups_use_score_universe_before_label_lookup(self):
        day = pd.Timestamp("2023-01-03")
        frame = pd.DataFrame(dict(trade_date=day, ticker=list("ABCDE"), lightgbm=[3., 1., 5., 2., 4.],
                                  label=[0., 1., np.nan, 2., 3.]))
        groups = quintile_assignments(frame, "lightgbm").set_index("ticker").group.to_dict()
        self.assertEqual(groups, {"B": 1, "D": 2, "A": 3, "E": 4, "C": 5})
        frame["label"] = np.nan
        self.assertEqual(quintile_assignments(frame, "lightgbm").set_index("ticker").group.to_dict(), groups)


class SharedBacktestTests(unittest.TestCase):
    def test_diagnostic_controls_do_not_enter_test_portfolios(self):
        from src.backtest import engine

        dates = pd.bdate_range("2024-01-02", periods=3)
        scores = pd.DataFrame({"trade_date": dates, "split": ["validation", "test", "test"],
                               "new": [1, 2, 3], "control": [3, 2, 1]})
        methods = ("new", "control")
        with patch.object(engine, "load_market_calendar", return_value=dates), \
             patch.object(engine, "prepare_test_panel", return_value=(None, dates[1:])), \
             patch.object(engine, "prediction_diagnostics") as diagnostics, patch.object(engine, "run_backtests") as backtest:
            result = engine.evaluate_signals(scores, {}, Path("."), methods, portfolio_methods=("new",))
        pd.testing.assert_index_equal(result, dates[1:])
        self.assertEqual(diagnostics.call_args.kwargs["strategies_by_split"], {"validation": methods, "test": methods})
        pd.testing.assert_frame_equal(backtest.call_args.args[3], scores.iloc[1:])
        self.assertEqual(backtest.call_args.kwargs["strategies"], ("new",))

    def test_native_topk_uses_close_lots_fees_and_includes_first_benchmark_return(self):
        dates = pd.bdate_range("2024-01-02", periods=7)
        prices = pd.DataFrame({"A": 10. + .2 * np.arange(7), "B": [10., 11., 11., 11., 11., 11., 11.],
                               "C": 10., "D": 10.}, index=dates)
        raw = prices.stack().rename("close").rename_axis(["trade_date", "ticker"]).reset_index()
        panel = prepare_panel(raw.assign(open=raw.close, high=raw.close + 2, low=raw.close - 2,
                                         vol=100., amount=10 * (raw.close - 1), adj_factor=1.), dates)
        scores = raw[["trade_date", "ticker"]].assign(model=[4., 3., 2., 1.] * 2 + [1., 2., 3., 4.] * 5)
        scores["factor"] = scores.model
        scores["sparse"] = scores.model.where(scores.trade_date.eq(dates[1]) & scores.ticker.eq("C"))
        scores.loc[scores.trade_date.eq(dates[1]) & scores.ticker.eq("A"), "sparse"] = np.inf
        config = dict(load_config(), topk=2, n_drop=1, rebalance_days=2, initial_capital_cny=100_000)
        with TemporaryDirectory() as temp:
            out = Path(temp)
            pd.DataFrame({"trade_date": dates.strftime("%Y%m%d"), "close": 100.,
                          "pre_close": [98., *([100.] * 6)]}).to_csv(out / "index.csv", index=False)
            run_backtests({"limits": out / "limits.parquet", "benchmark": out / "index.csv"},
                          config, dates, scores, panel, out, strategies=("model", "factor", "sparse"))
            frames = [pd.read_csv(out / "portfolio_metrics.csv")]
            frames += [pd.read_parquet(out / name) for name in ("ledger.parquet", "holdings.parquet", "trades.parquet")]
            coverage = json.loads((out / "execution_audit.json").read_text())["candidate_coverage"]
        self.assertEqual(coverage["model"]["scored_stock_days"], len(scores))
        self.assertEqual(coverage["sparse"], dict(scored_stock_days=1, min_daily_candidates=0, max_daily_candidates=1))
        sparse_trades = frames[-1].query('strategy == "sparse" and deal_amount != 0')
        self.assertEqual(sparse_trades[["trade_date", "ticker"]].values.tolist(), [[dates[2], "C"]])
        for frame in frames:
            model, factor = (frame.loc[frame.strategy.eq(name)].drop(columns="strategy").reset_index(drop=True)
                             for name in ("model", "factor"))
            self.assertFalse(model.empty)
            pd.testing.assert_frame_equal(model, factor)
        ledger = frames[1].query('strategy == "factor" and offset == 0')
        self.assertAlmostEqual(ledger.benchmark_return.iloc[0], 100 / 98 - 1)
        self.assertAlmostEqual(ledger.daily_return.iloc[0], 0.)
        benchmark = frames[0].loc[frames[0].strategy.eq("benchmark")].iloc[0]
        self.assertAlmostEqual(benchmark.total_return, 100 / 98 - 1)
        np.testing.assert_allclose(ledger.daily_return, ledger.daily_return_before_cost - ledger.cost_rate)
        np.testing.assert_allclose(ledger.nav, (1 + ledger.daily_return).cumprod())
        holdings = frames[2].query('strategy == "factor" and offset == 0')
        self.assertEqual(holdings.loc[holdings.trade_date.eq(dates[1]), "ticker"].tolist(), ["A"])
        self.assertEqual(holdings.loc[holdings.trade_date.eq(dates[2]), "ticker"].tolist(), ["A"])
        trades = frames[-1].query('strategy == "factor" and deal_amount != 0').merge(raw, on=["trade_date", "ticker"])
        for offset, fills in trades.groupby("offset"):
            self.assertTrue(fills.trade_date.isin(dates[offset + 1::2]).all())
            self.assertEqual(fills.trade_date.min(), dates[offset + 1])
        self.assertTrue(np.all(dates.get_indexer(trades.trade_date) - dates.get_indexer(trades.signal_date) == 1))
        buys = trades.deal_amount.gt(0)
        self.assertTrue((~buys).any())
        np.testing.assert_allclose(trades.trade_price, trades.close / 10, rtol=1e-6)
        np.testing.assert_allclose(trades.deal_amount / 1000, (trades.deal_amount / 1000).round(), atol=1e-6)
        np.testing.assert_allclose(trades.trade_cost, np.maximum(5., trades.trade_value.abs() * np.where(buys, .0005, .0015)))

    def test_qlib_excess_risk_uses_net_returns_and_aligns_benchmark_dates(self):
        from qlib.contrib.evaluate import risk_analysis

        dates = pd.bdate_range("2024-01-02", periods=4)
        gross = pd.Series([0., .039, -.015, .012], index=dates)
        fees = pd.Series([0., .009, .002, .004], index=dates)
        benchmark = pd.Series([0., .01, -.01, .002], index=dates)
        excess = gross - benchmark - fees
        expected = risk_analysis(excess, freq="day", mode="sum")["risk"]
        actual = qlib_excess_metrics(gross - fees, benchmark.iloc[::-1])
        self.assertAlmostEqual(actual["annualized_excess_return"], expected["annualized_return"])
        self.assertAlmostEqual(actual["information_ratio"], expected["information_ratio"])
        self.assertAlmostEqual(actual["max_excess_drawdown"], expected["max_drawdown"])
        self.assertAlmostEqual(actual["max_excess_drawdown"], -.007)

    def test_return_metrics_independent_example(self):
        ledger = pd.DataFrame(dict(trade_date=pd.bdate_range("2023-01-30", periods=3),
                                   nav=[1., 1.10, .99], daily_return=[0., .1, -.10],
                                   turnover_two_sided=[0., .5, 0.], cost_cny=[0., 1., 0.]))
        result = performance(ledger)
        self.assertAlmostEqual(result["total_return"], -.01)
        self.assertAlmostEqual(result["annual_return"], .99**(238 / 3) - 1)
        self.assertAlmostEqual(result["annual_volatility"], np.log1p(ledger.daily_return).std() * np.sqrt(238))
        self.assertAlmostEqual(result["sharpe"], 0.)
        self.assertAlmostEqual(result["max_drawdown"], -.10)
        self.assertAlmostEqual(result["annual_turnover_one_way"], .5 * 238 / 6)
        self.assertAlmostEqual(result["month_win_rate"], .5)


if __name__ == "__main__":
    unittest.main()
