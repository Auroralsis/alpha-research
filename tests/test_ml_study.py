"""验证 Alpha158、固定和滚动训练、预测评价及周频交易账本。"""

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

from src.backtest.factor_metrics import prediction_diagnostics, quintile_assignments
from src.backtest.portfolio import performance, qlib_excess_metrics, window_portfolio_returns
from src.config import load_config
from src.data.qlib_provider import qlib_bars, write_provider
from src.data.research_panel import execution_labels, prepare_membership, prepare_panel, segment_masks
from src.factors.alpha158 import initialize_qlib, native_features
from src.ml.dataset import daily_zscore, read_feature_split
from src.ml.models import fit_lightgbm, load_predictions, train_and_predict, training_settings
from src.ml.rolling import (
    check_prediction_windows, fit_window, init_rolling_qlib, merge_rolling_predictions, rolling_windows,
    train_rolling,
)


def synthetic_panel(dates: pd.DatetimeIndex, gap: int | None = None):
    price = 10. + np.arange(len(dates))
    shares = 200. + 100 * np.arange(len(dates))
    raw = pd.DataFrame(dict(ticker="A", trade_date=dates, open=price - .5, close=price, high=price + 1,
                            low=price - 1, vol=shares / 100, amount=(price + .25) * shares / 1000,
                            adj_factor=np.where(np.arange(len(dates)) < 7, 2., 4.)))
    return prepare_panel(raw.drop(index=gap) if gap is not None else raw, dates)


class TrainingTests(unittest.TestCase):
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
                      lightgbm_tuning=dict(search_space={"lambda_l1": [0., 30.], "max_depth": [2, 3, 2]}),
                      lightgbm=dict(objective="regression", metric="l2", lambda_l1=200., max_depth=8,
                                    min_data_in_leaf=2, num_threads=1, verbosity=-1))
        names = [f"F{i}" for i in range(158)]
        ticker = np.tile(np.arange(5), len(dates))
        frame = pd.DataFrame({name: (ticker + i).astype(np.float32) for i, name in enumerate(names)}).assign(
            trade_date=dates.repeat(5), ticker=ticker.astype(str), label=ticker * .01,
            label_end_date=pd.Series(dates).shift(-2).repeat(5).to_numpy(),
        )
        frame.loc[frame.trade_date.ge(dates[10]), names[0]] = np.inf
        frame = frame.sort_values(["ticker", "trade_date"])
        original_fit = LGBModel.fit
        output = StringIO()
        with TemporaryDirectory() as temp:
            out = Path(temp)
            (out / "alpha158.json").write_text(json.dumps([dict(name=name) for name in names]))
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
            curve = pd.read_csv(out / "best_training_mse.csv")
            self.assertAlmostEqual(curve.validation_mse.min(), best.validation_mse)
            study = optuna.load_study(study_name="alpha158_lightgbm", storage=f"sqlite:///{out / 'optuna.sqlite3'}")
            self.assertAlmostEqual(study.best_value, best.validation_mse)


class MLDataTests(unittest.TestCase):
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

    def test_factor_api_shares_keys_and_ml_adds_labels_after_computation(self):
        from src.factors import factor_batches
        from src.ml.dataset import build_features

        dates = pd.bdate_range("2023-01-03", periods=75)
        panel = synthetic_panel(dates, gap=71)
        members = pd.DataFrame(dict(ticker="A", trade_date=dates, source_snapshot_date=dates))
        config = load_config()
        keys = ["trade_date", "ticker"]
        signals = dates[65:70]
        requested = members.assign(label=999.)
        for family, column in (("alpha158", "OPEN0"), ("price_volume", "MOM20")):
            frame = pd.concat(factor_batches(family, panel, requested, signals, config))
            self.assertNotIn("label", frame)
            self.assertEqual(frame[keys].values.tolist(), members.loc[members.trade_date.isin(signals), keys].values.tolist())
            self.assertTrue(frame[column].notna().all())
            if family == "alpha158":
                expected = frame.set_index(keys).sort_index()

        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            build_features(panel, members, config, out, workers=1)
            samples = pd.read_parquet(out / "features.parquet").set_index(keys)
            actual = samples.loc[expected.index, expected.columns]
            np.testing.assert_allclose(actual, expected, equal_nan=True, rtol=1e-6)
            self.assertNotIn((dates[71], "A"), samples.index)
            self.assertTrue(pd.isna(samples.loc[(dates[69], "A"), "label"]))
            self.assertIn((dates[-1], "A"), samples.index)

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
        groups = quintile_assignments(frame, "lightgbm").set_index("ticker").group.to_dict()
        self.assertEqual(groups, {"B": 1, "D": 2, "A": 3, "E": 4, "C": 5})
        frame["label"] = np.nan
        self.assertEqual(quintile_assignments(frame, "lightgbm").set_index("ticker").group.to_dict(), groups)

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
                (out / "alpha158.json").write_text(json.dumps([{"name": feature} for feature in names]))
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


class MLPortfolioTests(unittest.TestCase):
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

if __name__ == "__main__":
    unittest.main()
