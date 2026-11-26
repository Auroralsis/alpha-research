"""验证从下载和历史成员到复权行情、收益标签和单因子分数的数据流程。"""

import importlib.util
import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.config import load_config
from src.data.download import merge_prices, write_parquet_batches
from src.data.qlib_provider import qlib_bars, write_provider
from src.data.research_panel import execution_labels, parse_dates, prepare_membership, prepare_panel, segment_masks
from src.data.universe_process import build_membership
from src.factors.alpha101 import formula_values
from src.factors.alpha158 import initialize_qlib, native_features
from src.factors.neutralization import align_industry_history, mad_clip, neutral_feature_matrix, neutralize
from src.factors.price_volume import (
    FACTORS, alpha3, alpha13, alpha44, choose, compute_factors, correlation, covariance, cross_sectional_rank,
    decay_linear, indneutralize, less, ts_argmax, ts_rank, window_correlation, window_covariance,
)


class DataPreparationTests(unittest.TestCase):
    def test_download_cli_runs_all_in_order_or_only_requested_step(self):
        from src.data import download

        steps = []
        with patch.object(download, "download_calendar", lambda: steps.append("calendar")), \
             patch("src.data.universe_process.main", lambda: steps.append("membership")), \
             patch.object(download, "download_prices", lambda: steps.append("prices")), \
             patch.object(download, "download_industry", lambda: steps.append("industry")):
            download.main([])
            self.assertEqual(steps, ["calendar", "membership", "prices", "industry"])
            for name in steps.copy():
                steps.clear()
                download.main([name])
                self.assertEqual(steps, [name])

    def test_interrupted_batch_write_preserves_existing_parquet(self):
        frame = pd.DataFrame({"score": [1.0]})

        def interrupted():
            yield frame.assign(score=2.0)
            raise ValueError("batch failed")

        with TemporaryDirectory() as temp:
            path = Path(temp) / "scores.parquet"
            write_parquet_batches([frame], path)
            with self.assertRaisesRegex(ValueError, "batch failed"):
                write_parquet_batches(interrupted(), path)
            pd.testing.assert_frame_equal(pd.read_parquet(path), frame)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_frequency_names_and_latest_run_keep_trading_periods_separate(self):
        from src.config import rebalance_frequency
        from src.scripts.run_all import latest_run, new_run

        for days, suffix in ((1, "daily"), (5, "weekly"), (10, "10d"), (20, "20d")):
            self.assertEqual(rebalance_frequency(days)[0], suffix)
            self.assertTrue(new_run(days).name.startswith(f"{suffix}_"))
        with TemporaryDirectory() as temp, patch("src.scripts.run_all.RUNS", Path(temp)):
            runs = [
                ("daily_20260101_100000_000000", 1),
                ("weekly_20260102_100000_000000", 5),
                ("20260103_100000_000000", 5),
            ]
            for name, days in runs:
                path = Path(temp) / name
                path.mkdir()
                (path / "config.json").write_text(json.dumps({"rebalance_days": days}))
                (path / "manifest.json").write_text("{}")
            self.assertEqual(latest_run("manifest.json", rebalance_days=1).name, runs[0][0])
            self.assertEqual(latest_run("manifest.json", rebalance_days=5).name, runs[2][0])
            with self.assertRaises(FileNotFoundError):
                latest_run("manifest.json", rebalance_days=10)

    def test_run_all_prepares_missing_inputs_once_in_dependency_order(self):
        from src.data.research_panel import prepare_inputs

        dates = pd.bdate_range("2013-09-02", periods=3)
        config = {"segments": {"test": ["2013-09-02", "2013-09-04"]}}
        with TemporaryDirectory() as temp:
            paths = {name: Path(temp) / f"{name}.parquet" for name in ("calendar", "membership", "prices", "industries")}
            steps = []

            def calendar():
                steps.append("calendar")
                pd.DataFrame({"cal_date": dates.strftime("%Y%m%d")}).to_parquet(paths["calendar"])

            def members():
                steps.append("membership")
                pd.DataFrame(dict(trade_date=dates, ticker="A")).to_parquet(paths["membership"])

            def prices():
                steps.append("prices")
                pd.DataFrame(dict(trade_date=dates, ticker="A", adj_factor=1.)).to_parquet(paths["prices"])

            def industry():
                steps.append("industry")
                paths["industries"].touch()

            with patch("src.data.research_panel.CSI500_INPUTS", paths), patch(
                "src.data.download.download_calendar", side_effect=calendar
            ), patch("src.data.universe_process.main", side_effect=members), patch(
                "src.data.download.download_prices", side_effect=prices
            ), patch(
                "src.data.download.download_industry", side_effect=industry
            ):
                prepare_inputs(config, need_industry=False)
                self.assertEqual(steps, ["calendar", "membership", "prices"])
                pd.testing.assert_index_equal(prepare_inputs(config), dates, check_names=False)
                pd.testing.assert_index_equal(prepare_inputs(config), dates, check_names=False)
            self.assertEqual(steps, ["calendar", "membership", "prices", "industry"])

    def test_early_membership_uses_the_last_known_snapshot(self):
        snapshots = pd.DataFrame(dict(
            con_code=["A", "B"], trade_date=["20130830", "20130903"], weight=[100., 100.],
            index_code="000905.SH",
        ))
        calendar = pd.DataFrame(dict(cal_date=["20130902", "20130903", "20130904"]))
        result = build_membership(snapshots, calendar)
        self.assertEqual(result.ticker.to_list(), ["A", "B", "B"])
        self.assertTrue(result.source_snapshot_date.le(result.trade_date).all())
        with self.assertRaises(ValueError):
            build_membership(snapshots.iloc[1:], calendar)

    def test_price_download_matches_adjustments_by_stock_and_date(self):
        daily = pd.DataFrame(dict(ts_code=["A", "A"], trade_date=["20140102", "20140103"], close=[10., 11.]))
        factors = pd.DataFrame(dict(ts_code=["A", "A"], trade_date=["20140103", "20140102"], adj_factor=[2., 1.]))
        result = merge_prices(daily, factors)
        self.assertEqual(result.adj_factor.to_list(), [1., 2.])
        with self.assertRaises(ValueError):
            merge_prices(daily, factors.iloc[:1])

    def test_industry_download_only_fills_confirmed_gaps(self):
        from src.data.download import supplement_history
        from src.factors.neutralization import align_industry_history

        keys = pd.DataFrame(dict(ticker="A", trade_date=pd.date_range("2023-01-01", periods=9)))
        primary = pd.DataFrame([
            ("old.SI", "20230101", "20230102"), ("new.SI", "20230107", None),
            ("old.SI", "20230108", "20230109"),
        ], columns=["l1_code", "in_date", "out_date"]).assign(ts_code="A", l1_name=lambda df: df.l1_code)
        secondary = pd.DataFrame([
            ("old", "2023-01-02", "2023-01-06"), ("new", "2023-01-05", "2023-01-06"),
            ("new", "1900-01-01", None), ("unrelated_index", "2023-01-01", None),
        ], columns=["index_code", "in_date", "out_date"])
        result, audit = supplement_history(keys, primary, lambda ticker: secondary)
        actual = align_industry_history(keys, result)
        self.assertEqual(actual.fillna("-").tolist(),
                         ["old.SI", "old.SI", "old.SI", "old.SI", "-", "-", "new.SI", "-", "new.SI"])
        self.assertEqual(audit.status.value_counts().to_dict(), {
            "已补一级行业": 3, "次源行业冲突": 1, "主源行业冲突": 1, "无历史一级行业记录": 1})
        self.assertEqual(len(result), len(primary) + 1)


def market_fixture():
    dates = pd.bdate_range("2023-01-02", periods=35)
    rows, members = [], []
    for j, ticker in enumerate("ABCDEF"):
        for i, day in enumerate(dates):
            if ticker == "B" and i == 20:  # 模拟停牌或原因未知的缺失记录。
                continue
            op = 10 + ((i * (j + 2) + j * 7) % 19) + i / 20
            vol = 100 + ((i * (j + 3) + j * 11) % 23)
            if i % 7 == 0:
                vol = 110.  # 构造同日截面上的并列值。
            rows.append(dict(ticker=ticker, trade_date=day, open=op, high=op+2,
                low=op-2, close=op+1, vol=vol, amount=op*vol/10, adj_factor=1. if i < 26 else 2.))
            if (ticker != "F" or i >= 24) and (ticker != "E" or i < 29):
                members.append(dict(ticker=ticker, trade_date=day, source_snapshot_date=dates[0], weight=1.))
    raw = pd.DataFrame(rows)
    raw.loc[(raw.ticker == "C") & (raw.trade_date == dates[18]), "high"] = np.inf
    # 即使缺少行情，成分表也应包含当日成员。
    members.append(dict(ticker="B", trade_date=dates[20], source_snapshot_date=dates[0], weight=1.))
    return raw, pd.DataFrame(members), dates


def synthetic_panel(dates: pd.DatetimeIndex, gap: int | None = None):
    price = 10. + np.arange(len(dates))
    shares = 200. + 100 * np.arange(len(dates))
    raw = pd.DataFrame(dict(ticker="A", trade_date=dates, open=price - .5, close=price, high=price + 1,
                            low=price - 1, vol=shares / 100, amount=(price + .25) * shares / 1000,
                            adj_factor=np.where(np.arange(len(dates)) < 7, 2., 4.)))
    return prepare_panel(raw.drop(index=gap) if gap is not None else raw, dates)


class DataTests(unittest.TestCase):
    def test_non_market_record_is_rejected(self):
        raw,_,dates=market_fixture()
        raw.loc[raw.index[0],"trade_date"]=pd.Timestamp("2023-01-07")
        with self.assertRaises(ValueError): prepare_panel(raw,dates)

    def test_split_adjustment_and_invalid_first_anchor(self):
        raw,_,dates=market_fixture()
        raw = raw.loc[raw["ticker"].eq("A"), :].copy()
        raw.loc[raw.trade_date<dates[1],["open","high","low","close"]]=[100.,102.,98.,100.]
        raw.loc[raw.trade_date==dates[1],["open","high","low","close","adj_factor"]]=[50.,51.,49.,50.,2.]
        panel=prepare_panel(raw,dates)
        self.assertAlmostEqual(panel.matrix("close_adj").iloc[1,0],panel.matrix("close_adj").iloc[0,0])
        raw.loc[raw.index[0],"adj_factor"]=np.nan
        with self.assertRaises(ValueError): prepare_panel(raw,dates)

    def test_strict_dates_keys_snapshot_and_original_immutability(self):
        raw, members, dates=market_fixture(); original=raw.copy(deep=True)
        prepare_panel(raw,dates)
        pd.testing.assert_frame_equal(raw,original)
        with self.assertRaises(ValueError): parse_dates(["2023013"])
        with self.assertRaises(ValueError): prepare_panel(pd.concat([raw,raw.iloc[:1]]),dates)
        members.loc[0,"source_snapshot_date"]=dates[-1]
        with self.assertRaises(ValueError): prepare_membership(members)

    def test_execution_labels_checks_keys_and_preserves_row_order(self):
        raw, _, dates = market_fixture()
        panel = prepare_panel(raw, dates)
        frame = pd.DataFrame(dict(trade_date=[dates[11], dates[10]], ticker=["B", "A"]), index=[17, 9])
        attached = execution_labels(panel, frame, 1, 5)
        self.assertEqual(attached.index.tolist(), [17, 9])
        self.assertEqual(attached.label_end_date.tolist(), [dates[16], dates[15]])
        close = panel.matrix("close_adj")
        self.assertAlmostEqual(attached.loc[17, "label"], close.loc[dates[16], "B"] / close.loc[dates[12], "B"] - 1)
        with self.assertRaises(ValueError):
            execution_labels(panel, frame.assign(ticker="unknown"), 1, 5)

    def test_panel_calculates_adjusted_vwap_and_masks_zero_volume(self):
        dates = pd.bdate_range("2023-01-03", periods=3)
        raw = pd.DataFrame({
            "ticker": "A", "trade_date": dates, "open": [10., 10., 10.],
            "high": [11., 11., 11.], "low": [9., 9., 9.], "close": [10., 10., 10.],
            "vol": [10., 10., 0.], "amount": [10., 10.5, 0.], "adj_factor": [2., 4., 4.],
        })
        panel = prepare_panel(raw, dates)
        np.testing.assert_allclose(panel.matrix("vwap_adj")["A"], [10., 21., np.nan], equal_nan=True)

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
        self.assertTrue(np.isnan(labeled.label.iloc[5]))  # 收益起点缺失时保留空值，不能顺延到第七天。
        self.assertAlmostEqual(labeled.label.iloc[1], 2 * 17 / 12 - 1)
        self.assertEqual(labeled.label_start_date.iloc[1], dates[2])
        parts = {"train": [str(dates[0].date()), str(dates[7].date())],
                 "validation": [str(dates[8].date()), str(dates[-1].date())]}
        self.assertEqual(segment_masks(labeled, parts)["train"].to_list(), [False, True] + [False] * 6)


def scalar_rank(items, key):
    finite = [v for v in items.values() if math.isfinite(v)]
    value = items.get(key, math.nan)
    if len(finite) < 2 or not math.isfinite(value):
        return math.nan
    return (sum(v < value for v in finite) + (sum(v == value for v in finite) + 1) / 2) / len(finite)


def scalar_stat(x, y, correlation):
    if not all(math.isfinite(v) for v in x + y):
        return math.nan
    mx, my = sum(x)/len(x), sum(y)/len(y)
    numerator = sum((a-mx)*(b-my) for a,b in zip(x,y))
    if not correlation:
        return numerator/(len(x)-1)
    denominator = math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
    return numerator/denominator if denominator > 1e-24 else math.nan


def scalar_reference(raw, membership, dates, day, factor):
    cohort = sorted(membership.loc[membership.trade_date == day, "ticker"])
    window = 10 if factor == "Alpha3" else 5
    field = {"Alpha3":"open", "Alpha13":"close", "Alpha44":"high"}[factor]
    data = raw.set_index(["trade_date", "ticker"])
    anchors = data.sort_index().groupby(level="ticker").adj_factor.first()
    start = dates.get_loc(day)-window+1
    prices, volumes = {t:[] for t in cohort}, {t:[] for t in cohort}
    for u in dates[start:dates.get_loc(day)+1]:
        p, v = {}, {}
        for t in cohort:
            if (u,t) in data.index:
                row = data.loc[(u,t)]
                if all(math.isfinite(row[c]) and row[c] > 0 for c in ["open","high","low","close","vol","amount","adj_factor"]):
                    p[t] = row[field] * row.adj_factor / anchors[t]
                    v[t] = row.vol
        for t in cohort:
            prices[t].append(p.get(t, math.nan) if factor == "Alpha44" else scalar_rank(p,t))
            volumes[t].append(scalar_rank(v,t))
    stat = {t:scalar_stat(prices[t], volumes[t], factor != "Alpha13") for t in cohort}
    return {t: -(scalar_rank(stat,t) if factor == "Alpha13" else stat[t]) for t in cohort}


class FactorTests(unittest.TestCase):
    def setUp(self):
        self.raw, self.members, self.dates = market_fixture()
        split = self.raw.ticker.eq("A") & self.raw.trade_date.ge(self.dates[26])
        self.raw.loc[split, "adj_factor"] *= 2
        self.panel = prepare_panel(self.raw.sample(frac=1,random_state=3), self.dates)
        self.members = prepare_membership(self.members)

    def check_factor(self, factor):
        days = self.dates[[22,24,28,32]]
        actual = compute_factors(self.panel, self.members, days)
        for day in days:
            reference = scalar_reference(self.raw, self.members, self.dates, day, factor)
            subset = actual[actual.trade_date == day].set_index("ticker")
            np.testing.assert_allclose(subset[factor], [reference[t] for t in subset.index], atol=1e-10, equal_nan=True)

    def test_alpha3_independent_reference(self): self.check_factor("Alpha3")
    def test_alpha13_independent_reference(self): self.check_factor("Alpha13")
    def test_alpha44_independent_reference(self): self.check_factor("Alpha44")

    def test_momentum_uses_adjusted_close_and_requires_complete_window(self):
        days = self.dates[[19, 20, 28]]
        result = compute_factors(self.panel, self.members, days).set_index(["trade_date", "ticker"])
        close = self.panel.matrix("close_adj")["A"]
        self.assertTrue(pd.isna(result.loc[(days[0], "A"), "MOM20"]))
        for pos in (20, 28):
            self.assertAlmostEqual(result.loc[(self.dates[pos], "A"), "MOM20"], close.iloc[pos] / close.iloc[pos - 20] - 1)
        self.assertTrue(pd.isna(result.loc[(days[2], "B"), "MOM20"]))
        self.assertEqual(result.loc[(days[2], "B"), "MOM20_reason"], "price_window_incomplete")

    def test_shuffle_and_future_append_do_not_change_past(self):
        day = cast(pd.Timestamp, self.dates[24])
        expected = compute_factors(self.panel,self.members,[day])
        extra = self.raw.iloc[:1].copy(); extra["ticker"] = "ZZ_future"; extra["trade_date"] = self.dates[-1]
        appended = pd.concat([self.raw,extra],ignore_index=True)
        altered = prepare_panel(appended.sample(frac=1,random_state=7),self.dates)
        actual = compute_factors(altered,self.members,[day])
        pd.testing.assert_frame_equal(actual,expected)
        # 后续价格和复权因子的变化不影响历史复权锚点。
        changed = self.raw.copy(); mask=changed.trade_date>day
        changed.loc[mask,"adj_factor"] *= 100
        a=compute_factors(prepare_panel(changed,self.dates),self.members,[day],price_mode="adjusted")
        b=compute_factors(self.panel,self.members,[day],price_mode="adjusted")
        pd.testing.assert_frame_equal(a,b)

    def test_new_member_uses_pre_entry_history_and_rank_policies_differ(self):
        day = cast(pd.Timestamp, self.dates[24])
        a=compute_factors(self.panel,self.members,[day])
        b=compute_factors(self.panel,self.members,[day],rank_policy="historical_members")
        self.assertTrue(a.loc[a.ticker.eq("F"),list(FACTORS)].notna().all().all())
        self.assertTrue(b.loc[b.ticker.eq("F"),["Alpha3","Alpha13","Alpha44"]].isna().all().all())

    def test_zero_variance_and_zero_covariance(self):
        x=np.ones((5,3)); y=np.tile([1.,2.,3.],(5,1))
        self.assertTrue(np.isnan(window_correlation(x,y)).all())
        np.testing.assert_array_equal(window_covariance(x,y),np.zeros(3))
        np.testing.assert_allclose(alpha13(x,y),[-2/3]*3)
        for func,size in [(alpha3,10),(alpha44,5)]:
            a=np.tile(np.arange(size)[:,None],(1,3))
            np.testing.assert_allclose(func(a,-a),np.ones(3))

    def test_rank_ties_and_invalid(self):
        np.testing.assert_allclose(cross_sectional_rank([1,1,3,np.inf]),[.5,.5,1,np.nan],equal_nan=True)
        self.assertTrue(np.isnan(cross_sectional_rank([1,np.nan])).all())


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


class Alpha158Tests(unittest.TestCase):
    def test_mad_uses_whole_days_across_stock_batches_and_preserves_labels(self):
        from src.ml.dataset import mad_feature_batches

        dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
        frame = pd.DataFrame({
            "trade_date": list(dates) * 5, "ticker": np.repeat(list("ABCDE"), 2),
            "feature": np.array([1, 100, 2, 101, 3, 102, 100, 103, np.nan, 104], dtype="float32"),
            "flat": [2.] * 8 + [20., 20.], "missing": np.nan, "label": np.arange(10.),
        })
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.parquet"
            write_parquet_batches([frame.iloc[:4], frame.iloc[4:]], path)
            result = pd.concat(mad_feature_batches(path, ["feature", "flat", "missing"]), ignore_index=True)
        # 第一天的有效值为 1、2、3、100，中位数 2.5，MAD 为 1。
        expected = frame.copy()
        expected.loc[6, "feature"] = 7.5
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)

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


class EqualWeightTests(unittest.TestCase):
    def test_train_direction_uses_pairwise_ranks_and_majority_of_years(self):
        from src.factors.alpha158 import factor_directions
        from src.backtest.factor_metrics import daily_ic
        from src.data.research_panel import stock_day_index

        dates = pd.to_datetime(["2017-01-03", "2018-01-03", "2019-01-03"])
        frame = pd.DataFrame(dict(trade_date=dates.repeat(5), ticker=list("ABCDE") * 3,
                                  label=[2, 1, 4, 0, 3] * 3, ROC5=[3, 3, np.nan, 4, 1] * 3, OPEN0=1.))
        table = factor_directions(frame, ["ROC5", "OPEN0"])
        key = stock_day_index(frame)
        expected = daily_ic(frame.ROC5.set_axis(key), frame.label.set_axis(key))[1].mean()
        self.assertAlmostEqual(table.at["ROC5", "train_rank_ic"], expected)
        self.assertEqual(table.at["ROC5", "direction"], -1)
        self.assertEqual(table.at["ROC5", "family"], "ROC")
        self.assertEqual(table.at["OPEN0", "direction"], 0)

    def test_direction_family_weights_ties_and_missing_scores(self):
        from src.factors.alpha158 import equal_weight_scores

        values = pd.DataFrame(dict(trade_date=["2023-01-03"] * 4, ROC5=[1, 2, 3, np.nan],
                                   ROC10=[1, 2, np.nan, np.nan], OPEN0=[1, 1, 3, np.nan]))
        directions = pd.DataFrame(dict(direction=[1, 1, -1], family=["ROC", "ROC", "OPEN0"]),
                                  index=["ROC5", "ROC10", "OPEN0"])
        expected = [((1/3 + 1/2)/2 + 2.5/3)/2, ((2/3 + 1)/2 + 2.5/3)/2, (1 + 1/3)/2, np.nan]
        np.testing.assert_allclose(equal_weight_scores(values, directions), expected, equal_nan=True)


class NeutralizationTests(unittest.TestCase):
    def test_neutralization_orthogonality_and_pit_guard(self):
        idx=range(30); cap=pd.Series(np.exp(np.arange(30)/10),index=idx)
        industry=pd.Series(["a"]*15+["b"]*15,index=idx)
        factor=pd.Series(np.sin(np.arange(30))+np.log(cap)*2,index=idx)
        result=neutralize(factor,cap,industry)
        self.assertAlmostEqual(result.std(),1.)
        self.assertAlmostEqual(result.iloc[:15].sum(),0.,places=9)
        self.assertAlmostEqual(float(np.dot(result,mad_clip(np.log(cap)))),0.,places=9)
        with self.assertRaises(ValueError): neutralize(factor,cap,industry,known_at=pd.Series(["2024-01-01"]),signal_date="2023-01-01")

    def test_matrix_neutralization_matches_single_factor_with_different_missing_masks(self):
        rng = np.random.default_rng(158)
        values = rng.normal(size=(45, 8)).astype(np.float32)
        values[:7, :2] = np.nan
        values[4:, 2] = np.nan
        values[:, 3] = 1
        values[10, 4] = 1e7
        caps = np.exp(rng.normal(size=45))
        caps[8] = 0
        industries = np.array(["A", "B", "C"] * 15, dtype=object)
        industries[12] = None
        source = values.copy()
        result = neutral_feature_matrix(values, caps, industries)
        reference = np.column_stack([neutralize(column, caps, industries) for column in values.T])
        np.testing.assert_allclose(result, reference, equal_nan=True, atol=2e-6)
        np.testing.assert_array_equal(values, source)

    def test_industry_interval_and_ambiguous_history(self):
        keys=pd.DataFrame({"ticker":["A"]*3,"trade_date":pd.to_datetime(["2023-01-01","2023-02-01","2023-03-01"])})
        history=pd.DataFrame({"ts_code":["A","A"],"in_date":["20220101","20230201"],"out_date":["20230201",None],"l1_code":["old","new"]})
        self.assertEqual(align_industry_history(keys,history).tolist(),["old","new","new"])
        history.loc[0,"out_date"]="20230215"
        self.assertTrue(pd.isna(align_industry_history(keys,history).iloc[1]))


class PublishedFinancialFactorTests(unittest.TestCase):
    def test_delayed_versions_and_quarter_end_equity(self):
        from src.scripts.factor_extensions import financial_factor

        quarters = pd.period_range("2015Q1", periods=18, freq="Q")
        dates = pd.bdate_range("2015-01-01", "2020-01-01")
        def records(field, values, kind):
            rows = []
            for period, value in zip(quarters, values):
                end = period.end_time.normalize()
                day = (end + pd.Timedelta(days=20)).strftime("%Y%m%d")
                rows.append(dict(ticker="A", end_date=end.strftime("%Y%m%d"), f_ann_date=day, ann_date=day,
                                 report_type=kind, **{field: value}))
            return pd.DataFrame(rows)

        eps = np.arange(len(quarters), dtype=float)**2
        income = records("basic_eps", eps, "2").assign(n_income_attr_p=eps)
        equity = records("total_hldr_eqy_exc_min_int", 100 + np.arange(len(quarters)), "1")
        date = pd.Timestamp(income.iloc[-1].f_ann_date)
        effective = dates[dates.searchsorted(date, side="right")]
        update = income.iloc[[-1]].assign(report_type="3", f_ann_date="20190801", ann_date="20190801",
                                          basic_eps=400, n_income_attr_p=400)
        events = pd.concat([income, update], ignore_index=True)
        streams = {"income": events, "equity": equity,
                   "eps": events.drop(columns="f_ann_date").rename(columns={"basic_eps": "q_eps"})}
        with patch("src.scripts.factor_extensions.cached", side_effect=streams.__getitem__):
            sue = financial_factor("SUE", dates).xs("A", level="ticker")
            roe = financial_factor("dROE", dates).xs("A", level="ticker")
        prior = eps[9:17] - eps[5:13]
        self.assertAlmostEqual(sue.at[effective], (eps[17]-eps[13]) / prior.std(ddof=1))
        self.assertAlmostEqual(roe.at[effective],eps[17]/116-eps[13]/112)
        self.assertAlmostEqual(roe.at[pd.Timestamp("20190801")],roe.at[effective])
        self.assertAlmostEqual(roe.at[pd.Timestamp("20190802")],400/116-eps[13]/112)


if __name__ == "__main__":
    unittest.main()
