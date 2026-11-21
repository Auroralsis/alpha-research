"""Independent scalar references, temporal integrity, inference and cash ledger."""
import json
import math
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtests
from src.config import load_config
from src.data.research_panel import execution_labels, parse_dates, prepare_membership, prepare_panel
from src.factors.neutralization import align_industry_history, mad_clip, neutralize
from src.factors.price_volume import (
    FACTORS, alpha13, alpha3, alpha44, compute_factors, cross_sectional_rank, window_correlation,
    window_covariance,
)


class SharedBacktestTests(unittest.TestCase):
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


def fixture():
    dates = pd.bdate_range("2023-01-02", periods=35)
    rows, members = [], []
    for j, ticker in enumerate("ABCDEF"):
        for i, day in enumerate(dates):
            if ticker == "B" and i == 20:  # suspension/unknown missing record
                continue
            op = 10 + ((i * (j + 2) + j * 7) % 19) + i / 20
            vol = 100 + ((i * (j + 3) + j * 11) % 23)
            if i % 7 == 0:
                vol = 110.  # cross-sectional ties
            rows.append(dict(ticker=ticker, trade_date=day, open=op, high=op+2,
                low=op-2, close=op+1, vol=vol, amount=op*vol/10, adj_factor=1. if i < 26 else 2.))
            if (ticker != "F" or i >= 24) and (ticker != "E" or i < 29):
                members.append(dict(ticker=ticker, trade_date=day, source_snapshot_date=dates[0], weight=1.))
    raw = pd.DataFrame(rows)
    raw.loc[(raw.ticker == "C") & (raw.trade_date == dates[18]), "high"] = np.inf
    # Membership must include a member even when its price record is missing.
    members.append(dict(ticker="B", trade_date=dates[20], source_snapshot_date=dates[0], weight=1.))
    return raw, pd.DataFrame(members), dates


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
        self.raw, self.members, self.dates = fixture()
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
        # Later price/corporate action changes do not alter a historical anchor.
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


class DataTests(unittest.TestCase):
    def test_non_market_record_is_rejected(self):
        raw,_,dates=fixture()
        raw.loc[raw.index[0],"trade_date"]=pd.Timestamp("2023-01-07")
        with self.assertRaises(ValueError): prepare_panel(raw,dates)

    def test_split_adjustment_and_invalid_first_anchor(self):
        raw,_,dates=fixture()
        raw = raw.loc[raw["ticker"].eq("A"), :].copy()
        raw.loc[raw.trade_date<dates[1],["open","high","low","close"]]=[100.,102.,98.,100.]
        raw.loc[raw.trade_date==dates[1],["open","high","low","close","adj_factor"]]=[50.,51.,49.,50.,2.]
        panel=prepare_panel(raw,dates)
        self.assertAlmostEqual(panel.matrix("close_adj").iloc[1,0],panel.matrix("close_adj").iloc[0,0])
        raw.loc[raw.index[0],"adj_factor"]=np.nan
        with self.assertRaises(ValueError): prepare_panel(raw,dates)

    def test_strict_dates_keys_snapshot_and_original_immutability(self):
        raw, members, dates=fixture(); original=raw.copy(deep=True)
        prepare_panel(raw,dates)
        pd.testing.assert_frame_equal(raw,original)
        with self.assertRaises(ValueError): parse_dates(["2023013"])
        with self.assertRaises(ValueError): prepare_panel(pd.concat([raw,raw.iloc[:1]]),dates)
        members.loc[0,"source_snapshot_date"]=dates[-1]
        with self.assertRaises(ValueError): prepare_membership(members)

    def test_execution_labels_checks_keys_and_preserves_row_order(self):
        raw, _, dates = fixture()
        panel = prepare_panel(raw, dates)
        frame = pd.DataFrame(dict(trade_date=[dates[11], dates[10]], ticker=["B", "A"]), index=[17, 9])
        attached = execution_labels(panel, frame, 1, 5)
        self.assertEqual(attached.index.tolist(), [17, 9])
        self.assertEqual(attached.label_end_date.tolist(), [dates[16], dates[15]])
        close = panel.matrix("close_adj")
        self.assertAlmostEqual(attached.loc[17, "label"], close.loc[dates[16], "B"] / close.loc[dates[12], "B"] - 1)
        with self.assertRaises(ValueError):
            execution_labels(panel, frame.assign(ticker="unknown"), 1, 5)


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

    def test_industry_interval_and_ambiguous_history(self):
        keys=pd.DataFrame({"ticker":["A"]*3,"trade_date":pd.to_datetime(["2023-01-01","2023-02-01","2023-03-01"])})
        history=pd.DataFrame({"ts_code":["A","A"],"in_date":["20220101","20230201"],"out_date":["20230201",None],"l1_code":["old","new"]})
        self.assertEqual(align_industry_history(keys,history).tolist(),["old","new","new"])
        history.loc[0,"out_date"]="20230215"
        self.assertTrue(pd.isna(align_industry_history(keys,history).iloc[1]))


class QuoteTests(unittest.TestCase):
    def test_panel_calculates_adjusted_vwap_and_masks_zero_volume(self):
        dates = pd.bdate_range("2023-01-03", periods=3)
        raw = pd.DataFrame({
            "ticker": "A", "trade_date": dates, "open": [10., 10., 10.],
            "high": [11., 11., 11.], "low": [9., 9., 9.], "close": [10., 10., 10.],
            "vol": [10., 10., 0.], "amount": [10., 10.5, 0.], "adj_factor": [2., 4., 4.],
        })
        panel = prepare_panel(raw, dates)
        np.testing.assert_allclose(panel.matrix("vwap_adj")["A"], [10., 21., np.nan], equal_nan=True)


if __name__ == "__main__":
    unittest.main()
