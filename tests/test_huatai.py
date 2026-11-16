"""Independent scalar references, temporal integrity, inference and cash ledger."""
import math
import unittest

import numpy as np
import pandas as pd

from src.data.research_panel import prepare_panel, prepare_membership, parse_dates
from src.features.huatai import compute_factors, alpha3, alpha13, alpha44, FACTORS
from src.features.huatai import cross_sectional_rank, window_covariance, window_correlation, quantile_groups
from src.features.targets import market_forward_returns
from src.features.neutralization import neutralize, mad_clip, align_industry_history
from src.evaluation.huatai_study import hac_mean, holm_adjust
from src.evaluation.huatai_study import evaluate_panel
from src.evaluation.portfolio import backtest_long_only, performance, trading_prices
from src.evaluation.diagnostics import distribution, diagnose


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
    start = dates.get_loc(day)-window+1
    prices, volumes = {t:[] for t in cohort}, {t:[] for t in cohort}
    for u in dates[start:dates.get_loc(day)+1]:
        p, v = {}, {}
        for t in cohort:
            if (u,t) in data.index:
                row = data.loc[(u,t)]
                if all(math.isfinite(row[c]) and row[c] > 0 for c in ["open","high","low","close","vol","amount","adj_factor"]):
                    p[t], v[t] = row[field], row.vol
        for t in cohort:
            prices[t].append(p.get(t, math.nan) if factor == "Alpha44" else scalar_rank(p,t))
            volumes[t].append(scalar_rank(v,t))
    stat = {t:scalar_stat(prices[t], volumes[t], factor != "Alpha13") for t in cohort}
    return {t: -(scalar_rank(stat,t) if factor == "Alpha13" else stat[t]) for t in cohort}


class FactorTests(unittest.TestCase):
    def setUp(self):
        self.raw, self.members, self.dates = fixture()
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

    def test_shuffle_and_future_append_do_not_change_past(self):
        day = self.dates[24]
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
        day=self.dates[24]
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

    def test_rank_ties_invalid_and_grouping(self):
        np.testing.assert_allclose(cross_sectional_rank([1,1,3,np.inf]),[.5,.5,1,np.nan],equal_nan=True)
        self.assertTrue(np.isnan(cross_sectional_rank([1,np.nan])).all())
        self.assertEqual(len(np.unique(quantile_groups(np.ones(10)))),1)
        np.testing.assert_array_equal(quantile_groups(np.arange(10)),[1,1,2,2,3,3,4,4,5,5])


class DataTests(unittest.TestCase):
    def test_non_market_record_is_rejected(self):
        raw,_,dates=fixture()
        raw.loc[raw.index[0],"trade_date"]=pd.Timestamp("2023-01-07")
        with self.assertRaises(ValueError): prepare_panel(raw,dates)

    def test_split_adjustment_and_invalid_first_anchor(self):
        raw,_,dates=fixture()
        raw=raw[raw.ticker.eq("A")].copy()
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

    def test_market_endpoint_missing_and_post_exit_label(self):
        dates=pd.bdate_range("2023-01-02",periods=10)
        prices=pd.DataFrame({"A":np.arange(10,20,dtype=float),"B":np.arange(20,30,dtype=float)},index=dates)
        prices.loc[dates[5],"A"]=np.nan
        labels,end,reason=market_forward_returns(prices.drop(dates[3]),dates,5)
        self.assertTrue(pd.isna(labels.iloc[0,0])); self.assertEqual(reason.iloc[0,0],"endpoint_missing_unknown")
        self.assertEqual(end.iloc[0],dates[5]);self.assertAlmostEqual(labels.iloc[0,1],25/20-1)
        self.assertTrue(end.iloc[-5:].isna().all());self.assertEqual(reason.iloc[-1,0],"calendar_boundary")
        # Caller membership is absent from endpoint computation by design.
        self.assertAlmostEqual(labels.loc[dates[1],"A"],16/11-1)


class StatisticalTests(unittest.TestCase):
    def test_cross_sectional_incremental_coefficient(self):
        mom=np.arange(10,dtype=float);alpha=np.array([3,8,1,9,2,7,0,6,4,5.],float)
        z_m=(mom-mom.mean())/mom.std(ddof=1);z_a=(alpha-alpha.mean())/alpha.std(ddof=1)
        frame=pd.DataFrame({"trade_date":pd.Timestamp("2023-01-03"),"MOM20":mom,"Alpha13":alpha,
            "future_ret_5":.02+.1*z_m+.03*z_a,"label_end_5":pd.Timestamp("2023-01-10")})
        *_,reg=evaluate_panel(frame,["Alpha13","MOM20"],horizons=[5],min_pairs=3)
        self.assertAlmostEqual(reg.alpha_beta.iloc[0],.03)

    def test_hac_manual_calendar_gaps(self):
        x=np.array([1.,2.,np.nan,4.,3.,2.]);valid=np.isfinite(x);n=valid.sum()
        e=np.where(valid,x-np.nanmean(x),0)
        expected=(np.dot(e,e)+np.dot(e[1:],e[:-1]))/n**2*n/(n-1)
        result=hac_mean(x,lag=1,min_obs=2,min_coverage=.5)
        self.assertAlmostEqual(result["hac_se"]**2,expected)
        self.assertNotAlmostEqual(result["hac_se"],hac_mean(x[valid],lag=1,min_obs=2)["hac_se"])

    def test_holm_and_insufficient_coverage(self):
        np.testing.assert_allclose(holm_adjust([.01,.04,.03]),[.03,.06,.06])
        self.assertEqual(hac_mean([1,np.nan,2])["inference_status"],"insufficient_coverage")
        self.assertTrue(np.isnan(hac_mean(np.ones(200))["p_value"]))

    def test_serial_correlation_increases_uncertainty(self):
        rng=np.random.default_rng(19);x=np.zeros(2000)
        for i in range(1,len(x)): x[i]=.8*x[i-1]+rng.normal()
        self.assertGreater(hac_mean(x,lag=20)["hac_se"],hac_mean(x,lag=0)["hac_se"]*2)

    def test_labels_cannot_change_group_membership(self):
        frame=pd.DataFrame({"trade_date":pd.Timestamp("2023-01-03"),"Alpha13":np.arange(10.),
            "future_ret_5":np.arange(10.)/100,"label_end_5":pd.Timestamp("2023-01-10")})
        a=evaluate_panel(frame,["Alpha13"],horizons=[5],min_pairs=3)
        frame.future_ret_5=np.nan
        b=evaluate_panel(frame,["Alpha13"],horizons=[5],min_pairs=3)
        np.testing.assert_array_equal(a[1].n_assigned,b[1].n_assigned)
        self.assertAlmostEqual(a[0].rank_ic.iloc[0],1.)
        self.assertEqual(b[0].n_pairs.iloc[0],0)

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


class PortfolioTests(unittest.TestCase):
    def test_opening_quotes_do_not_use_future_closing_fields(self):
        raw,_,dates=fixture()
        raw=raw[raw.ticker.eq("A")].copy()
        changed=raw.copy()
        changed.loc[changed.trade_date.eq(dates[1]),["high","low","close","vol"]]=[np.inf,np.nan,np.nan,0.]
        args=[]
        for data in [raw,changed]:
            panel=prepare_panel(data,dates)
            opening,closing,original=trading_prices(panel,dates,panel.tickers)
            _,trades,_=backtest_long_only(opening,closing,original,original*2,original*.5,{dates[0]:["A"]},fee=0)
            args.append(trades)
        pd.testing.assert_frame_equal(args[0],args[1])

    def test_current_open_marks_existing_position_when_close_missing(self):
        dates=pd.bdate_range("2023-01-02",periods=4)
        opening=pd.DataFrame({"A":[1.,1.,2.,3.]},index=dates)
        closing=opening.copy();closing.iloc[2]=np.nan
        ledger,_,_=backtest_long_only(opening,closing,opening,opening*2,opening*.5,{dates[0]:["A"]},fee=0)
        self.assertAlmostEqual(ledger.nav.iloc[2],2.)
        self.assertEqual(ledger.n_stale.iloc[2],1)

    def test_unknown_limit_is_not_assumed_executable(self):
        dates=pd.bdate_range("2023-01-02",periods=3)
        p=pd.DataFrame({"A":[1.,1.,1.]},index=dates)
        unknown=p*np.nan
        ledger,trades,_=backtest_long_only(p,p,p,unknown,unknown,{dates[0]:["A"]})
        self.assertTrue(trades.empty)
        np.testing.assert_allclose(ledger.nav,1.)
        self.assertEqual(ledger.n_blocked_buys.iloc[1],1)

    def test_buy_and_sell_costs_both_debit_cash(self):
        dates=pd.bdate_range("2023-01-02",periods=4)
        p=pd.DataFrame({"A":np.ones(4)},index=dates)
        ledger,trades,_=backtest_long_only(p,p,p,p*2,p*.5,{dates[0]:["A"],dates[1]:[]},fee=.01,rebalance_every=1)
        self.assertAlmostEqual(ledger.nav.iloc[-1],(1-.01)/(1+.01))
        self.assertEqual(len(trades),2)
        self.assertAlmostEqual(ledger.cost.sum(),1-ledger.nav.iloc[-1])

    def run_model(self, fee=0, blocked=False):
        dates=pd.bdate_range("2023-01-02",periods=5)
        prices=pd.DataFrame({"A":[1.,2.,4.,4.,4.]},index=dates)
        up=prices*2;down=prices*.5
        if blocked: up.iloc[1]=prices.iloc[1]
        return backtest_long_only(prices,prices,prices,up,down,{dates[0]:["A"]},fee=fee)

    def test_next_day_execution_and_exact_fees(self):
        ledger,trades,_=self.run_model()
        self.assertEqual(ledger.nav.iloc[0],1.)
        self.assertEqual(ledger.nav.iloc[1],1.)
        self.assertEqual(ledger.nav.iloc[2],2.)
        costly,_,_=self.run_model(.01)
        self.assertAlmostEqual(costly.nav.iloc[1],1/1.01)
        self.assertEqual(len(trades),1)
        self.assertAlmostEqual(performance(ledger)["total_return"],1.)

    def test_limit_blocked_order_stays_cash(self):
        ledger,trades,_=self.run_model(blocked=True)
        np.testing.assert_array_equal(ledger.nav,np.ones(len(ledger)))
        self.assertTrue(trades.empty)

    def test_missing_close_is_stale_not_a_trade(self):
        dates=pd.bdate_range("2023-01-02",periods=5)
        p=pd.DataFrame({"A":[1.,1.,np.nan,2.,2.]},index=dates)
        ledger,_,_=backtest_long_only(p,p,p,p*2,p*.5,{dates[0]:["A"],dates[1]:[]},fee=0,rebalance_every=1)
        self.assertEqual(ledger.n_stale.iloc[2],1)
        self.assertEqual(ledger.pending_exit_value.iloc[2],1.)
        self.assertEqual(ledger.n_holdings.iloc[3],0)


class DiagnosticTests(unittest.TestCase):
    def test_positive_ic_can_coexist_with_negative_arithmetic_spread(self):
        day=pd.Timestamp("2023-01-03");x=np.arange(100,dtype=float);y=x/1000;y[0]=10.
        frame=pd.DataFrame({"trade_date":day,"ticker":[str(i) for i in range(100)],"Alpha13":x,"Alpha44":x,
            "future_ret_5":y,"future_ret_20":y,"label_end_5":day+pd.Timedelta(days=7),
            "label_end_20":day+pd.Timedelta(days=30),"industry":"a","total_mv":100.,"close_raw":10.})
        daily,_,_,_=evaluate_panel(frame,["Alpha13"],horizons=[5],min_pairs=3)
        details,summary,stocks=diagnose(frame,pd.DatetimeIndex([day]),frame[["ticker","trade_date"]],min_pairs=3)
        s=summary[(summary.analysis=="spread")&(summary.period=="all")&(summary['sample']=="all")&(summary.factor=="Alpha13")&(summary.horizon==5)].set_index("metric")['mean']
        self.assertGreater(daily.rank_ic.iloc[0],0);self.assertLess(s['mean'],0)
        self.assertGreater(s['median'],0);self.assertGreater(s['trimmed_mean'],0)
        self.assertAlmostEqual(stocks[(stocks.factor=="Alpha13")&(stocks.horizon==5)].contribution.sum(),s['mean'])
        self.assertAlmostEqual(details[(details.factor=="Alpha13")&(details.horizon==5)].ic_contribution.sum(),daily.rank_ic.iloc[0])
        frame.future_ret_5=np.nan;frame.future_ret_20=np.nan
        missing,_,_=diagnose(frame,pd.DatetimeIndex([day]),frame[["ticker","trade_date"]],min_pairs=3)
        np.testing.assert_array_equal(missing.n_assigned,details.n_assigned)

    def test_trim_definition_and_nonfinite_values(self):
        stats=distribution([0,1,2,3,100,np.nan,np.inf])
        self.assertAlmostEqual(stats["mean"],21.2)
        self.assertEqual(stats["median"],2.)
        self.assertEqual(stats["trimmed_mean"],2.)
        self.assertEqual(stats["n_observed"],5)


if __name__ == "__main__":
    unittest.main()
