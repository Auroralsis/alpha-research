import json
import logging
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/huatai_matplotlib")
import numpy as np
import pandas as pd

from src.data.research_panel import prepare_panel, prepare_membership, parse_dates
from src.features.huatai import compute_factors, FACTORS, quantile_groups
from src.features.targets import market_forward_returns
from src.features.neutralization import align_industry_history, neutralize
from src.evaluation.huatai_study import evaluate_panel, summarize_ic, summarize_quantiles, hac_mean, holm_adjust
from src.evaluation.diagnostics import diagnose
from src.evaluation.portfolio import backtest_long_only, performance, trading_prices
from src.scripts.huatai_report import write_reports, plot_results, validate_results

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/huatai_alpha"


def attach_labels(frame, panel, horizons):
    """输入的frame是实际研究的股票和日期(这里没用限制是否在membership中)，
        panel是总的研究数据，包括全部的时间段和股票
        horizons通常是[5,20]，用于计算未来收益
    """
    frame = frame.copy()
    prices = panel.matrix("close_adj")
    # 分别取出frame.trade_date和columns在price中对应的位置索引
    # 如果get_indexer没找到相应索引，就会返回-1，在np后续访问时会访问最后一个元素
    # 不过由于这里的frame是panel生成的，所以预期一定能够找到
    row, col = prices.index.get_indexer(frame.trade_date), prices.columns.get_indexer(frame.ticker)
    if (row == -1).any() or (col == -1).any():
        raise ValueError("input frame index not found in panel")
    for h in horizons:
        labels, end, reason = market_forward_returns(prices, panel.calendar, h)
        # 这里得到的结果都是一个一维np数组，[row,col]是一一配对，不是全部交叉组合
        frame[f"future_ret_{h}"] = labels.to_numpy()[row,col]
        frame[f"label_end_{h}"] = frame.trade_date.map(end)
        frame[f"label_reason_{h}"] = reason.to_numpy()[row,col]
    frame["observation_status"] = panel.observation_status().to_numpy()[row,col]
    frame["close_raw"] = panel.matrix("close",clean=False).to_numpy()[row,col]
    return frame


def add_controls(primary):
    caps = pd.read_parquet(OUT/"auxiliary/daily_basic.parquet").rename(columns={"ts_code":"ticker"})
    caps["trade_date"] = parse_dates(caps.trade_date).to_numpy()
    frame = primary.merge(caps[["ticker","trade_date","total_mv"]],on=["ticker","trade_date"],how="left",validate="one_to_one")
    frame["industry"] = align_industry_history(frame,pd.read_parquet(OUT/"auxiliary/industry_history.parquet"))
    for factor in FACTORS:
        frame[factor+"_neutral"] = np.nan
    for _,group in frame.groupby("trade_date"):
        for factor in FACTORS:
            frame.loc[group.index,factor+"_neutral"] = neutralize(group[factor],group.total_mv,group.industry)
    return frame


def portfolios(frame, panel, dates, config):
    limits = pd.read_parquet(OUT/"auxiliary/stk_limit.parquet").rename(columns={"ts_code":"ticker"})
    limits["trade_date"] = parse_dates(limits.trade_date).to_numpy()
    tickers = pd.Index(sorted(frame.ticker.unique()),name="ticker")
    matrices = trading_prices(panel,dates,tickers)
    up,down = [limits.pivot(index="trade_date",columns="ticker",values=c).reindex(index=dates,columns=tickers)
               for c in ["up_limit","down_limit"]]
    source = pd.read_csv(OUT/"auxiliary/benchmark.csv",dtype={"trade_date":str})
    benchmark = pd.Series(source.close.to_numpy(),index=pd.DatetimeIndex(parse_dates(source.trade_date))).sort_index()
    signals = {day:g.loc[quantile_groups(g[config["portfolio"]["factor"]])==config["portfolio"]["group"],"ticker"].tolist()
               for day,g in frame.groupby("trade_date")}
    stats,books,fills,positions = [],[],[],[]
    # 20d/offset0 remains the original main strategy. 5d and every other offset
    # are complete diagnostic comparisons; never select an optimal trajectory.
    for holding_days in [20,5]:
        for offset in range(holding_days):
            fees = config["portfolio"]["fee_bps"] if offset==0 else [15]
            for bps in fees:
                ledger,trades,held = backtest_long_only(*matrices,up,down,signals,fee=bps/10000,
                                                       rebalance_every=holding_days,offset=offset)
                tags = dict(holding_days=holding_days,offset=offset,fee_bps=bps)
                stats.append(dict(**tags,**performance(ledger,benchmark)))
                if offset==0:
                    books.append(ledger.assign(**tags));fills.append(trades.assign(**tags));positions.append(held.assign(**tags))
    for name,frames in [("portfolio",books),("trades",fills),("holdings",positions)]:
        pd.concat(frames,ignore_index=True).to_parquet(OUT/f"details/{name}.parquet",index=False)
    return pd.DataFrame(stats),benchmark


def main():
    for folder in [OUT/"details",OUT/"figures",OUT/"evidence"]:
        folder.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s",handlers=[
        logging.FileHandler(OUT/"evidence/run.log",mode="w"),logging.StreamHandler()])
    with (OUT/"evidence/tests.txt").open("w") as test_log:
        subprocess.run([sys.executable,"-B","-m","unittest","discover","-s","tests","-v"],
                       cwd=ROOT,stdout=test_log,stderr=subprocess.STDOUT,check=True)
    config = json.loads((OUT/"config.json").read_text())
    cal = pd.DatetimeIndex(parse_dates(pd.read_parquet(ROOT/"data/raw/calender/trade_calender_2016_2025.parquet").cal_date)).sort_values()
    working = cal[(cal>=config["working_start"]) & (cal<=config["calendar_end"])]
    dates = working[(working>=config["signal_start"]) & (working<=config["signal_end"])]
    common = dates[dates<=config["common_signal_end"]]
    members = prepare_membership(pd.read_parquet(ROOT/"data/processed/universe/csi500_membership.parquet"))
    panel = prepare_panel(pd.read_parquet(ROOT/"data/processed/daily/csi500_processed_daily.parquet"),working)
    logging.info("Computing frozen primary specification")
    primary = attach_labels(compute_factors(panel,members,dates,config["price_mode"],config["rank_policy"]),panel,config["horizons"])
    primary = add_controls(primary)
    primary.to_parquet(OUT/"details/research_panel.parquet",index=False)
    daily,quantiles,correlations,incremental = evaluate_panel(primary,FACTORS,min_pairs=config["min_pairs"])
    summaries,daily_tables,quantile_tables = [],[daily.assign(specification="primary")],[quantiles.assign(specification="primary")]
    base = summarize_ic(daily[daily.trade_date.isin(common)],common,bootstrap=True)
    summaries.append(base.assign(analysis="primary",specification="primary",period="all"))
    for year in [2023,2024,2025]:
        selected = common[common.year==year]
        summaries.append(summarize_ic(daily[daily.trade_date.isin(selected)],selected).assign(
            analysis="yearly",specification="primary",period=str(year)))
    summaries.append(summarize_ic(daily[daily.trade_date.isin(common)],common,lag=config["hac_sensitivity_lag"]).assign(
        analysis="hac60",specification="primary",period="all"))
    for h in config["horizons"]:
        selected = dates[working.get_indexer(dates)+h<len(working)]
        summaries.append(summarize_ic(daily[(daily.horizon==h)&daily.trade_date.isin(selected)],selected).assign(
            analysis="own_horizon",specification="primary",period="all"))
    for metric in ["residual_ic","matched_raw_ic","delta_ic","alpha_beta"]:
        for (factor,h),group in incremental.groupby(["factor","horizon"]):
            series = group.set_index("trade_date")[metric].reindex(common)
            stats = hac_mean(series,lag=config["hac_lag"])
            summaries.append(pd.DataFrame([dict(analysis="incremental",specification="primary",period="all",
                factor=factor,horizon=h,metric=metric,icir=series.mean()/series.std() if metric in ["residual_ic","matched_raw_ic"] else np.nan,**stats)]))
    incremental = incremental.merge(correlations,on=["trade_date","factor"],how="left",validate="many_to_one")
    incremental.to_parquet(OUT/"details/incremental.parquet",index=False)
    for price_mode,policy in [("raw","historical_members"),("adjusted","current_cohort")]:
        logging.info("Frozen sensitivity: %s / %s",policy,price_mode)
        frame = attach_labels(compute_factors(panel,members,dates,price_mode,policy),panel,config["horizons"])
        dd,_,_,_ = evaluate_panel(frame,FACTORS,min_pairs=config["min_pairs"])
        name = f"{policy}_{price_mode}"
        daily_tables.append(dd.assign(specification=name))
        summaries.append(summarize_ic(dd[dd.trade_date.isin(common)],common).assign(analysis="robustness",specification=name,period="all"))
    supported = primary.copy()
    supported.loc[~supported[list(FACTORS)].notna().all(axis=1),list(FACTORS)] = np.nan
    dd,_,_,_ = evaluate_panel(supported,FACTORS,min_pairs=config["min_pairs"])
    summaries.append(summarize_ic(dd[dd.trade_date.isin(common)],common).assign(analysis="robustness",specification="common_factor_support",period="all"))
    neutral_factors = []
    for factor in FACTORS:
        primary[factor+"_matched"] = primary[factor].where(primary[factor+"_neutral"].notna())
        neutral_factors.extend([factor+"_matched",factor+"_neutral"])
    dd,qq,_,_ = evaluate_panel(primary,neutral_factors,min_pairs=config["min_pairs"])
    daily_tables.append(dd.assign(specification="neutralization"));quantile_tables.append(qq.assign(specification="neutralization"))
    summaries.append(summarize_ic(dd[dd.trade_date.isin(common)],common).assign(analysis="neutralization",specification="SW_effective_history",period="all"))
    summary = pd.concat(summaries,ignore_index=True)
    summary["p_holm_expanded"] = holm_adjust(summary.p_value)
    summary.to_csv(OUT/"ic_summary.csv",index=False,float_format="%.15g")
    pd.concat(daily_tables,ignore_index=True).to_parquet(OUT/"details/daily_ic.parquet",index=False)
    all_quantiles = pd.concat(quantile_tables,ignore_index=True)
    all_quantiles.to_parquet(OUT/"details/daily_quantiles.parquet",index=False)
    qm = []
    for name,group in all_quantiles[all_quantiles.trade_date.isin(common)].groupby("specification"):
        means,spreads = summarize_quantiles(group)
        means = means.merge(spreads,on=["factor","horizon"],validate="many_to_one")
        qm.append(means.assign(specification=name))
    pd.concat(qm,ignore_index=True).to_csv(OUT/"quantile_summary.csv",index=False,float_format="%.15g")
    logging.info("Diagnosing distributions and attribution; original samples unchanged")
    detail,diagnostics,influence = diagnose(primary[primary.trade_date.isin(common)],working,members,config["min_pairs"])
    detail.to_parquet(OUT/"details/diagnostics.parquet",index=False)
    diagnostics.to_csv(OUT/"diagnostics.csv",index=False,float_format="%.15g")
    influence.to_csv(OUT/"influential_stocks.csv",index=False,float_format="%.15g")
    portfolio,benchmark = portfolios(primary,panel,dates,config)
    portfolio.to_csv(OUT/"portfolio_summary.csv",index=False,float_format="%.15g")
    plot_results(OUT,benchmark)
    write_reports(ROOT,OUT)
    validate_results(ROOT,OUT)
    logging.info("Complete: reports, regression comparisons and input hashes verified")


if __name__=="__main__":
    main()
