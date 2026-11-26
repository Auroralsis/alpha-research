"""用固定的四个 LightGBM 预测评价六种市场状态融合规则。"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest.engine import run_backtests
from src.backtest.portfolio import performance, QLIB_DAILY_ANNUALIZATION
from src.data.research_panel import CSI500_INPUTS, ResearchPanel, load_market_calendar, prepare_membership, prepare_panel
from src.ml.rank_validation import BASES, validation_loss
from src.scripts.run_all import RUNS, new_run

ARMS = tuple("ABCDEF")
REFERENCE = RUNS / "daily_pair_tuning_20261122_100328_550844"
VALID_BENCH = CSI500_INPUTS["benchmark"].with_name("benchmark_20220104_20221230.csv")


def indicators(close, prices, members):
    """所有滚动计算以完整交易日为索引，先算个股均线，再取当日成员。"""
    state = close.rename("close").to_frame()
    state["ma20"] = close.rolling(20).mean()
    state["ma60"] = close.rolling(60).mean()
    logprice = np.log(close.where(close.gt(0)))
    state["r20"] = logprice.diff(20)
    state["sigma20"] = logprice.diff().rolling(20).std(ddof=0)
    state["z"] = state.r20 / (np.sqrt(20) * state.sigma20.where(state.sigma20.gt(0)))
    ma = prices.rolling(20).mean()
    valid = ma.notna() & prices.notna() & members
    state["members"] = members.sum(axis=1)
    state["valid"] = valid.sum(axis=1)
    state["above"] = (prices.gt(ma) & valid).sum(axis=1)
    state["coverage"] = state.valid / state.members.where(state.members.gt(0))
    state["breadth20"] = state.above / state.valid.where(state.valid.gt(0))
    state["A"], state["B"], state["C"] = 0.5, 0.7, 0.3
    state["D"] = np.select([(close > state.ma20) & (state.ma20 > state.ma60),
                            (close < state.ma20) & (state.ma20 < state.ma60)], [.7, .3], default=.5)
    trend = np.tanh(state.z)
    state["E"] = (.5 + .2 * trend).fillna(.5).clip(.3, .7)
    state["F"] = (.5 + .1 * (trend + 2 * state.breadth20 - 1)).fillna(.5).clip(.3, .7)
    return state


def predict(frame, state):
    ranks = frame.groupby("trade_date")[list(BASES)].rank(pct=True)
    assert np.isfinite(ranks).all().all()
    aggressive, defensive = ranks.iloc[:, :2].mean(axis=1), ranks.iloc[:, 2:].mean(axis=1)
    baseline = ranks.mean(axis=1)
    np.testing.assert_array_equal(baseline, frame.equal)
    scores = frame[["trade_date", "ticker"]].copy()
    for arm in ARMS:
        w = frame.trade_date.map(state[arm])
        assert w.notna().all()
        # 权重中性时直接使用基线分数，避免浮点运算改变排序或打破并列。
        scores[arm] = (w * aggressive + (1 - w) * defensive).where(w.ne(.5), baseline)
    return scores


def nplhr_predictions(folder, cfg, split, frame):
    from src.ml.dataset import alpha158_columns, read_feature_split
    from src.ml.rank_validation import equal_ranks
    from src.ml.tuning import frozen_scores

    keys = ["trade_date", "ticker"]
    if split == "test":
        columns = {f"lightgbm_{name}": name for name in BASES}
        raw = pd.read_parquet(folder / "predictions.parquet", columns=[*keys, *columns, "lightgbm"])
        raw = raw.rename(columns={**columns, "lightgbm": "equal"}).set_index(keys)
    else:
        source = Path(cfg["feature_source"])
        bounds = cfg["segments"][split]
        x = read_feature_split(source / "features.parquet", bounds, split).set_index(keys)
        factor = pd.read_parquet(folder / "factor.parquet", columns=[*keys, "nPLHR"],
                                 filters=[("trade_date", ">=", pd.Timestamp(bounds[0])),
                                          ("trade_date", "<=", pd.Timestamp(bounds[1]))]).set_index(keys)
        x = x[alpha158_columns(source)].join(factor, validate="one_to_one")
        raw = frozen_scores(folder, "lightgbm", cfg["lightgbm"], x).reset_index()
        raw["equal"] = equal_ranks(raw, list(BASES))
        raw = raw.set_index(keys)
    aligned = frame.drop(columns=[*BASES, "equal"]).set_index(keys)
    pd.testing.assert_index_equal(aligned.sort_index().index, raw.sort_index().index)
    return aligned.join(raw, validate="one_to_one").reset_index()


def signal_metrics(frame, scores, cfg):
    gains = {t: cfg["label_gains"][t] for t in BASES}
    valid = frame.loc[frame.evaluable]
    ranked_label = valid.groupby("trade_date").label.rank()
    rows = []
    for arm in ARMS:
        loss = validation_loss(frame, scores[arm], gains, cfg["topk"]).set_index("trade_date").loss
        rank = scores.loc[valid.index].groupby("trade_date")[arm].rank()
        ic = rank.groupby(valid.trade_date).corr(ranked_label)
        rows.append(pd.DataFrame(dict(ndcg50=1-loss, rank_ic=ic, strategy=arm)))
    return pd.concat(rows).rename_axis("trade_date").reset_index()


def summarize(out, book, signals, state, split):
    records, periods = [], []
    for arm, block in book.groupby("strategy"):
        block = block.sort_values("trade_date").set_index("trade_date")
        windows = {"2022": block} if split == "validation" else {
            "2023-2025": block, **{str(y): g for y, g in block.groupby(block.index.year)}}
        comparison = signals[signals.strategy.eq(arm)].set_index("trade_date")
        for label, g in windows.items():
            r, b = g.daily_return, g.benchmark_return
            metrics = performance(g.assign(nav=(1+r).cumprod()).reset_index(), b)
            beta = r.cov(b) / b.var()
            w = state.loc[g.index, arm]
            records.append(dict(strategy=arm, period=label, days=len(g), **metrics, beta=beta,
                                alpha_annual=(r.mean()-beta*b.mean())*QLIB_DAILY_ANNUALIZATION,
                                calmar=metrics["annual_return"]/abs(metrics["max_drawdown"]),
                                **comparison.reindex(g.index)[["ndcg50","rank_ic"]].mean(),
                                w_mean=w.mean(), w_std=w.std(ddof=0), fee_rate_sum=g.cost_rate.sum()))
        if split == "test":
            windows.update(rally=block.loc["2025-07-15":"2025-12-31"],
                           drawdown=block.loc["2023-08-02":"2024-02-05"])
        for label, g in windows.items():
            for side, mask in {"all": g.daily_return.notna(), "up": g.benchmark_return.gt(0),
                               "down": g.benchmark_return.lt(0)}.items():
                sub = g[mask]
                r,b = sub.daily_return,sub.benchmark_return
                periods.append(dict(strategy=arm, period=label, side=side, days=len(sub),
                                    total_return=(1+r).prod()-1, index_return=(1+b).prod()-1,
                                    excess_mean=(r-b).mean(), excess_sum=(r-b).sum(),
                                    capture=r.mean()/b.mean() if b.mean() else np.nan))
    stats = pd.DataFrame(records).set_index(["period","strategy"])
    for key in ("cost_cny","annual_turnover_one_way","fee_rate_sum"):
        stats[key+"_vs_A"] = stats[key]-stats[key].xs("A",level="strategy").reindex(stats.index,level="period")
    stats.to_csv(out/"metrics.csv")
    pd.DataFrame(periods).to_csv(out/"market_days.csv", index=False)


def charts(out, book, state):
    r = book.pivot(index="trade_date", columns="strategy",values="daily_return")
    b = book.drop_duplicates("trade_date").set_index("trade_date").benchmark_return.reindex(r.index)
    nav = (1+r).cumprod()
    beta = r.rolling(60).cov(b).div(b.rolling(60).var(), axis=0)
    beta.to_csv(out/"rolling_beta60.csv")
    def draw(filename, panels):
        fig, axes = plt.subplots(len(panels), 1, figsize=(11, 4*len(panels)), squeeze=False)
        for ax, (label, data) in zip(axes.flat, panels):
            data.plot(ax=ax, linewidth=1.1)
            ax.set(title=label, xlabel="", ylabel="")
            ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(out/filename, dpi=145)
        plt.close(fig)
    draw("nav_excess.png", [("Net asset value (after fees)",nav.assign(CSI500=(1+b).cumprod())),
                            ("Cumulative arithmetic excess",r.sub(b,axis=0).cumsum())])
    draw("drawdown.png",[("Portfolio drawdown",nav/nav.cummax()-1)])
    draw("weights.png",[("Aggressive weight on signal date",state[list("DEF")])])
    draw("beta.png",[("Trailing 60-session beta",beta)])
    draw("breadth.png",[("CSI 500 members above their 20-session MA",state.breadth20)])


def write_report(out):
    from src.report.research import markdown_table

    metrics = pd.concat([pd.read_csv(out / part / "metrics.csv", dtype={"period": str})
                         for part in ("validation", "test")], ignore_index=True)
    metrics.to_csv(out / "metrics_all.csv", index=False)
    experiment = json.loads((out / "config.json").read_text())
    feature_set = "Alpha158 + nPLHR" if experiment.get("nplhr_archive") else "Alpha158"
    audit = ("A 测试分数与 159 列归档纯 LightGBM 四等权逐股票日核对。"
             if experiment.get("nplhr_archive") else "A 测试账本已与归档逐日对齐核验。")
    parts = [f"# 中证500四排序器市场状态融合（{feature_set}）",
             "A 等权、B 固定进攻、C 固定防守、D 双均线、E 波动率标准化趋势、F 趋势加宽度。"
             "复用同一批 LightGBM 分数。T 日收盘定权，T+1 收盘按原 Top50/Drop5 执行。"
             "2022 年固定验证与 2023–2025 年历史诊断分别从 1 亿元建仓；测试期跨年连续持仓。",
             "## 全期与逐年指标",
             "相对中证500价格指数计算算术年化超额、IR 和最大超额回撤。绝对收益与净值回撤按复利计算。"
             "采用项目的 238 日年化。Alpha 为扣费日收益对指数日收益的带截距回归截距乘 238，"
             "无风险收益取零。Calmar 为绝对年化收益除以最大回撤绝对值。费用为人民币实付金额。"]
    full = metrics[metrics.period.eq("2023-2025")].set_index("strategy")
    valid = metrics[metrics.period.eq("2022")].set_index("strategy")
    market = pd.read_csv(out / "test/market_days.csv")
    rally = market[market.period.eq("rally") & market.side.eq("up")].set_index("strategy")
    parts[2:2] = ["## 六组比较",
                  f"2022 年固定验证中，年化超额最高为 {valid.annualized_excess_return.idxmax()}，"
                  f"IR 最高为 {valid.information_ratio.idxmax()}，NDCG@50 最高为 {valid.ndcg50.idxmax()}。"
                  f"2023–2025 年历史诊断中，年化超额最高为 {full.annualized_excess_return.idxmax()}，"
                  f"IR 最高为 {full.information_ratio.idxmax()}，净值最大回撤最浅为 {full.max_drawdown.idxmax()}。",
                  f"B 对 A：2025 年 7 月 15 日至年底的上涨日捕获率从 {rally.at['A','capture']:.1%} "
                  f"变为 {rally.at['B','capture']:.1%}。C 对 A：全测试期最大回撤从 "
                  f"{full.at['A','max_drawdown']:.2%} 变为 {full.at['C','max_drawdown']:.2%}。",
                  f"D 对 E：双均线与连续趋势的年化超额分别为 {full.at['D','annualized_excess_return']:.2%}、"
                  f"{full.at['E','annualized_excess_return']:.2%}；加入宽度后的 F 为 "
                  f"{full.at['F','annualized_excess_return']:.2%}。各方案在下表按同一年度、同一账本口径比较。"]
    for period, g in metrics.groupby("period", sort=False):
        parts.extend([f"### {period}", markdown_table(
            ["方案","累计","绝对年化","年化超额","IR","Sharpe","Calmar","最大回撤","最大超额回撤","年化波动"],
            [[r.strategy,f"{r.total_return:.2%}",f"{r.annual_return:.2%}",f"{r.annualized_excess_return:.2%}",
              f"{r.information_ratio:.2f}",f"{r.sharpe:.2f}",f"{r.calmar:.2f}",f"{r.max_drawdown:.2%}",
              f"{r.max_excess_drawdown:.2%}",f"{r.annual_volatility:.2%}"] for r in g.itertuples()]),
            markdown_table(["方案","年化 Alpha","Beta","RankIC","NDCG@50","年化单边换手（倍）","费用（万元）",
                            "进攻权重均值","权重标准差","较 A 换手差（倍）","较 A 费用差（万元）"],
                [[r.strategy,f"{r.alpha_annual:.2%}",f"{r.beta:.3f}",f"{r.rank_ic:.4f}",f"{r.ndcg50:.5f}",
                  f"{r.annual_turnover_one_way:.2f}",f"{r.cost_cny/1e4:.2f}",f"{r.w_mean:.1%}",f"{r.w_std:.1%}",
                  f"{r.annual_turnover_one_way_vs_A:+.2f}",f"{r.cost_cny_vs_A/1e4:+.2f}"] for r in g.itertuples()])])
    for key, title in [("2023-2025","三年测试期"),("rally","2025-07-15 至 2025-12-31"),
                       ("drawdown","2023-08-01 收盘高点至 2024-02-05 收盘低点")]:
        sample = market[market.period.eq(key)].set_index(["strategy","side"])
        rows = []
        for arm in ARMS:
            full,up,down = (sample.loc[arm,side] for side in ("all","up","down"))
            rows.append([arm,f"{full.total_return:.2%}",f"{full.index_return:.2%}",f"{up.capture:.1%}",
                         f"{1e4*up.excess_mean:+.1f}",f"{1e4*down.excess_mean:+.1f}"])
        parts.extend([f"### {title}",markdown_table(["方案","累计收益","指数累计","上涨捕获率","上涨日均超额 bp",
                                                    "下跌日均超额 bp"],rows)])
    parts.extend(["上涨与下跌由账本当日指数收益符号划分，收益按实际发生日对齐。上涨捕获率等于上涨日策略日均收益除以指数日均收益。"
                  "峰谷归因从高点后的第一个交易日开始。逐年及验证期分别见 test/market_days.csv 和 validation/market_days.csv。",
                  "## 图表",
                  *[f"![{title}]({file}.png)" for title,file in [("测试净值与累计算术超额","nav_excess"),
                    ("测试净值回撤","drawdown"),("每日进攻权重","weights"),("测试滚动60日 Beta","beta"),
                    ("历史成员市场宽度","breadth")]],
                  "## 数据与复核",
                  "daily_weights.csv 记录各日指数、两条均线、20 日收益与波动、z、成员总数、有效数、高于均线数、覆盖率及进攻权重。"
                  "每个进攻模型权重为 w/2，每个防守模型为 (1−w)/2。宽度仅使用当日历史成员；"
                  "股票均线要求连续 20 个交易日有有效复权行情，分母只计有效成员。"
                  "指数预热不足回到等权，2022 年初 D 有 59 天、E/F 有 20 天。"
                  "两个回测目录各自保存股票分数、交易、持仓、账本及逐日 NDCG 和 RankIC。" + audit,
                  "研究背景见 [Factor Timing](https://doi.org/10.1093/rfs/hhaa017)、"
                  "[Time Series Momentum](https://doi.org/10.1016/j.jfineco.2011.11.003) 和 "
                  "[Momentum Crashes](https://doi.org/10.1016/j.jfineco.2015.12.002)。六种融合规则及参数为本项目设定。"])
    (out / "report.md").write_text("\n\n".join(parts)+"\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference",type=Path,default=REFERENCE)
    parser.add_argument("--nplhr-archive", type=Path, help="159 列 nPLHR 因子、冻结模型和测试分数目录")
    parser.add_argument("--out",type=Path)
    args = parser.parse_args(argv)
    out = args.out or new_run(1,tag="regime_fusion")
    out.mkdir(parents=True,exist_ok=False)
    cfg = json.loads((args.reference/"config.json").read_text())
    if args.nplhr_archive:
        extended = json.loads((args.nplhr_archive.parent / "config.json").read_text())
        winner = json.loads((args.reference / "winners.json").read_text())["lightgbm"]
        assert extended["lightgbm"] == winner
        for key in ("segments","label_gains","topk","n_drop","seed","buy_price","buy_fee_bps","sell_fee_bps",
                    "min_trade_cost_cny","initial_capital_cny","feature_source","rebalance_days","trade_unit","limit_threshold"):
            assert extended[key] == cfg[key], key
        cfg = extended
    (out/"config.json").write_text(json.dumps(dict(reference=str(args.reference.resolve()), config=cfg,
        nplhr_archive=str(args.nplhr_archive.resolve()) if args.nplhr_archive else None,
        arms=dict(zip(ARMS,["Baseline","Static-Aggressive","Static-Defensive","Dynamic-MA","Dynamic-Trend",
                            "Dynamic-Breadth"])), trend_days=20, slow_ma=60, defensive_weight=.3,
        aggressive_weight=.7, trend_share_in_F=.5),indent=2))
    calendar = load_market_calendar(CSI500_INPUTS["calendar"])
    start,end = cfg["segments"]["validation"][0],cfg["segments"]["test"][1]
    dates = calendar[max(0,calendar.searchsorted(start)-60):calendar.searchsorted(end,side="right")]
    panel = prepare_panel(pd.read_parquet(CSI500_INPUTS["prices"]), dates)
    prices = panel.matrix("close_adj")
    members = prepare_membership(pd.read_parquet(CSI500_INPUTS["membership"]))
    members = members[members.trade_date.isin(dates)].assign(member=True).pivot(
        index="trade_date",columns="ticker",values="member").eq(True).reindex(index=dates,fill_value=False)
    prices = prices.reindex(columns=members.columns)
    index = pd.concat([pd.read_csv(f,usecols=["trade_date","close"],parse_dates=["trade_date"])
                       for f in (VALID_BENCH,CSI500_INPUTS["benchmark"])]).drop_duplicates()
    close = index.set_index("trade_date").close.reindex(dates)
    state = indicators(close,prices,members).loc[start:end]
    state.to_csv(out/"daily_weights.csv",index_label="trade_date")
    cols=["trade_date","ticker","label","evaluable","grade","equal",*BASES]
    for split in ("validation","test"):
        folder = out/split
        folder.mkdir()
        frame=pd.read_parquet(args.reference/"predictions.parquet",columns=cols,filters=[("split","==",split)])
        if args.nplhr_archive:
            frame = nplhr_predictions(args.nplhr_archive, cfg, split, frame)
        scores=predict(frame,state)
        scores.to_parquet(folder/"scores.parquet",index=False)
        signal=signal_metrics(frame,scores,cfg)
        signal.to_csv(folder/"daily_signal_metrics.csv",index=False)
        lo,hi=cfg["segments"][split]
        sessions=calendar[(calendar>=lo)&(calendar<=hi)]
        inputs=dict(CSI500_INPUTS, benchmark=VALID_BENCH if split=="validation" else CSI500_INPUTS["benchmark"])
        # 与原回测一致，为 Qlib 的价格归一化预留 21 个交易日的行情。
        first = calendar.get_loc(sessions[0])
        trading_dates = calendar[max(first-21, 0):first+len(sessions)]
        trading_panel = ResearchPanel(panel.daily.loc[panel.daily.trade_date.isin(trading_dates)],
                                      trading_dates, panel.tickers)
        run_backtests(inputs,cfg,sessions,scores,trading_panel,folder,strategies=ARMS)
        book=pd.read_parquet(folder/"ledger.parquet")
        summarize(folder,book,signal,state,split)
        if split=="test" and not args.nplhr_archive:
            original=pd.read_parquet(args.reference/"ledger.parquet",filters=[("strategy","==","equal")])
            numeric=["daily_return","benchmark_return","nav","cost_cny","turnover_two_sided"]
            pd.testing.assert_frame_equal(book[book.strategy.eq("A")].set_index("trade_date")[numeric],
                                          original.set_index("trade_date")[numeric],rtol=1e-10,atol=1e-10)
        if split=="test":
            charts(out,book,state)
    write_report(out)
    print(out,flush=True)


if __name__=="__main__":
    main()
