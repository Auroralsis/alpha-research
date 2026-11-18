"""Run the documented single-factor replication on frozen CSI 500 inputs."""

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/factor_study_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.research_panel import parse_dates, prepare_membership, prepare_panel
from src.evaluation.cross_section import daily_factor_tests, summarize_factor_tests
from src.evaluation.layer_backtest import run_layer_tracks, simulate_track, summarize_tracks, track_metrics
from src.evaluation.portfolio import trading_prices
from src.features.price_volume import compute_factors, quantile_groups
from src.features.neutralization import align_industry_history, mad_clip, neutralize, standardize
from src.features.targets import attach_labels

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/factor_study"
REPORT = ROOT / "docs/factor_study.md"
AUX = ROOT / "data/reference/csi500"
INPUTS = dict(prices=ROOT / "data/processed/daily/csi500_processed_daily.parquet",
              membership=ROOT / "data/processed/universe/csi500_membership.parquet",
              calendar=ROOT / "data/raw/calender/trade_calender_2016_2025.parquet",
              caps=AUX / "daily_basic.parquet", industries=AUX / "industry_history.parquet",
              limits=AUX / "stk_limit.parquet", benchmark=AUX / "benchmark.csv")
SOURCE = "https://crm.htsc.com.cn/doc/2019/10750101/dae9cd9e-c2cd-4f05-bb6c-e5ac8d216d3d.pdf"


def prepare_study(design):
    calendar = pd.DatetimeIndex(parse_dates(pd.read_parquet(INPUTS["calendar"]).cal_date)).sort_values()
    working = calendar[(calendar >= design["warmup_start"]) & (calendar <= design["end"])]
    dates = working[working >= design["signal_start"]]
    members = prepare_membership(pd.read_parquet(INPUTS["membership"]))
    panel = prepare_panel(pd.read_parquet(INPUTS["prices"]), working)
    frame = compute_factors(panel, members, dates, design["factor_price"], design["rank_policy"])
    frame = attach_labels(frame, panel, design["horizons"])
    caps = pd.read_parquet(INPUTS["caps"]).rename(columns={"ts_code": "ticker"})
    caps["trade_date"] = parse_dates(caps.trade_date).to_numpy()
    frame = frame.merge(caps, on=["ticker", "trade_date"], how="left", validate="one_to_one")
    frame["industry"] = align_industry_history(frame, pd.read_parquet(INPUTS["industries"]))
    observed = panel.observation_status().eq("observed").shift(-1, fill_value=False)
    frame["next_session_observed"] = observed.to_numpy()[observed.index.get_indexer(frame.trade_date),
                                                         observed.columns.get_indexer(frame.ticker)]
    for factor in design["factors"]:
        frame[f"{factor}_standardized"] = frame.groupby("trade_date")[factor].transform(
            lambda values: standardize(mad_clip(values)))
        frame[f"{factor}_neutral"] = np.nan
        for _, group in frame.groupby("trade_date"):
            frame.loc[group.index, f"{factor}_neutral"] = neutralize(group[factor], group.total_mv, group.industry)
    return frame, panel, dates


def market_inputs(frame, panel, dates):
    tickers = pd.Index(sorted(frame.ticker.unique()), name="ticker")
    limits = pd.read_parquet(INPUTS["limits"]).rename(columns={"ts_code": "ticker"})
    limits["trade_date"] = parse_dates(limits.trade_date).to_numpy()
    bounds = [limits.pivot(index="trade_date", columns="ticker", values=column).reindex(index=dates, columns=tickers)
              for column in ("up_limit", "down_limit")]
    loose = trading_prices(panel, dates, tickers)
    adjusted, raw = [panel.matrix(column).reindex(index=dates, columns=tickers) for column in ("close_adj", "close")]
    benchmark = pd.read_csv(INPUTS["benchmark"], dtype={"trade_date": str})
    benchmark = pd.Series(benchmark.close.to_numpy(), index=pd.DatetimeIndex(parse_dates(benchmark.trade_date)))
    return (*loose, *bounds), (adjusted, adjusted, raw, *bounds), benchmark


def compare_execution(frame, open_quotes, close_quotes, benchmark, design):
    holding, fee = design["main_holding_days"], design["main_fee_bps"]
    rows = []
    for score, quote_sets in [("Alpha13", {"open": open_quotes, "close": close_quotes}),
                              ("Alpha13_neutral", {"close": close_quotes})]:
        signals = {day: group.loc[quantile_groups(group[score]) == 5, "ticker"].tolist()
                   for day, group in frame.groupby("trade_date")}
        for execution, quotes in quote_sets.items():
            for offset in range(holding):
                ledger = simulate_track(quotes, signals, holding, offset, fee)
                rows.append(dict(score=score, execution=execution, offset=offset, **track_metrics(ledger, benchmark)))
    return pd.DataFrame(rows)


def markdown_table(frame):
    header = "| " + " | ".join(frame.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(frame.columns)) + " |"
    body = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *body])


def percent(values):
    return values.map(lambda value: f"{value:.2%}" if pd.notna(value) else "—")


def write_report(summary, aggregate, tracks, nav, benchmark, dates, ranges, frame, daily, design):
    artifacts = Path(os.path.relpath(OUT, REPORT.parent)).as_posix()
    base_fee, main_horizon, layer_count = (design[key] for key in ("main_fee_bps", "main_holding_days", "layers"))
    columns = {"factor": "因子", "treatment": "处理", "horizon": "预测日数", "mean_abs_t": "平均|t|",
               "share_abs_t_gt_2": "|t|>2占比", "mean_t": "平均t", "mean_factor_return": "平均回归斜率",
               "mean_rank_ic": "平均IC", "rank_ic_std": "IC标准差", "icir": "ICIR", "share_ic_positive": "正IC占比"}
    regressions = summary[list(columns)].copy()
    for col in ("share_abs_t_gt_2", "mean_factor_return", "mean_rank_ic", "rank_ic_std", "share_ic_positive"):
        regressions[col] = percent(regressions[col])
    for col in ("mean_abs_t", "mean_t", "icir"):
        regressions[col] = regressions[col].map(lambda value: f"{value:.3f}")
    regressions = regressions.rename(columns=columns)
    regressions.columns = regressions.columns.str.replace("|", "&#124;", regex=False)
    selected = aggregate[(aggregate.kind.eq("layer") & aggregate.number.eq(1)) | aggregate.kind.eq("spread")]
    selected = selected[selected.fee_bps.eq(base_fee)].copy()
    portfolios = selected.assign(组合=np.where(selected.kind.eq("spread"), "最高减最低", "最高分层"),
                                 持有日数=selected.holding_days, 因子=selected.factor,
                                 平均年化=percent(selected.mean_annual), 最低年化=percent(selected.min_annual),
                                 最高年化=percent(selected.max_annual), 平均累计=percent(selected.mean_total),
                                 平均最大回撤=percent(selected.mean_drawdown),
                                 平均相对年化=percent(selected.mean_annual_relative),
                                 跑赢指数轨道=selected.outperform_tracks.map(lambda value: f"{value:.0f}" if pd.notna(value) else "—"))
    portfolios = portfolios[["因子", "持有日数", "组合", "平均年化", "最低年化", "最高年化", "平均累计",
                             "平均最大回撤", "平均相对年化", "跑赢指数轨道"]]
    benchmark_total = benchmark.loc[dates[-1]] / benchmark.loc[dates[0]] - 1
    benchmark_annual = (1 + benchmark_total) ** (252 / len(dates)) - 1
    inputs_with_next_session = int(frame.trade_date.lt(dates[-1]).sum())
    missing_next = inputs_with_next_session - int(frame.next_session_observed.sum())
    main = summary[summary.horizon.eq(main_horizon)]
    main_tracks = selected[selected.holding_days.eq(main_horizon) & selected.kind.eq("layer")]
    check = daily.groupby("horizon").agg(min_ic=("n_ic", "min"), min_wls=("n_wls", "min"))
    coverage = ", ".join(f"{int(horizon)}日={len(window)}个信号日" for horizon, window in ranges.items())
    ic_passed = summary.loc[summary.ic_holm_p.lt(.05), ["factor", "treatment", "horizon"]]
    passed = "、".join(f"{factor} {kind} {horizon}日" for factor, kind, horizon in ic_passed.itertuples(index=False, name=None))
    slope_passed = int(summary.return_holm_p.lt(.05).sum())
    top_risk = selected.loc[selected.holding_days.eq(main_horizon)].copy()
    index_nav = benchmark.reindex(dates) / benchmark.loc[dates[0]]
    primary_nav = nav.query('holding_days == @main_horizon and fee_bps == @base_fee and kind == "layer" and number == 1')
    relative_rows = []
    for (factor, offset), path in primary_nav.groupby(["factor", "offset"]):
        path = path.set_index("trade_date").reindex(dates)
        excess = path.daily_return - index_nav.pct_change().fillna(0)
        relative_nav = path.nav / index_nav
        relative_dd = (relative_nav / relative_nav.cummax() - 1).min()
        relative_annual = relative_nav.iloc[-1] ** (252 / len(dates)) - 1
        relative_rows.append(dict(factor=factor, offset=offset, relative_dd=relative_dd,
                                  information_ratio=np.sqrt(252) * excess.mean() / excess.std(ddof=1),
                                  relative_calmar=relative_annual / abs(relative_dd) if relative_dd < 0 else np.nan))
    relative_stats = pd.DataFrame(relative_rows).groupby("factor").mean()
    primary_tracks = tracks.query('holding_days == @main_horizon and fee_bps == @base_fee and kind == "layer" and number == 1')
    relative_stats["two_way_turnover"] = 2 * primary_tracks.groupby("factor").annual_turnover_one_way.mean()
    main_relative = relative_stats.reset_index().assign(
        因子=lambda table: table.factor, 相对最大回撤=lambda table: percent(table.relative_dd),
        信息比率=lambda table: table.information_ratio.map("{:.2f}".format),
        相对Calmar=lambda table: table.relative_calmar.map("{:.2f}".format),
        年均双边换手_倍=lambda table: table.two_way_turnover.map("{:.2f}".format))
    main_relative = main_relative[["因子", "相对最大回撤", "信息比率", "相对Calmar", "年均双边换手_倍"]]
    top_risk = top_risk.assign(因子=top_risk.factor, 组合=np.where(top_risk.kind.eq("spread"), "最高减最低", "最高分层"),
                               平均夏普=top_risk.mean_sharpe.map(lambda value: f"{value:.2f}"),
                               平均月胜率=percent(top_risk.mean_month_win))[["因子", "组合", "平均夏普", "平均月胜率"]]
    lines = ["# 中证500：量价因子方法复现", "",
             f"本实验在 {dates[0]:%Y-%m-%d}—{dates[-1]:%Y-%m-%d} 的动态中证500上，对三个量价因子应用[原报告]({SOURCE}#page=10)的回归与分层方法。原报告为2010—2019年全A股；本实验不是对其数字的重现。",
             "", "## 每日回归与 IC", "",
             f"使用复权收盘收益，按各期限剔除日历越界：{coverage}。`raw` 为中位数去极值后标准化，`neutral` 再剔除行业和对数总市值暴露后标准化。收益回归控制行业和对数总市值，以流通市值平方根作为 WLS 权重；每日 t 值用该日回归的常规 WLS 标准误。平均 IC 以小数转百分数展示，不是收益率。ICIR 不年化。",
             "", markdown_table(regressions), "",
             f"[每日数据]({artifacts}/daily_factor_tests.csv)包含样本数、斜率、标准误及 t 值；[完整汇总]({artifacts}/factor_summary.csv)另给均值的 HAC(20) 标准误和 p 值。IC 和斜率各自对全部18项做 Holm 校正。它们是附加检验，不等于原报告的每日 t 值。",
             f"通过 0.05 门槛的平均 IC：{passed or '无'}。平均回归斜率通过校正的规格为 {slope_passed} 项。",
             "", "## 分层回测", "",
             f"行业市值中性化后按分数从高到低均分 {layer_count} 层，同分按股票代码排序；第一层最高。每层目标等权，按下一交易日收盘尝试换仓。默认单边 {base_fee}bp；每个调仓间隔跑完所有起始日。按252日折算年化，表中的均值是各轨道结果的算术平均。相对收益按策略/基准净值计算。同期中证500价格指数累计 {benchmark_total:.2%}、年化 {benchmark_annual:.2%}。",
             "", markdown_table(portfolios), "",
             f"{main_horizon}日主设置的风险和月胜率（按各轨道取均值，无风险利率为零）：", "", markdown_table(top_risk), "",
             "最高分层相对指数的风险和换手（轨道均值，信息比率用日主动收益计算）：", "", markdown_table(main_relative), "",
             f"[全部分层与费率结果]({artifacts}/portfolio_summary.csv) · [逐轨道指标]({artifacts}/portfolio_tracks.csv) · [执行对照]({artifacts}/execution_comparison.csv) · [极端层及价差日净值]({artifacts}/nav.parquet)",
             "", f"![{main_horizon}日分层年化]({artifacts}/layer_returns.svg)", "",
             f"![最高分层和模拟价差净值]({artifacts}/portfolio_curves.svg)", "",
             f"![{main_horizon}日最高分层的调仓起始日对照]({artifacts}/rebalance_offsets.svg)", "",
             f"每个点是一条 {main_horizon} 日调仓、单边 {base_fee}bp 的最高分层轨道。横轴从第一个信号日开始，按相差的交易日数编号；彩色实线是各轨道的平均年化，黑色虚线是同期基准年化。",
             "最高减最低按每日两条组合净收益之差复利，为研究性价差。两条腿各自为扣费多头，减去最低层净收益不等同真实做空费用，且不含融券可得性与借券成本。附表也固定比较了不同持股数以及最高层的10/15/20bp费用情景。执行对照保留原Alpha13五分组，在15bp下依次比较原分数开盘、原分数收盘及中性分数收盘；进入二十层还会改变组合集中度。",
             "", "## 口径与数据限制", "",
             f"- 无完整历史ST/PT和停牌事件表。回归和IC以次日有正常行情近似排除停牌（在有下一交易日的范围内，{missing_next:,}条成员记录没有通过该代理）；只作为事后统计样本掩码。因子计算、预处理及组合目标不读取次日数据。报告不把缺行情认定为确切停牌。",
             f"- 单日IC、WLS均要求至少{design['min_pairs']}对；各期限有效天内，最少 IC 配对和 WLS 配对分别为 {int(check.min_ic.min())} 和 {int(check.min_wls.min())}。行情用原项目的动态成员、原始价格和信号日成员排名参照。成员和行业只有近似的历史可得性；本样本曾用于研究，不是完全未接触的样本外区间。",
             "- 收盘涨停不买、跌停不卖，缺有效收盘价或官方限价不成交。卖不出的清仓单以后重试，未成交的新买单不补买；缺有效收盘用上一个可用收盘估值。不重建集合竞价可成交量、冲击、整手、公司行动现金与退市清算。原文未完整披露这些成交细节，属本实验约定。",
             "- 原文也考查其他中性化组合及参数；本轮只比较不中性与行业/市值中性，不复制依赖额外风格数据的设定，也不调参去追原报告收益。指数是价格指数。",
             "", f"主20日口径共 {len(main)} 个因子/预处理回归及 IC 组合、{int(main_tracks.n_tracks.sum())} 条最高分层轨道。",
             "", f"复现：`python -B -m src.scripts.run_factor_study`。参数在 [design.json]({artifacts}/design.json)，输入版本在 [manifest.json]({artifacts}/manifest.json)。"]
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_results(aggregate, nav, benchmark, design):
    main_horizon, fee = design["main_holding_days"], design["main_fee_bps"]
    layers = aggregate.query('kind == "layer" and holding_days == @main_horizon and fee_bps == @fee')
    fig, axis = plt.subplots(figsize=(8, 4))
    for factor, group in layers.groupby("factor"):
        axis.plot(group.number, group.mean_annual * 100, marker="o", label=factor)
    axis.axhline(0, color="0.7", linewidth=1)
    axis.set(xlabel="Layer (1 = highest score)", ylabel="Mean annualized return (%)", xticks=range(1, 21))
    axis.legend()
    fig.savefig(OUT / "layer_returns.svg", bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    view = nav.query('holding_days == @main_horizon and fee_bps == @fee')
    for (kind, number), axis in zip([("layer", 1), ("spread", 0)], axes):
        subset = view[(view.kind.eq(kind)) & view.number.eq(number)]
        for factor, group in subset.groupby("factor"):
            path = group.groupby("trade_date").nav.mean()
            axis.plot(path.index, path, label=factor)
        axis.axhline(1, color="0.7", linewidth=1)
        axis.set_ylabel("Top layer NAV" if kind == "layer" else "Spread NAV")
        axis.legend(loc="best")
    base = benchmark.reindex(path.index)
    axes[0].plot(base.index, base / base.iloc[0], color="black", linestyle="--", label="CSI 500 price index")
    axes[0].legend(loc="best")
    fig.savefig(OUT / "portfolio_curves.svg", bbox_inches="tight")
    plt.close(fig)


def plot_rebalance_offsets(tracks, benchmark, dates, design):
    factors = sorted(design["factors"])
    holding, fee = design["main_holding_days"], design["main_fee_bps"]
    top = tracks.query('kind == "layer" and number == 1 and holding_days == @holding and fee_bps == @fee')
    index_annual = ((benchmark.loc[dates[-1]] / benchmark.loc[dates[0]]) ** (252 / len(dates)) - 1) * 100
    fig, axes = plt.subplots(len(factors), 1, figsize=(8, 6), sharex=True, sharey=True, squeeze=False)
    for color_number, (factor, axis) in enumerate(zip(factors, axes[:, 0])):
        results = top[top.factor.eq(factor)]
        annual = results.annual_return * 100
        color = f"C{color_number}"
        wins = results.relative_total_return.gt(0).sum()
        axis.scatter(results.offset, annual, color=color, label="Individual track", zorder=3)
        axis.axhline(annual.mean(), color=color, linewidth=1.2, label="Mean across tracks")
        axis.axhline(index_annual, color="0.2", linestyle="--", linewidth=1, label="CSI 500 price index")
        axis.set_title(f"{factor}  ({wins}/{len(results)} beat the index)", loc="left", fontsize=10)
        axis.grid(axis="y", alpha=.2)
    axes[-1, 0].set(xlabel="Rebalance start offset (trading days)", xticks=range(holding))
    fig.supylabel("Annualized return (%)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3, frameon=False)
    fig.tight_layout(rect=(.02, 0, 1, .93))
    fig.savefig(OUT / "rebalance_offsets.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    design_file = OUT / "design.json"
    design = json.loads(design_file.read_text())
    print("Preparing factor and market data", flush=True)
    frame, panel, dates = prepare_study(design)
    ranges = {horizon: dates[:-horizon] for horizon in design["horizons"]}
    print("Computing daily rank IC and weighted regressions", flush=True)
    daily = daily_factor_tests(frame, design["factors"], design["horizons"], design["min_pairs"])
    daily = pd.concat([daily[daily.horizon.eq(horizon) & daily.trade_date.isin(window)]
                       for horizon, window in ranges.items()], ignore_index=True)
    summary = summarize_factor_tests(daily, dates)
    daily.to_csv(OUT / "daily_factor_tests.csv", index=False)
    summary.to_csv(OUT / "factor_summary.csv", index=False)
    open_quotes, close_quotes, benchmark = market_inputs(frame, panel, dates)
    tracks, nav = run_layer_tracks(frame, close_quotes, benchmark, design)
    aggregate = summarize_tracks(tracks)
    tracks.to_csv(OUT / "portfolio_tracks.csv", index=False)
    aggregate.to_csv(OUT / "portfolio_summary.csv", index=False)
    nav.to_parquet(OUT / "nav.parquet", index=False)
    compare_execution(frame, open_quotes, close_quotes, benchmark, design).to_csv(OUT / "execution_comparison.csv", index=False)
    plot_results(aggregate, nav, benchmark, design)
    plot_rebalance_offsets(tracks, benchmark, dates, design)
    write_report(summary, aggregate, tracks, nav, benchmark, dates, ranges, frame, daily, design)
    manifest = {str(path.relative_to(ROOT)): hashlib.file_digest(path.open("rb"), "sha256").hexdigest()
                for path in [*INPUTS.values(), design_file]}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Finished: {REPORT}", flush=True)


if __name__ == "__main__":
    main()
