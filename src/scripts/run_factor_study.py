"""运行基于本地中证500数据的单因子研究。"""

import json
import os
from pathlib import Path
from typing import cast

os.environ.setdefault("MPLCONFIGDIR", "/tmp/factor_study_matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.research_panel import CSI500_INPUTS as INPUTS
from src.data.research_panel import ROOT, ResearchPanel, load_market_calendar, prepare_membership, prepare_panel
from src.evaluation.cross_section import daily_factor_tests, summarize_factor_tests
from src.evaluation.layer_backtest import build_targets, run_layer_tracks, summarize_tracks
from src.evaluation.portfolio import BuyQuotes, QuoteGrid, backtest_long_only, benchmark_nav, close_market_inputs, load_benchmark, performance
from src.features.price_volume import compute_factors
from src.features.neutralization import attach_neutral_scores, mad_clip, standardize
from src.features.targets import attach_labels
from src.reporting import markdown_table as format_table
from src.reporting import number_table_captions, sha256

OUT = ROOT / "results/factor_study"
REPORT = ROOT / "docs/ml_study.md"
FACTOR_HEADING = "## 传统单因子研究"
SOURCE = "https://crm.htsc.com.cn/doc/2019/10750101/dae9cd9e-c2cd-4f05-bb6c-e5ac8d216d3d.pdf"


def prepare_study(design: dict) -> tuple[pd.DataFrame, ResearchPanel, pd.DatetimeIndex]:
    """读取 design.json 的日期、因子及期限设置，返回（股票日长表、行情容器、信号日索引）。"""
    calendar = load_market_calendar(INPUTS["calendar"])
    working = calendar[(calendar >= design["warmup_start"]) & (calendar <= design["end"])]
    dates = working[working >= design["signal_start"]]
    members = prepare_membership(pd.read_parquet(INPUTS["membership"]))
    panel = prepare_panel(pd.read_parquet(INPUTS["prices"]), working)
    frame = compute_factors(panel, members, dates, design["factor_price"], design["rank_policy"])
    frame = attach_labels(frame, panel, design["horizons"])
    frame = attach_neutral_scores(frame, pd.read_parquet(INPUTS["caps"]), pd.read_parquet(INPUTS["industries"]),
                                  design["factors"])
    observed = panel.observation_status().eq("observed").shift(-1, fill_value=False)
    frame["next_session_observed"] = observed.to_numpy()[observed.index.get_indexer(frame.trade_date),
                                                         observed.columns.get_indexer(frame.ticker)]
    for factor in design["factors"]:
        frame[f"{factor}_standardized"] = frame.groupby("trade_date")[factor].transform(
            lambda values: standardize(mad_clip(values)))
    return frame, panel, dates


def market_inputs(
    frame: pd.DataFrame, panel: ResearchPanel, dates: pd.DatetimeIndex,
) -> tuple[QuoteGrid, BuyQuotes, pd.Series]:
    """按 frame.ticker 和 dates 对齐行情，返回收盘行情、VWAP 买入行情和指数点位。

    买入行情由复权 VWAP 和原始最低价组成，原始最低价用于判断全天锁涨停。
    """
    tickers = pd.Index(sorted(frame.ticker.unique()), name="ticker")
    close_quotes, benchmark = close_market_inputs(panel, dates, tickers, INPUTS["limits"], INPUTS["benchmark"])
    vwap, low = [panel.matrix(column).reindex(index=dates, columns=tickers) for column in ("vwap_adj", "low")]
    return close_quotes, (vwap, low), benchmark


def compare_execution(
    frame: pd.DataFrame, close_quotes: QuoteGrid, buy_quotes: BuyQuotes, benchmark: pd.Series, design: dict,
) -> pd.DataFrame:
    """frame 含 trade_date、ticker、Alpha13、Alpha13_neutral；对最高分层返回逐口径逐偏移的绩效表。"""
    holding, fee = design["main_holding_days"], design["main_fee_bps"]
    rows = []
    for score, quote_sets in [("Alpha13", {"close": None, "vwap_buy": buy_quotes}),
                              ("Alpha13_neutral", {"vwap_buy": buy_quotes})]:
        layers = build_targets(frame, score, design["layers"])
        for execution, quotes in quote_sets.items():
            for offset in range(holding):
                result = backtest_long_only(close_quotes, layers[0], fee_bps=fee, rebalance_every=holding,
                                            offset=offset, record_trades=False, buy_quotes=quotes)
                rows.append(dict(score=score, execution=execution, offset=offset, **performance(result.ledger, benchmark)))
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    """用 DataFrame 的列名作表头，将每行转换成 Markdown。"""
    return format_table(frame.columns, frame.itertuples(index=False, name=None))


def percent(values: pd.Series) -> pd.Series:
    """格式化一列小数，例如 0.15 → '15.00%'；保留索引。"""
    return values.map(lambda value: f"{value:.2%}" if pd.notna(value) else "—")


def factor_section(
    summary: pd.DataFrame, aggregate: pd.DataFrame, tracks: pd.DataFrame, nav: pd.DataFrame,
    benchmark: pd.Series, dates: pd.DatetimeIndex, design: dict, report: Path,
) -> str:
    """用已计算的结果生成统一报告中的传统单因子部分。

    summary 是因子检验汇总表，aggregate/tracks 是汇总和逐轨道回测表。
    nav 是逐轨道逐日净值表，benchmark 按交易日索引，design 对应 design.json。
    """
    artifacts = Path(os.path.relpath(OUT, report.parent)).as_posix()
    factor_prices = "复权行情" if design["factor_price"] == "adjusted" else "原始行情"
    base_fee, main_horizon, layer_count = (design[key] for key in ("main_fee_bps", "main_holding_days", "layers"))
    short_horizon = min(design["horizons"])
    short = summary.query('treatment == "neutral" and horizon == @short_horizon').sort_values("icir", ascending=False)
    signals = pd.DataFrame({
        "因子": short.factor,
        "平均 Rank IC": short.mean_rank_ic.map("{:.4f}".format),
        "ICIR": short.icir.map("{:.3f}".format),
        "正 IC 占比": percent(short.share_ic_positive),
        "单日 &#124;t&#124; > 2 占比": percent(short.share_abs_t_gt_2),
    })
    leaders = " 和 ".join(short.factor.head(2))
    index_nav = benchmark_nav(benchmark, dates)
    benchmark_total = index_nav.iloc[-1] - 1
    benchmark_annual = (1 + benchmark_total) ** (252 / len(dates)) - 1
    top_filter = 'kind == "layer" and number == 1 and fee_bps == @base_fee'
    primary_filter = top_filter + ' and holding_days == @main_horizon'
    top_layers = aggregate.query(top_filter).sort_values(["holding_days", "mean_annual"], ascending=[True, False])
    primary = top_layers.loc[top_layers.holding_days.eq(main_horizon)].set_index("factor", drop=False)
    best = primary.iloc[0]
    alpha13 = primary.loc["Alpha13"]
    portfolios = pd.DataFrame({
        "换仓周期": top_layers.holding_days.map("{} 日".format),
        "因子": top_layers.factor,
        "平均年化": percent(top_layers.mean_annual),
        "平均累计收益": percent(top_layers.mean_total),
        "平均最大回撤": percent(top_layers.mean_drawdown),
        "跑赢指数的轨道": top_layers.outperform_tracks.map("{:.0f}".format) + "/" + top_layers.n_tracks.astype(str),
    })
    primary_nav = nav.query(primary_filter)
    relative_rows = []
    for key, path in primary_nav.groupby(["factor", "offset"]):
        factor, offset = cast(tuple[str, int], key)
        path = path.set_index("trade_date").reindex(dates)
        excess = path.daily_return - index_nav.pct_change().fillna(0)
        relative_nav = path.nav / index_nav
        relative_dd = (relative_nav / relative_nav.cummax() - 1).min()
        relative_annual = relative_nav.iloc[-1] ** (252 / len(dates)) - 1
        relative_rows.append(dict(factor=factor, offset=offset, relative_dd=relative_dd,
                                  information_ratio=np.sqrt(252) * excess.mean() / excess.std(ddof=1),
                                  relative_calmar=relative_annual / abs(relative_dd) if relative_dd < 0 else np.nan))
    relative_stats = cast(pd.DataFrame, pd.DataFrame(relative_rows).groupby("factor").mean())
    primary_tracks = tracks.query(primary_filter)
    relative_stats["two_way_turnover"] = 2 * primary_tracks.groupby("factor").annual_turnover_one_way.mean()
    main_relative = relative_stats.reset_index().assign(
        因子=lambda table: table.factor, 相对最大回撤=lambda table: percent(table.relative_dd),
        信息比率=lambda table: table.information_ratio.map("{:.2f}".format),
        相对Calmar=lambda table: table.relative_calmar.map("{:.2f}".format),
        年均双边换手=lambda table: table.two_way_turnover.map("{:.2f} 倍".format))
    main_relative = main_relative.loc[:, ["因子", "相对最大回撤", "信息比率", "相对Calmar", "年均双边换手"]]
    return f"""{FACTOR_HEADING}

本节使用 {layer_count} 层等权和下述成交口径。参与五日 Top-50 对比的同名因子采用对应的 Top-50 策略规则。

**{leaders} 对短期收益的排序表现更好。** 按每 {main_horizon} 个交易日调仓、单边 {base_fee} bp 的主设定买入最高分层，表现最好的 {best.factor} 平均年化为 **{best.mean_annual:.2%}**，低于同期中证500价格指数的 **{benchmark_annual:.2%}**。

研究使用 {dates[0]:%Y-%m-%d} 至 {dates[-1]:%Y-%m-%d} 的动态中证500成分股，用{factor_prices}计算 Alpha3、Alpha13 和 Alpha44。回归、Rank IC 和分层回测参照华泰证券的单因子研究方法。 [华泰原研报]({SOURCE}#page=10)

### 因子分数能否排对顺序

Rank IC 衡量因子分数与未来收益的排序是否一致，加权回归则检验控制行业和市值后，分数与未来收益的关系。下表列出剔除行业和市值影响后的 {short_horizon} 日检验结果。前两名因子的 Rank IC 在超过六成交易日为正，而各因子的回归 `|t|` 超过 2 的日期约占一到两成。

**表　传统单因子的横截面回归与 Rank IC 检验**

{markdown_table(signals)}

### 买入最高分层后赚了多少

股票按中性化后的分数分成 {layer_count} 层，每层目标等权。信号发出后，下一交易日按复权 VWAP 买入，按复权收盘价卖出并估值。每个换仓周期都跑完所有调仓起始日，再汇总最高分层各轨道的收益、最大回撤和跑赢指数的次数。同期中证500价格指数累计上涨 **{benchmark_total:.2%}**。

**表　传统单因子最高分层在各调仓周期的绩效**

{markdown_table(portfolios)}

![{main_horizon} 日换仓下的逐日平均净值]({artifacts}/portfolio_curves.svg)

**图 F1　最高分层与模拟价差的逐日平均净值。** 换仓周期为 {main_horizon} 个交易日，单边费用为 {base_fee} bp。每条因子曲线取 `offset=0` 至 `{main_horizon - 1}` 的 {main_horizon} 条轨道在同一交易日的平均净值。上图为最高分层，黑色虚线为中证500价格指数。下图为最高层减最低层的模拟价差。

各层收益起伏明显，有些低分层的年化也高于最高分层。即使只看最高层，起始日不同，结果也相差很大。例如 Alpha13 的年化从 **{alpha13.min_annual:.2%}** 到 **{alpha13.max_annual:.2%}**。

![各层平均年化收益]({artifacts}/layer_returns.svg)

**图 F2　各因子分层的平均年化收益。** 换仓周期为 {main_horizon} 个交易日，单边费用为 {base_fee} bp。横轴为分层编号，第 1 层的因子分数最高。每个点是对应层在 {main_horizon} 个调仓起始日下的年化收益均值。

![不同调仓起始日的年化收益]({artifacts}/rebalance_offsets.svg)

**图 F3　最高分层在不同调仓起始日下的年化收益。** 换仓周期为 {main_horizon} 个交易日，单边费用为 {base_fee} bp。横轴为距首个信号日的交易日数，每个点是一条轨道的年化收益。彩色实线为各因子所有轨道的平均年化，黑色虚线为同期中证500价格指数的年化收益。子图标题列出跑赢指数的轨道数和总轨道数。

### 数据与复现

结论对应本研究的历史样本与模拟成交条件。回测考虑交易费率和涨跌停限制，历史成分、行业和停牌状态受数据可得性约束。图中的价差来自最高层和最低层的日收益之差。

VWAP 按当日成交额除以成交股数计算。全天处于涨停的股票不买入。账本按日结算，同日卖出款可用于买入，目标份额按收盘价估算。执行对照固定收盘卖出，比较收盘买入与 VWAP 买入。

- [因子检验汇总]({artifacts}/factor_summary.csv) · [逐日回归与 Rank IC]({artifacts}/daily_factor_tests.csv)
- [分层回测汇总]({artifacts}/portfolio_summary.csv) · [逐轨道结果]({artifacts}/portfolio_tracks.csv) · [执行对照]({artifacts}/execution_comparison.csv) · [逐日净值]({artifacts}/nav.parquet)
- 复现命令为 `python -B -m src.scripts.run_factor_study`。 [研究参数]({artifacts}/design.json) · [输入版本]({artifacts}/manifest.json)

<details>
<summary>最高分层相对指数的补充指标</summary>

**表　传统单因子最高分层的相对绩效与换手率**

{markdown_table(main_relative)}

</details>
"""


def saved_factor_section(report: Path) -> str:
    """读取已保存的单因子结果，供模型研究更新统一报告时使用。"""
    files = ["factor_summary.csv", "portfolio_summary.csv", "portfolio_tracks.csv", "nav.parquet", "design.json"]
    if not all((OUT / name).is_file() for name in files):
        return ""
    summary, aggregate, tracks = (pd.read_csv(OUT / name) for name in files[:3])
    nav = pd.read_parquet(OUT / files[3])
    dates = pd.DatetimeIndex(nav.trade_date.unique()).sort_values()
    design = json.loads((OUT / files[4]).read_text())
    return factor_section(summary, aggregate, tracks, nav, load_benchmark(INPUTS["benchmark"]), dates, design, report)


def update_factor_section(report: Path, section: str) -> None:
    """在统一报告中更新末尾的单因子结果，保留前面的模型研究。"""
    current = report.read_text(encoding="utf-8") if report.exists() else "# 中证500选股研究"
    previous = current.partition(f"\n{FACTOR_HEADING}\n")[0]
    report.parent.mkdir(exist_ok=True)
    report.write_text(number_table_captions(previous.rstrip() + "\n\n" + section), encoding="utf-8")


def plot_results(aggregate: pd.DataFrame, nav: pd.DataFrame, benchmark: pd.Series, design: dict) -> None:
    """用分层回测汇总、逐轨道日净值和基准点位画年化收益与净值图。"""
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
    base = benchmark_nav(benchmark, pd.DatetimeIndex(view["trade_date"].unique()).sort_values())
    axes[0].plot(base.index, base, color="black", linestyle="--", label="CSI 500 price index")
    axes[0].legend(loc="best")
    fig.savefig(OUT / "portfolio_curves.svg", bbox_inches="tight")
    plt.close(fig)


def plot_rebalance_offsets(
    tracks: pd.DataFrame, benchmark: pd.Series, dates: pd.DatetimeIndex, design: dict,
) -> None:
    """tracks 每行是一条回测轨道；用其中的 offset 和 annual_return 画起始日对照。"""
    factors = sorted(design["factors"])
    holding, fee = design["main_holding_days"], design["main_fee_bps"]
    top = tracks.query('kind == "layer" and number == 1 and holding_days == @holding and fee_bps == @fee')
    index_end = benchmark_nav(benchmark, dates).iloc[-1]
    index_annual = (index_end ** (252 / len(dates)) - 1) * 100
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


def main() -> None:
    """读取研究配置，依次生成单因子检验、分层回测和报告。"""
    design_file = OUT / "design.json"
    design = json.loads(design_file.read_text())
    print("Preparing factor and market data", flush=True)
    # 这里frame是每只股票每天一行的长表df
    # panel是ResearchPanel，可以根据需要取出不同数据的行情矩阵
    # dates是交易日历DatetimeIndex
    frame, panel, dates = prepare_study(design)
    ranges = {horizon: dates[:-horizon] for horizon in design["horizons"]}
    print("Computing daily rank IC and weighted regressions", flush=True)
    daily = daily_factor_tests(frame, design["factors"], design["horizons"], design["min_pairs"])
    daily = pd.concat([daily.loc[daily["horizon"].eq(horizon) & daily["trade_date"].isin(window.to_list()), :]
                       for horizon, window in ranges.items()], ignore_index=True)
    summary = summarize_factor_tests(daily)
    daily.to_csv(OUT / "daily_factor_tests.csv", index=False)
    summary.to_csv(OUT / "factor_summary.csv", index=False)
    close_quotes, buy_quotes, benchmark = market_inputs(frame, panel, dates)
    # tracks和nav均包含factor, holding_days, offset, fee_bps, kind, number
    # 分别指因子、调仓周期、偏移、单边交易费率、组合类型（layer 或 spread）和组合编号
    # tracks除了这6个含义列，还有15个指标列，包括年化收益、夏普等等。只要确定这6个含义就能唯一确定一条轨道
    # nav除了这6个含义列，还包括交易日历。每日收益
    tracks, nav = run_layer_tracks(frame, close_quotes, benchmark, design, buy_quotes=buy_quotes)
    aggregate = summarize_tracks(tracks)
    tracks.to_csv(OUT / "portfolio_tracks.csv", index=False)
    aggregate.to_csv(OUT / "portfolio_summary.csv", index=False)
    nav.to_parquet(OUT / "nav.parquet", index=False)
    compare_execution(frame, close_quotes, buy_quotes, benchmark, design).to_csv(OUT / "execution_comparison.csv", index=False)
    plot_results(aggregate, nav, benchmark, design)
    plot_rebalance_offsets(tracks, benchmark, dates, design)
    manifest = {str(path.relative_to(ROOT)): sha256(path) for path in [*INPUTS.values(), design_file]}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    section = factor_section(summary, aggregate, tracks, nav, benchmark, dates, design, REPORT)
    update_factor_section(REPORT, section)
    print(f"Finished: {REPORT}", flush=True)


if __name__ == "__main__":
    main()
