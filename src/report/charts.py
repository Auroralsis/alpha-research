"""绘制统一研究报告使用的 SVG 图。"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from matplotlib.ticker import PercentFormatter


def save_chart(ax, path: Path, *, legend_y: float, columns: int) -> None:
    """统一将图例放在图下，保存并释放画布。"""
    ax.legend(loc="upper center", bbox_to_anchor=(.5, legend_y), ncol=columns, frameon=False)
    ax.figure.tight_layout()
    ax.figure.savefig(path)
    plt.close(ax.figure)


def plot_training_mse(out: Path, curve: pd.DataFrame, best: int, validation_mse: float, *, raw_labels: bool) -> None:
    """画逐轮训练与验证损失，并标出最佳迭代轮数。"""
    _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(curve.iteration, curve.train_mse, label="Train")
    ax.plot(curve.iteration, curve.validation_mse, label="Validation")
    ax.scatter(best, validation_mse, color="C1", zorder=3)
    ax.axvline(best, color="0.5", linestyle="--", label=f"Best: round {best}")
    ax.set(xlabel="Boosting round", ylabel="MSE on raw returns" if raw_labels else "MSE on daily Z-scored labels")
    save_chart(ax, out / "training_mse.svg", legend_y=-.17, columns=3)


def plot_rank_ic(out: Path, daily_ic: pd.DataFrame, switch_dates: list[pd.Timestamp],
                 strategies: tuple[str, ...], display: dict[str, str], colors: dict[str, str]) -> None:
    """画各方法的 20 日滚动平均 Rank IC 及模型切换日期。"""
    _, ax = plt.subplots(figsize=(9.5, 4.2))
    for name in strategies:
        days = daily_ic.loc[daily_ic.strategy.eq(name)].sort_values("trade_date")
        ax.plot(days.trade_date, days.rank_ic.rolling(20).mean(), color=colors[name], linewidth=1.2,
                label=display.get(name, name))
    for date in switch_dates:
        ax.axvline(date, color=".8", linewidth=.8, zorder=0)
    ax.axhline(0, color=".55", linewidth=.8)
    ax.set(ylabel="20-session mean Rank IC")
    save_chart(ax, out / "rank_ic_timeline.svg", legend_y=-.11, columns=3)


def grouped_returns(ax, values: pd.DataFrame, strategies, display, colors) -> None:
    """行是横坐标上的组，列是策略；同组收益并排画成百分比柱。"""
    positions = np.arange(len(values))
    width = .8 / len(strategies)
    for i, name in enumerate(strategies):
        centers = positions + (i - (len(strategies) - 1) / 2) * width
        ax.bar(centers, values[name], width=width, color=colors[name], label=display.get(name, name))
    ax.set(xticks=positions, xticklabels=values.index)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))


def plot_quintiles(out: Path, all_groups: pd.DataFrame, buckets: list[str], horizon: int,
                   strategies: tuple[str, ...], display: dict[str, str], colors: dict[str, str]) -> None:
    """画各方法的五组平均远期收益。"""
    _, ax = plt.subplots(figsize=(9, 4))
    grouped_returns(ax, all_groups[buckets].T.astype(float), strategies, display, colors)
    ax.set_ylabel(f"Mean {horizon}-session return")
    ax.axhline(0, color=".55", linewidth=.8)
    save_chart(ax, out / "quintile_returns.svg", legend_y=-.12, columns=3)


def plot_portfolio_nav(out: Path, ledger: pd.DataFrame, switch_dates: list[pd.Timestamp],
                       strategies: tuple[str, ...], display: dict[str, str], colors: dict[str, str],
                       *, monthly_axis: bool = False, benchmark_width: float = 1.5) -> None:
    """画各方法的跨轨道平均净值与指数净值。"""
    _, ax = plt.subplots(figsize=(9.5, 4.4))
    mean_nav = ledger.groupby(["strategy", "trade_date"])["nav"].mean()
    for name in strategies:
        path = mean_nav.loc[name]
        ax.plot(path.index, path, label=display.get(name, name), color=colors[name], linewidth=1.3)
    benchmark_returns = ledger.drop_duplicates("trade_date").set_index("trade_date").benchmark_return.sort_index()
    index_nav = (1 + benchmark_returns).cumprod()
    ax.plot(index_nav.index, index_nav, "--", color=".15" if benchmark_width > 1.5 else ".36", linewidth=benchmark_width, label=display["benchmark"])
    for date in switch_dates:
        ax.axvline(date, color=".8", linewidth=.8, zorder=0)
    ax.set(ylabel="NAV")
    if monthly_axis:
        start = index_nav.index[0].to_period("M").start_time
        end = index_nav.index[-1] + pd.offsets.MonthBegin(1)
        ax.set(xticks=pd.date_range(start, end, freq="MS"), xlim=(start, end))
        ax.xaxis.set_major_formatter(DateFormatter("%Y-%m"))
        ax.tick_params(axis="x", labelrotation=90, labelsize=8)
    save_chart(ax, out / "portfolio_nav.svg", legend_y=-.23 if monthly_axis else -.11, columns=4)


def plot_offset_returns(out: Path, returns_by_offset: pd.DataFrame, strategies: tuple[str, ...],
                        display: dict[str, str], colors: dict[str, str]) -> None:
    """画各调仓偏移的年化超额收益。"""
    _, ax = plt.subplots(figsize=(9, 4.2))
    grouped_returns(ax, returns_by_offset.rename(index=int), strategies, display, colors)
    ax.axhline(0, color=".36", linestyle="--", label=display["benchmark"])
    ax.set(xlabel="Rebalance offset", ylabel="Annualized excess return (net of fees)")
    save_chart(ax, out / "offset_returns.svg", legend_y=-.2, columns=4)


def plot_rank_feature_paths(out: Path, ledger: pd.DataFrame, methods: dict[str, str],
                            filename: str = "feature_comparison.png") -> None:
    """画固定各半组合的扣费净值和逐日累加超额，两幅图共用指数与日期轴。"""
    selected = ledger.loc[ledger.strategy.isin(methods)]
    returns = selected.pivot(index="trade_date", columns="strategy", values="daily_return").sort_index()
    benchmark = selected.pivot(index="trade_date", columns="strategy", values="benchmark_return").sort_index()
    if returns.isna().any(axis=None) or benchmark.isna().any(axis=None) or not benchmark.nunique(axis=1).eq(1).all():
        raise ValueError("各组合必须使用相同交易日与指数收益")
    returns["benchmark"] = benchmark.iloc[:, 0]
    start = returns.index[0].to_period("M").start_time
    end = returns.index[-1] + pd.offsets.MonthBegin(1)
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6.6))
    panels = [(1 + returns).cumprod(), returns.sub(returns.benchmark, axis=0).cumsum()]
    for ax, values, initial in zip(axes, panels, (1., 0.)):
        paths = pd.concat([pd.DataFrame(initial, index=[start], columns=values.columns), values])
        for (method, title), color in zip(methods.items(), ("#773db6", "#2166a5", "#00876e", "#c26a20")):
            ax.plot(paths.index, paths[method], label=title, linewidth=1.4, color=color)
        ax.plot(paths.index, paths.benchmark, "--", label="CSI 500", linewidth=2.4, color=".15")
        ax.grid(axis="y", linewidth=.5, alpha=.25)
    axes[0].set_ylabel("Net asset value")
    axes[1].set_ylabel("Cumulative net excess return")
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    axes[1].set(xticks=pd.date_range(start, end, freq="MS"), xlim=(start, end))
    axes[1].xaxis.set_major_formatter(DateFormatter("%Y-%m"))
    axes[1].tick_params(axis="x", rotation=90, labelsize=8)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, .055, 1, 1))
    fig.savefig(out / filename, dpi=180)
    plt.close(fig)
