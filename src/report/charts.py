"""绘制统一研究报告使用的 SVG 图。"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
                       strategies: tuple[str, ...], display: dict[str, str], colors: dict[str, str]) -> None:
    """画各方法的跨轨道平均净值与指数净值。"""
    _, ax = plt.subplots(figsize=(9.5, 4.4))
    mean_nav = ledger.groupby(["strategy", "trade_date"])["nav"].mean()
    for name in strategies:
        path = mean_nav.loc[name]
        ax.plot(path.index, path, label=display.get(name, name), color=colors[name], linewidth=1.3)
    benchmark_returns = ledger.drop_duplicates("trade_date").set_index("trade_date").benchmark_return.sort_index()
    index_nav = (1 + benchmark_returns).cumprod()
    ax.plot(index_nav.index, index_nav, "--", color=".36", label=display["benchmark"])
    for date in switch_dates:
        ax.axvline(date, color=".8", linewidth=.8, zorder=0)
    ax.set(ylabel="NAV")
    save_chart(ax, out / "portfolio_nav.svg", legend_y=-.11, columns=4)


def plot_offset_returns(out: Path, returns_by_offset: pd.DataFrame, strategies: tuple[str, ...],
                        display: dict[str, str], colors: dict[str, str]) -> None:
    """画各调仓偏移的年化超额收益。"""
    _, ax = plt.subplots(figsize=(9, 4.2))
    grouped_returns(ax, returns_by_offset.rename(index=int), strategies, display, colors)
    ax.axhline(0, color=".36", linestyle="--", label=display["benchmark"])
    ax.set(xlabel="Rebalance offset", ylabel="Annualized excess return (net of fees)")
    save_chart(ax, out / "offset_returns.svg", legend_y=-.2, columns=4)
