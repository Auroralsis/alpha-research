"""共同研究口径下的指数与组合绩效。"""

import json
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd

from src.backtest.factor_metrics import summarize_ic
from src.data.research_panel import parse_dates

# 与项目锁定的 Qlib 0.9.7 risk_analysis(freq="day") 一致。
QLIB_DAILY_ANNUALIZATION = 238


def load_benchmark_returns(path: Path, dates: pd.DatetimeIndex) -> pd.Series:
    """按当日收盘点位除以前收盘点位计算指数日收益，保留测试首日涨跌。"""
    data = pd.read_csv(path, dtype={"trade_date": str}, usecols=["trade_date", "close", "pre_close"])
    data.index = pd.DatetimeIndex(parse_dates(data.trade_date))
    aligned = data[["close", "pre_close"]].reindex(dates)
    if not (np.isfinite(aligned) & aligned.gt(0)).all(axis=None):
        raise ValueError("指数日收益缺少有效的当日或前一交易日收盘点位")
    return aligned.close / aligned.pre_close - 1


def qlib_return_risk(returns: pd.Series, mode: Literal["sum", "product"] = "sum") -> pd.Series:
    """通过 Qlib 评价每日收益序列。"""
    from qlib.contrib.evaluate import risk_analysis

    with np.errstate(divide="ignore", invalid="ignore"):
        return risk_analysis(returns, freq="day", mode=mode)["risk"]


def qlib_excess_metrics(net_return: pd.Series, benchmark_return: pd.Series) -> dict[str, float]:
    """用日期对齐的扣费日收益和基准日收益计算算术超额绩效。"""
    risk = qlib_return_risk((net_return - benchmark_return).sort_index())
    return {
        "annualized_excess_return": risk.loc["annualized_return"],
        "information_ratio": risk.loc["information_ratio"],
        "max_excess_drawdown": risk.loc["max_drawdown"],
    }


def qlib_relative_risk(net_return: pd.Series, benchmark_return: pd.Series) -> pd.Series:
    """评价策略净值除以基准净值的复利相对路径。"""
    relative = (1 + net_return) / (1 + benchmark_return) - 1
    return qlib_return_risk(relative.sort_index(), mode="product")


def performance(ledger: pd.DataFrame, benchmark_returns: pd.Series | None = None) -> dict[str, float]:
    """每日账本含交易日、净值、扣费日收益、双边换手率和当日人民币费用。

    benchmark_returns 是以交易日为索引的基准日收益。
    """
    returns = cast(pd.Series, ledger.set_index("trade_date")["daily_return"])
    total = ledger.nav.iloc[-1]
    compound = qlib_return_risk(returns, mode="product")
    arithmetic = qlib_return_risk(returns)
    result = dict(
        annual_return=compound.loc["annualized_return"], total_return=total - 1,
        annual_volatility=compound.loc["std"] * np.sqrt(QLIB_DAILY_ANNUALIZATION),
        sharpe=arithmetic.loc["information_ratio"], max_drawdown=compound.loc["max_drawdown"],
        annual_turnover_one_way=ledger.turnover_two_sided.sum(min_count=1) * QLIB_DAILY_ANNUALIZATION / (2 * len(ledger)),
        cost_cny=ledger.cost_cny.sum(min_count=1),
    )
    if benchmark_returns is not None:
        index_return = benchmark_returns.reindex(returns.index)
        index_nav = (1 + index_return).cumprod()
        relative = qlib_relative_risk(returns, index_return)
        result.update(
            benchmark_total_return=index_nav.iloc[-1] - 1,
            relative_total_return=total / index_nav.iloc[-1] - 1,
            annual_relative_return=relative.loc["annualized_return"],
            **qlib_excess_metrics(returns, index_return),
        )
    # 同一个月内的日收益连乘，统计月收益大于零的比例。
    months = pd.PeriodIndex(ledger["trade_date"].to_numpy(), freq="M")
    monthly = (1 + returns).groupby(months).prod() - 1
    result["month_win_rate"] = monthly.gt(0).mean()
    return result


STRATEGIES = ("lightgbm", "rolling_lightgbm")


def window_portfolio_returns(ledger: pd.DataFrame, windows: list[dict], rebalance_days: int,
                             strategies: tuple[str, ...] = STRATEGIES) -> pd.DataFrame:
    """按测试窗口分别计算各偏移轨道的年化超额收益，再对轨道等权平均。"""
    books = ledger.loc[ledger.strategy.isin(strategies)]
    rows = []
    for window in windows:
        begin, end = map(pd.Timestamp, window["effective"]["test"])
        segment = books.loc[books.trade_date.between(begin, end)].set_index("trade_date")
        for strategy in strategies:
            tracks = segment.loc[segment.strategy.eq(strategy)].groupby("offset")
            if set(tracks.groups) != set(range(rebalance_days)):
                raise ValueError("窗口缺少调仓偏移轨道或包含意外轨道")
            returns = []
            for _, book in tracks:
                if len(book) != window["test_sessions"] or book.index.has_duplicates:
                    raise ValueError("窗口轨道账本缺少交易日或包含重复记录")
                if not np.isfinite(book[["daily_return", "benchmark_return"]]).all(axis=None):
                    raise ValueError("窗口包含无效的组合或基准日收益")
                returns.append(qlib_excess_metrics(book.daily_return, book.benchmark_return)["annualized_excess_return"])
            rows.append(dict(window=window["window"], strategy=strategy, annualized_excess_return=np.mean(returns),
                             portfolio_days=window["test_sessions"], n_tracks=len(returns)))
    return pd.DataFrame(rows)


def evaluate_windows(windows: list[dict], rebalance_days: int, dates: pd.DatetimeIndex,
                     n_test: int, daily: pd.DataFrame, out: Path) -> None:
    """从统一的逐日表和账本生成逐窗排序指标、超额收益和样本审计。"""
    daily = daily.loc[daily.split.eq("test") & daily.strategy.isin(STRATEGIES)]
    daily_by_strategy = list(daily.set_index("trade_date").groupby("strategy"))
    window_ic = []
    for window in windows:
        begin, end = map(pd.Timestamp, window["effective"]["test"])
        window_dates = dates[(dates >= begin) & (dates <= end)]
        for strategy, group in daily_by_strategy:
            window_ic.append(dict(strategy=strategy, **summarize_ic(group.reindex(window_dates)), window=window["window"]))
    returns = window_portfolio_returns(pd.read_parquet(out / "ledger.parquet"), windows, rebalance_days,
                                       tuple(name for name, _ in daily_by_strategy))
    summary = pd.DataFrame(window_ic).merge(returns, on=["window", "strategy"], validate="one_to_one")
    summary.to_csv(out / "window_rank_ic.csv", index=False)

    audit = dict(test_predictions=n_test, ic_pairs_by_strategy=daily.groupby("strategy").n_pairs.sum().to_dict())
    (out / "window_sample_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
