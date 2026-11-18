"""Small long-only ledger in fixed-anchor adjusted units, never live trading.

This is a simplified model: finite liquidity/auction allocation, lots, corporate
action cash flows and delisting settlements are not reconstructed. Official
limit prices block orders at the opening limit; missing quotes never fill.
"""
import numpy as np
import pandas as pd

from src.data.research_panel import ResearchPanel


TradeSignals = dict[pd.Timestamp, list[str]]
QuoteGrid = tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]


def trading_prices(panel: ResearchPanel, dates: pd.DatetimeIndex, tickers: pd.Index) -> list[pd.DataFrame]:
    """返回复权开盘价、复权收盘价、原始开盘价，均按 dates 为行、tickers 为列对齐。"""
    return [panel.matrix(column, clean=False).reindex(index=dates, columns=tickers)
            .where(lambda x: np.isfinite(x) & x.gt(0))
            for column in ["open_adj", "close_adj", "open"]]


def backtest_long_only(
    open_adj: pd.DataFrame, close_adj: pd.DataFrame, raw_open: pd.DataFrame,
    up_limit: pd.DataFrame, down_limit: pd.DataFrame, signals: TradeSignals,
    fee: float = .0015, rebalance_every: int = 20, offset: int = 0, *,
    record_trades: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """模拟一条多头组合轨道，返回（每日账本、逐笔成交、期末持仓）。

    五张行情表都是同一 [交易日 × 股票] 网格。open_adj、raw_open 保留旧参数名，
    分别表示复权和原始成交价；次日收盘口径传入收盘价。close_adj 是收盘估值价，
    up_limit、down_limit 是原始价格口径的涨跌停价。
    signals 如 {pd.Timestamp("2025-01-06"): ["000001.SZ"]}，键是信号日。
    fee 是每边小数费率；rebalance_every 和 offset 都以市场交易日为单位。
    """
    if fee < 0 or rebalance_every <= 0 or offset < 0:
        raise ValueError("Invalid trading parameters")
    dates, columns = open_adj.index, open_adj.columns
    if any(set(targets) - set(columns) for targets in signals.values()):
        raise ValueError("Signal contains securities outside the price grid")
    for matrix in [close_adj, raw_open, up_limit, down_limit]:
        if not matrix.index.equals(dates) or not matrix.columns.equals(columns):
            raise ValueError("Trading matrices must have identical grids")
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("Invalid trading calendar")
    # signals are fixed before observing any t+1 execution data.
    schedule = {}
    for day in dates[offset::rebalance_every]:
        if day in signals:
            row = dates.get_loc(day)
            if row + 1 < len(dates):
                schedule[dates[row+1]] = set(signals[day])
    units = np.zeros(len(columns))
    cash, previous_equity = 1., 1.
    marks = np.full(len(columns), np.nan)
    pending_exits = np.zeros(len(columns), dtype=bool)
    bought_on = np.full(len(columns), -1)
    records, trades = [], []
    last_signal_date = pd.NaT
    quotes = [matrix.to_numpy(dtype=float) for matrix in (open_adj, close_adj, raw_open, up_limit, down_limit)]
    for row, (day, op, cl, ro, up, down) in enumerate(zip(dates, *quotes)):
        finite_open = np.isfinite(op) & (op > 0)
        at_open = np.where(finite_open, op, marks)
        if ((units > 0) & ~np.isfinite(at_open)).any():
            raise ValueError("Held position has no historical mark")
        equity_open = cash + np.dot(units, np.nan_to_num(at_open))
        buyable = finite_open & np.isfinite(ro) & np.isfinite(up) & (ro < up - 1e-8)
        sellable = finite_open & np.isfinite(ro) & np.isfinite(down) & (ro > down + 1e-8) & (bought_on < row)
        desired = units.copy()
        rebalance = day in schedule
        target_without_open = 0
        if rebalance:
            last_signal_date = dates[row-1]
            target = np.array([ticker in schedule[day] for ticker in columns])
            n = int(target.sum())
            target_without_open = int((target & ~finite_open).sum())
            desired = np.zeros(len(columns))
            # Missing opening quotes retain their allocation in cash, not in peers.
            np.divide(equity_open / max(n, 1), op, out=desired, where=target & finite_open)
            pending_exits = (units > 0) & ~target
        desired[pending_exits] = 0
        blocked_sells = int(((units > desired + 1e-12) & ~sellable).sum())
        blocked_buys = int(((desired > units + 1e-12) & ~buyable).sum()) if rebalance else 0
        sell_units = np.where(sellable, np.maximum(units - desired, 0), 0)
        sell_value = sell_units * np.nan_to_num(op)
        units -= sell_units
        cash += sell_value.sum() * (1 - fee)
        buy_units = np.zeros(len(columns))
        if rebalance:
            buy_units = np.where(buyable, np.maximum(desired - units, 0), 0)
            needed = np.dot(buy_units, np.nan_to_num(op)) * (1 + fee)
            buy_units *= min(1., max(cash, 0) / needed) if needed > 0 else 0
            units += buy_units
            bought_on[buy_units > 0] = row
        buy_value = buy_units * np.nan_to_num(op)
        cash -= buy_value.sum() * (1 + fee)
        if cash < -1e-10 or (units < -1e-10).any():
            raise AssertionError("Negative cash or holdings")
        cash = max(cash, 0)
        traded = buy_value.sum() + sell_value.sum()
        cost = traded * fee
        # Self financing identity at execution prices, including blocked holdings.
        after_trade = cash + np.dot(units, np.nan_to_num(at_open))
        if not np.isclose(after_trade, equity_open - cost, atol=1e-10):
            raise AssertionError("Ledger does not conserve cash and holdings")
        # The current open is more recent than yesterday's close, including for
        # existing holdings. This is marking only, never a fabricated fill.
        marks = np.where(np.isfinite(cl), cl, np.where(finite_open, op, marks))
        equity = cash + np.dot(units, np.nan_to_num(marks))
        stale = (units > 0) & ~np.isfinite(cl)
        records.append(dict(trade_date=day, nav=equity, daily_return=equity/previous_equity-1,
            cash=cash, cost=cost, traded_notional=traded, turnover_two_sided=traded/equity_open,
            stale_value=float(np.dot(units[stale], np.nan_to_num(marks[stale]))),
            n_stale=int(stale.sum()), n_holdings=int((units > 0).sum()), rebalance=rebalance,
            n_blocked_buys=blocked_buys, n_blocked_sells=blocked_sells, n_target_without_open=target_without_open,
            pending_exit_value=float(np.dot(units[pending_exits], np.nan_to_num(marks[pending_exits])))))
        if record_trades:
            for col in np.flatnonzero((buy_units > 0) | (sell_units > 0)):
                trades.append(dict(trade_date=day, ticker=columns[col], buy_notional=buy_value[col],
                                   sell_notional=sell_value[col], fee=(buy_value[col]+sell_value[col])*fee,
                                   signal_date=last_signal_date, execution_price=op[col],
                                   bought_units=buy_units[col], sold_units=sell_units[col]))
        previous_equity = equity
    holdings = pd.DataFrame({"ticker": columns, "adjusted_units": units, "last_mark": marks})
    return pd.DataFrame(records), pd.DataFrame(trades), holdings[holdings.adjusted_units > 0]


def performance(ledger: pd.DataFrame, benchmark: pd.Series | None = None) -> dict[str, float]:
    """每日账本需含 trade_date、nav、daily_return、turnover_two_sided、cost、stale_value。

    benchmark 是以交易日为索引的基准收盘点位列，不是日收益列。
    """
    returns = ledger.daily_return
    total = ledger.nav.iloc[-1]
    sd = returns.std(ddof=1)
    nav_with_initial = np.r_[1., ledger.nav.to_numpy()]
    result = dict(annual_return=total**(252/len(ledger))-1, total_return=total-1,
        annual_volatility=sd*np.sqrt(252), sharpe=np.sqrt(252)*returns.mean()/sd if sd > 0 else np.nan,
        max_drawdown=np.min(nav_with_initial/np.maximum.accumulate(nav_with_initial)-1),
        annual_turnover_one_way=ledger.turnover_two_sided.sum()/2*252/len(ledger),
        cost_initial_nav=ledger.cost.sum(), terminal_stale_value=ledger.stale_value.iloc[-1],
        max_stale_value=ledger.stale_value.max())
    if benchmark is not None:
        benchmark = benchmark.reindex(pd.DatetimeIndex(ledger.trade_date))
        if benchmark.isna().any():
            raise ValueError("Benchmark coverage incomplete")
        benchmark_nav = benchmark / benchmark.iloc[0]
        result["benchmark_total_return"] = benchmark_nav.iloc[-1]-1
        result["relative_total_return"] = total/benchmark_nav.iloc[-1]-1
        result["annual_relative_return"] = (total/benchmark_nav.iloc[-1])**(252/len(ledger))-1
    return result
