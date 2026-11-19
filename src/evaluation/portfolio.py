from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple, cast

import numpy as np
import pandas as pd

from src.data.research_panel import ResearchPanel, parse_dates


TradeSignals = dict[pd.Timestamp, list[str]]
BuyQuotes = tuple[pd.DataFrame, pd.DataFrame]
TargetSelector = Callable[[pd.Timestamp, set[str], set[str]], list[str]]
HoldingsRecorder = Callable[[pd.Timestamp, pd.Index, np.ndarray, np.ndarray], None]


class QuoteGrid(NamedTuple):
    """每张行情表均以交易日为行、股票代码为列。"""

    sell_adj: pd.DataFrame
    mark_adj: pd.DataFrame
    sell_raw: pd.DataFrame
    up_limit: pd.DataFrame
    down_limit: pd.DataFrame


class BacktestResult(NamedTuple):
    """每日一行的账本、逐笔成交表，以及每只期末持仓股票的份额和账面估值。"""

    ledger: pd.DataFrame
    trades: pd.DataFrame
    final_holdings: pd.DataFrame


def load_benchmark(path: Path) -> pd.Series:
    """读取以交易日为索引的指数收盘点位。"""
    data = pd.read_csv(path, dtype={"trade_date": str})
    return pd.Series(data.close.to_numpy(), index=pd.DatetimeIndex(parse_dates(data.trade_date)))


def benchmark_nav(benchmark: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """把指数点位对齐到指定交易日，检查日期覆盖，返回首日为 1 的净值列。"""
    aligned = benchmark.reindex(dates)
    if aligned.isna().any():
        raise ValueError("Benchmark coverage incomplete")
    return aligned / aligned.iloc[0]


def close_market_inputs(
    panel: ResearchPanel, dates: pd.DatetimeIndex, tickers: pd.Index,
    limit_path: Path, benchmark_path: Path,
) -> tuple[QuoteGrid, pd.Series]:
    """按指定的交易日和股票顺序对齐收盘成交价、估值价、交易限价和指数点位。"""
    limits = pd.read_parquet(limit_path).rename(columns={"ts_code": "ticker"})
    limits["trade_date"] = parse_dates(limits.trade_date).to_numpy()
    up, down = (limits.pivot(index="trade_date", columns="ticker", values=column)
                .reindex(index=dates, columns=tickers) for column in ("up_limit", "down_limit"))
    adjusted = panel.matrix("close_adj").reindex(index=dates, columns=tickers)
    raw = panel.matrix("close").reindex(index=dates, columns=tickers)
    return QuoteGrid(adjusted, adjusted, raw, up, down), load_benchmark(benchmark_path)


def backtest_long_only(
    quotes: QuoteGrid, signals: TradeSignals, fee_bps: float = 15,
    rebalance_every: int = 20, offset: int = 0, *,
    sell_fee_bps: float | None = None, record_trades: bool = True, buy_quotes: BuyQuotes | None = None,
    select_targets: TargetSelector | None = None, target_count: int | None = None,
    retry_exits: bool = True, record_holdings: HoldingsRecorder | None = None,
    record_orders: Callable[[pd.DataFrame], None] | None = None,
) -> BacktestResult:
    """模拟一条多头组合轨道，返回（每日账本、逐笔成交、期末持仓）。

    quotes 包含同一 [交易日 x 股票] 网格上的复权卖出价、复权估值价、
    原始卖出价和原始涨跌停价。用复权卖出价确定等权目标份额。
    buy_quotes 可另传两张同网格的表，分别是复权买入价和用于涨停检查的原始价格。
    默认买卖共用成交价。日频账本允许同日的卖出款用于买入。
    select_targets 可在调仓日接收信号日期、当前持仓和可卖集合，返回本次目标名单。
    target_count 可固定等权分母，record_holdings 在日终接收当天持仓份额及估值。
    record_orders 可记录预定调仓日的计划份额、实际成交份额及成交受阻原因。
    signals 如 {pd.Timestamp("2025-01-06"): ["000001.SZ"]}，键是信号日。
    fee_bps 是买入费率，sell_fee_bps 默认同买入费率，1 bp = 0.01%。
    rebalance_every 和 offset 以市场交易日为单位。
    """
    buy_fee = fee_bps / 10000
    sell_fee = buy_fee if sell_fee_bps is None else sell_fee_bps / 10000
    if (not np.isfinite([buy_fee, sell_fee]).all() or min(buy_fee, sell_fee) < 0
            or rebalance_every <= 0 or offset < 0 or (target_count is not None and target_count <= 0)):
        raise ValueError("Invalid trading parameters")
    sell_adj, mark_adj, sell_raw, up_limit, down_limit = quotes
    dates, columns = sell_adj.index, sell_adj.columns
    universe = set(columns)
    if any(set(targets) - universe for targets in signals.values()):
        raise ValueError("Signal contains securities outside the price grid")
    buy_adj, buy_raw_reference = buy_quotes if buy_quotes is not None else (sell_adj, sell_raw)
    for matrix in [mark_adj, sell_raw, up_limit, down_limit, buy_adj, buy_raw_reference]:
        if not matrix.index.equals(dates) or not matrix.columns.equals(columns):
            raise ValueError("Trading matrices must have identical grids")
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("Invalid trading calendar")
    # 这里给出执行日schedule，每一个执行日day作为key，对应一个执行可用的股票列表
    schedule = {}
    for day in dates[offset::rebalance_every]:
        if day in signals:
            # cast是类型指示器，表明这里的结果是一个整数
            row = cast(int, dates.get_loc(day))
            if row + 1 < len(dates):
                # 这里使用set是为了确保当天的股票不重复的同时，后续的查找更快O(1)
                schedule[dates[row+1]] = set(signals[day])
    units = np.zeros(len(columns))
    cash, previous_equity = 1., 1.
    marks = np.full(len(columns), np.nan)
    pending_exits = np.zeros(len(columns), dtype=bool)
    bought_on = np.full(len(columns), -1)
    records: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []
    last_signal_date = pd.NaT
    quote_arrays = [matrix.to_numpy(dtype=float) for matrix in (*quotes, buy_adj, buy_raw_reference)]

    # 回测的全部计算流程
    for row, (day, op, cl, ro, up, down, buy_price, buy_reference) in enumerate(zip(dates, *quote_arrays)):
        finite_open = np.isfinite(op) & (op > 0)
        at_open = np.where(finite_open, op, marks)
        if ((units > 0) & ~np.isfinite(at_open)).any():
            raise ValueError("Held position has no historical mark")
        equity_open = cash + np.dot(units, np.nan_to_num(at_open))
        finite_buy = np.isfinite(buy_price) & (buy_price > 0)
        buyable = finite_open & finite_buy & np.isfinite(buy_reference) & np.isfinite(up) & (buy_reference < up - 1e-8)
        sellable = finite_open & np.isfinite(ro) & np.isfinite(down) & (ro > down + 1e-8) & (bought_on < row)
        # desired是账户希望持有多少，units是账户现在持有多少
        desired = units.copy()
        rebalance = day in schedule
        target_without_open = 0
        # 如果day in schedule，那么说明当天是执行日
        if rebalance:
            last_signal_date = dates[row-1]
            target_names = schedule[day]
            if select_targets is not None:
                target_names = set(select_targets(last_signal_date, set(columns[units > 0]), set(columns[sellable])))
                if target_names - universe:
                    raise ValueError("Selected tickers outside the price grid")
            target = np.array([ticker in target_names for ticker in columns])
            n = target_count if target_count is not None else int(target.sum())
            target_without_open = int((target & ~(finite_open & finite_buy)).sum())
            desired = np.zeros(len(columns))
            # 这里根据交易前的账户价值除以名单中的股票数去让股票平分
            # 如果有股票没有报价，他依然会占用这个n中的一个，也会影响其他股票的分配份额
            np.divide(equity_open / max(n, 1), op, out=desired, where=target & finite_open)
            # 这里找的是账户仍然持有，但是新名单已经不需要的股票
            pending_exits = (units > 0) & ~target
        if rebalance or retry_exits:
            desired[pending_exits] = 0
        requested = desired - units if rebalance and record_orders is not None else None
        blocked_sells = int(((units > desired + 1e-12) & ~sellable).sum())
        blocked_buys = int(((desired > units + 1e-12) & ~buyable).sum()) if rebalance else 0
        # 计算每只股票卖出的份额和金额，op 是复权卖出价。
        sell_units = np.where(sellable, np.maximum(units - desired, 0), 0)
        sell_value = sell_units * np.nan_to_num(op)
        units -= sell_units
        # 扣除交易费后换算成现金
        sell_total = sell_value.sum()
        cash += sell_total * (1 - sell_fee)
        # rebalance时买入
        buy_units = np.zeros(len(columns))
        if rebalance:
            buy_units = np.where(buyable & (sell_units == 0), np.maximum(desired - units, 0), 0)
            # 计算买入股票需要花费的总金额，当现金不足时乘一个系数降低购买量
            needed = np.dot(buy_units, np.nan_to_num(buy_price)) * (1 + buy_fee)
            buy_units *= min(1., max(cash, 0) / needed) if needed > 0 else 0
            units += buy_units
            bought_on[buy_units > 0] = row
        buy_value = buy_units * np.nan_to_num(buy_price)
        buy_total = buy_value.sum()
        cash -= buy_total * (1 + buy_fee)
        if cash < -1e-10 or (units < -1e-10).any():
            raise AssertionError("Negative cash or holdings")
        cash = max(cash, 0)
        traded = buy_total + sell_total
        cost = buy_total * buy_fee + sell_total * sell_fee
        # 买入份额按同一估值价记账，买入成交价与估值价的差额体现在净值里。
        after_trade = cash + np.dot(units, np.nan_to_num(at_open))
        buy_price_effect = np.dot(buy_units, np.nan_to_num(at_open)) - buy_total
        if not np.isclose(after_trade, equity_open - cost + buy_price_effect, atol=1e-10):
            raise AssertionError("Ledger does not conserve cash and holdings")
        # marks是对当前股票价格的估值，如果有有效收盘价，就用cl，没有有效收盘价，但是
        # 有当日成交时点报价op就沿用。都没有就沿用上一次账本对该股票的估值
        marks = np.where(np.isfinite(cl), cl, np.where(finite_open, op, marks))
        equity = cash + np.dot(units, np.nan_to_num(marks))
        # stale表示仍然持有，但没有有效收盘价的股票。
        stale = (units > 0) & ~np.isfinite(cl)
        if record_holdings is not None:
            record_holdings(day, columns, units, marks)
        if requested is not None and record_orders is not None:
            selected = np.flatnonzero(np.abs(requested) > 1e-12)
            buy = requested[selected] > 0
            intended = np.abs(requested[selected])
            filled = (buy_units + sell_units)[selected]
            tradable = np.where(buy, buyable[selected], sellable[selected])
            limit_reference = np.where(buy, buy_reference[selected], ro[selected])
            has_quote = (finite_open[selected] & (~buy | finite_buy[selected]) & np.isfinite(limit_reference)
                         & np.isfinite(np.where(buy, up[selected], down[selected])))
            status = np.select(
                [~tradable & ~has_quote, ~tradable, np.isclose(intended, filled, rtol=1e-10, atol=1e-12), filled > 0],
                ["missing_quote_or_limit", "limit_or_hold_constraint", "filled", "partial_cash"],
                default="unfilled_cash",
            )
            record_orders(pd.DataFrame({"trade_date": day, "signal_date": last_signal_date,
                                        "ticker": columns.take(selected), "side": np.where(buy, "buy", "sell"),
                                        "requested_adjusted_units": intended, "filled_adjusted_units": filled,
                                        "raw_limit_reference": limit_reference, "status": status}))
        # records是一个字典列表，转换为df时将每个字典变成一行，字典的key变成列名
        # nav是净值，绘制其随交易日的变化可以得到净值曲线
        records.append(dict(trade_date=day, nav=equity, daily_return=equity/previous_equity-1,
            cash=cash, cost=cost, traded_notional=traded, turnover_two_sided=traded/equity_open,
            stale_value=float(np.dot(units[stale], np.nan_to_num(marks[stale]))),
            n_stale=int(stale.sum()), n_holdings=int((units > 0).sum()), rebalance=rebalance,
            n_blocked_buys=blocked_buys, n_blocked_sells=blocked_sells, n_target_without_open=target_without_open,
            pending_exit_value=float(np.dot(units[pending_exits], np.nan_to_num(marks[pending_exits])))))
        if record_trades:
            for position in np.flatnonzero((buy_units > 0) | (sell_units > 0)):
                col = int(position)
                ticker = cast(str, columns[col])
                trades.append(dict(trade_date=day, ticker=ticker, buy_notional=buy_value[col],
                                   sell_notional=sell_value[col], fee=buy_value[col]*buy_fee+sell_value[col]*sell_fee,
                                   signal_date=last_signal_date,
                                   execution_price=buy_price[col] if buy_units[col] > 0 else op[col],
                                   bought_units=buy_units[col], sold_units=sell_units[col]))
        previous_equity = equity
    holdings = pd.DataFrame({"ticker": columns, "adjusted_units": units, "last_mark": marks})
    return BacktestResult(pd.DataFrame(records), pd.DataFrame(trades), holdings.loc[holdings["adjusted_units"].gt(0), :])


def performance(ledger: pd.DataFrame, benchmark: pd.Series | None = None) -> dict[str, float]:
    """每日账本ledger需含 trade_date、nav、daily_return、turnover_two_sided、cost、stale_value。

    benchmark 是以交易日为索引的基准收盘点位列，不是日收益列。
    """
    returns = cast(pd.Series, ledger["daily_return"])
    total = ledger.nav.iloc[-1]
    # 计算方差时除以(有效个数-1)
    sd = returns.std(ddof=1)
    nav_with_initial = np.r_[1., ledger.nav.to_numpy()]
    # annual_return: 年化收益率 total_return: 净收益率
    # annual_volatility: 年化波动率 sharpe: 年化夏普率
    # max_drawdown: 最大回撤 annual_turnover_one_way: 年化单边换手率
    result = dict(annual_return=total**(252/len(ledger))-1, total_return=total-1,
        annual_volatility=sd*np.sqrt(252), sharpe=np.sqrt(252)*returns.mean()/sd if sd > 0 else np.nan,
        max_drawdown=np.min(nav_with_initial/np.maximum.accumulate(nav_with_initial)-1),
        annual_turnover_one_way=ledger.turnover_two_sided.sum(min_count=1)/2*252/len(ledger),
        cost_initial_nav=ledger.cost.sum(min_count=1), terminal_stale_value=ledger.stale_value.iloc[-1],
        max_stale_value=ledger.stale_value.max())
    if benchmark is not None:
        index_end = benchmark_nav(benchmark, pd.DatetimeIndex(ledger.trade_date)).iloc[-1]
        result["benchmark_total_return"] = index_end-1
        result["relative_total_return"] = total/index_end-1
        result["annual_relative_return"] = (total/index_end)**(252/len(ledger))-1
    # 同一个月内的日收益连乘，统计月收益大于零的比例。
    months = pd.PeriodIndex(ledger["trade_date"].to_numpy(), freq="M")
    monthly = (1 + returns).groupby(months).prod() - 1
    result["month_win_rate"] = monthly.gt(0).mean()
    if "n_blocked_buys" in ledger:
        result["blocked_buys"] = ledger.n_blocked_buys.sum()
        result["blocked_sells"] = ledger.n_blocked_sells.sum()
    return result
