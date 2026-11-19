"""将模型和单因子分数转为 Top-K 换股信号，交给共用回测引擎。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.research_panel import ResearchPanel, prepare_membership, prepare_panel
from src.evaluation.portfolio import backtest_long_only, benchmark_nav, close_market_inputs, performance
from src.features.neutralization import attach_neutral_scores
from src.features.price_volume import compute_factors
from src.ml.models import FACTOR_SCORES, SCORE_COLUMNS, common_scores


def dropout_orders(ranked: list[str], held: set[str], topk: int, n_drop: int) -> tuple[set[str], list[str]]:
    """输入按分数降序排列的股票代码及当前持仓，返回拟卖出集合和候选买入顺序。"""
    if not 0 <= n_drop <= topk or len(held) > topk:
        raise ValueError("Invalid dropout inventory or parameters")
    fresh = [ticker for ticker in ranked if ticker not in held][:n_drop + topk - len(held)]
    combined = held | set(fresh)
    ordered = [ticker for ticker in ranked if ticker in combined] + sorted(held - set(ranked))
    sells = held.intersection(ordered[-n_drop:]) if n_drop else set()
    return sells, fresh[:len(sells) + topk - len(held)]


def dropout_targets(ranked: list[str], held: set[str], sellable: set[str], topk: int, n_drop: int) -> list[str]:
    """根据可卖出的持仓估计能腾出的仓位，返回本次目标持仓名单。"""
    sells, buys = dropout_orders(ranked, held, topk, n_drop)
    slots = topk - len(held) + len(sells & sellable)
    return sorted(held - sells) + buys[:slots]


def compare_scores(
    predictions: pd.DataFrame, inputs: dict[str, Path], config: dict, market_dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, ResearchPanel, pd.DatetimeIndex]:
    """按交易日和股票合并模型预测与中性化单因子分数，另返回行情容器和测试期交易日。"""
    start, end = config["segments"]["test"]
    dates = market_dates[market_dates.to_series().between(start, end).to_numpy()]
    if dates.empty:
        raise ValueError("No test trading sessions")
    first = market_dates.get_indexer(dates[:1])[0]
    working = market_dates[max(first - 21, 0):first + len(dates)]
    panel = prepare_panel(pd.read_parquet(inputs["prices"]), working)
    members = prepare_membership(pd.read_parquet(inputs["membership"]))
    factors = compute_factors(panel, members, dates)
    factors = attach_neutral_scores(factors, pd.read_parquet(inputs["caps"]), pd.read_parquet(inputs["industries"]),
                                    FACTOR_SCORES)
    columns = {f"{name}_neutral": name for name in FACTOR_SCORES}
    factors = factors[["trade_date", "ticker", *columns]].rename(columns=columns)
    scores = predictions.drop(columns=list(FACTOR_SCORES), errors="ignore").merge(
        factors, on=["trade_date", "ticker"], how="left", validate="one_to_one",
    )
    return scores, panel, dates


def run_backtests(
    inputs: dict[str, Path], config: dict, dates: pd.DatetimeIndex, scores: pd.DataFrame, panel: ResearchPanel, out: Path,
    *, strategies: tuple[str, ...] = SCORE_COLUMNS,
) -> None:
    """输入测试期交易日、各方法的打分和行情，逐方法写出同口径周频回测结果。"""
    test = scores.loc[scores.split.eq("test"), ["trade_date", "ticker", *strategies]]
    if test.empty or test.duplicated(["trade_date", "ticker"]).any():
        raise ValueError("Invalid test score keys")
    prediction = common_scores(test, strategies)
    candidates = prediction.groupby("trade_date").size().reindex(dates, fill_value=0)
    if candidates.lt(config["topk"]).any():
        raise ValueError("Some test sessions have fewer common finite scores than the target portfolio size")
    tickers = pd.Index(sorted(prediction.ticker.unique()), name="ticker")
    quotes, benchmark = close_market_inputs(panel, dates, tickers, inputs["limits"], inputs["benchmark"])
    factor = quotes.sell_adj / quotes.sell_raw
    mark_factor = factor.ffill().to_numpy(dtype=float)
    factor_values, raw_values = factor.to_numpy(dtype=float), quotes.sell_raw.to_numpy(dtype=float)
    capital = config["initial_capital_cny"]
    topk, n_drop, every = (config[key] for key in ("topk", "n_drop", "rebalance_days"))
    buy_fee, sell_fee = config["buy_fee_bps"], config["sell_fee_bps"]
    if capital <= 0:
        raise ValueError("Initial capital must be positive")
    summary, ledgers, executions, orders, inventory = [], [], [], [], []
    index_nav = benchmark_nav(benchmark, dates)
    index_ledger = pd.DataFrame(dict(trade_date=dates, nav=index_nav.to_numpy(),
                                     daily_return=index_nav.pct_change().fillna(0).to_numpy(),
                                     turnover_two_sided=np.nan, cost=np.nan, stale_value=np.nan))
    summary.append(dict(strategy="benchmark", **performance(index_ledger, benchmark)))

    for name in strategies:
        ordered = prediction.sort_values(["trade_date", name, "ticker"], ascending=[True, False, True])
        ranked = {day: group.ticker.to_list() for day, group in ordered.groupby("trade_date")}

        def select(day: pd.Timestamp, held: set[str], sellable: set[str]) -> list[str]:
            """按信号日取得排序，并用执行日的可卖集合生成目标名单。"""
            return dropout_targets(ranked[day], held, sellable, topk, n_drop)

        for offset in range(every):
            holdings, requests = [], []

            def record(day: pd.Timestamp, columns: pd.Index, units: np.ndarray, marks: np.ndarray) -> None:
                """把日终实际持仓转成等效股数和人民币市值。"""
                active = np.flatnonzero(units > 0)
                factors = mark_factor[dates.get_loc(day), active]
                holdings.append(pd.DataFrame({"trade_date": day, "ticker": columns.take(active),
                                             "adjusted_units": units[active] * capital,
                                             "raw_share_equivalent": units[active] * factors * capital,
                                             "raw_mark_price": marks[active] / factors,
                                             "holding_value_cny": units[active] * marks[active] * capital}))

            ledger, trades, _ = backtest_long_only(
                quotes, ranked, fee_bps=buy_fee, sell_fee_bps=sell_fee,
                rebalance_every=every, offset=offset, select_targets=select,
                target_count=topk, retry_exits=False, record_holdings=record, record_orders=requests.append,
            )
            tags = dict(strategy=name, buy_fee_bps=buy_fee, sell_fee_bps=sell_fee, offset=offset)
            stats = performance(ledger, benchmark)
            summary.append(dict(**tags, **stats, annual_turnover_two_way=2 * stats["annual_turnover_one_way"],
                                cost_cny=capital * stats["cost_initial_nav"], rebalance_days=int(ledger.rebalance.sum())))
            ledger = ledger.assign(**tags, cash_cny=capital * ledger.cash, equity_cny=capital * ledger.nav,
                                   cost_cny=capital * ledger.cost, traded_notional_cny=capital * ledger.traded_notional)
            ledgers.append(ledger)
            orders.append(pd.concat(requests, ignore_index=True).assign(**tags))
            inv = pd.concat(holdings, ignore_index=True).assign(**tags)
            values = inv.groupby("trade_date").holding_value_cny.sum().reindex(dates, fill_value=0).to_numpy()
            np.testing.assert_allclose(ledger.cash_cny + values, ledger.equity_cny, atol=1e-5)
            assert ledger.n_holdings.le(topk).all()
            inventory.append(inv)
            if not trades.empty:
                row = dates.get_indexer(trades.trade_date)
                col = tickers.get_indexer(trades.ticker)
                trade_factor = factor_values[row, col]
                trades = trades.rename(columns={"execution_price": "adjusted_unit_price"}).assign(
                    **tags, raw_close_price=raw_values[row, col],
                    bought_share_equivalent=trades.bought_units * trade_factor * capital,
                    sold_share_equivalent=trades.sold_units * trade_factor * capital,
                    buy_cny=trades.buy_notional * capital, sell_cny=trades.sell_notional * capital,
                    fee_cny=trades.fee * capital)
                np.testing.assert_allclose(trades.bought_share_equivalent * trades.raw_close_price, trades.buy_cny, atol=1e-6)
                np.testing.assert_allclose(trades.sold_share_equivalent * trades.raw_close_price, trades.sell_cny, atol=1e-6)
                np.testing.assert_allclose(trades.fee_cny, (trades.buy_cny * buy_fee + trades.sell_cny * sell_fee) / 10000)
                executions.append(trades)
        print(f"Backtest: {name}, buy {buy_fee} bp / sell {sell_fee} bp, all {every} offsets", flush=True)
    pd.DataFrame(summary).to_csv(out / "portfolio_metrics.csv", index=False)
    for filename, frames in (("ledger", ledgers), ("trades", executions), ("orders", orders), ("holdings", inventory)):
        pd.concat(frames, ignore_index=True).to_parquet(out / f"{filename}.parquet", index=False)
    missing_limits = quotes.sell_raw.notna() & (quotes.up_limit.isna() | quotes.down_limit.isna())
    (out / "execution_audit.json").write_text(json.dumps({
        "calendar_sessions": len(dates), "quoted_tickers": len(tickers),
        "test_stock_days": len(test), "common_scored_stock_days": len(prediction),
        "min_daily_candidates": int(candidates.min()), "max_daily_candidates": int(candidates.max()),
        "observed_close_missing_limit": int(missing_limits.to_numpy().sum()),
        "bench_price_start": float(benchmark.loc[dates[0]]), "bench_price_end": float(benchmark.loc[dates[-1]]),
    }, indent=2) + "\n")
