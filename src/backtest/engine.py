"""用 Qlib 原生 TopkDropoutStrategy 回测各方法的股票日分数。"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.data.qlib_provider import qlib_bars, register_provider, write_provider
from src.data.research_panel import CSI500_INPUTS, ResearchPanel, load_market_calendar, parse_dates, prepare_test_panel
from src.backtest.factor_metrics import prediction_diagnostics
from src.backtest.portfolio import load_benchmark_returns, performance


def exchange_bars(panel: ResearchPanel, limits_path: Path, tickers: pd.Index, buy_field: str,
                  limit_threshold: float | None) -> pd.DataFrame:
    """准备复权量价。使用固定阈值时，涨跌幅相对上一有效复权收盘价计算。"""
    daily = panel.daily.loc[panel.daily.ticker.isin(tickers)]
    bars = qlib_bars(ResearchPanel(daily, panel.calendar, tickers))
    bars["change"] = bars.groupby("ticker").close.pct_change(fill_method=None)
    if limit_threshold is not None:
        return bars
    # 兼容按逐股逐日原始限价设置的历史实验配置。
    keys = ["ticker", "trade_date"]
    limits = pd.read_parquet(limits_path, columns=["ts_code", "trade_date", "up_limit", "down_limit"])
    limits = limits.rename(columns={"ts_code": "ticker"})
    limits["trade_date"] = parse_dates(limits.trade_date).to_numpy()
    raw = daily[[*keys, "close", "low"]].rename(columns={"close": "raw_close", "low": "raw_low"})
    market = bars.merge(raw, on=keys, validate="one_to_one").merge(limits, on=keys, how="left", validate="one_to_one")
    buy_reference = market.raw_low if buy_field == "vwap" else market.raw_close
    market["buy_blocked"] = market.up_limit.isna() | market[buy_field].isna() | buy_reference.ge(market.up_limit - 1e-8)
    market["sell_blocked"] = market.down_limit.isna() | market.raw_close.le(market.down_limit + 1e-8)
    return market.drop(columns=["raw_close", "raw_low", "up_limit", "down_limit"])


def daily_ledger(report: pd.DataFrame, capital: float) -> pd.DataFrame:
    """把 Qlib 的日频账户报表转换为现有研究使用的净值与收益格式。"""
    day_cost = report.total_cost.diff().fillna(report.total_cost.iloc[0])
    return pd.DataFrame({
        "trade_date": report.index,
        "nav": report.account.to_numpy() / capital,
        "daily_return": (report["return"] - report.cost).to_numpy(),
        "daily_return_before_cost": report["return"].to_numpy(),
        "cost_rate": report.cost.to_numpy(),
        "cost_cny": day_cost.to_numpy(),
        "turnover_two_sided": report.turnover.to_numpy(),
        "equity_cny": report.account.to_numpy(),
        "cash_cny": report.cash.to_numpy(),
        "benchmark_return": report.bench.to_numpy(),
    })


def run_backtests(
    inputs: dict[str, Path], config: dict, dates: pd.DatetimeIndex, scores: pd.DataFrame, panel: ResearchPanel, out: Path,
    *, strategies: tuple[str, ...],
) -> None:
    """各方法用自己的有效分数选股，共用 Qlib 策略、交易所和模拟执行器。"""
    from qlib.backtest import backtest, get_exchange
    from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy

    if scores.empty or scores.duplicated(["trade_date", "ticker"]).any() or not scores.trade_date.isin(dates).all():
        raise ValueError("测试期打分缺失、日期越界或股票日键重复")
    signal_frame = scores.set_index(["trade_date", "ticker"])[list(strategies)]
    signal_frame = signal_frame.rename_axis(["datetime", "instrument"]).sort_index()
    finite = np.isfinite(signal_frame)
    signal_frame = signal_frame.where(finite)
    tickers = signal_frame.index[finite.any(axis=1)].get_level_values("instrument").unique().sort_values()
    if tickers.empty:
        raise ValueError("所有方法均无有效测试期分数")
    common_counts = finite.all(axis=1).groupby(level="datetime").sum().reindex(dates, fill_value=0)
    capital = config["initial_capital_cny"]
    topk, n_drop, every = (config[key] for key in ("topk", "n_drop", "rebalance_days"))
    buy_field = config.get("buy_price", "vwap")
    threshold = config.get("limit_threshold")
    min_cost = config.get("min_trade_cost_cny", 0)
    trade_unit = config.get("trade_unit")
    bars = exchange_bars(panel, inputs["limits"], tickers, buy_field, threshold)
    index_returns = load_benchmark_returns(inputs["benchmark"], dates)
    index_ledger = pd.DataFrame(dict(trade_date=dates, nav=(1 + index_returns).cumprod().to_numpy(),
                                     daily_return=index_returns.to_numpy(), turnover_two_sided=np.nan, cost_cny=np.nan))
    summary = [dict(strategy="benchmark", **performance(index_ledger, index_returns))]
    ledgers, trades, holdings = [], [], []
    candidate_coverage = {}
    prior_session = pd.Series(dates[:-1], index=dates[1:])

    with TemporaryDirectory(prefix="csi500-qlib-backtest-") as temp:
        provider = Path(temp)
        write_provider(bars, panel.calendar, provider)
        # Qlib 用下一条日历记录界定当日末尾，首日也会访问前一条记录。
        calendar = panel.calendar
        if calendar[0] == dates[0]:
            calendar = calendar.insert(0, dates[0] - pd.Timedelta(days=1))
        if calendar[-1] == dates[-1]:
            calendar = calendar.append(pd.DatetimeIndex([dates[-1] + pd.Timedelta(days=1)]))
        (provider / "calendars/day_future.txt").write_text("\n".join(calendar.strftime("%Y-%m-%d")) + "\n")
        register_provider(provider, config["qlib_version"])
        exchange = get_exchange(
            freq="day", start_time=dates[0], end_time=dates[-1], codes=tickers.tolist(),
            deal_price=(f"${buy_field}", "$close"),
            limit_threshold=float(threshold) if threshold is not None else ("$buy_blocked", "$sell_blocked"),
            open_cost=config["buy_fee_bps"] / 10_000, close_cost=config["sell_fee_bps"] / 10_000,
            min_cost=min_cost, trade_unit=trade_unit,
        )
        executor = {"class": "SimulatorExecutor", "module_path": "qlib.backtest.executor",
                    "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True}}

        for name in strategies:
            available = signal_frame[name].dropna()
            available_dates = available.index.get_level_values("datetime")
            counts = available.groupby(level="datetime").size().reindex(dates, fill_value=0)
            candidate_coverage[name] = dict(scored_stock_days=len(available), min_daily_candidates=int(counts.min()),
                                            max_daily_candidates=int(counts.max()))
            for offset in range(every):
                signal_dates = dates[offset:-1:every]
                signal = available.loc[available_dates.isin(signal_dates)]
                strategy = TopkDropoutStrategy(
                    signal=signal, topk=topk, n_drop=n_drop, forbid_all_trade_at_limit=threshold is not None,
                    # Qlib 0.9.7 的 bottom 路径会把 [-0:] 当全量。random 的零个卖单保留原生零换股语义。
                    method_sell="bottom" if n_drop else "random",
                )
                portfolio, indicators = backtest(
                    start_time=dates[0], end_time=dates[-1], strategy=strategy, executor=executor,
                    benchmark=index_returns, account=float(capital), exchange_kwargs={"exchange": exchange},
                )
                report, positions = portfolio["1day"]
                if not report.index.equals(dates):
                    raise ValueError("Qlib 账本日期与研究日历不一致")
                tags = dict(strategy=name, offset=offset)
                ledger = daily_ledger(report, capital)
                summary.append(dict(**tags, **performance(ledger, index_returns), rebalance_days=len(signal_dates)))
                ledgers.append(ledger.assign(**tags))
                for day, indicator in indicators["1day"][1].order_indicator_his.items():
                    # Qlib 0.9.7 的 Index 需要转成列表才能交给 pandas 2.2。
                    orders = pd.DataFrame({name: pd.Series(metric.data, index=metric.index.tolist())
                                           for name, metric in indicator.data.items()})
                    if not orders.empty:
                        trades.append(orders.rename_axis("ticker").reset_index().assign(
                            trade_date=day, signal_date=prior_session.loc[day], **tags,
                        ))
                holdings.extend(dict(trade_date=day, ticker=code, amount=position.get_stock_amount(code),
                                     price=position.get_stock_price(code), **tags)
                                for day, position in positions.items() for code in position.get_stock_list())
            print(f"{name} Qlib Top-{topk} 回测完成，各 {every} 个偏移", flush=True)
    pd.DataFrame(summary).to_csv(out / "portfolio_metrics.csv", index=False)
    pd.concat(ledgers, ignore_index=True).to_parquet(out / "ledger.parquet", index=False)
    orders = pd.concat(trades, ignore_index=True) if trades else pd.DataFrame(
        columns=["ticker", "trade_date", "signal_date", "strategy", "offset", "amount", "deal_amount", "trade_price",
                 "trade_value", "trade_cost", "trade_dir"],
    )
    orders.to_parquet(out / "trades.parquet", index=False)
    pd.DataFrame(holdings, columns=["trade_date", "ticker", "amount", "price", "strategy", "offset"]).to_parquet(
        out / "holdings.parquet", index=False,
    )
    (out / "execution_audit.json").write_text(json.dumps({
        "engine": "qlib.TopkDropoutStrategy", "calendar_sessions": len(dates), "quoted_tickers": len(tickers),
        "common_scored_stock_days": int(common_counts.sum()), "min_daily_candidates": int(common_counts.min()),
        "max_daily_candidates": int(common_counts.max()), "candidate_coverage": candidate_coverage,
        "buy_price": buy_field, "limit_threshold": threshold,
        "min_trade_cost_cny": min_cost, "trade_unit": trade_unit, "benchmark_uses_pre_close": True,
    }, indent=2) + "\n")


def evaluate_signals(scores: pd.DataFrame, config: dict, out: Path, methods: tuple[str, ...],
                     *, portfolio_methods: tuple[str, ...] | None = None) -> pd.DatetimeIndex:
    """在验证和测试期评价分数，仅用测试期分数回测，返回账本交易日。"""
    calendar = load_market_calendar(CSI500_INPUTS["calendar"])
    prediction_diagnostics(scores, config, calendar, out, strategies_by_split={"validation": methods, "test": methods})
    panel, dates = prepare_test_panel(config, calendar)
    run_backtests(CSI500_INPUTS, config, dates, scores.loc[scores.split.eq("test")], panel, out,
                  strategies=methods if portfolio_methods is None else portfolio_methods)
    return dates
