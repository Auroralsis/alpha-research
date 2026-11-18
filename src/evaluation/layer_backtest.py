from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.evaluation.portfolio import QuoteGrid, TradeSignals, backtest_long_only, performance


def build_targets(
    frame: pd.DataFrame, score: str, layer_count: int, top_counts: Sequence[int],
) -> tuple[list[TradeSignals], dict[int, TradeSignals]]:
    """frame 是股票日长表，至少有 trade_date、ticker 和 score 指定的分数列。

    layer_count 如 20；top_counts 如 [25, 50]，表示另取前 25 或 50 只。
    返回 layers、top：layers[0][day] 是最高层名单，top[25][day] 是前 25 只名单。
    day 是 pd.Timestamp；每份名单都是股票代码列表，如 ["000001.SZ", "600000.SH"]。
    """
    layers = [{} for _ in range(layer_count)]
    top = {count: {} for count in top_counts}
    # 依靠交易日期分组，每一组包含相应的股票和因子
    for day, group in frame.groupby("trade_date", sort=True):
        usable = group.loc[np.isfinite(group[score]), ["ticker", score]]
        tickers = usable.sort_values([score, "ticker"], ascending=[False, True]).ticker.to_numpy()
        for targets, partition in zip(layers, np.array_split(tickers, layer_count)):
            targets[day] = partition.tolist()
        for count, targets in top.items():
            targets[day] = tickers[:count].tolist()
    return layers, top


def simulate_track(
    quotes: QuoteGrid, signals: TradeSignals, holding: int, offset: int, fee_bps: int,
) -> pd.DataFrame:
    """quotes 依次是复权成交价、复权估值价、原始成交价、涨停价、跌停价。

    每张表都是 [交易日 × 股票]；signals 是 {信号日: 股票名单}。
    holding、offset 以市场交易日计，fee_bps 是单边费用（1bp = 0.01%）。
    """
    return backtest_long_only(*quotes, signals, fee=fee_bps / 10000,
                              rebalance_every=holding, offset=offset, record_trades=False)[0]


def spread_track(top: pd.DataFrame, bottom: pd.DataFrame) -> pd.DataFrame:
    """top、bottom 都是每日账本，须按相同日期和行索引排列，含 trade_date、daily_return。"""
    spread = top.daily_return - bottom.daily_return
    if (spread <= -1).any():
        raise ValueError("Spread NAV cannot be compounded after losing 100%")
    return top[["trade_date"]].assign(daily_return=spread, nav=(1 + spread).cumprod(),
                                    turnover_two_sided=np.nan, cost=np.nan, stale_value=np.nan)


def track_metrics(ledger: pd.DataFrame, benchmark: pd.Series | None = None) -> dict[str, float]:
    """输入每日账本和可选的 [交易日 → 指数点位] 序列；输出本轨道的指标字典。"""
    stats = performance(ledger, benchmark)
    monthly = ledger.groupby(ledger.trade_date.dt.to_period("M")).daily_return.apply(
        lambda returns: (1 + returns).prod() - 1)
    stats["month_win_rate"] = monthly.gt(0).mean()
    if benchmark is None:
        stats["annual_turnover_one_way"] = stats["cost_initial_nav"] = np.nan
    else:
        stats["blocked_buys"] = ledger.n_blocked_buys.sum()
        stats["blocked_sells"] = ledger.n_blocked_sells.sum()
    return stats


def run_layer_tracks(
    frame: pd.DataFrame, close_quotes: QuoteGrid, benchmark: pd.Series, design: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """frame 是含 trade_date、ticker、各因子 _neutral 列的股票日长表。

    close_quotes 是五张 [交易日 × 股票] 行情表，benchmark 是按交易日索引的点位。
    design 来自 design.json。返回逐轨道统计表和含 trade_date、nav、daily_return 的净值长表。
    """
    rows, curves = [], []
    layers_count = design["layers"]
    base_fee = design["main_fee_bps"]

    def store(
        ledger: pd.DataFrame, kind: str, number: int, fee: int, relative: bool = True, save_curve: bool = False,
    ) -> None:
        """把一条轨道的账本归入指定组合类型；需要时保留逐日净值。"""
        tags = dict(factor=factor, holding_days=holding, offset=offset,
                    fee_bps=fee, kind=kind, number=number)
        rows.append(dict(**tags, **track_metrics(ledger, benchmark if relative else None)))
        if save_curve:
            curves.append(ledger[["trade_date", "nav", "daily_return"]].assign(**tags))

    for factor in design["factors"]:
        print(f"Backtesting {factor}", flush=True)
        layers, top_counts = build_targets(frame, f"{factor}_neutral", layers_count, design["top_counts"])
        for holding in design["horizons"]:
            for offset in range(holding):
                for number, signals in enumerate(layers, start=1):
                    ledger = simulate_track(close_quotes, signals, holding, offset, base_fee)
                    extreme = number in (1, layers_count)
                    store(ledger, "layer", number, base_fee, save_curve=extreme)
                    if number == 1:
                        top_layer = ledger
                store(spread_track(top_layer, ledger), "spread", 0, base_fee, relative=False, save_curve=True)
                for fee in design["top_layer_fee_bps"]:
                    ledger = simulate_track(close_quotes, layers[0], holding, offset, fee)
                    store(ledger, "layer", 1, fee)
                if holding == design["main_holding_days"]:
                    for count, signals in top_counts.items():
                        ledger = simulate_track(close_quotes, signals, holding, offset, base_fee)
                        store(ledger, "top_n", count, base_fee)
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)


def summarize_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    """输入每行一条调仓偏移轨道的统计表，按因子、周期、费率和组合取汇总。"""
    grouped = tracks.groupby(["factor", "holding_days", "fee_bps", "kind", "number"], sort=True)
    return grouped.agg(n_tracks=("offset", "size"), mean_annual=("annual_return", "mean"),
                       min_annual=("annual_return", "min"), max_annual=("annual_return", "max"),
                       mean_total=("total_return", "mean"), mean_sharpe=("sharpe", "mean"),
                       mean_drawdown=("max_drawdown", "mean"), mean_month_win=("month_win_rate", "mean"),
                       mean_annual_relative=("annual_relative_return", "mean"),
                       mean_total_relative=("relative_total_return", "mean"),
                       outperform_tracks=("relative_total_return", lambda returns: returns.gt(0).sum(min_count=1)
                                          if returns.notna().any() else np.nan)).reset_index()
