from typing import cast

import numpy as np
import pandas as pd

from src.evaluation.portfolio import BuyQuotes, QuoteGrid, TradeSignals, backtest_long_only, performance


def build_targets(frame: pd.DataFrame, score: str, layer_count: int) -> list[TradeSignals]:
    """frame 是股票日长表，至少有 trade_date、ticker 和 score 指定的分数列。

    layer_count 如 20。返回各层的信号字典，layers[0][day] 是最高层在当天的名单。
    day 是 pd.Timestamp，名单是股票代码列表，如 ["000001.SZ", "600000.SH"]。
    """
    layers: list[TradeSignals] = [{} for _ in range(layer_count)]
    # 依靠交易日期分组，每一组包含相应的股票和因子
    for day, group in frame.groupby("trade_date", sort=True):
        # 对每一个不同的交易日，股票应该是不一样的
        # 先按因子分数从高到低排序，同分时按股票代码升序排列
        usable = group.loc[np.isfinite(group[score]), ["ticker", score]]
        tickers = usable.sort_values([score, "ticker"], ascending=[False, True]).ticker.to_numpy()
        signal_day = cast(pd.Timestamp, day)
        for targets, partition in zip(layers, np.array_split(tickers, layer_count)):
            # 这里target = {}，对应第n层，每个不同的day作为key对应一个列表
            # 这个列表就是每天的第n层所对应的股票列表
            targets[signal_day] = partition.tolist()
    return layers


def spread_track(top: pd.DataFrame, bottom: pd.DataFrame) -> pd.DataFrame:
    """top、bottom 都是每日账本，须按相同日期和行索引排列，含 trade_date、daily_return。"""
    spread = top.daily_return - bottom.daily_return
    if (spread <= -1).any():
        raise ValueError("Spread NAV cannot be compounded after losing 100%")
    return top[["trade_date"]].assign(daily_return=spread, nav=(1 + spread).cumprod(),
                                    turnover_two_sided=np.nan, cost=np.nan, stale_value=np.nan)


def run_layer_tracks(
    frame: pd.DataFrame, close_quotes: QuoteGrid, benchmark: pd.Series, design: dict, *, buy_quotes: BuyQuotes,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """frame 是含 trade_date、ticker、各因子 _neutral 列的股票日长表。

    close_quotes 是五张 [交易日 x 股票] 行情表，benchmark 是按交易日索引的点位。
    buy_quotes 提供同一网格下的复权买入价和原始涨停判断价。
    design 来自 design.json。返回逐轨道统计表和含 trade_date、nav、daily_return 的净值长表。
    """
    rows, curves = [], []
    layers_count = design["layers"]
    if layers_count < 2:
        raise ValueError("At least two layers are required for the top-minus-bottom spread")
    base_fee = design["main_fee_bps"]

    def store(
        ledger: pd.DataFrame, kind: str, number: int, relative: bool = True, save_curve: bool = False,
    ) -> None:
        """把一条轨道的账本归入指定组合类型；需要时保留逐日净值。"""
        tags = dict(factor=factor, holding_days=holding, offset=offset,
                    fee_bps=base_fee, kind=kind, number=number)
        rows.append(dict(**tags, **performance(ledger, benchmark if relative else None)))
        if save_curve:
            curves.append(ledger[["trade_date", "nav", "daily_return"]].assign(**tags))

    for factor in design["factors"]:
        print(f"Backtesting {factor}", flush=True)
        layers = build_targets(frame, f"{factor}_neutral", layers_count)
        for holding in design["horizons"]:
            for offset in range(holding):
                ledgers = [
                    backtest_long_only(close_quotes, signals, fee_bps=base_fee, rebalance_every=holding,
                                       offset=offset, record_trades=False, buy_quotes=buy_quotes).ledger
                    for signals in layers
                ]
                for number, ledger in enumerate(ledgers, start=1):
                    extreme = number in (1, layers_count)
                    store(ledger, "layer", number, save_curve=extreme)
                store(spread_track(ledgers[0], ledgers[-1]), "spread", 0, relative=False, save_curve=True)
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
