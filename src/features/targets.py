from collections.abc import Iterable

import pandas as pd

from src.data.research_panel import DateValues, ResearchPanel


def calc_future_return(prices: pd.Series, horizon: int) -> pd.Series:
    """输入已按日期排序的单只股票价格；horizon 是向后取价的行数。"""
    return prices.shift(-horizon) / prices - 1


def market_forward_returns(
    prices: pd.DataFrame, calendar: DateValues, horizon: int,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """输入 [交易日 × 股票] 复权收盘价、完整交易日历和未来交易日数。

    返回 labels、end、reason：收益和缺失原因都是 [交易日 × 股票] 宽表，
    end 是以信号日为索引的一维终点日期列。
    """
    # 要求期限horizon是正整数
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    dates = pd.DatetimeIndex(calendar)
    # 要求交易日历递增不重复
    if not dates.is_monotonic_increasing or dates.has_duplicates:
        raise ValueError("Calendar must be increasing and unique")
    if prices.index.has_duplicates:
        raise ValueError("Duplicate price dates")
    aligned = prices.reindex(dates)
    aligned = aligned.where(aligned.gt(0) & aligned.lt(float("inf")))
    end = pd.Series(dates, index=dates).shift(-horizon)
    future = aligned.shift(-horizon)
    labels = future / aligned - 1
    reason = pd.DataFrame("available", index=dates, columns=aligned.columns)
    reason = reason.mask(future.isna(), "endpoint_missing_unknown")
    reason = reason.mask(aligned.isna(), "signal_price_missing")
    reason.loc[end.isna()] = "calendar_boundary"
    return labels, end, reason


def attach_labels(frame: pd.DataFrame, panel: ResearchPanel, horizons: Iterable[int]) -> pd.DataFrame:
    """frame 是带 trade_date、ticker 的长表；两列共同定位 panel 的一个价格格子。

    horizons 如 [5, 10, 20]。返回原表副本，为每个期限添加收益、终点日期及缺失原因。
    """
    frame = frame.copy()
    prices = panel.matrix("close_adj")
    rows = prices.index.get_indexer(frame.trade_date)
    columns = prices.columns.get_indexer(frame.ticker)
    if (rows == -1).any() or (columns == -1).any():
        raise ValueError("input frame index not found in panel")
    for horizon in horizons:
        labels, end, reason = market_forward_returns(prices, panel.calendar, horizon)
        frame[f"future_ret_{horizon}"] = labels.to_numpy()[rows, columns]
        frame[f"label_end_{horizon}"] = frame.trade_date.map(end)
        frame[f"label_reason_{horizon}"] = reason.to_numpy()[rows, columns]
    return frame
