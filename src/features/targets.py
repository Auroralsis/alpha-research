import pandas as pd

def calc_future_return(prices : pd.Series , horizon):
    return prices.shift(-horizon) / prices - 1


def market_forward_returns(prices: pd.DataFrame, calendar, horizon: int):
    """输入复权收盘价(prices)，日期为行，股票为列，交易日历(calender)，未来期限(horizon)

        返回收益标签矩阵(labels)、收益计算的终点日期(end)，以及终点数据可用性及缺失原因(reason)。
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


def attach_labels(frame, panel, horizons):
    """Join forward labels by the signal date and ticker of each row."""
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
