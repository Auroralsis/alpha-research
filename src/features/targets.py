import pandas as pd

def calc_future_return(prices : pd.Series , horizon):
    return prices.shift(-horizon) / prices - 1


def market_forward_returns(prices: pd.DataFrame, calendar, horizon: int):
    """Reindex first: shift then counts market days, not observed stock records.

    Return the label matrix, endpoint dates, and endpoint availability reasons.
    Missing prices are never forward-filled here.
    """
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be a positive integer")
    dates = pd.DatetimeIndex(calendar)
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
