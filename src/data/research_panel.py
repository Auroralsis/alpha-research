"""Read-only, calendar-aligned research views; source files are never rewritten."""
from dataclasses import dataclass

import numpy as np
import pandas as pd


def parse_dates(values):
    values = pd.Series(values).reset_index(drop=True)
    if pd.api.types.is_datetime64_any_dtype(values):
        result = pd.to_datetime(values, errors="raise")
    else:
        text = values.astype("string")
        if not text.str.fullmatch(r"\d{8}").fillna(False).all():
            raise ValueError("Dates must be datetime values or exactly YYYYMMDD")
        result = pd.to_datetime(text, format="%Y%m%d", errors="raise")
    if result.isna().any() or (result != result.dt.normalize()).any():
        raise ValueError("Missing dates and intraday timestamps are not allowed")
    return result


@dataclass
class ResearchPanel:
    daily: pd.DataFrame
    calendar: pd.DatetimeIndex
    tickers: pd.Index

    def matrix(self, column, clean=True):
        data = self.daily
        values = data[column]
        if clean:
            values = values.where(data.observation_status.eq("observed"))
        return (data[["ticker", "trade_date"]].assign(value=values).pivot(index="trade_date", columns="ticker", values="value")
                .reindex(index=self.calendar, columns=self.tickers))

    def observation_status(self):
        return (self.daily.pivot(index="trade_date", columns="ticker", values="observation_status")
                .reindex(index=self.calendar, columns=self.tickers).fillna("no_record_unknown"))


def prepare_panel(daily, calendar):
    """Anchor before calendar clipping, preserving pre-entry and post-exit history."""
    dates = pd.DatetimeIndex(parse_dates(calendar)).sort_values()
    if dates.has_duplicates or len(dates) == 0:
        raise ValueError("Calendar must have unique dates")
    data = daily.copy()
    data["trade_date"] = parse_dates(data.trade_date).to_numpy()
    if data.ticker.isna().any() or data.duplicated(["ticker", "trade_date"]).any():
        raise ValueError("Missing ticker or duplicate ticker/date")
    data = data.sort_values(["ticker", "trade_date"]).reset_index(drop=True)
    inside = data.trade_date.between(dates.min(), dates.max())
    if (inside & ~data.trade_date.isin(dates)).any():
        raise ValueError("Observed date inside the working interval is not a market day")
    columns = ["open", "high", "low", "close", "vol", "amount", "adj_factor"]
    data[columns] = data[columns].astype(float)
    valid = np.isfinite(data[columns]).all(axis=1) & data[columns].gt(0).all(axis=1)
    valid &= (data.low.le(data.open) & data.low.le(data.close)
              & data.high.ge(data.open) & data.high.ge(data.close))
    # The anchor is the first available observation, never the most recent factor.
    anchor = data.ticker.map(data.drop_duplicates("ticker").set_index("ticker").adj_factor)
    if (~np.isfinite(anchor) | anchor.le(0)).any():
        raise ValueError("Invalid fixed adjustment anchor")
    data["adjustment_anchor"] = anchor
    for column in ["open", "high", "low", "close"]:
        data[column + "_adj"] = data[column] * data.adj_factor / anchor
    data["observation_status"] = np.where(valid, "observed", "invalid_input")
    data.loc[data.vol.eq(0), "observation_status"] = "no_trade"
    # Records outside the requested working grid remain in the source file.
    data = data[data.trade_date.isin(dates)].copy()
    return ResearchPanel(data, dates, pd.Index(sorted(daily.ticker.unique()), name="ticker"))


def prepare_membership(membership):
    data = membership.copy()
    for column in ["trade_date", "source_snapshot_date"]:
        data[column] = parse_dates(data[column]).to_numpy()
    if data.ticker.isna().any() or data.duplicated(["ticker", "trade_date"]).any():
        raise ValueError("Invalid membership key")
    if (data.source_snapshot_date > data.trade_date).any():
        raise ValueError("Future snapshot used for membership")
    data["approximate_pit"] = True
    return data.sort_values(["trade_date", "ticker"]).reset_index(drop=True)
