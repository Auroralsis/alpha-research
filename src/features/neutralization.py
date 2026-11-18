import numpy as np
import pandas as pd


def mad_clip(values, multiple=5):
    values = pd.Series(values, copy=True, dtype=float)
    median = values.median()
    mad = (values - median).abs().median()
    # A zero MAD does not justify destroying the remaining cross-sectional spread.
    return values if not np.isfinite(mad) or mad == 0 else values.clip(median - multiple * mad, median + multiple * mad)


def standardize(values):
    values = pd.Series(values, dtype=float)
    sd = values.std(ddof=1)
    return (values - values.mean()) / sd if np.isfinite(sd) and sd > 1e-12 else values * np.nan


def neutralize(factor, total_mv, industry, known_at=None, signal_date=None):
    """MAD clip -> OLS residual on industry and log total cap -> z score.

    An explicit known_at check is available for event tables. An in_date alone
    certifies effective history, not publication-time PIT; callers must disclose.
    """
    frame = pd.concat([pd.Series(factor).rename("factor"), pd.Series(total_mv).rename("cap"),
                       pd.Series(industry).rename("industry")], axis=1)
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    if known_at is not None:
        if signal_date is None or (pd.to_datetime(known_at).dropna() > pd.Timestamp(signal_date)).any():
            raise ValueError("Controls are not known by the signal time")
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    frame = frame[frame.cap > 0]
    if len(frame) < 3:
        return result
    y = mad_clip(frame.factor)
    x = pd.get_dummies(frame.industry, dtype=float)
    x["log_cap"] = mad_clip(np.log(frame.cap))
    design = x.to_numpy(float)
    if len(frame) <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        return result
    residual = y.to_numpy() - design @ np.linalg.lstsq(design, y.to_numpy(), rcond=None)[0]
    result.loc[frame.index] = standardize(pd.Series(residual, index=frame.index))
    return result


def align_industry_history(keys, history):
    """Use effective [in_date,out_date) intervals; ambiguous overlaps stay missing.

    Returns industry only; does not invent an announcement known_at timestamp.
    """
    output = pd.Series(pd.NA, index=keys.index, dtype="string")
    history = history.copy()
    history["in_date"] = pd.to_datetime(history.in_date, format="%Y%m%d", errors="coerce")
    history["out_date"] = pd.to_datetime(history.out_date, format="%Y%m%d", errors="coerce")
    joined = keys[["ticker", "trade_date"]].assign(key_index=keys.index).merge(
        history, left_on="ticker", right_on="ts_code", how="left")
    valid = joined[(joined.in_date <= joined.trade_date)
                   & (joined.out_date.isna() | (joined.trade_date < joined.out_date))]
    distinct = valid.groupby("key_index").l1_code.agg(["nunique", "first"])
    unique = distinct[distinct["nunique"].eq(1)]
    output.loc[unique.index] = unique["first"].astype("string")
    return output
