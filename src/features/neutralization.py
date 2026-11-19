from collections.abc import Sequence
from datetime import date

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from src.data.research_panel import DateValues, parse_dates


def mad_clip(values: ArrayLike, multiple: float = 5) -> pd.Series:
    """输入一维截面数值，按中位数 ± multiple x MAD 截断，保留原行索引。"""
    values = pd.Series(values, copy=True, dtype=float)
    median = values.median()
    mad = (values - median).abs().median()
    # A zero MAD does not justify destroying the remaining cross-sectional spread.
    return values if not np.isfinite(mad) or mad == 0 else values.clip(median - multiple * mad, median + multiple * mad)


def standardize(values: ArrayLike) -> pd.Series:
    """输入一维截面数值（每项一只股票）；返回同索引的标准分数。"""
    values = pd.Series(values, dtype=float)
    sd = values.std(ddof=1)
    return (values - values.mean()) / sd if np.isfinite(sd) and sd > 1e-12 else values * np.nan


def neutralize(
    factor: ArrayLike, total_mv: ArrayLike, industry: pd.Series | Sequence[str],
    known_at: DateValues | None = None, signal_date: date | str | None = None,
) -> pd.Series:
    """输入同一天的三列 [股票] 数据：因子分数、总市值、行业代码。

    用 Series 时三列按索引对齐。先去极值，再对行业和对数总市值回归，
    返回标准化残差；缺少控制变量的股票保留缺失。
    可选 known_at 是控制变量的可知日期；提供时须给 signal_date（信号日），
    仅有行业生效日不能替代公告可知日期。
    """
    frame = pd.concat([pd.Series(factor).rename("factor"), pd.Series(total_mv).rename("cap"),
                       pd.Series(industry).rename("industry")], axis=1)
    result = pd.Series(np.nan, index=frame.index, dtype=float)
    if known_at is not None:
        if signal_date is None or (pd.to_datetime(pd.Series(known_at)).dropna() > pd.Timestamp(signal_date)).any():
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


def attach_neutral_scores(
    frame: pd.DataFrame, caps: pd.DataFrame, history: pd.DataFrame, factors: Sequence[str],
) -> pd.DataFrame:
    """给逐股因子长表补上信号日的行业、市值和各因子的中性化分数。"""
    caps = caps.rename(columns={"ts_code": "ticker"})
    caps["trade_date"] = parse_dates(caps.trade_date).to_numpy()
    frame = frame.merge(caps, on=["ticker", "trade_date"], how="left", validate="one_to_one")
    frame["industry"] = align_industry_history(frame, history)
    for factor in factors:
        frame[f"{factor}_neutral"] = np.nan
    for _, group in frame.groupby("trade_date"):
        for factor in factors:
            frame.loc[group.index, f"{factor}_neutral"] = neutralize(group[factor], group.total_mv, group.industry)
    return frame


def align_industry_history(keys: pd.DataFrame, history: pd.DataFrame) -> pd.Series:
    """给 keys 的每条（ticker, trade_date）匹配当日行业。

    history 至少含 ts_code、in_date、out_date、l1_code，按 [in_date, out_date) 生效。
    返回与 keys 同索引的行业代码列；没有匹配或多个行业互相冲突时保留缺失。
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
