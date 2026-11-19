"""量价因子公式，以及横截面排名和滚动统计。"""

from typing import cast

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import rankdata

from src.data.research_panel import DateValues, ResearchPanel, as_date_index
from .momentum import calc_momentum


def cross_sectional_rank(values: ArrayLike) -> np.ndarray:
    """输入 [股票] 向量或 [交易日 x 股票] 矩阵，每个日期内部排名；返回同形状数组。"""
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values), values, np.nan)
    n = np.isfinite(values).sum(axis=-1, keepdims=True)
    # 这里计算排名时直接忽略NaN的股票
    result = rankdata(values, method="average", axis=-1, nan_policy="omit") / np.maximum(n, 1)
    # 只对至少有两个股票的横截面给出排名结果
    return np.where(n >= 2, result, np.nan)


def window_covariance(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """x、y 都是 [窗口交易日 x 股票] 矩阵，行列须逐格对应；返回 [股票] 协方差向量。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or x.ndim != 2 or len(x) < 2:
        raise ValueError("Expected matching windows with at least two rows")
    complete = (np.isfinite(x) & np.isfinite(y)).all(axis=0)
    xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
    return np.where(complete, (xc * yc).sum(axis=0) / (len(x) - 1), np.nan)


def window_correlation(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """x、y 都是 [窗口交易日 x 股票] 矩阵，沿日期计算；返回 [股票] 相关系数向量。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    covariance = window_covariance(x, y)
    xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
    denominator = np.sqrt((xc * xc).sum(axis=0) * (yc * yc).sum(axis=0))
    # np.full是根据输入shape和填充值填充一个数组
    out = np.full(covariance.shape, np.nan)
    # 这里为了防止denominator过小(比如某一个股票的某一组数据标准差为0时)而设置阈值
    np.divide(covariance * (len(x) - 1), denominator, out=out, where=denominator > 1e-24)
    # 将输出限制在[-1,1]之间，超出的结果赋值为-1或1
    return np.clip(out, -1, 1)


FACTORS = ("Alpha3", "Alpha13", "Alpha44", "MOM20")


def _pair_ranks(price: np.ndarray, volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """两个输入都是同形状的 [交易日 x 股票] 矩阵；返回价格排名和成交量排名矩阵。"""
    # 只有当price和volume两个量均不是NaN时才会参与计算横截面
    valid = np.isfinite(price) & np.isfinite(volume)
    return (cross_sectional_rank(np.where(valid, price, np.nan)),
            cross_sectional_rank(np.where(valid, volume, np.nan)))


def alpha3(open_rank: ArrayLike, volume_rank: ArrayLike) -> np.ndarray:
    """输入 [10 个交易日 x 股票] 的开盘价排名和成交量排名；返回 [股票] 分数。"""
    open_rank = np.asarray(open_rank, dtype=float)
    if len(open_rank) != 10:
        raise ValueError("Alpha3 requires ten market days")
    return -window_correlation(open_rank, volume_rank)


def alpha13(close_rank: ArrayLike, volume_rank: ArrayLike) -> np.ndarray:
    """输入 [5 个交易日 x 股票] 的收盘价排名和成交量排名；返回 [股票] 分数。"""
    close_rank = np.asarray(close_rank, dtype=float)
    if len(close_rank) != 5:
        raise ValueError("Alpha13 requires five market days")
    return -cross_sectional_rank(window_covariance(close_rank, volume_rank))


def alpha44(high: ArrayLike, volume_rank: ArrayLike) -> np.ndarray:
    """输入 [5 个交易日 x 股票] 的最高价数值和成交量排名；返回 [股票] 分数。"""
    high = np.asarray(high, dtype=float)
    if len(high) != 5:
        raise ValueError("Alpha44 requires five market days")
    return -window_correlation(high, volume_rank)


def compute_factors(
    panel: ResearchPanel, membership: pd.DataFrame, signal_dates: DateValues,
    price_mode: str = "adjusted", rank_policy: str = "current_cohort",
) -> pd.DataFrame:
    """为每个信号日的成员计算分数，返回每行一个（trade_date, ticker）的长表。

    panel 提供含历史窗口的行情；membership 至少有 trade_date、ticker 两列。
    signal_dates 是研究日期序列，不含前置缓冲日期。
    price_mode 默认用 adjusted 复权行情，也可指定 raw。current_cohort 用信号日名单给历史窗口排名；
    historical_members 用窗口内各历史日自己的成员名单。
    """
    if price_mode not in ("raw", "adjusted") or rank_policy not in ("current_cohort", "historical_members"):
        raise ValueError("Unknown factor specification")
    suffix = "" if price_mode == "raw" else "_adj"
    arrays = {name: panel.matrix(name + suffix).to_numpy() for name in ["open", "close", "high"]}
    arrays["vol"] = panel.matrix("vol").to_numpy()
    close = panel.matrix("close_adj")
    observations = cast(pd.DataFrame, close.rolling(21, min_periods=21).count())
    momentum = calc_momentum(close, 20).where(observations.eq(21)).to_numpy(dtype=float)
    members = {d: g for d, g in membership.groupby("trade_date", sort=True)}
    ticker_pos = {t: i for i, t in enumerate(panel.tickers)}
    ranks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if rank_policy == "historical_members":
        for field in ["open", "close", "high"]:
            rp, rv = np.full_like(arrays[field], np.nan), np.full_like(arrays[field], np.nan)
            for row, day in enumerate(panel.calendar):
                if day not in members:
                    continue
                idx = [ticker_pos[t] for t in members[day].ticker]
                a, b = _pair_ranks(arrays[field][row:row+1, idx], arrays["vol"][row:row+1, idx])
                rp[row, idx], rv[row, idx] = a[0], b[0]
            ranks[field] = rp, rv
    output = []
    ordered_dates = as_date_index(signal_dates).sort_values()
    signal_days: list[pd.Timestamp] = ordered_dates.to_list()
    for day in signal_days:
        if day not in members or day not in panel.calendar:
            raise ValueError("Signal date lacks membership or independent calendar")
        row = cast(int, panel.calendar.get_loc(day))
        if row < 9:
            raise ValueError("Provide at least nine buffer days")
        frame = members[day].copy()
        idx = [ticker_pos[t] for t in frame.ticker]
        pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for field in ["open", "close", "high"]:
            if rank_policy == "current_cohort":
                pairs[field] = _pair_ranks(arrays[field][row-9:row+1, idx], arrays["vol"][row-9:row+1, idx])
            else:
                price_rank, volume_rank = ranks[field]
                pairs[field] = price_rank[row-9:row+1, idx], volume_rank[row-9:row+1, idx]
        frame["Alpha3"] = alpha3(*pairs["open"])
        close_rank, volume_rank = pairs["close"]
        frame["Alpha13"] = alpha13(close_rank[-5:], volume_rank[-5:])
        # high是直接数值，而不是排名。成交量volume是排名
        frame["Alpha44"] = alpha44(arrays["high"][row-4:row+1, idx], pairs["high"][1][-5:])
        frame["MOM20"] = momentum[row, idx]
        for field, factor, window in [("open", "Alpha3", 10), ("close", "Alpha13", 5), ("high", "Alpha44", 5)]:
            p, v = (a[-window:] for a in pairs[field])
            complete = (np.isfinite(p) & np.isfinite(v)).all(axis=0)
            frame[factor + "_reason"] = np.where(frame[factor].notna(), "available",
                np.where(complete, "zero_variance_or_insufficient_cross_section", "input_or_rank_window_incomplete"))
        frame["MOM20_reason"] = np.where(frame.MOM20.notna(), "available", "price_window_incomplete")
        output.append(frame)
    return pd.concat(output, ignore_index=True).sort_values(["trade_date", "ticker"]).reset_index(drop=True)
