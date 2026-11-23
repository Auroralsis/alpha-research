"""量价因子公式，以及横截面排名和滚动统计。"""

from typing import cast

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import ArrayLike
from scipy.stats import rankdata

from src.data.research_panel import DateValues, ResearchPanel, as_date_index


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
    constant = (x == x[0]).all(axis=0) | (y == y[0]).all(axis=0)
    cov = (xc * yc).sum(axis=0) / (len(x) - 1)
    return np.where(complete, np.where(constant, 0, cov), np.nan)


def window_correlation(x: ArrayLike, y: ArrayLike) -> np.ndarray:
    """x、y 都是 [窗口交易日 x 股票] 矩阵，沿日期计算；返回 [股票] 相关系数向量。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    covariance = window_covariance(x, y)
    xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
    denominator = np.sqrt((xc * xc).sum(axis=0) * (yc * yc).sum(axis=0))
    # np.full是根据输入shape和填充值填充一个数组
    out = np.full(covariance.shape, np.nan)
    # 这里为了防止denominator过小(比如某一个股票的某一组数据标准差为0时)而设置阈值
    variable = (x != x[0]).any(axis=0) & (y != y[0]).any(axis=0)
    np.divide(covariance * (len(x) - 1), denominator, out=out, where=variable & (denominator > 1e-24))
    if len(x) == 2:
        return np.sign(out)
    # 将输出限制在[-1,1]之间，超出的结果赋值为-1或1
    return np.clip(out, -1, 1)


FACTORS = ("Alpha3", "Alpha13", "Alpha44", "MOM20")
SINGLE_FACTOR_SCORES = ("Alpha13", "Alpha3", "Alpha44")


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
    momentum = close.pct_change(20, fill_method=None).where(observations.eq(21)).to_numpy(dtype=float)
    members = {d: g for d, g in membership.groupby("trade_date", sort=True)}
    ticker_pos = {t: i for i, t in enumerate(panel.tickers)}
    ranks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if rank_policy == "historical_members":
        # 先遮掉历史日的非成分股，再在每天自己的股票池中排名。
        active = np.zeros_like(arrays["vol"], dtype=bool)
        for day, cohort in members.items():
            if day in panel.calendar:
                active[panel.calendar.get_loc(day), [ticker_pos[t] for t in cohort.ticker]] = True
        volume = np.where(active, arrays["vol"], np.nan)
        ranks = {field: _pair_ranks(arrays[field], volume) for field in ("open", "close", "high")}
    output = []
    for day in as_date_index(signal_dates).sort_values():
        row = cast(int, panel.calendar.get_loc(day))
        if row < 9:
            raise ValueError("Provide at least nine buffer days")
        frame = members[day].copy()
        idx = [ticker_pos[t] for t in frame.ticker]
        reasons = {}
        for field, name, days, formula in (("open", "Alpha3", 10, alpha3), ("close", "Alpha13", 5, alpha13),
                                           ("high", "Alpha44", 5, alpha44)):
            span = slice(row - days + 1, row + 1)
            price = arrays[field][span, idx]
            if rank_policy == "current_cohort":
                price_rank, volume_rank = _pair_ranks(price, arrays["vol"][span, idx])
            else:
                price_rank, volume_rank = (matrix[span, idx] for matrix in ranks[field])
            # Alpha44 用最高价原值；三项的窗口覆盖诊断都按配对排名判断。
            frame[name] = formula(price if name == "Alpha44" else price_rank, volume_rank)
            complete = (np.isfinite(price_rank) & np.isfinite(volume_rank)).all(axis=0)
            reasons[name + "_reason"] = np.where(frame[name].notna(), "available",
                np.where(complete, "zero_variance_or_insufficient_cross_section", "input_or_rank_window_incomplete"))
        frame = frame.assign(MOM20=momentum[row, idx], **reasons)
        frame["MOM20_reason"] = np.where(frame.MOM20.notna(), "available", "price_window_incomplete")
        output.append(frame)
    return pd.concat(output, ignore_index=True).sort_values(["trade_date", "ticker"]).reset_index(drop=True)


def divide(x, y):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = np.divide(x, y)
    return np.where(np.isfinite(result), result, np.nan)


def less(x, y):
    return np.where(np.isfinite(x) & np.isfinite(y), np.less(x, y), np.nan)


def at_most(x, y):
    return np.where(np.isfinite(x) & np.isfinite(y), np.less_equal(x, y), np.nan)


def choose(condition, yes, no):
    """条件缺失时保留缺失，其余取所选分支。"""
    return np.where(np.isfinite(condition), np.where(condition, yes, no), np.nan)


def delay(x, days):
    n = int(days)
    out = np.full_like(x, np.nan)
    out[n:] = x[:-n]
    return out


def delta(x, days):
    return x - delay(x, days)


def signedpower(x, exponent):
    return np.sign(x) * np.power(np.abs(x), exponent)


def scale(x):
    return divide(x, np.nansum(np.abs(x), axis=1, keepdims=True))


def indneutralize(x, industry):
    """减去当日同一行业有效因子值的均值，行业编码 -1 表示缺失。"""
    out = np.full_like(x, np.nan)
    for code in np.unique(industry):
        if code < 0:
            continue
        mask = (industry == code) & np.isfinite(x)
        values = np.where(mask, x, 0)
        mean = divide(values.sum(axis=1), mask.sum(axis=1))
        np.subtract(x, mean[:, None], out=out, where=mask)
    return out


def rolling(x, days):
    return pd.DataFrame(x).rolling(int(days), min_periods=int(days))


def ts_sum(x, days):
    return rolling(x, days).sum().to_numpy()


def ts_min(x, days):
    return rolling(x, days).min().to_numpy()


def ts_max(x, days):
    return rolling(x, days).max().to_numpy()


def ts_rank(x, days):
    return rolling(x, days).rank(method="average", pct=True).to_numpy()


def stddev(x, days):
    return rolling(x, days).std(ddof=1).to_numpy()


def _rolling_pair(function, x, y, days):
    n = int(days)
    out = np.full_like(x, np.nan)
    for end in range(n, len(x) + 1):
        out[end - 1] = function(x[end-n:end], y[end-n:end])
    return out


def correlation(x, y, days):
    return _rolling_pair(window_correlation, x, y, days)


def covariance(x, y, days):
    return _rolling_pair(window_covariance, x, y, days)


def _rolling_reduce(x, days, reduce):
    n = int(days)
    out = np.full_like(x, np.nan)
    if len(x) >= n:
        windows = sliding_window_view(x, n, axis=0)
        values = reduce(windows)
        out[n-1:] = np.where(np.isfinite(windows).all(axis=-1), values, np.nan)
    return out


def ts_argmax(x, days):
    return _rolling_reduce(x, days, lambda w: w.argmax(axis=-1) + 1)


def ts_argmin(x, days):
    return _rolling_reduce(x, days, lambda w: w.argmin(axis=-1) + 1)


def product(x, days):
    return _rolling_reduce(x, days, lambda w: w.prod(axis=-1))


def decay_linear(x, days):
    weights = np.arange(1, int(days) + 1, dtype=float)
    weights /= weights.sum()
    return _rolling_reduce(x, days, lambda w: np.einsum("ijk,k->ij", w, weights))
