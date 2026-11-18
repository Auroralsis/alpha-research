import numpy as np
import pandas as pd

from .momentum import calc_momentum
from scipy.stats import rankdata

def cross_sectional_rank(values):
    """横截面排名，输入是[trade_date][ticker]的矩阵"""
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values), values, np.nan)
    n = np.isfinite(values).sum(axis=-1, keepdims=True)
    # 这里计算排名时直接忽略NaN的股票
    result = rankdata(values, method="average", axis=-1, nan_policy="omit") / np.maximum(n, 1)
    # 只对至少有两个股票的横截面给出排名结果
    return np.where(n >= 2, result, np.nan)


def window_covariance(x, y):
    """计算协方差，输入x和y都是[trade_date][ticker]的矩阵，与上面函数一致"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape or x.ndim != 2 or len(x) < 2:
        raise ValueError("Expected matching windows with at least two rows")
    complete = (np.isfinite(x) & np.isfinite(y)).all(axis=0)
    xc, yc = x - x.mean(axis=0), y - y.mean(axis=0)
    return np.where(complete, (xc * yc).sum(axis=0) / (len(x) - 1), np.nan)


def window_correlation(x, y):
    """计算相关系数，输入与上面的函数一致"""
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


def quantile_groups(values, n_groups=5):
    ranks = cross_sectional_rank(values)
    return np.ceil(ranks * n_groups).clip(1, n_groups)


FACTORS = ("Alpha3", "Alpha13", "Alpha44", "MOM20")


def _pair_ranks(price, volume):
    """计算两个量的排名结果"""
    # 只有当price和volume两个量均不是NaN时才会参与计算横截面
    valid = np.isfinite(price) & np.isfinite(volume)
    return (cross_sectional_rank(np.where(valid, price, np.nan)),
            cross_sectional_rank(np.where(valid, volume, np.nan)))


def alpha3(open_rank, volume_rank):
    """输入行数为10的开盘价和成交量的归一化排名，这里的输入数据结构依然为[trade_date][ticker]"""
    if len(open_rank) != 10:
        raise ValueError("Alpha3 requires ten market days")
    return -window_correlation(open_rank, volume_rank)


def alpha13(close_rank, volume_rank):
    if len(close_rank) != 5:
        raise ValueError("Alpha13 requires five market days")
    return -cross_sectional_rank(window_covariance(close_rank, volume_rank))


def alpha44(high, volume_rank):
    if len(high) != 5:
        raise ValueError("Alpha44 requires five market days")
    return -window_correlation(high, volume_rank)


def compute_factors(panel, membership, signal_dates, price_mode="raw", rank_policy="current_cohort"):
    """Compute from intact market-grid prices, then emit the date's members.

    current_cohort reranks each historical row against M_t, not the full-history
    union. historical_members ranks against M_u and preserves absent ranks.
    """
    if price_mode not in ("raw", "adjusted") or rank_policy not in ("current_cohort", "historical_members"):
        raise ValueError("Unknown factor specification")
    suffix = "" if price_mode == "raw" else "_adj"
    arrays = {name: panel.matrix(name + suffix).to_numpy() for name in ["open", "close", "high"]}
    arrays["vol"] = panel.matrix("vol").to_numpy()
    close = panel.matrix("close_adj")
    # 这里要求
    mom = calc_momentum(close, 20).where(close.rolling(21, min_periods=21).count().eq(21))
    members = {d: g for d, g in membership.groupby("trade_date", sort=True)}
    ticker_pos = {t: i for i, t in enumerate(panel.tickers)}
    ranks = {}
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
    for day in pd.DatetimeIndex(signal_dates).sort_values():
        if day not in members or day not in panel.calendar:
            raise ValueError("Signal date lacks membership or independent calendar")
        row = panel.calendar.get_loc(day)
        if row < 9:
            raise ValueError("Provide at least nine buffer days")
        frame = members[day].copy()
        idx = [ticker_pos[t] for t in frame.ticker]
        pairs = {}
        for field in ["open", "close", "high"]:
            if rank_policy == "current_cohort":
                pairs[field] = _pair_ranks(arrays[field][row-9:row+1, idx], arrays["vol"][row-9:row+1, idx])
            else:
                pairs[field] = tuple(a[row-9:row+1, idx] for a in ranks[field])
        frame["Alpha3"] = alpha3(*pairs["open"])
        frame["Alpha13"] = alpha13(*(a[-5:] for a in pairs["close"]))
        # high是直接数值，而不是排名。成交量volume是排名
        frame["Alpha44"] = alpha44(arrays["high"][row-4:row+1, idx], pairs["high"][1][-5:])
        frame["MOM20"] = mom.iloc[row, idx].to_numpy()
        frame["eligible_t"] = True
        for field, factor, window in [("open", "Alpha3", 10), ("close", "Alpha13", 5), ("high", "Alpha44", 5)]:
            p, v = (a[-window:] for a in pairs[field])
            complete = (np.isfinite(p) & np.isfinite(v)).all(axis=0)
            frame[factor + "_reason"] = np.where(frame[factor].notna(), "available",
                np.where(complete, "zero_variance_or_insufficient_cross_section", "input_or_rank_window_incomplete"))
        frame["MOM20_reason"] = np.where(frame.MOM20.notna(), "available", "price_window_incomplete")
        output.append(frame)
    return pd.concat(output, ignore_index=True).sort_values(["trade_date", "ticker"]).reset_index(drop=True)
