"""逐日加权横截面回归、RankIC 及跨日统计。"""

from collections.abc import Sequence
from typing import cast

import numpy as np
import pandas as pd

from src.evaluation.factor_metrics import calc_ic_rank


def weighted_slope(
    design: np.ndarray, returns: np.ndarray, weights: np.ndarray,
) -> tuple[float, float]:
    """design 是 [股票 x 解释变量] 矩阵，第 0 列是目标因子。

    returns、weights 都是 [股票] 向量，顺序必须与 design 的行一致；
    weights 是正的回归权重。返回第 0 列的系数和当日 WLS 标准误。
    """
    if len(returns) <= design.shape[1]:
        return np.nan, np.nan
    scale = np.sqrt(weights / weights.mean())
    weighted_design = design * scale[:, None]
    weighted_returns = returns * scale
    coefficients, _, rank, _ = np.linalg.lstsq(weighted_design, weighted_returns, rcond=None)
    if rank != design.shape[1]:
        return np.nan, np.nan
    residual = weighted_returns - weighted_design @ coefficients
    variance = (residual @ residual) / (len(returns) - rank)
    covariance = np.linalg.solve(weighted_design.T @ weighted_design, np.eye(design.shape[1])[:, 0])
    return coefficients[0], np.sqrt(variance * covariance[0])


def daily_factor_tests(
    frame: pd.DataFrame, factors: Sequence[str], horizons: Sequence[int], min_pairs: int = 100,
) -> pd.DataFrame:
    """输入股票日长表，每行一个（trade_date, ticker）。

    必需列包括 trade_date、next_session_observed、industry、total_mv、circ_mv，
    每个因子的 _standardized / _neutral 分数列，以及各期限的 future_ret_{h}。
    返回每行一个（信号日, 因子, 处理方式, 期限）的 Rank IC 和 WLS 结果。
    """
    rows = []
    for day, group in frame.groupby("trade_date", sort=True):
        research = group.loc[group.next_session_observed]
        controls = research.loc[research.industry.notna() & research.total_mv.gt(0) & research.circ_mv.gt(0)
                                & np.isfinite(research.total_mv) & np.isfinite(research.circ_mv)]
        for factor in factors:
            for treatment, column in [("raw", f"{factor}_standardized"), ("neutral", f"{factor}_neutral")]:
                for horizon in horizons:
                    label = f"future_ret_{horizon}"
                    pairs = research[[column, label]].dropna()
                    regression = controls[[column, label, "industry", "total_mv", "circ_mv"]].dropna()
                    rank_ic = calc_ic_rank(pairs[column], pairs[label], min_pairs)
                    slope, se = np.nan, np.nan
                    if len(regression) >= min_pairs:
                        design = np.column_stack([regression[column], np.log(regression.total_mv),
                                                  pd.get_dummies(regression.industry, dtype=float)])
                        slope, se = weighted_slope(design, regression[label].to_numpy(),
                                                   np.sqrt(regression.circ_mv.to_numpy()))
                    rows.append(dict(trade_date=day, factor=factor, treatment=treatment, horizon=horizon,
                                     n_members=len(group), n_eligible=len(research), n_ic=len(pairs),
                                     n_wls=len(regression), rank_ic=rank_ic, factor_return=slope, daily_se=se,
                                     daily_t=slope / se if se > 1e-12 else np.nan))
    return pd.DataFrame(rows)


def summarize_factor_tests(daily: pd.DataFrame) -> pd.DataFrame:
    """输入已按预测期限截尾的逐日检验长表；每个（因子, 处理方式, 期限）汇总一行。"""
    rows = []
    for key, group in daily.groupby(["factor", "treatment", "horizon"]):
        factor, treatment, horizon = cast(tuple[str, str, int], key)
        ic, slope, t_stats = group.rank_ic, group.factor_return, group.daily_t.dropna()
        sd = ic.std(ddof=1)
        rows.append(dict(factor=factor, treatment=treatment, horizon=horizon,
                         n_ic_days=ic.count(), n_wls_days=slope.count(), n_t_days=len(t_stats),
                         mean_n_ic=group.n_ic.mean(), mean_n_wls=group.n_wls.mean(),
                         mean_abs_t=t_stats.abs().mean(), share_abs_t_gt_2=t_stats.abs().gt(2).mean(),
                         mean_t=t_stats.mean(), mean_factor_return=slope.mean(), mean_rank_ic=ic.mean(),
                         rank_ic_std=sd, icir=ic.mean() / sd if sd > 1e-12 else np.nan,
                         share_ic_positive=ic.dropna().gt(0).mean()))
    return pd.DataFrame(rows)
