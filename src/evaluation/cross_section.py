"""Daily weighted factor regressions, rank evidence and across-date summaries."""

import numpy as np
import pandas as pd

from src.evaluation.factor_metrics import calc_ic_rank
from src.evaluation.inference import hac_mean, holm_adjust


def weighted_slope(design, returns, weights):
    """Estimate the first coefficient and its conventional within-day WLS SE."""
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
    covariance = np.linalg.solve(weighted_design.T @ weighted_design, np.eye(rank)[:, 0])
    return coefficients[0], np.sqrt(variance * covariance[0])


def daily_factor_tests(frame, factors, horizons, min_pairs=100):
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
                    rank_ic = np.nan
                    if len(pairs) >= min_pairs and pairs[column].nunique() > 1 and pairs[label].nunique() > 1:
                        rank_ic = calc_ic_rank(pairs[column], pairs[label])
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


def summarize_factor_tests(daily, dates):
    rows = []
    for (factor, treatment, horizon), group in daily.groupby(["factor", "treatment", "horizon"]):
        group = group.set_index("trade_date").reindex(dates)
        ic, slope, t_stats = group.rank_ic, group.factor_return, group.daily_t.dropna()
        sd = ic.std(ddof=1)
        ic_hac, slope_hac = hac_mean(ic), hac_mean(slope)
        rows.append(dict(factor=factor, treatment=treatment, horizon=horizon,
                         n_ic_days=ic.count(), n_wls_days=slope.count(), n_t_days=len(t_stats),
                         mean_n_ic=group.n_ic.mean(), mean_n_wls=group.n_wls.mean(),
                         mean_abs_t=t_stats.abs().mean(), share_abs_t_gt_2=t_stats.abs().gt(2).mean(),
                         mean_t=t_stats.mean(), mean_factor_return=slope.mean(), mean_rank_ic=ic.mean(),
                         rank_ic_std=sd, icir=ic.mean() / sd if sd > 1e-12 else np.nan,
                         share_ic_positive=ic.dropna().gt(0).mean(),
                         ic_hac_se=ic_hac["hac_se"], ic_hac_p=ic_hac["p_value"],
                         return_hac_se=slope_hac["hac_se"], return_hac_p=slope_hac["p_value"]))
    summary = pd.DataFrame(rows)
    summary["ic_holm_p"] = holm_adjust(summary.ic_hac_p)
    summary["return_holm_p"] = holm_adjust(summary.return_hac_p)
    return summary
