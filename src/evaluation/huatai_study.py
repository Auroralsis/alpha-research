import numpy as np
import pandas as pd

from .factor_metrics import calc_ic_rank
from .quantile_analysis import calc_quantile_returns
from scipy.stats import norm
from src.features.huatai import quantile_groups, cross_sectional_rank


def hac_mean(values, lag=20, min_obs=100, min_coverage=.8):
    """计算时间序列均值及其 HAC 稳健统计检验结果。

    输入参数：
        values: 一维数组或序列，通常为按市场交易日升序排列的每日 RankIC。
            每个元素对应一个市场交易日，缺失观测使用 NaN 表示。
            输入应已对齐完整市场交易日历，不能直接删除缺失日期。
        lag: int，HAC 估计考虑的最大滞后交易日数，默认20。
        min_obs: int，进行统计推断所需的最少有效观测数，默认100。
        min_coverage: float，最小有效数据覆盖率，默认0.8。
    """
    values = np.asarray(values, float)
    valid = np.isfinite(values)
    n = int(valid.sum())
    mean = values[valid].mean() if n else np.nan
    result = dict(mean=mean, n_days=n, calendar_days=len(values), hac_lag=lag,
                  hac_se=np.nan, t_hac=np.nan, p_value=np.nan, ci_low=np.nan, ci_high=np.nan,
                  inference_status="insufficient_coverage")
    if n < min_obs or n / max(len(values), 1) < min_coverage or len(values) <= lag:
        return result
    score = np.where(valid, values - mean, 0.)
    # 初始meat就是偏差平方，用于计算方差
    meat = np.dot(score, score)
    # 这里加入时间协方差修正，因为这么多个交易日的IC值是有相关性的，不能够直接计算
    for k in range(1, min(lag, len(values) - 1) + 1):
        meat += 2 * (1 - k / (lag + 1)) * np.dot(score[k:], score[:-k])
    variance = meat / n**2 * n / (n - 1)
    if not np.isfinite(variance) or variance <= 1e-24:
        result["inference_status"] = "degenerate_variance"
        return result
    se = np.sqrt(variance)
    result.update(hac_se=se, t_hac=mean / se, p_value=2 * norm.sf(abs(mean / se)),
                  ci_low=mean - norm.ppf(.975) * se, ci_high=mean + norm.ppf(.975) * se,
                  inference_status="ok")
    return result


def holm_adjust(p_values):
    p = np.asarray(p_values, float)
    if ((p[np.isfinite(p)] < 0) | (p[np.isfinite(p)] > 1)).any():
        raise ValueError("Invalid p value")
    # Unestimable family members remain counted, conservatively, as p=1.
    order = np.argsort(np.where(np.isfinite(p), p, 1), kind="stable")
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        np.where(np.isfinite(p[order]), p[order], 1) * np.arange(len(p), 0, -1)))
    return np.where(np.isfinite(p), adjusted, np.nan)


def block_intervals(values, block=60, repeats=500, seed=20261111):
    values = np.asarray(values, float)
    if len(values) < block or np.isfinite(values).sum() < 100:
        return dict(block_mean_low=np.nan, block_mean_high=np.nan,
                    positive_ratio_low=np.nan, positive_ratio_high=np.nan)
    rng = np.random.default_rng(seed)
    means, positives = [], []
    for _ in range(repeats):
        starts = rng.integers(0, len(values) - block + 1, int(np.ceil(len(values) / block)))
        sample = np.concatenate([values[s:s+block] for s in starts])[:len(values)]
        sample = sample[np.isfinite(sample)]
        if len(sample):
            means.append(sample.mean())
            positives.append(np.mean(sample > 0))
    return dict(block_mean_low=np.quantile(means, .025), block_mean_high=np.quantile(means, .975),
                positive_ratio_low=np.quantile(positives, .025), positive_ratio_high=np.quantile(positives, .975))


def evaluate_panel(frame, factors, horizons=(5, 20), min_pairs=100):
    daily, quantiles, correlations, regressions = [], [], [], []
    for day, group in frame.groupby("trade_date", sort=True):
        for factor in factors:
            # 这里取出每个特定trade_date对应的一组因子的数据
            x = group[factor]
            # Assignment sees factors only, never labels or label availability.
            q = pd.Series(quantile_groups(x), index=group.index)
            full = q.nunique() == 5
            for h in horizons:
                y = group[f"future_ret_{h}"]
                # mask确保只使用有限数值的因子计算，排除NaN和无穷大的数值
                mask = np.isfinite(x) & np.isfinite(y)
                n = int(mask.sum())
                usable = n >= min_pairs and x[mask].nunique() > 1 and y[mask].nunique() > 1
                ic = calc_ic_rank(x[mask], y[mask]) if usable else np.nan
                daily.append(dict(trade_date=day, factor=factor, horizon=h, rank_ic=ic,
                    n_eligible=len(group), n_factor=int(x.notna().sum()), n_label=int(y.notna().sum()),
                    n_pairs=n, factor_coverage=x.notna().mean(), label_coverage=y.notna().mean(),
                    pair_coverage=n / len(group), valid_ic=usable,
                    label_in_calendar=bool(group[f"label_end_{h}"].notna().all())))
                grouped = pd.DataFrame({"q": q, "y": y})
                means = calc_quantile_returns(grouped, "q", "y")
                for k in range(1, 6):
                    members = q.eq(k)
                    quantiles.append(dict(trade_date=day, factor=factor, horizon=h, quantile=k,
                        mean_return=means.get(k, np.nan), n_assigned=int(members.sum()),
                        n_observed=int((members & y.notna()).sum()), full_five_groups=full))
        if "MOM20" not in group:
            continue
        for factor in [f for f in factors if f in ("Alpha3", "Alpha13", "Alpha44")]:
            mask = group[[factor, "MOM20"]].notna().all(axis=1)
            data = group.loc[mask]
            if len(data) < min_pairs or data[factor].nunique() < 2 or data.MOM20.nunique() < 2:
                continue
            correlations.append(dict(trade_date=day, factor=factor,
                correlation_mom20=calc_ic_rank(data[factor], data.MOM20), n=len(data)))
            a, m = cross_sectional_rank(data[factor]), cross_sectional_rank(data.MOM20)
            design = np.column_stack([np.ones(len(data)), m])
            residual = a - design @ np.linalg.lstsq(design, a, rcond=None)[0]
            z = data[["MOM20", factor]].to_numpy()
            z = (z - z.mean(axis=0)) / z.std(axis=0, ddof=1)
            for h in horizons:
                y = data[f"future_ret_{h}"].to_numpy()
                ok = np.isfinite(y)
                enough = ok.sum() >= min_pairs
                r = pd.Series(residual[ok])
                target = pd.Series(y[ok])
                r_ic = calc_ic_rank(r, target) if enough and r.std() > 1e-12 and target.nunique() > 1 else np.nan
                original = calc_ic_rank(data.loc[ok, factor], data.loc[ok, f"future_ret_{h}"]) if enough else np.nan
                beta = np.nan
                if enough:
                    model = np.column_stack([np.ones(ok.sum()), z[ok]])
                    if np.linalg.matrix_rank(model) == 3:
                        beta = np.linalg.lstsq(model, y[ok], rcond=None)[0][-1]
                regressions.append(dict(trade_date=day, factor=factor, horizon=h, residual_ic=r_ic,
                    matched_raw_ic=original, delta_ic=r_ic-original, alpha_beta=beta, n_pairs=int(ok.sum())))
    return tuple(pd.DataFrame(rows) for rows in (daily, quantiles, correlations, regressions))


def summarize_ic(daily, calendar, lag=20, bootstrap=False):
    rows = []
    # 根据daily数据的factor和horizon分组。这里daily数据不包括ticker，因为
    # 每一天的rank_ic是所有股票当天的因子和未来收益率的rank_ic。
    for (factor, h), group in daily.groupby(["factor", "horizon"], sort=True):
        # 将每一组的交易日期作为索引后，再去reindex(calender)
        series = group.set_index("trade_date").rank_ic.reindex(calendar)
        stats = hac_mean(series.to_numpy(), lag=lag)
        sd = series.std(ddof=1)
        stats.update(factor=factor, horizon=h, rank_ic_std=sd,
            icir=series.mean()/sd if sd > 1e-12 else np.nan,
            positive_ratio=(series.dropna() > 0).mean(),
            mean_n_pairs=group.n_pairs.mean(), mean_factor_coverage=group.factor_coverage.mean(),
            mean_label_coverage=group.label_coverage.mean(), mean_pair_coverage=group.pair_coverage.mean())
        if bootstrap:
            stats.update(block_intervals(series.to_numpy()))
        rows.append(stats)
    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result.p_value)
    return result


def summarize_quantiles(quantiles):
    # Equal date weights, never weighted by the number of observed stocks.
    means = quantiles.groupby(["factor", "horizon", "quantile"]).agg(
        mean_return=("mean_return", "mean"), n_dates=("mean_return", "count"),
        mean_assigned=("n_assigned", "mean"), mean_observed=("n_observed", "mean")).reset_index()
    rows = []
    for (factor, h), g in quantiles.groupby(["factor", "horizon"]):
        p = g.pivot(index="trade_date", columns="quantile", values="mean_return")
        full = g.groupby("trade_date").full_five_groups.first() & p.notna().all(axis=1)
        p = p.loc[full]
        av = p.mean()
        rows.append(dict(factor=factor, horizon=h, n_full_dates=len(p), q5_minus_q1=(p[5]-p[1]).mean(),
            mean_group_monotonicity=av.corr(pd.Series(range(1,6), index=range(1,6)), method="spearman"),
            adjacent_12=av.get(2,np.nan)-av.get(1,np.nan), adjacent_23=av.get(3,np.nan)-av.get(2,np.nan),
            adjacent_34=av.get(4,np.nan)-av.get(3,np.nan), adjacent_45=av.get(5,np.nan)-av.get(4,np.nan)))
    return means, pd.DataFrame(rows)
