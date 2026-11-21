"""评价股票日分数与事后收益的关系。"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from src.data.download import write_parquet_batches
from src.data.research_panel import CSI500_INPUTS, stock_day_index
from src.factors.neutralization import attach_controls, neutral_scores


def daily_ic(prediction: pd.Series, label: pd.Series) -> tuple[pd.Series, pd.Series]:
    """对 (datetime, instrument) 索引的分数和原始收益调用 Qlib。"""
    from qlib.contrib.eva.alpha import calc_ic

    return calc_ic(prediction, label)


def daily_score_metrics(scores: pd.Series, target: pd.Series) -> pd.DataFrame:
    """按股票日键计算 Qlib 日相关系数与有效配对数。"""
    ic, rank_ic = daily_ic(scores, target)
    pairs = target.where(scores.notna()).groupby(level="datetime").count()
    return pd.DataFrame(dict(ic=ic, rank_ic=rank_ic, n_pairs=pairs)).rename_axis("trade_date").reset_index()


def ic_statistics(values: pd.Series) -> tuple[float, float, float]:
    """按 Qlib 的 IC 记录口径返回日均值、样本标准差和 ICIR。"""
    mean, std = values.mean(), values.std()
    with np.errstate(divide="ignore", invalid="ignore"):
        return mean, std, mean / std


def summarize_ic(daily: pd.DataFrame) -> dict:
    """汇总已按统计区间日历对齐的逐日 IC 和 Rank IC。"""
    rank = daily.rank_ic.dropna()
    mean_ic, _, icir = ic_statistics(daily.ic)
    mean_rank_ic, rank_std, rank_icir = ic_statistics(rank)
    return dict(mean_ic=mean_ic, icir=icir, mean_rank_ic=mean_rank_ic, rank_ic_std=rank_std,
                rank_icir=rank_icir, positive_rate=rank.gt(0).mean(), n_days=len(rank), calendar_days=len(daily),
                n_pairs=int(daily.n_pairs.sum()))


def common_scores(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.DataFrame:
    """输入逐股打分表，只保留指定方法都有有限分数的股票日。"""
    return frame.loc[np.isfinite(frame[list(names)]).all(axis=1)]


def quintile_assignments(scores: pd.DataFrame, score: str) -> pd.DataFrame:
    """按指定分数分成 Q1 至 Q5，返回 trade_date、ticker 和 group 三列。

    每天从低分排到高分，同分时按股票代码排序。
    """
    ranked = scores[["trade_date", "ticker", score]].sort_values(["trade_date", score, "ticker"]).copy()
    groups = ranked.groupby("trade_date", sort=False)[score]
    sizes = groups.transform("size").to_numpy()
    positions = groups.cumcount().to_numpy()
    # 与 np.array_split 一致，多出来的股票分给靠前的组。
    width, remainder = sizes // 5, sizes % 5
    wide_end = (width + 1) * remainder
    ranked["group"] = np.where(positions < wide_end, positions // (width + 1),
                                remainder + (positions - wide_end) // np.maximum(width, 1)) + 1
    return ranked[["trade_date", "ticker", "group"]]


def prediction_diagnostics(
    predictions: pd.DataFrame, config: dict, dates: pd.DatetimeIndex, out: Path, *,
    strategies_by_split: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    """按方法计算 Qlib IC 和 Rank IC，在共同股票池上汇总五组收益。"""
    daily_metrics, daily_groups = [], []
    for split, frame in predictions.groupby("split", sort=False):
        names = strategies_by_split[split]
        common = common_scores(frame, names)
        labels = frame[["trade_date", "ticker", "label"]].copy()
        labels["label"] = labels.label.where(frame.evaluable & np.isfinite(labels.label))
        index = stock_day_index(frame)
        target = labels.label.set_axis(index)
        for name in names:
            groups = quintile_assignments(common, name).merge(labels, on=["trade_date", "ticker"], validate="one_to_one")
            daily_groups.append(groups.groupby(["trade_date", "group"])["label"].agg(
                n_assigned="size", n_labeled="count", mean_return="mean").reset_index().assign(split=split, strategy=name))
            scores = frame[name].where(np.isfinite(frame[name]))
            metrics = daily_score_metrics(scores.set_axis(index), target)
            daily_metrics.append(metrics.assign(split=split, strategy=name))

    ic = pd.concat(daily_metrics, ignore_index=True)
    q = pd.concat(daily_groups, ignore_index=True).sort_values(["split", "strategy", "group", "trade_date"])
    ic_summary, group_summary = summarize_predictions(ic, q, config, dates)
    for name, table in (("daily_rank_ic", ic), ("rank_ic_summary", ic_summary),
                        ("quintile_daily", q), ("quintile_summary", group_summary)):
        table.to_csv(out / f"{name}.csv", index=False)
    return ic


def summarize_predictions(ic: pd.DataFrame, q: pd.DataFrame, config: dict, dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """按 Qlib 口径汇总两种日 IC，并统计给定日期区间的五组收益。"""
    ic_rows, group_rows = [], []
    for (split, name), group in ic.groupby(["split", "strategy"]):
        begin, end = config["segments"][split]
        calendar = dates[dates.to_series().between(begin, end).to_numpy()]
        by_date = group.set_index("trade_date")
        for year in ["all", *sorted(set(calendar.year))]:
            segment = calendar if year == "all" else calendar[calendar.year == year]
            ic_rows.append(dict(split=split, strategy=name, year=year, **summarize_ic(by_date.reindex(segment))))
            partition = q.loc[q.split.eq(split) & q.strategy.eq(name) & q.trade_date.isin(segment)]
            wide = partition.pivot(index="trade_date", columns="group", values="mean_return").reindex(columns=range(1, 6))
            complete = wide.dropna()
            mean = wide.mean()
            coverage = partition.groupby("group")[["n_assigned", "n_labeled"]].sum()
            group_rows.append(dict(split=split, strategy=name, year=year,
                                   **{f"Q{i}": mean[i] for i in range(1, 6)},
                                   spread=(wide[5] - wide[1]).mean(), monotonic_means=bool(mean.is_monotonic_increasing),
                                   n_complete_days=len(complete),
                                   daily_monotonic_rate=complete.diff(axis=1).iloc[:, 1:].ge(0).all(axis=1).mean(),
                                   **{f"Q{i}_coverage": coverage.n_labeled.get(i, 0) / max(coverage.n_assigned.get(i, 0), 1)
                                      for i in range(1, 6)}))
    return pd.DataFrame(ic_rows), pd.DataFrame(group_rows)


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


def evaluate_regression(
    scores: pd.DataFrame, controls: pd.DataFrame, strategies: tuple[str, ...], min_pairs: int, out: Path,
) -> None:
    """对测试期有效收益回归：解释变量为各方法的分数、对数总市值和行业，权重为流通市值平方根。"""
    frame = scores.loc[scores.evaluable].merge(controls, on=["trade_date", "ticker"], validate="one_to_one")
    frame = frame.loc[frame.industry.notna() & frame.total_mv.gt(0) & frame.circ_mv.gt(0)
                      & np.isfinite(frame.total_mv) & np.isfinite(frame.circ_mv)]
    rows = []
    for day, group in frame.groupby("trade_date", sort=True):
        if len(group) < min_pairs:
            continue
        control_matrix = np.column_stack([np.log(group.total_mv), pd.get_dummies(group.industry, dtype=float)])
        returns, weights = group.label.to_numpy(), np.sqrt(group.circ_mv.to_numpy())
        for strategy in strategies:
            design = np.column_stack([group[strategy], control_matrix])
            slope, se = weighted_slope(design, returns, weights)
            rows.append(dict(trade_date=day, strategy=strategy, n_pairs=len(group),
                             slope=slope, standard_error=se, t_stat=slope / se if se > 1e-12 else np.nan))
    daily = pd.DataFrame(rows, columns=["trade_date", "strategy", "n_pairs", "slope", "standard_error", "t_stat"])
    daily.to_csv(out / "regression_daily.csv", index=False)
    tests = daily.assign(abs_t=daily.t_stat.abs()).groupby("strategy")
    summary = tests.agg(n_valid_days=("t_stat", "count"), mean_pairs=("n_pairs", "mean"),
                        mean_abs_t=("abs_t", "mean"),
                        share_abs_t_gt_2=("abs_t", lambda values: values.dropna().gt(2).mean()))
    summary.reindex(strategies).to_csv(out / "regression_summary.csv")


def _evaluate_month(frame: pd.DataFrame, names: list[str]):
    """按完整信号日做截面回归，再用有真实标签的股票评价。"""
    neutral = neutral_scores(frame, names)
    keys = frame[["trade_date", "ticker"]]
    index = stock_day_index(keys)
    target = frame.label.set_axis(index)
    daily = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"(numpy|pandas)\.")
        for name in names:
            for version, data in [("raw", frame), ("neutral", neutral)]:
                stats = daily_score_metrics(data[name].set_axis(index), target)
                daily.append(stats.assign(factor=name, version=version))
    return pd.concat([keys, neutral], axis=1), pd.concat(daily, ignore_index=True)


def evaluate_factors(frame: pd.DataFrame, names: list[str], out: Path, workers: int) -> pd.DataFrame:
    """写出处理后的样本、逐日 IC 和逐因子四项指标，保留符号和绝对值排名。"""
    caps = pd.read_parquet(CSI500_INPUTS["caps"], columns=["trade_date", "ts_code", "total_mv"])
    frame = attach_controls(frame, caps, pd.read_parquet(CSI500_INPUTS["industries"]))
    daily = []
    with parallel_config(backend="loky", inner_max_num_threads=1):
        months = Parallel(n_jobs=workers, return_as="generator", pre_dispatch=workers)(
            delayed(_evaluate_month)(block, names) for _, block in frame.groupby(frame.trade_date.dt.to_period("M")))

        def score_batches():
            for neutral, stats in months:
                daily.append(stats)
                yield neutral

        write_parquet_batches(score_batches(), out / "neutral.parquet")
    days = pd.concat(daily, ignore_index=True)
    days.to_parquet(out / "daily_ic.parquet", index=False)
    summary = pd.DataFrame([
        dict(factor=name, version=version, **summarize_ic(group), ic_days=group.ic.count())
        for (name, version), group in days.groupby(["factor", "version"])])
    for metric in ("mean_ic", "icir", "mean_rank_ic", "rank_icir"):
        summary[f"abs_{metric}_rank"] = summary[metric].abs().groupby(summary.version).rank(
            ascending=False, method="min").astype("Int64")
    summary = summary.sort_values(["version", "abs_rank_icir_rank"], kind="stable")
    summary.to_csv(out / "factor_metrics.csv", index=False)
    return summary
