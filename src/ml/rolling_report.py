"""在共同样本上评价滚动预测，并生成统一报告的滚动部分。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.ml.models import ROLLING_SCORE, SCORE_COLUMNS, common_scores, summarize_predictions
from src.reporting import markdown_table


STRATEGIES = ("lightgbm", ROLLING_SCORE)
NAMES = {"lightgbm": "固定 LightGBM", "rolling_lightgbm": "滚动 LightGBM"}


def hac_mean(series: pd.Series, lag: int) -> dict:
    """估计日 RankIC 均值的 Bartlett–Newey–West 标准误，保留缺失日的时间间隔。"""
    observed = series.notna().to_numpy()
    n = int(observed.sum())
    mean = series.mean()
    result = dict(n_days=n, lag=lag, mean=mean, se=np.nan, t_stat=np.nan, p_value=np.nan,
                  ci_lower=np.nan, ci_upper=np.nan)
    if n < 2:
        return result
    deviations = np.where(observed, series.to_numpy(dtype=float) - mean, 0.)
    covariance = deviations @ deviations
    for k in range(1, min(lag + 1, len(deviations))):
        covariance += 2 * (1 - k / (lag + 1)) * (deviations[k:] @ deviations[:-k])
    se = np.sqrt(max(covariance, 0.) / (n * (n - 1)))
    if se > 0:
        t = mean / se
        result.update(se=se, t_stat=t, p_value=2 * norm.sf(abs(t)),
                      ci_lower=mean - norm.ppf(.975) * se, ci_upper=mean + norm.ppf(.975) * se)
    return result


def window_portfolio_returns(ledger: pd.DataFrame, windows: list[dict], offset: int) -> pd.DataFrame:
    """按测试窗口截取主轨道的扣费日收益，保留跨窗口继承的持仓与资金。"""
    primary = ledger.loc[ledger.offset.eq(offset) & ledger.strategy.isin(STRATEGIES)]
    rows = []
    for window in windows:
        begin, end = map(pd.Timestamp, window["effective"]["test"])
        segment = primary.loc[primary.trade_date.between(begin, end)]
        for strategy in STRATEGIES:
            book = segment.loc[segment.strategy.eq(strategy)].sort_values("trade_date")
            if len(book) != window["test_sessions"] or book.trade_date.duplicated().any():
                raise ValueError("窗口的主轨道账本缺少交易日或包含重复记录")
            if not np.isfinite(book.daily_return).all() or book.daily_return.le(-1).any():
                raise ValueError("窗口包含无效的组合日收益")
            annual_return = (1 + book.daily_return).prod() ** (252 / len(book)) - 1
            rows.append(dict(window=window["window"], strategy=strategy, annual_return=annual_return,
                             portfolio_days=len(book)))
    return pd.DataFrame(rows)


def evaluate_windows(config: dict, dates: pd.DatetimeIndex, predictions: pd.DataFrame,
                     daily: pd.DataFrame, groups: pd.DataFrame, out: Path) -> None:
    """从统一的逐日表生成逐窗对比和 HAC 检验。"""
    specs = json.loads((out / "windows.json").read_text())
    windows = specs["windows"]
    daily = daily.loc[daily.split.eq("test") & daily.strategy.isin(STRATEGIES)]
    groups = groups.loc[groups.split.eq("test") & groups.strategy.isin(STRATEGIES)]
    ic = daily.pivot(index="trade_date", columns="strategy", values="rank_ic").reindex(dates)
    ic["rolling_minus_fixed"] = ic.rolling_lightgbm - ic.lightgbm
    lag = config["label_end_days"] - config["label_start_days"]
    window_ic, window_groups, inference = [], [], []
    periods = [("all", dates)]
    for window in windows:
        number = window["window"]
        begin, end = map(pd.Timestamp, window["effective"]["test"])
        window_dates = dates[(dates >= begin) & (dates <= end)]
        periods.append((str(number), window_dates))
        period_config = dict(config, segments={"test": [str(begin.date()), str(end.date())]})
        summary, spread = summarize_predictions(
            daily.loc[daily.trade_date.isin(window_dates)], groups.loc[groups.trade_date.isin(window_dates)],
            period_config, window_dates,
        )
        for tables, frame in ((window_ic, summary), (window_groups, spread)):
            tables.append(frame.loc[frame.year.eq("all")].drop(columns=["split", "year"]).assign(window=number))
    for name, calendar in periods:
        for strategy in (*STRATEGIES, "rolling_minus_fixed"):
            inference.append(dict(period=name, strategy=strategy, **hac_mean(ic[strategy].reindex(calendar), lag)))
    returns = window_portfolio_returns(pd.read_parquet(out / "ledger.parquet"), windows, config["main_offset"])
    summary = pd.concat(window_ic, ignore_index=True).merge(
        returns, on=["window", "strategy"], validate="one_to_one",
    )
    summary.to_csv(out / "window_rank_ic.csv", index=False)
    pd.concat(window_groups, ignore_index=True).to_csv(out / "window_quintile_summary.csv", index=False)
    pd.DataFrame(inference).to_csv(out / "hac_rank_ic.csv", index=False)

    test = predictions.loc[predictions.split.eq("test")]
    cohort = common_scores(test, (*SCORE_COLUMNS, ROLLING_SCORE))
    audit = dict(test_predictions=len(test), common_stock_days=len(cohort), common_labels=int(cohort.evaluable.sum()))
    (out / "window_sample_audit.json").write_text(json.dumps(audit, indent=2) + "\n")


def rolling_section(out: Path, links: str, windows_spec: dict) -> str:
    """用滚动研究产物生成统一报告中的训练窗口与预测对比。"""
    audit = json.loads((out / "window_sample_audit.json").read_text())
    window_ic = pd.read_csv(out / "window_rank_ic.csv")
    window_q = pd.read_csv(out / "window_quintile_summary.csv")
    hac = pd.read_csv(out / "hac_rank_ic.csv", dtype={"period": str}).query('period == "all"')
    rows = []
    for window in windows_spec["windows"]:
        number = window["window"]
        folder = f"models/window_{number:02d}"
        log = json.loads((out / folder / "training_log.json").read_text())
        ranges = [" 至 ".join(value[:10] for value in log["effective"][split])
                  for split in ("train", "validation", "test")]
        fit = log["fit_audit"]
        samples = f'{fit["train"]["fitted_rows"]:,} / {fit["validation"]["fitted_rows"]:,} / {log["test_rows"]:,}'
        rows.append([number, *ranges, log["test_sessions"], samples,
                     f'[{log["best_iteration"]}]({links}/{folder}/training_log.json)',
                     f'[模型]({links}/{folder}/lightgbm.txt)'])
    window_table = markdown_table(
        ["窗口", "训练信号", "验证信号", "测试信号", "测试天数", "训练 / 验证 / 预测行数", "最佳轮", "训练产物"], rows,
    )
    test_table = markdown_table(
        ["窗口", "方法", "有效天数", "Mean RankIC", "RankIC 标准差", "ICIR", "主轨道扣费年化", "正 IC 比例"],
        [[row.window, NAMES[row.strategy], row.n_days, f"{row.mean:.4f}", f"{row.rank_ic_std:.4f}",
          f"{row.icir:.4f}", f"{row.annual_return:.2%}", f"{row.positive_rate:.2%}"]
         for row in window_ic.itertuples(index=False)],
    )
    hac_table = markdown_table(
        ["方法", "Mean RankIC", "HAC 标准误", "t", "p", "均值 95% 区间"],
        [[NAMES.get(row.strategy, "滚动减固定"), f"{row.mean:.4f}", f"{row.se:.5f}", f"{row.t_stat:.3f}",
          f"{row.p_value:.4f}", f"[{row.ci_lower:.4f}, {row.ci_upper:.4f}]"]
         for row in hac.itertuples(index=False)],
    )
    group_table = markdown_table(
        ["窗口", "方法", "Q1", "Q2", "Q3", "Q4", "Q5", "Q5−Q1", "均值依次非降"],
        [[row.window, NAMES[row.strategy], *(f"{getattr(row, f'Q{i}'):.3%}" for i in range(1, 6)),
          f"{row.spread:.3%}", "是" if row.monotonic_means else "否"]
         for row in window_q.itertuples(index=False)],
    )
    delta = hac.loc[hac.strategy.eq("rolling_minus_fixed")].iloc[0]
    return f"""## 滚动 LightGBM 的逐窗结果

固定 LightGBM 在整个测试期使用同一个模型。滚动 LightGBM 使用 Qlib 的 `RollingGen` 生成窗口，按 **{windows_spec['step']} 个交易日**推进。每窗训练和验证分别使用下表的信号日期，收益标签终点须早于下一阶段首日。新模型在对应测试段接管打分，组合延续原有持仓、现金和调仓日程。

**表　滚动模型的训练、验证与测试窗口**

{window_table}

滚动模型共生成 {audit['test_predictions']:,} 条股票日预测，同口径比较使用其中 {audit['common_stock_days']:,} 条，具有有效未来收益标签的有 {audit['common_labels']:,} 条。

**表　固定与滚动 LightGBM 在各窗口的 RankIC 与年化收益**

{test_table}

年化收益取 offset {windows_spec['config']['main_offset']} 主轨道在各测试窗口内的扣费日收益连乘，再按 252 个交易日折算。窗口首日收益衔接前一交易日的收盘资产，组合沿用连续回测中的持仓与现金。年化分母使用窗口全部交易日，末端缺少未来标签的日期仍计入组合收益，因此可能多于表中的 IC 有效天数。ICIR 衡量整个候选池的排序能力，组合年化还受到 Top-50 持仓、换股和费用的影响。

滚动减固定的日 RankIC 均值为 **{delta['mean']:+.4f}**。用滞后 {delta.lag} 个交易日的 HAC 标准误处理相邻五日收益重叠后，均值差的 95% 区间为 **[{delta.ci_lower:.4f}, {delta.ci_upper:.4f}]**。

**表　测试期日 RankIC 均值及配对差异的 HAC 检验**

{hac_table}

<details>
<summary>展开各测试窗口的五组收益</summary>

**表　固定与滚动 LightGBM 在各窗口的五组收益**

{group_table}

</details>

[窗口原始与有效日期]({links}/windows.json) · [统一训练损失表]({links}/training_mse.csv) · [统一预测表]({links}/predictions.parquet) · [窗口样本审计]({links}/window_sample_audit.json) · [逐窗检验]({links}/hac_rank_ic.csv)"""

