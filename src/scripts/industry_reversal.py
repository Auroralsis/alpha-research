"""在固定树模型的前两档内，检验五日行业相对反转。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.research_panel import (
    CSI500_INPUTS, ROOT, execution_labels, load_market_calendar, prepare_membership, prepare_test_panel, segment_masks,
)
from src.factors.neutralization import align_industry_history
from src.ml.rank_validation import BASES, equal_ranks
from src.report.research import markdown_table
from src.scripts.run_all import FACTOR_RUNS, RUNS, new_run

FIXED_RUN = RUNS / "daily_pair_tuning_20261122_100328_550844"
KEY = ["trade_date", "ticker"]
FACTOR = "industry_reversal_5d"
WINDOW, MIN_PEERS, NW_LAGS = 5, 3, 5


def check_keys(frame: pd.DataFrame, source: str) -> None:
    """检查股票日主键或已经冻结的组内记录。"""
    if frame[KEY].isna().any().any() or frame.duplicated(KEY).any():
        raise ValueError(f"{source} 股票日主键缺失或重复")


def test_scores(source: Path, reference: Path, eligible: pd.DataFrame) -> pd.DataFrame:
    """只读推理分数。用两个归档的八列原始输出分别验证日内百分位。"""
    light, xgb = list(BASES), [f"xgb_{name}" for name in BASES]
    filters = [("split", "==", "test")]
    scores = pd.read_parquet(source, columns=[*KEY, *light, "equal", "xgb_equal", "blend_equal"], filters=filters)
    other = pd.read_parquet(reference, columns=[*KEY, *xgb], filters=filters)
    check_keys(scores, "正式预测")
    scored = scores.merge(other, on=KEY, how="left", validate="one_to_one")
    if not np.isfinite(scored[[*light, *xgb, "blend_equal"]]).all().all():
        raise ValueError("正式预测有非有限分数或缺少对应的 XGBoost 原始输出")
    pd.testing.assert_index_equal(
        pd.MultiIndex.from_frame(scored[KEY]).sort_values(),
        pd.MultiIndex.from_frame(eligible[KEY]).sort_values(),
    )
    np.testing.assert_allclose(equal_ranks(scored, light), scored.equal, atol=1e-14, rtol=0)
    np.testing.assert_allclose(equal_ranks(scored, xgb), scored.xgb_equal, atol=1e-14, rtol=0)
    np.testing.assert_allclose((scored.equal + scored.xgb_equal) / 2, scored.blend_equal, atol=1e-14, rtol=0)
    return scored[KEY + ["blend_equal"]]


def peer_statistics(components: pd.DataFrame) -> pd.DataFrame:
    """给每个有效股票日计算当日同业中证500成员数和排除自身的中位数。"""
    group_keys = ["trade_date", "industry"]
    peers = components.loc[components.member & components.industry.notna()].sort_values(
        [*group_keys, "stock_return", "ticker"]
    ).copy()
    by_industry = peers.groupby(group_keys, sort=False)
    totals = by_industry.stock_return.agg(peer_total="size", inclusive_median="median")

    # 排序后，删除自身会使其右侧的顺位左移一位。按这一偏移取剩余样本的中间一项或两项。
    position = by_industry.cumcount().to_numpy()
    first = np.arange(len(peers)) - position
    n = by_industry.stock_return.transform("size").to_numpy()
    lower = np.maximum((n - 2) // 2, 0)
    upper = (n - 1) // 2
    values = peers.stock_return.to_numpy()
    loo = (values[first + lower + (position <= lower) * (n > 1)]
           + values[first + upper + (position <= upper) * (n > 1)]) / 2
    peers["exclusive_median"] = np.where(n > 1, loo, np.nan)
    stats = components.merge(totals, on=group_keys, how="left", validate="many_to_one").merge(
        peers[KEY + ["exclusive_median"]], on=KEY, how="left", validate="one_to_one")
    self_in_peers = stats.member & stats.industry.notna()
    stats["peer_count"] = stats.peer_total.fillna(0).astype(int) - self_in_peers.astype(int)
    stats["peer_median"] = stats.inclusive_median.where(~self_in_peers, stats.exclusive_median)
    stats["relative_return"] = (stats.stock_return - stats.peer_median).where(stats.peer_count.ge(MIN_PEERS))
    return stats.drop(columns=["peer_total", "inclusive_median", "exclusive_median"])


def reversal(panel, members: pd.DataFrame, history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回逐股票日因子和可追溯的日收益分项，保留连续市场日上的缺口。"""
    close = panel.matrix("close_adj").rename_axis(index="trade_date", columns="ticker")
    close = close.where(np.isfinite(close) & close.gt(0))
    daily = close.div(close.shift(1)).sub(1)
    daily = daily.where(np.isfinite(daily))
    components = daily.stack(future_stack=True).dropna().rename("stock_return").reset_index()
    components = components.merge(members[KEY].assign(member=True), on=KEY, how="left", validate="one_to_one")
    components["member"] = components.member.eq(True)
    components["industry"] = align_industry_history(components, history)
    parts = peer_statistics(components)
    relative = (parts.pivot(index="trade_date", columns="ticker", values="relative_return")
                .reindex(panel.calendar).rename_axis(index="trade_date"))
    values = -relative.rolling(WINDOW, min_periods=WINDOW).sum()
    factor = values.stack(future_stack=True).dropna().rename(FACTOR).reset_index()
    return factor, parts


def freeze_groups(ranked: pd.DataFrame, days: pd.DatetimeIndex, *, matched: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """只使用日期、代码、当日基线名次和因子。先生成覆盖分母，再冻结两端。"""
    rows = ranked.loc[ranked.band.eq("B")].copy() if matched else ranked.copy()
    bucket = "bucket" if matched else "band"
    if matched:
        rows[bucket] = (rows.baseline_rank - 51) // 10
    index = ["trade_date", bucket]
    buckets = range(5) if matched else ["A", "B"]
    capacity, count = (10, 2) if matched else (50, 10)
    coverage = pd.MultiIndex.from_product([days, buckets], names=index)
    daily = rows.groupby(index).agg(n_band=("ticker", "size"), n_factor=(FACTOR, "count")).reindex(
        coverage, fill_value=0).reset_index()
    qualifying = daily.loc[daily.n_band.eq(capacity) & daily.n_factor.ge(2 * count), index]
    available = rows.dropna(subset=[FACTOR]).merge(qualifying, on=index, validate="many_to_one")
    available = available.sort_values([*index, FACTOR, "ticker"], ascending=[True, True, False, True])
    available["factor_rank"] = available.groupby(index).cumcount() + 1
    groups = available.groupby(index, sort=False)
    picks = pd.concat([groups.head(count).assign(side="H"), groups.tail(count).assign(side="L")])
    check_keys(picks, "H/L 冻结名单")
    return picks.sort_values([*index, "side", "factor_rank"]), daily


def daily_results(coverage: pd.DataFrame, selected: pd.DataFrame, group: str) -> pd.DataFrame:
    """逐日计算每端的可评估收益、名次、覆盖和两端收益差。"""
    index = ["trade_date", group]
    counts = selected.groupby([*index, "side"]).agg(
        selected=("ticker", "size"), n_label=("label", "count"), ret=("label", "mean"),
        rank=("baseline_rank", "mean"),
    ).unstack("side")
    counts = counts.reindex(columns=pd.MultiIndex.from_product(
        [["selected", "n_label", "ret", "rank"], ["H", "L"]]))
    counts.columns = counts.columns.map("_".join)
    daily = coverage.merge(counts.reset_index(), on=index, how="left", validate="one_to_one")
    count_cols = ["selected_H", "selected_L", "n_label_H", "n_label_L"]
    daily[count_cols] = daily[count_cols].fillna(0).astype(int)
    daily["spread"] = daily.ret_H - daily.ret_L
    daily["full_labels"] = daily.selected_H.gt(0) & daily.selected_H.eq(daily.n_label_H) & daily.selected_L.eq(daily.n_label_L)
    return daily


def matched_dates(small: pd.DataFrame) -> pd.DataFrame:
    """五个小档同日都有收益差才取等权平均。名次按全部入选的小档成对股票汇总。"""
    daily = small.groupby("trade_date").agg(
        **{col: (col, "sum") for col in ("n_band", "n_factor", "selected_H", "selected_L", "n_label_H", "n_label_L")},
        ret_H=("ret_H", "mean"), ret_L=("ret_L", "mean"), rank_H=("rank_H", "mean"), rank_L=("rank_L", "mean"),
        buckets_evaluable=("spread", "count"), full_labels=("full_labels", "all"),
    )
    daily.loc[daily.buckets_evaluable.ne(5), ["ret_H", "ret_L"]] = np.nan
    daily["spread"] = daily.ret_H - daily.ret_L
    return daily.reset_index().assign(band="B")


def nw_interval(spread: pd.Series) -> tuple[float, float, float]:
    """全市场日历上的 Bartlett Newey–West 均值区间，5 个交易日滞后。"""
    n = spread.count()
    if n < 2:
        return np.nan, np.nan, np.nan
    mean = spread.mean()
    residual = (spread - mean).fillna(0).to_numpy(dtype=float)
    variance_total = residual @ residual
    for lag in range(1, min(NW_LAGS + 1, len(residual))):
        variance_total += 2 * (1 - lag / (NW_LAGS + 1)) * (residual[lag:] @ residual[:-lag])
    se = np.sqrt(max(variance_total, 0)) / n
    return se, mean - 1.96 * se, mean + 1.96 * se


def summarize(daily: pd.DataFrame, matched: pd.DataFrame) -> pd.DataFrame:
    """按日等权汇总，覆盖率使用股票数作分母。"""
    rows = []
    for variant, frame in (("main", daily), ("full_labels", daily), ("matched", matched)):
        for band, series in frame.groupby("band", sort=False):
            for period, block in [("全期", series), *series.groupby(series.trade_date.dt.year)]:
                selected = block.loc[block.selected_H.gt(0)]
                evaluable = block.spread.notna()
                if variant == "full_labels":
                    evaluable &= block.full_labels
                paired = block.loc[evaluable]
                se, low, high = nw_interval(block.spread.where(evaluable)) if period == "全期" else (np.nan,) * 3
                rank_days = paired if variant == "full_labels" else selected
                coverage = paired if variant == "full_labels" else block
                rows.append(dict(
                    variant=variant, band=band, period=str(period), start=block.trade_date.min(), end=block.trade_date.max(),
                    evaluated_start=paired.trade_date.min(), evaluated_end=paired.trade_date.max(),
                    signal_days=len(block), evaluated_days=len(paired),
                    factor_coverage=coverage.n_factor.sum() / max(coverage.n_band.sum(), 1) if len(coverage) else np.nan,
                    label_coverage_H=coverage.n_label_H.sum() / coverage.selected_H.sum() if coverage.selected_H.sum() else np.nan,
                    label_coverage_L=coverage.n_label_L.sum() / coverage.selected_L.sum() if coverage.selected_L.sum() else np.nan,
                    ret_H_bp=paired.ret_H.mean() * 1e4, ret_L_bp=paired.ret_L.mean() * 1e4,
                    spread_bp=paired.spread.mean() * 1e4, positive_days=paired.spread.gt(0).mean(),
                    baseline_rank_H=rank_days.rank_H.mean(), baseline_rank_L=rank_days.rank_L.mean(),
                    nw_se_bp=se * 1e4, ci_low_bp=low * 1e4, ci_high_bp=high * 1e4,
                ))
    return pd.DataFrame(rows)


def results_table(summary: pd.DataFrame) -> str:
    """汇总表保留收益、覆盖和各端名次，全期展示区间。"""
    columns = ["口径", "档", "期间", "有效天数", "因子覆盖", "H/L 标签覆盖", "H / L 收益 bp", "H−L bp",
               "正差比例", "H / L 原名次", "95% 区间 bp"]
    modes = {"main": "主规格", "full_labels": "全标签", "matched": "小档对照"}
    return markdown_table(columns, [
        (modes[r.variant], r.band, r.period, r.evaluated_days, f"{r.factor_coverage:.2%}",
         f"{r.label_coverage_H:.2%} / {r.label_coverage_L:.2%}", f"{r.ret_H_bp:.2f} / {r.ret_L_bp:.2f}",
         f"{r.spread_bp:+.2f}", f"{r.positive_days:.2%}", f"{r.baseline_rank_H:.2f} / {r.baseline_rank_L:.2f}",
         f"[{r.ci_low_bp:+.2f}, {r.ci_high_bp:+.2f}]" if np.isfinite(r.ci_low_bp) else "—")
        for r in summary.itertuples(index=False)
    ])


def write_examples(out: Path, main: pd.DataFrame, selections: pd.DataFrame, parts: pd.DataFrame,
                   calendar: pd.DatetimeIndex) -> tuple[str, str]:
    """选 B 档首个全标签日，按原行情重核当天名单与一只股票的五日窗口。"""
    day = main.loc[main.band.eq("B") & main.full_labels].iloc[0]
    picks = selections.loc[selections.trade_date.eq(day.trade_date) & selections.band.eq("B")]
    high = picks.loc[picks.side.eq("H")].iloc[0]
    pos = calendar.get_loc(day.trade_date)
    window = calendar[pos - WINDOW + 1:pos + 1]
    example = parts.loc[parts.ticker.eq(high.ticker) & parts.trade_date.isin(window)].sort_values("trade_date")
    assert len(example) == WINDOW and example.relative_return.notna().all()
    for row in example.itertuples(index=False):
        peers = parts.loc[(parts.trade_date.eq(row.trade_date) & parts.industry.eq(row.industry)
                           & parts.member & parts.ticker.ne(row.ticker)), "stock_return"]
        assert len(peers) == row.peer_count
        np.testing.assert_allclose(peers.median(), row.peer_median, atol=1e-14)
    np.testing.assert_allclose(-example.relative_return.sum(), high[FACTOR], atol=1e-14)
    check_keys(picks, "抽查日 H/L")
    np.testing.assert_allclose(picks.loc[picks.side.eq("H"), "label"].mean()
                               - picks.loc[picks.side.eq("L"), "label"].mean(), day.spread, atol=1e-14)
    example.to_csv(out / "example_factor.csv", index=False)
    picks.to_csv(out / "example_groups.csv", index=False)
    sample_tables = (
        f"信号日 **{day.trade_date.date()}**，股票 **{high.ticker}**，"
        f"原始因子为 **{high[FACTOR]:.8f}**（收益小数）。\n\n"
        + markdown_table(["交易日", "行业代码", "个股收益", "同行数", "同行中位数", "收益差"],
                         [(r.trade_date.date(), r.industry, f"{r.stock_return:.6%}", r.peer_count,
                           f"{r.peer_median:.6%}", f"{r.relative_return:.6%}") for r in example.itertuples()])
        + f"\n\n同日 B 档 H、L 各有 {day.n_label_H}、{day.n_label_L} 个有效收益，"
          f"日收益差为 **{day.spread * 1e4:+.2f} bp**。\n\n"
        + markdown_table(["组", "股票", "原基线名次", "因子（小数）", "未来收益"],
                         [(r.side, r.ticker, r.baseline_rank, f"{getattr(r, FACTOR):.8f}", f"{r.label:.6%}")
                          for r in picks.itertuples()])
    )
    return sample_tables, f"{day.trade_date.date()} B 档 H / L 各 10 只且无重叠，{high.ticker} 的五日因子与逐日分项一致"


def main(argv=None) -> None:
    """对正式冻结分数执行一次独立的年度诊断，归档可复核结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-run", type=Path, default=FIXED_RUN, help="正式固定组合的实验目录")
    parser.add_argument("--out", type=Path, help="新的独立结果目录")
    args = parser.parse_args(argv)
    fixed = args.fixed_run.resolve()
    config = json.loads((fixed / "config.json").read_text())
    source = fixed / "predictions.parquet"
    reference = Path(config["pair_reference"]) / source.name
    if (config["label_start_days"], config["label_end_days"]) != (1, 2):
        raise ValueError("正式标签不符合 T+1 至 T+2 收盘收益")
    calendar = load_market_calendar(CSI500_INPUTS["calendar"])
    panel, days = prepare_test_panel(config, calendar)
    members = prepare_membership(pd.read_parquet(CSI500_INPUTS["membership"]))
    members = members.loc[members.trade_date.isin(panel.calendar)].copy()
    signal_members = members.loc[members.trade_date.isin(days)]
    if not signal_members.groupby("trade_date").size().reindex(days).eq(500).all():
        raise ValueError("信号日日历没有匹配到完整的每日历史成员")
    observed = panel.daily.loc[panel.daily.observation_status.eq("observed"), KEY]
    eligible = signal_members[KEY].merge(observed, on=KEY, validate="one_to_one")
    scores = test_scores(source, reference, eligible)
    history = pd.read_parquet(CSI500_INPUTS["industries"], columns=["ts_code", "in_date", "out_date", "l1_code"])
    factors, parts = reversal(panel, members, history)
    ranked = scores.sort_values(["trade_date", "blend_equal", "ticker"], ascending=[True, False, True])
    ranked["baseline_rank"] = ranked.groupby("trade_date").cumcount() + 1
    ranked = ranked.loc[ranked.baseline_rank.le(100)].merge(factors, on=KEY, how="left", validate="one_to_one")
    ranked["band"] = np.where(ranked.baseline_rank.le(50), "A", "B")
    picks, coverage = freeze_groups(ranked, days)
    small_picks, small_coverage = freeze_groups(ranked, days, matched=True)

    out = (args.out or new_run(tag=FACTOR, root=FACTOR_RUNS)).resolve()
    out.mkdir(parents=True, exist_ok=False)
    ranked.to_parquet(out / "baseline_bands.parquet", index=False)
    picks.to_parquet(out / "selections.parquet", index=False)
    small_picks.to_parquet(out / "matched_selections.parquet", index=False)

    # 两份名单已锁定。未来端点只影响接下来的绩效评价。
    labels = execution_labels(panel, scores[KEY], config["label_start_days"], config["label_end_days"])
    valid = segment_masks(labels, {"test": config["segments"]["test"]})["test"]
    archived = pd.read_parquet(source, columns=[*KEY, "label", "evaluable", "label_end_date"],
                               filters=[("split", "==", "test")])
    audit = labels.assign(evaluable=valid).merge(archived, on=KEY, validate="one_to_one", suffixes=("", "_archive"))
    if len(audit) != len(labels):
        raise ValueError("重算标签与正式档案的股票日主键不一致")
    np.testing.assert_array_equal(audit.evaluable, audit.evaluable_archive)
    np.testing.assert_array_equal(audit.label_end_date.to_numpy().view("i8"),
                                  audit.label_end_date_archive.to_numpy().view("i8"))
    np.testing.assert_allclose(audit.label, audit.label_archive, atol=1e-14, rtol=0, equal_nan=True)
    labels["label"] = labels.label.where(valid)
    picks = picks.merge(labels.drop(columns="feature_date"), on=KEY, how="left", validate="one_to_one")
    small_picks = small_picks.merge(labels.drop(columns="feature_date"), on=KEY, how="left", validate="one_to_one")
    daily = daily_results(coverage, picks, "band")
    buckets = daily_results(small_coverage, small_picks, "bucket")
    matched = matched_dates(buckets)
    summary = summarize(daily, matched)
    summary.to_csv(out / "summary.csv", index=False)
    daily.to_csv(out / "daily.csv", index=False)
    matched.to_csv(out / "matched_daily.csv", index=False)
    buckets.to_csv(out / "matched_buckets.csv", index=False)
    picks.to_parquet(out / "selection_outcomes.parquet", index=False)
    small_picks.to_parquet(out / "matched_outcomes.parquet", index=False)
    signal_members[KEY].merge(factors, on=KEY, how="left", validate="one_to_one").to_parquet(
        out / "factor_values.parquet", index=False)
    date_coverage = signal_members.groupby("trade_date").agg(
        members=("ticker", "size"), snapshot=("source_snapshot_date", "first"))
    date_coverage["scored"] = scores.groupby("trade_date").size()
    date_coverage["evaluable"] = labels.groupby("trade_date").label.count()
    date_coverage.to_csv(out / "score_coverage.csv")

    audit_summary = {
        "factor": FACTOR, "window_market_days": WINDOW, "minimum_other_members": MIN_PEERS,
        "nw_market_lags": NW_LAGS, "label_offsets": [1, 2], "research_segments": config["segments"],
        "calendar": str(CSI500_INPUTS["calendar"].relative_to(ROOT)),
        "membership": str(CSI500_INPUTS["membership"].relative_to(ROOT)),
        "prices": str(CSI500_INPUTS["prices"].relative_to(ROOT)),
        "industry": str(CSI500_INPUTS["industries"].relative_to(ROOT)),
        "predictions": str(source), "xgboost_raw_predictions": str(reference),
        "signal_days": len(days), "first_signal": str(days[0].date()), "last_signal": str(days[-1].date()),
        "member_rows": len(signal_members), "scored_rows": len(scores),
        "scored_without_evaluable_label": int((~valid).sum()),
        "same_day_member_snapshot_days": int(date_coverage.snapshot.eq(date_coverage.index).sum()),
        "archive_check": "八个排序器均值、合格股票日主键、标签端点、原始收益和有效性一致",
    }
    (out / "audit.json").write_text(json.dumps(audit_summary, ensure_ascii=False, indent=2) + "\n")
    sample_tables, sample_check = write_examples(out, daily, picks, parts, panel.calendar)
    table = results_table(summary)
    full = summary.loc[summary.period.eq("全期")].set_index(["variant", "band"])
    b = full.loc[("main", "B")]
    a = full.loc[("main", "A")]
    matched_b = full.loc[("matched", "B")]
    full_b = full.loc[("full_labels", "B")]
    annual_b = summary.query("variant == 'main' and band == 'B' and period != '全期'")
    annual = "、".join(f"{r.period} 年 {r.spread_bp:+.2f} bp" for r in annual_b.itertuples(index=False))
    (out / "report.md").write_text(f"""# 中证500行业内五日相对反转

正式固定八模型的 `blend_equal` 是 LightGBM 与 XGBoost 各四个排序器当日百分位的等权平均。使用供回测的测试期打分，{days[0].date()} 至 {days[-1].date()} 共 {len(days)} 个交易日。{len(signal_members):,} 条成员股票日中有 {len(scores):,} 条具备当日有效行情并完成打分，包含 {int((~valid).sum()):,} 条没有有效未来标签的股票日。全部推理股票日主键与当日可观察成员一致。原实验用 2014–2020 年训练，2021 年早停，2022 年选参，2023–2025 年已用于多轮研究。较早的 2022 年存档预测属于选参样本。本次表格只列正式测试期的历史诊断。

## 因子与分组

成员来自 TuShare 中证500月度历史权重快照，沿用最近一个日期不晚于 d 的快照。区间内有 {audit_summary['same_day_member_snapshot_days']} 天使用同日快照。快照缺少公告可知时间，属于近似 PIT。行业使用单因子中性化的申万一级历史映射，以 TuShare `index_member_all` 为主，新浪历史指数成分补足部分缺口。区间为 `[in_date, out_date)`，冲突或缺行业留空，历史记录版本的公告时点未验证。

先按市场日历算复权收盘的单日简单收益。两端都要求项目定义的 `observed` 且复权收盘有限、为正。缺价和零成交量不产生收益，正常交易且收盘未变的收益为零。同行是 d 日同业至少 3 个有有效收益的中证500成员，排除自己。被考察股票在窗口内尚未入池的日期，也可以用该日它自己的有效收益和当日在池同行比较。对 T−4 至 T 五项日收益差取负的算术和，任一项缺失则因子缺失。档内排序使用原始因子，由高到低，以股票代码升序破同分。

第一层按 `blend_equal` 降序、股票代码升序锁定 A（1–50）、B（51–100）。再对每个完整基线档，按因子序列头尾各取 10 只，至少要有 20 只有效。冻结名单到 Parquet 后才调用项目 `execution_labels`。标签用 T+2 / T+1 复权收盘 − 1，两端要求有效观测，标签终点在测试期内。涨跌停当天有有效行情时照常计入。缺标签留空，组内对有效标签等权，两端均可评时才取差。小档对照将 B 分成五个连续名次区间，每档头尾各 2 只，五档当日都能作差才平均。

## 表 1　年度及全期收益差

{table}

收益列为同一批可比日期的日均值，1 bp 为 0.01 个百分点。主规格因子覆盖以档内股票日为分母，标签覆盖以已冻结全部选入数为分母，基线名次也来自冻结名单。全标签口径使用 H、L 各 10 个有效标签的日期，并在这批日期重新统计覆盖与排名。小档口径的覆盖与原名次统计纳入所有已冻结的小档，两端收益、收益差和正差比例只基于当日五档都可评价的日期。全期均值的 95% 区间采用 Newey–West Bartlett 权重、固定 5 个市场交易日滞后和正态近似。无可评价收益差的日期保留其市场日历位置，协方差中该日中心化残差按零贡献处理。

主检验 B 档日均差为 **{b.spread_bp:+.2f} bp**（{b.evaluated_days} 天），年度结果为 {annual}。A 档日均差为 **{a.spread_bp:+.2f} bp**（{a.evaluated_days} 天）。B 档全标签日为 **{full_b.spread_bp:+.2f} bp**（{full_b.evaluated_days} 天），固定小档对照为 **{matched_b.spread_bp:+.2f} bp**（{matched_b.evaluated_days} 天）。收益差为未扣费的预测诊断。

## 抽查

{sample_tables}

完整数据见 [汇总](summary.csv)、[逐日](daily.csv)、[名次小档](matched_buckets.csv) 和 [输入核验](audit.json)。
""", encoding="utf-8")
    print(table)
    print(f"抽查通过。{sample_check}。冻结名单与原始收益已核对。")
    print(out)


if __name__ == "__main__":
    main()
