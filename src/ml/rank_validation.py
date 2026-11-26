"""排序研究共用的股票样本、收益分档、名单命中率和验证损失。"""

import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.research_panel import segment_masks
from src.ml.dataset import read_feature_split

BASES = ("top50", "top100", "avoid50", "avoid100")
METHODS = (*BASES, "linear", "equal", "logistic")


def fixed_grades(frame: pd.DataFrame) -> pd.Series:
    """按每日真实名次划档，边界并列同档。"""
    group = frame.groupby("trade_date").label
    high = group.rank(method="min", ascending=False)
    low = group.rank(method="min")
    if (low.le(100) & high.le(100)).any():
        raise ValueError("前后100名出现重叠，无法定义互斥标签")
    grades = pd.Series(np.select([low.le(50), low.le(100), high.le(50), high.le(100)],
                                 [0, 1, 4, 3], default=2), index=frame.index, dtype="int8")
    return grades


def read_period(source: Path, bounds: list[str], cfg=None) -> pd.DataFrame:
    """按信号日读全股票池，仅对收益终点也落入本期的股票生成标签。"""
    frame = read_feature_split(source / "features.parquet", bounds, "排名实验", cfg).sort_values(["trade_date", "ticker"])
    frame["evaluable"] = segment_masks(frame, {"period": bounds})["period"]
    frame["grade"] = fixed_grades(frame.loc[frame.evaluable]).reindex(frame.index).astype("Int8")
    return frame


def candidates_from_grid(params, space):
    """从起始配置开始遍历各候选，只比较不同的完整配置。"""
    if not space.keys() <= params.keys() or any(not isinstance(v, list) or not v for v in space.values()):
        raise ValueError("搜索网格须提供已有模型参数及其非空候选列表")
    candidates = [params]
    for values in product(*space.values()):
        candidate = {**params, **dict(zip(space, values))}
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def archive_periods(reference, segments):
    """核验训练、早停、验证和测试的日期边界，返回归档的逐日标签分布。"""
    archive = reference / "final"
    periods = json.loads((archive / "periods.json").read_text())
    start, end = segments["validation"]
    if ([p["period"] for p in periods] != ["train", "early_stop"]
            or not periods[0]["end"] < periods[1]["start"] <= periods[1]["end"] < start <= end < segments["test"][0]):
        raise ValueError("训练、早停、验证与测试的时间交叉")
    return periods, pd.read_csv(archive / "grade_counts.csv", parse_dates=["trade_date"])


def check_grades(rows, period, archived):
    """验证有效股票日、最后的标签终点及每日五档样本数。"""
    if len(rows) != period["stock_days"] or str(rows.label_end_date.max().date()) != period["last_label_end"]:
        raise ValueError("有效样本与归档不一致")
    counts = pd.crosstab(rows.trade_date, rows.grade)
    counts.columns = counts.columns.astype(str)
    expected = archived.loc[archived.period.eq(period["period"])].set_index("trade_date").drop(columns="period")
    pd.testing.assert_frame_equal(counts.reindex(columns=expected.columns, fill_value=0), expected,
                                  check_names=False, check_dtype=False)
    return counts.sum(axis=1).to_numpy()


def training_periods(source, reference, segments, early_stop=None, cfg=None):
    """逐段读取训练与早停样本，按归档核对交易日、标签边界和分档人数。"""
    if early_stop is not None:
        for name, bounds in (("train", segments["train"]), ("early_stop", early_stop)):
            frame = read_period(source, bounds, cfg=cfg)
            rows = frame.loc[frame.evaluable]
            yield name, rows, rows.groupby("trade_date", sort=False).size().to_numpy()
        return
    periods, counts = archive_periods(reference, segments)
    for period in periods:
        frame = read_period(source, [period["start"], period["end"]], cfg=cfg)
        rows = frame.loc[frame.evaluable]
        del frame
        sizes = (rows.groupby("trade_date", sort=False).size().to_numpy() if cfg and cfg.get("label_end_days", 2) != 2
                 else check_grades(rows, period, counts))
        yield period["period"], rows, sizes
        del rows


def equal_ranks(frame, columns):
    """先在全部打分股票上逐日取百分位，再给各目标相同权重。"""
    return frame.groupby("trade_date")[columns].rank(pct=True).mean(axis=1)


def fuse_rankers(frame, columns, mode="percentile"):
    """用同日截面的原始预测做标准化，四个排序目标等权。"""
    if mode == "percentile":
        return equal_ranks(frame, columns)
    if mode != "zscore":
        raise ValueError(f"未知的排序器融合方式 {mode}")
    values = frame[columns].astype("float64")
    if not np.isfinite(values).all().all():
        raise ValueError("排序器原始预测缺分")
    groups = values.groupby(frame.trade_date)
    std = groups.transform("std", ddof=1)
    return ((values - groups.transform("mean")) / std.where(std.gt(0))).fillna(0.).mean(axis=1)


def validation_loss(frame, score, gains, k):
    """有效标签截面按最终分数排序，同分按代码；逐日、逐目标等权计算 1 − NDCG。"""
    columns = list(gains)
    gain_table = np.asarray([gains[name] for name in columns], dtype=float).T
    if not score.index.equals(frame.index) or not np.isfinite(score.to_numpy()).all():
        raise ValueError("预测分数缺失或与股票日未对齐")
    valid = frame.loc[frame.evaluable, ["trade_date", "ticker", "grade"]].assign(score=score)
    if valid.empty or valid.grade.isna().any():
        raise ValueError("验证损失需要对齐的预测与有效收益档位")
    valid = valid.sort_values(["trade_date", "score", "ticker"], ascending=[True, False, True])
    rows = []
    for day, group in valid.groupby("trade_date", sort=False):
        values = gain_table[group.grade.to_numpy(dtype=int)]
        n = min(k, len(values))
        discount = 1 / np.log2(np.arange(2, n + 2))
        ideal = discount @ np.sort(values, axis=0)[::-1][:n]
        if (ideal <= 0).any():
            raise ValueError(f"{day} 的理想增益为零")
        rows.append([day, *(1 - (discount @ values[:n]) / ideal)])
    daily = pd.DataFrame(rows, columns=["trade_date", *columns])
    daily["loss"] = daily[columns].mean(axis=1)
    return daily


def aligned_reference(path, frame, split, columns, *, check_labels=False):
    """只读指定期间，逐股票日核对档位与可评价标记后提取基线分数。"""
    keys = ["trade_date", "ticker"]
    left = frame.set_index(keys, verify_integrity=True)
    controls = ["grade", "evaluable", "label"] if check_labels else ["grade", "evaluable"]
    old = pd.read_parquet(path / "predictions.parquet", filters=[("split", "==", split)],
                          columns=[*keys, *controls, *columns]).set_index(keys, verify_integrity=True)
    pd.testing.assert_index_equal(left.index.sort_values(), old.index.sort_values())
    old = old.reindex(left.index)
    pd.testing.assert_frame_equal(left[controls], old[controls], check_dtype=False)
    if not np.isfinite(old[columns].to_numpy()).all():
        raise ValueError("归档预测缺分")
    return old[columns].set_axis(frame.index)


def selection_stats(scores: pd.DataFrame, methods: tuple[str, ...] = METHODS) -> pd.DataFrame:
    """在每日有效标签内选预测前50，记录两种头部命中、两种尾部误选和平均收益。"""
    valid = scores.loc[scores.evaluable]
    pieces = []
    for name in methods:
        top = valid.sort_values(["trade_date", name, "ticker"], ascending=[True, False, True]).groupby("trade_date").head(50)
        flags = top.assign(hit50=top.grade.ge(4), hit100=top.grade.ge(3),
                           bottom50=top.grade.eq(0), bottom100=top.grade.le(1))
        pieces.append(flags.groupby(["split", "trade_date"])[["hit50", "hit100", "bottom50", "bottom100", "label"]]
                      .mean().rename(columns={"label": "top_return"}).assign(strategy=name))
    return pd.concat(pieces).reset_index()


def bottom_members(frame: pd.DataFrame, k: int) -> pd.Series:
    """按信号日识别真实收益最低的 k 只，边界并列全部计入。"""
    return frame.groupby("trade_date").label.rank(method="min").le(k)


def holding_bottom_rates(holdings: pd.DataFrame, scores: pd.DataFrame, dates: pd.DatetimeIndex, k: int) -> pd.DataFrame:
    """用 T 日标签评价 T+1 收盘实持股票，逐日保留可评价及全部持仓数。"""
    ranked = scores.loc[scores.evaluable, ["trade_date", "ticker", "label"]].copy()
    ranked["bottom"] = bottom_members(ranked, k)
    execution = pd.Series(dates[1:], index=dates[:-1])
    ranked["trade_date"] = ranked.trade_date.map(execution)
    held = holdings.loc[holdings.amount.gt(0)].merge(ranked[["trade_date", "ticker", "bottom"]],
                                                     on=["trade_date", "ticker"], how="left", validate="many_to_one")
    daily = held.groupby(["strategy", "offset", "trade_date"]).bottom.agg(
        held="size", evaluable="count", in_bottom="sum").reset_index()
    daily["held_bottom_rate"] = daily.in_bottom / daily.evaluable.where(daily.evaluable.gt(0))
    return daily
