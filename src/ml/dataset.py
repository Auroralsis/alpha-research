"""把项目的股票日特征与收益接到 Qlib DatasetH。"""

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from src.data.download import write_parquet_batches
from src.data.research_panel import (
    CSI500_INPUTS as INPUTS, ResearchPanel, execution_labels, prepare_membership, prepare_panel,
    segment_masks, stock_day_index,
)
from src.factors import factor_batches
from src.factors.alpha158 import alpha158_spec


def prepare_features(config: dict, dates: pd.DatetimeIndex, out: Path, workers: int, *, through: str | None = None) -> None:
    """准备交易日行情和成分股，生成全实验共用的 Alpha158 特征。"""
    daily = pd.read_parquet(INPUTS["prices"])
    members = prepare_membership(pd.read_parquet(INPUTS["membership"]))
    if not pd.DatetimeIndex(members.trade_date.unique()).as_unit("ns").equals(dates.as_unit("ns")):
        raise ValueError("Membership does not cover the market calendar")
    if through is not None:
        cutoff = pd.Timestamp(through)
        dates = dates[dates <= cutoff]
        members = members.loc[members.trade_date.le(cutoff)]
    panel = prepare_panel(daily, dates)
    input_audit = dict(
        raw_price_rows=len(daily), raw_start=str(daily.trade_date.min()), raw_end=str(daily.trade_date.max()),
        raw_tickers=int(daily.ticker.nunique()), market_sessions=len(dates),
        raw_input_sorted_by_ticker_date=pd.MultiIndex.from_frame(daily[["ticker", "trade_date"]]).is_monotonic_increasing,
        panel_observation_status=panel.daily.observation_status.value_counts().to_dict(),
        price_keys_unique=True, member_keys_unique=True, snapshot_dates_at_or_before_signals=True,
    )
    build_features(panel, members, config, out, workers, input_audit=input_audit)


def build_features(panel: ResearchPanel, members: pd.DataFrame, config: dict, out: Path, workers: int,
                   *, input_audit: dict | None = None) -> None:
    """用行情、成员长表与研究配置生成 Alpha158 特征和标签，并写入 out。"""
    bars = panel.daily.loc[panel.daily.observation_status.eq("observed"), ["ticker", "trade_date", "vwap_adj"]]
    eligible = members[["ticker", "trade_date", "source_snapshot_date"]].merge(
        bars[["ticker", "trade_date"]], on=["ticker", "trade_date"], how="inner", validate="one_to_one")
    eligible = eligible.loc[eligible.trade_date.ge(panel.calendar[config["feature_warmup_days"]])]
    keys = execution_labels(panel, eligible, config["label_start_days"], config["label_end_days"])
    labels = keys.set_index(["ticker", "trade_date"])
    infinite_counts = np.zeros(158, dtype=np.int64)
    spec = alpha158_spec()
    names = [entry["name"] for entry in spec]

    def labeled_batches():
        for values in factor_batches("alpha158", panel, eligible, panel.calendar, config, workers=workers):
            block = values.join(labels, on=["ticker", "trade_date"], validate="many_to_one")[[*keys.columns, *names]]
            inputs = block[names].to_numpy(dtype=np.float32, copy=True)
            infinite = np.isinf(inputs)
            infinite_counts[:] += infinite.sum(axis=0)
            inputs[infinite] = np.nan
            block[names] = inputs
            yield block

    write_parquet_batches(labeled_batches(), out / "features.parquet")
    (out / "alpha158.json").write_text(json.dumps(spec, indent=2) + "\n")
    audit = {
        "members": len(members), "observed_panel_rows": len(bars), "feature_rows": len(keys),
        "valid_vwap_rows": int(bars.vwap_adj.notna().sum()), "feature_count": len(spec),
        "feature_infinities_stored_as_missing": {name: int(n) for name, n in zip(names, infinite_counts) if n},
        "first_feature_date": str(keys.trade_date.min().date()),
        "last_feature_date": str(keys.trade_date.max().date()),
        "eligible_members_missing_observation": len(members) - len(eligible) - int(
            members.trade_date.lt(panel.calendar[config["feature_warmup_days"]]).sum()),
        "usable_labels_by_split": {name: int(mask.sum()) for name, mask in segment_masks(keys, config["segments"]).items()},
    }
    audit.update(input_audit or {})
    (out / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")


def alpha158_columns(out: Path) -> list[str]:
    """读取当前实验的 Alpha158 列顺序。"""
    names = [entry["name"] for entry in json.loads((out / "alpha158.json").read_text())]
    if len(names) != 158 or len(set(names)) != 158:
        raise ValueError("Expected 158 unique Alpha158 columns")
    return names


def read_feature_split(path: Path, bounds: Sequence, split: str) -> pd.DataFrame:
    """从特征表读取指定区间并检查股票日键。"""
    begin, end = map(pd.Timestamp, bounds)
    rows = pd.read_parquet(path, filters=[("trade_date", ">=", begin), ("trade_date", "<=", end)])
    if rows.empty or rows.duplicated(["trade_date", "ticker"]).any():
        raise ValueError(f"Empty or duplicated feature rows in {split}")
    return rows


def daily_zscore(values: pd.Series, dates: pd.Series) -> pd.Series:
    """按信号日用样本标准差做 Z-score，截面退化时返回 NaN。"""
    groups = values.groupby(dates)
    sd = groups.transform("std", ddof=1)
    return (values - groups.transform("mean")) / sd.where(sd.gt(0) & np.isfinite(sd))


def fitting_targets(eligible: pd.DataFrame, mode: str) -> pd.Series:
    """将通过日期与有限性筛选的原始收益转换为拟合目标。"""
    if mode == "raw":
        return eligible.label
    target = daily_zscore(eligible.label, eligible.trade_date)
    return target.loc[np.isfinite(target)]


def qlib_segment(rows: pd.DataFrame, names: list[str], labels: pd.Series) -> pd.DataFrame:
    """组装 Qlib 的 feature 与 label 列组，标签索引决定纳入哪些股票日。"""
    features = rows.loc[labels.index, names]
    return pd.concat({"feature": features, "label": labels.to_frame("label")}, axis=1).set_axis(
        stock_day_index(rows.loc[labels.index, ["trade_date", "ticker"]]),
    )


def fitting_segment(
    rows: pd.DataFrame, names: list[str], valid: pd.Series, mode: str, split: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """按该时段的有效股票日生成拟合数据，另返回保留原行索引的目标供 Ridge 与样本核验使用。"""
    target = fitting_targets(rows.loc[valid, ["trade_date", "label"]], mode)
    if target.empty:
        raise ValueError(f"No usable fitting labels in {split}")
    return qlib_segment(rows, names, target), target


def make_dataset(**frames: pd.DataFrame) -> DatasetH:
    """用完整股票日键保留各阶段的输入顺序，供 Qlib 拟合和预测。"""
    segments = {name: frame.index for name, frame in frames.items()}
    return DatasetH(handler=DataHandlerLP.from_df(pd.concat(frames.values())), segments=segments)
