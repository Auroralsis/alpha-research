"""用 Qlib 的 Alpha158 表达式计算中证500特征和收益标签。"""

import json
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.data.research_panel import ResearchPanel


PRICE_FIELDS = ("open", "high", "low", "close", "vwap")


def qlib_bars(panel: ResearchPanel) -> pd.DataFrame:
    """从行情容器取出有效股票日，返回 Qlib 使用的归一化量价长表。

    每只股票用首个有效复权收盘价归一化，成交量用价格倍率的倒数调整。
    """
    daily = panel.daily
    columns = ["ticker", "trade_date", "close", "vol", *(f"{name}_adj" for name in PRICE_FIELDS)]
    observed = daily.loc[daily.observation_status.eq("observed"), columns].copy()
    first_close = observed.groupby("ticker", sort=False)["close_adj"].transform("first")
    factor = observed.close_adj / observed.close / first_close
    result = observed[["ticker", "trade_date"]].copy()
    for name in PRICE_FIELDS:
        result[name] = observed[f"{name}_adj"] / first_close
    result["factor"] = factor
    result["volume"] = 100 * observed.vol / factor
    return result


def write_provider(bars: pd.DataFrame, dates: pd.DatetimeIndex, path: Path) -> None:
    """把量价长表对齐到交易日历，并按股票和字段写出 float32 二进制文件。"""
    (path / "calendars").mkdir(parents=True, exist_ok=True)
    (path / "calendars" / "day.txt").write_text("\n".join(dates.strftime("%Y-%m-%d")) + "\n")
    for ticker, group in bars.groupby("ticker", sort=False):
        stock_dir = path / "features" / ticker.lower()
        stock_dir.mkdir(parents=True, exist_ok=True)
        aligned = group.set_index("trade_date").reindex(dates)
        for name in (*PRICE_FIELDS, "volume", "factor"):
            values = np.empty(len(dates) + 1, dtype="<f4")
            values[0] = 0  # Qlib 用第一个数记录日历起始位置。
            values[1:] = aligned[name].to_numpy(dtype=np.float32)
            values.tofile(stock_dir / f"{name}.day.bin")


def initialize_qlib(path: Path, version: str) -> list[dict[str, str]]:
    """加载指定版本的 Qlib 本地数据源，返回 158 个名称与公式组成的列表。"""
    import qlib
    from qlib.config import C
    from qlib.contrib.data.loader import Alpha158DL
    from qlib.data.cache import H
    from qlib.data.data import ExpressionD, register_all_wrappers
    from qlib.data.ops import register_all_ops

    if qlib.__version__ != version:
        raise RuntimeError(f"Expected Qlib {version}, found {qlib.__version__}")
    # 只注册 Qlib 数据算子和读取接口，实验记录由项目保存。
    H.clear()
    C.set("client", provider_uri=str(path), region="cn", expression_cache=None, dataset_cache=None)
    register_all_ops(C)
    register_all_wrappers(C)
    fields, names = Alpha158DL.get_feature_config()
    if len(fields) != 158 or len(set(names)) != 158:
        raise RuntimeError("Unexpected default Alpha158 specification")
    for field in fields:
        if ExpressionD.get_expression_instance(field).get_extended_window_size()[1] != 0:
            raise RuntimeError(f"Alpha158 expression reads forward: {field}")
    return [{"name": name, "expression": field} for name, field in zip(names, fields)]


def native_features(ticker: str, spec: list[dict[str, str]], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """计算一只股票在 start 至 end 间的特征，返回含日期、股票代码和特征列的长表。"""
    from qlib.data.data import Cal, ExpressionD

    features = pd.DataFrame({entry["name"]: ExpressionD.expression(ticker, entry["expression"], start, end)
                             for entry in spec})
    features.index = pd.DatetimeIndex(Cal.calendar(freq="day")[features.index.to_numpy(dtype=int)], name="trade_date")
    return features.reset_index().assign(ticker=ticker)


def execution_labels(panel: ResearchPanel, keys: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """给逐股信号长表补上收益端点和复权收盘收益。

    start、end 表示相对信号日的交易日偏移，按市场日历定位价格。
    """
    date_pos = panel.calendar.get_indexer(keys.trade_date)
    ticker_pos = panel.tickers.get_indexer(keys.ticker)
    if (date_pos < 0).any() or (ticker_pos < 0).any() or not 0 < start < end:
        raise ValueError("Invalid label keys or offsets")
    close = panel.matrix("close_adj")
    calendar = pd.Series(panel.calendar, index=panel.calendar)
    start_close = close.shift(-start).to_numpy()[date_pos, ticker_pos]
    end_close = close.shift(-end).to_numpy()[date_pos, ticker_pos]
    return keys.assign(feature_date=keys.trade_date, label_start_date=keys.trade_date.map(calendar.shift(-start)),
                       label_end_date=keys.trade_date.map(calendar.shift(-end)), label=end_close / start_close - 1)


def segment_masks(rows: pd.DataFrame, segments: dict[str, list[str]]) -> dict[str, pd.Series]:
    """按阶段返回布尔掩码，要求信号日与收益终点同属该阶段且收益有效。"""
    boundaries = [(pd.Timestamp(start), pd.Timestamp(end)) for start, end in segments.values()]
    if any(lo > hi for lo, hi in boundaries) or any(a[1] >= b[0] for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("Research splits must be chronologically ordered and disjoint")
    return {name: rows.trade_date.between(lo, hi) & rows.label_end_date.le(hi) & np.isfinite(rows.label)
            for name, (lo, hi) in zip(segments, boundaries)}


def build_features(panel: ResearchPanel, members: pd.DataFrame, config: dict, out: Path, workers: int) -> None:
    """用行情、成员长表与研究配置生成 Alpha158 特征和标签，并写入 out。"""
    import pyarrow as pa
    import pyarrow.parquet as pq
    bars = qlib_bars(panel)
    eligible = members[["ticker", "trade_date", "source_snapshot_date"]].merge(
        bars[["ticker", "trade_date"]], on=["ticker", "trade_date"], how="inner", validate="one_to_one")
    eligible = eligible.loc[eligible.trade_date.ge(panel.calendar[config["feature_warmup_days"]])]
    keys = execution_labels(panel, eligible, config["label_start_days"], config["label_end_days"])
    batches = sorted(keys.ticker.unique())
    if not len(batches):
        raise ValueError("No eligible feature stock-days")
    pending = out / "features.pending"
    writer = None
    infinite_counts = np.zeros(158, dtype=np.int64)
    with TemporaryDirectory(prefix="alpha158-provider-") as tmp:
        provider = Path(tmp)
        write_provider(bars, panel.calendar, provider)
        spec = initialize_qlib(provider, config["qlib_version"])
        names = [entry["name"] for entry in spec]
        evaluate = partial(native_features, spec=spec, start=keys.trade_date.min(), end=panel.calendar[-1])
        try:
            with ProcessPoolExecutor(max_workers=workers, initializer=initialize_qlib,
                                     initargs=(provider, config["qlib_version"])) as pool:
                for start in range(0, len(batches), 64):
                    symbols = batches[start:start + 64]
                    values = pd.concat(pool.map(evaluate, symbols), ignore_index=True)
                    block = keys.loc[keys.ticker.isin(symbols)].merge(values, on=["ticker", "trade_date"],
                                                                     how="left", validate="one_to_one")
                    inputs = block[names].to_numpy(dtype=np.float32, copy=True)
                    infinite = np.isinf(inputs)
                    infinite_counts += infinite.sum(axis=0)
                    # 将无穷值记为缺失，并保留各特征的异常计数。
                    inputs[infinite] = np.nan
                    block[names] = inputs
                    table = pa.Table.from_pandas(block, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(pending, table.schema, compression="zstd")
                    writer.write_table(table)
                    print(f"Alpha158: {min(start + 64, len(batches))}/{len(batches)} tickers", flush=True)
        finally:
            if writer is not None:
                writer.close()
    pending.replace(out / "features.parquet")
    (out / "alpha158.json").write_text(json.dumps(spec, indent=2) + "\n")
    audit = {
        "members": len(members), "observed_panel_rows": len(bars), "feature_rows": len(keys),
        "valid_vwap_rows": int(bars.vwap.notna().sum()), "feature_count": len(spec),
        "feature_infinities_stored_as_missing": {name: int(n) for name, n in zip(names, infinite_counts) if n},
        "first_feature_date": str(keys.trade_date.min().date()),
        "last_feature_date": str(keys.trade_date.max().date()),
        "eligible_members_missing_observation": len(members) - len(eligible) - int(
            members.trade_date.lt(panel.calendar[config["feature_warmup_days"]]).sum()),
        "usable_labels_by_split": {name: int(mask.sum()) for name, mask in segment_masks(keys, config["segments"]).items()},
    }
    (out / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
