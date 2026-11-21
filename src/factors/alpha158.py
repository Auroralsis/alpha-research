"""用 Qlib 表达式分批生成 Alpha158 因子。"""

from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.data.qlib_provider import qlib_bars, register_provider, write_provider
from src.data.research_panel import ResearchPanel


def alpha158_spec() -> list[dict[str, str]]:
    """读取 Qlib 官方 Alpha158 的列名和表达式。"""
    from qlib.contrib.data.loader import Alpha158DL

    fields, names = Alpha158DL.get_feature_config()
    if len(fields) != 158 or len(set(names)) != 158:
        raise RuntimeError("Unexpected default Alpha158 specification")
    return [{"name": name, "expression": field} for name, field in zip(names, fields)]


def initialize_qlib(path: Path, version: str, *, workers: int = 1) -> list[dict[str, str]]:
    """加载本地数据源，返回 Alpha158 的特征名称与公式。"""
    from qlib.data.data import ExpressionD

    register_provider(path, version, workers=workers)
    spec = alpha158_spec()
    for entry in spec:
        field = entry["expression"]
        if ExpressionD.get_expression_instance(field).get_extended_window_size()[1] != 0:
            raise RuntimeError(f"Alpha158 expression reads forward: {field}")
    return spec


def native_features(tickers: str | list[str], spec: list[dict[str, str]], start: pd.Timestamp,
                    end: pd.Timestamp) -> pd.DataFrame:
    """经 Qlib D.features 加载 Alpha158 批次，并返回项目的股票日长表。"""
    from qlib.data import D

    instruments = [tickers] if isinstance(tickers, str) else tickers
    features = D.features(instruments, [entry["expression"] for entry in spec], start, end)
    features.columns = [entry["name"] for entry in spec]
    return features.rename_axis(index=["ticker", "trade_date"]).reset_index()


def alpha158_batches(panel: ResearchPanel, keys: pd.DataFrame, config: dict, workers: int) -> Iterator[pd.DataFrame]:
    """按股票分批计算，保留请求的股票日键。"""
    if keys.empty:
        raise ValueError("No eligible feature stock-days")
    tickers = sorted(keys.ticker.unique())
    batch_size = config.get("feature_batch_size", 64)
    start_date, end_date = keys.trade_date.min(), keys.trade_date.max()
    with TemporaryDirectory(prefix="alpha158-provider-") as tmp:
        provider = Path(tmp)
        write_provider(qlib_bars(panel), panel.calendar, provider)
        spec = initialize_qlib(provider, config["qlib_version"], workers=workers)
        for start in range(0, len(tickers), batch_size):
            symbols = tickers[start:start + batch_size]
            values = native_features(symbols, spec, start_date, end_date)
            yield keys.loc[keys.ticker.isin(symbols)].merge(
                values, on=["ticker", "trade_date"], how="left", validate="one_to_one")
