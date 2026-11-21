"""为 Qlib 特征和执行器准备本地量价数据源。"""

import logging
from pathlib import Path

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
    fields = bars.columns.difference(["ticker", "trade_date"], sort=False)
    for ticker, group in bars.groupby("ticker", sort=False):
        stock_dir = path / "features" / ticker.lower()
        stock_dir.mkdir(parents=True, exist_ok=True)
        aligned = group.set_index("trade_date").reindex(dates)
        for name in fields:
            values = np.empty(len(dates) + 1, dtype="<f4")
            values[0] = 0  # Qlib 用第一个数记录日历起始位置。
            values[1:] = aligned[name].to_numpy(dtype=np.float32)
            values.tofile(stock_dir / f"{name}.day.bin")


def register_provider(path: Path, version: str, *, workers: int = 1) -> None:
    """注册本地 Qlib 数据源，共供特征计算和回测使用。"""
    import qlib

    if qlib.__version__ != version:
        raise RuntimeError(f"Expected Qlib {version}, found {qlib.__version__}")
    qlib.init(provider_uri=str(path), region="cn", expression_cache=None, dataset_cache=None, kernels=workers,
              logging_level=logging.WARNING,
              exp_manager={"class": "MLflowExpManager", "module_path": "qlib.workflow.expm",
                           "kwargs": {"uri": (path.resolve() / "mlruns").as_uri(), "default_exp_name": "csi500"}})
