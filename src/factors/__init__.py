"""因子计算入口。每批返回按 trade_date、ticker 标识的股票日长表。"""

from collections.abc import Iterator
from typing import Literal

import pandas as pd

from src.data.research_panel import ResearchPanel
from src.factors.alpha101 import alpha101_batches
from src.factors.alpha158 import alpha158_batches
from src.factors.price_volume import compute_factors


def factor_batches(
    family: Literal["alpha158", "price_volume", "alpha101"], panel: ResearchPanel, members: pd.DataFrame,
    dates: pd.DatetimeIndex, config: dict, *, workers: int = 1,
) -> Iterator[pd.DataFrame]:
    """输入带预热期的行情、成员表和信号日，每批返回股票日键和因子列。"""
    keys = members.loc[members.trade_date.isin(dates), ["trade_date", "ticker"]]
    if keys.duplicated().any():
        raise ValueError("因子输入包含重复股票日")
    if family == "alpha158":
        yield from alpha158_batches(panel, keys, config, workers)
    elif family == "alpha101":
        yield from alpha101_batches(panel, keys)
    elif family == "price_volume":
        yield compute_factors(panel, keys, dates)
    else:
        raise ValueError(f"未知因子组 {family}")
