from typing import overload

import pandas as pd


@overload
def calc_momentum(prices: pd.Series, window: int) -> pd.Series: ...


@overload
def calc_momentum(prices: pd.DataFrame, window: int) -> pd.DataFrame: ...


# calculate momentum, the adjusted prices must be sorted according to trade date
# window > 0
def calc_momentum(prices: pd.Series | pd.DataFrame, window: int) -> pd.Series | pd.DataFrame:
    """prices 是按日期排序的单股价格列，或 [交易日 x 股票] 宽表。

    window 是向前比较的行数，返回与输入相同结构的涨幅。
    """
    return prices.pct_change(window)
