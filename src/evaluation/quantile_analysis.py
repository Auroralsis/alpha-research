import pandas as pd

def assign_quantiles(factor: pd.Series, n_quantiles: int = 5) -> pd.Series:
    """输入一天的 [股票] 分数列，返回与原索引对应的组号。"""
    if factor.count() == 0:
        return pd.Series(index=factor.index, dtype='float64')
    return pd.qcut(factor, n_quantiles, labels=False) + 1

def calc_quantile_returns(df: pd.DataFrame, quantile_type: str, future_ret_type: str) -> pd.Series:
    """输入一行一只股票的表，以及组号列名和未来收益列名；按组返回平均收益。"""
    return df.groupby(quantile_type)[future_ret_type].mean()
