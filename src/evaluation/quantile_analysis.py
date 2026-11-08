import pandas as pd

def assign_quantiles(factor: pd.Series, n_quantiles=5) -> pd.Series :
    if factor.count() == 0:
        return pd.Series(index=factor.index, dtype='float64')
    return pd.qcut(factor, n_quantiles, labels=False) + 1

def calc_quantile_returns(df: pd.DataFrame, quantile_type: str, future_ret_type: str) -> pd.Series :
    return df.groupby(quantile_type)[future_ret_type].mean()