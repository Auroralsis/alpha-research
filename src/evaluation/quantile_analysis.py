import pandas as pd

def assign_quantiles(factor : pd.Series, n_quantile=5) -> pd.Series :
    if factor.count() == 0:
        return pd.Series(index=factor.index, dtype='float64')
    return pd.qcut(factor, n_quantile, labels=False) + 1

def calc_quantile_returns(df : pd.DataFrame) -> pd.Series :
    return (df.groupby('quantiles')['future_ret_5']
            .apply(lambda x : x.sum() / x.count()))