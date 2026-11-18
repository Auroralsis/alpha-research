import pandas as pd

def calc_ic_rank(alpha : pd.Series, future_ret : pd.Series) -> float :
    return alpha.corr(future_ret, method='spearman')

def calc_ic(alpha : pd.Series, future_ret : pd.Series) -> float :
    return alpha.corr(future_ret, method='pearson')