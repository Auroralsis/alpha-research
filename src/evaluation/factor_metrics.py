import pandas as pd

def calc_ic_rank(mom : pd.Series, future_ret : pd.Series) -> float :
    return mom.corr(future_ret, method='spearman')

def calc_ic(mom : pd.Series, future_ret : pd.Series) -> float :
    return mom.corr(future_ret, method='pearson')