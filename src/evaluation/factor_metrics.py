import pandas as pd

def calc_ic_rank(alpha: pd.Series, future_ret: pd.Series) -> float:
    """输入同一个信号日的两列 [股票] 数据：因子分数和未来收益，索引须对应。"""
    return alpha.corr(future_ret, method='spearman')

def calc_ic(alpha: pd.Series, future_ret: pd.Series) -> float:
    """输入同一个信号日、索引对应的分数与未来收益，直接对数值算 Pearson IC。"""
    return alpha.corr(future_ret, method='pearson')
