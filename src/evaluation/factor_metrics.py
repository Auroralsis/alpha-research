import pandas as pd


def calc_ic_rank(alpha: pd.Series, future_ret: pd.Series, min_pairs: int = 2) -> float:
    """输入同一信号日的股票分数和未来收益，按索引配对后返回 Spearman 相关系数。

    任一列为常数或有效配对数少于 min_pairs 时，返回 NaN。
    """
    if min(len(alpha), len(future_ret)) < min_pairs or alpha.nunique() < 2 or future_ret.nunique() < 2:
        return float("nan")
    return alpha.corr(future_ret, method="spearman", min_periods=min_pairs)
