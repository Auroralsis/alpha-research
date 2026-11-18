import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm


def hac_mean(
    values: ArrayLike, lag: int = 20, min_obs: int = 100, min_coverage: float = .8,
) -> dict[str, float | int | str]:
    """计算时间序列均值及其 HAC 稳健统计检验结果。

    输入参数：
        values: 一维数组或序列，通常为按市场交易日升序排列的每日 RankIC。
            每个元素对应一个市场交易日，缺失观测使用 NaN 表示。
            输入应已对齐完整市场交易日历，不能直接删除缺失日期。
        lag: int，HAC 估计考虑的最大滞后交易日数，默认20。
        min_obs: int，进行统计推断所需的最少有效观测数，默认100。
        min_coverage: float，最小有效数据覆盖率，默认0.8。
    """
    values = np.asarray(values, float)
    valid = np.isfinite(values)
    n = int(valid.sum())
    mean = values[valid].mean() if n else np.nan
    result = dict(mean=mean, n_days=n, calendar_days=len(values), hac_lag=lag,
                  hac_se=np.nan, t_hac=np.nan, p_value=np.nan, ci_low=np.nan, ci_high=np.nan,
                  inference_status="insufficient_coverage")
    if n < min_obs or n / max(len(values), 1) < min_coverage or len(values) <= lag:
        return result
    score = np.where(valid, values - mean, 0.)
    # 初始meat就是偏差平方，用于计算方差
    meat = np.dot(score, score)
    # 这里加入时间协方差修正，因为这么多个交易日的IC值是有相关性的，不能够直接计算
    for k in range(1, min(lag, len(values) - 1) + 1):
        meat += 2 * (1 - k / (lag + 1)) * np.dot(score[k:], score[:-k])
    variance = meat / n**2 * n / (n - 1)
    if not np.isfinite(variance) or variance <= 1e-24:
        result["inference_status"] = "degenerate_variance"
        return result
    se = np.sqrt(variance)
    # hac_se是hac方法得到的标准误
    # t_hac是t统计量，越大说明反对真实平均RankIC等于0的统计证据越强，表述是否真实期望会显著偏离0.
    # p_value表述如果真实RankIC是0，那么得到如此偏离结果概率多大，越小说明越不太可能是0
    # ci_low和ci_high给出了测得的区间。长期来看有95%的区间应该包含真实的期望。换句话
    # 只有5%的概率构造的区间不包含真实期望
    result.update(hac_se=se, t_hac=mean / se, p_value=2 * norm.sf(abs(mean / se)),
                  ci_low=mean - norm.ppf(.975) * se, ci_high=mean + norm.ppf(.975) * se,
                  inference_status="ok")
    return result


def holm_adjust(p_values: ArrayLike) -> np.ndarray:
    """输入长度为检验项数的 p 值向量；返回按原顺序排列的校正值。"""
    p = np.asarray(p_values, float)
    if ((p[np.isfinite(p)] < 0) | (p[np.isfinite(p)] > 1)).any():
        raise ValueError("Invalid p value")
    # Unestimable family members remain counted, conservatively, as p=1.
    order = np.argsort(np.where(np.isfinite(p), p, 1), kind="stable")
    adjusted = np.empty(len(p))
    # 计算累计最大值，再将超过1的结果截断为1
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        np.where(np.isfinite(p[order]), p[order], 1) * np.arange(len(p), 0, -1)))
    return np.where(np.isfinite(p), adjusted, np.nan)
