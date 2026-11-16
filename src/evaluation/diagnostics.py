"""Distribution and sample diagnostics; never changes signals or main labels."""
import numpy as np
import pandas as pd

from src.features.huatai import quantile_groups
from .huatai_study import hac_mean, holm_adjust


def distribution(values):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(n_observed=0, **{k: np.nan for k in ["mean", "median", "trimmed_mean", "winsor_mean",
            "p01", "p10", "p90", "p99", "minimum", "maximum", "std", "tail_contribution"]})
    lo, hi = np.quantile(values, [.01, .99])
    trimmed = values[(values >= lo) & (values <= hi)]
    return dict(n_observed=len(values), mean=values.mean(), median=np.median(values),
        trimmed_mean=trimmed.mean(), winsor_mean=np.clip(values, lo, hi).mean(),
        p01=lo, p10=np.quantile(values,.1), p90=np.quantile(values,.9), p99=hi,
        minimum=values.min(), maximum=values.max(), std=values.std(ddof=1) if len(values)>1 else np.nan,
        tail_contribution=values[np.abs(values)>.10].sum()/len(values))


def diagnose(frame, calendar, membership, min_pairs=100):
    """Daily equal-weight summaries; trim at group 1/99 percentiles, descriptive only.

    Observation contributions sum to Q5-Q1 exactly; no stock is removed from the
    original evaluation or portfolio. Missing labels never determine assignment.
    """
    calendar = pd.DatetimeIndex(calendar)
    sets = {d:set(g.ticker) for d,g in membership.groupby("trade_date")}
    changed = pd.Series([sets.get(d,set()) != sets.get(calendar[i-1],set()) if i else False
                         for i,d in enumerate(calendar)], index=calendar)
    stable = changed.rolling(10,min_periods=10).sum().eq(0)
    daily, contributions = [], []
    for day, group in frame.groupby("trade_date",sort=True):
        for factor in ["Alpha13","Alpha44"]:
            if group[factor].notna().sum()<min_pairs:
                continue
            q = pd.Series(quantile_groups(group[factor]),index=group.index)
            for h in [5,20]:
                y = group[f"future_ret_{h}"].where(lambda x: np.isfinite(x))
                paired = np.isfinite(group[factor]) & np.isfinite(y)
                grouped_pairs = q.notna() & np.isfinite(y)
                if not paired.equals(grouped_pairs):
                    raise AssertionError("IC and quantile paired universes differ")
                rx, ry = group.loc[paired,factor].rank(), y[paired].rank()
                sx, sy = rx-rx.mean(), ry-ry.mean()
                denominator = np.sqrt((sx*sx).sum()*(sy*sy).sum())
                ic_parts = sx*sy/denominator if denominator>0 else sx*np.nan
                industry_y = y.groupby(group.industry).transform("mean")
                adjusted = y-industry_y
                for k in range(1,6):
                    assigned = q.eq(k)
                    values = y[assigned]
                    stats = distribution(values)
                    record = dict(trade_date=day,factor=factor,horizon=h,quantile=k,
                        n_assigned=int(assigned.sum()),n_pairs=int(paired.sum()),
                        ic_contribution=ic_parts.reindex(group.index[assigned]).sum(min_count=1),
                        mean_return_rank=ry.reindex(group.index[assigned]).mean()/max(len(ry),1),
                        industry_adjusted_mean=adjusted[assigned].mean(),
                        industry_matched_mean=y[assigned & adjusted.notna()].mean(),
                        median_total_mv=group.loc[assigned,"total_mv"].median(),
                        median_price=group.loc[assigned,"close_raw"].median(),
                        stable_membership=bool(stable.get(day,False)), **stats)
                    daily.append(record)
                    if k in [1,5] and stats["n_observed"]:
                        cells=group.loc[assigned & paired,["ticker"]].copy()
                        cells["contribution"]=y.loc[cells.index]/stats["n_observed"]*(1 if k==5 else -1)
                        cells["trade_date"],cells["factor"],cells["horizon"],cells["quantile"]=day,factor,h,k
                        contributions.append(cells)
    daily=pd.DataFrame(daily)
    rows=[]
    for period in ["all","2023","2024","2025"]:
        subset=daily if period=="all" else daily[daily.trade_date.dt.year.eq(int(period))]
        for (factor,h),g in subset.groupby(["factor","horizon"]):
            for k,q in g.groupby("quantile"):
                r=q.select_dtypes("number").mean().to_dict()
                r.update(analysis="distribution",period=period,factor=factor,horizon=h,quantile=k,n_dates=len(q))
                rows.append(r)
            for sample,mask in [("all",pd.Series(True,index=g.index)),("stable_membership",g.stable_membership)]:
                selected=g[mask]
                metrics=["mean","median","trimmed_mean","winsor_mean","tail_contribution",
                         "industry_adjusted_mean","industry_matched_mean","mean_return_rank"]
                for metric in metrics:
                    p=selected.pivot(index="trade_date",columns="quantile",values=metric).reindex(columns=range(1,6))
                    spread=p[5]-p[1]
                    dates=pd.DatetimeIndex(sorted(subset.trade_date.unique()))
                    stats=dict(mean=spread.mean(),n_days=int(spread.notna().sum()))
                    if period=="all" and sample=="all" and metric=="mean":
                        stats=hac_mean(spread.reindex(dates),lag=20)
                    rows.append(dict(analysis="spread",period=period,sample=sample,factor=factor,horizon=h,
                        metric=metric,positive_ratio=spread.dropna().gt(0).mean(),**stats))
    summary=pd.DataFrame(rows)
    family=summary.analysis.eq("spread") & summary.period.eq("all") & summary["sample"].eq("all") & summary.metric.eq("mean")
    summary.loc[family,"p_holm_diagnostic4"]=holm_adjust(summary.loc[family,"p_value"])
    if not contributions:
        return daily,summary,pd.DataFrame(columns=["factor","horizon","ticker","quantile","contribution","n_labels"])
    cells=pd.concat(contributions,ignore_index=True)
    n_dates=frame.trade_date.nunique()
    stocks=cells.groupby(["factor","horizon","ticker","quantile"]).agg(
        contribution=("contribution","sum"),n_labels=("contribution","size")).reset_index()
    stocks.contribution/=n_dates
    # Keep all stocks: top contributors are views of this table, not new filters.
    return daily,summary,stocks
