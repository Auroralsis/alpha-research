"""每次向归档 Alpha158 追加一个特征，用固定八排序器及正式 Qlib 账本评价。"""
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.engine import run_backtests
from src.backtest.portfolio import QLIB_DAILY_ANNUALIZATION, qlib_excess_metrics
from src.data.factor_disclosures import CACHE
from src.data.research_panel import CSI500_INPUTS as INPUTS, load_market_calendar, prepare_test_panel
from src.factors.neutralization import mad_clip
from src.ml.dataset import alpha158_columns
from src.ml.rank_validation import BASES, equal_ranks, training_periods
from src.scripts.run_all import RUNS, new_run
from src.scripts.run_pair_tuning import quantile_matrices, train_candidate
from src.report.research import markdown_table

BASE = RUNS / "daily_pair_tuning_20261122_100328_550844"
FACTORS = ("ABTURN", "SUE", "dROE", "ILLIQ20", "nPLHR")
KEY = ["trade_date", "ticker"]


def cached(kind):
    """读取全历史股票缓存，缺少下载文件时拒绝误用部分样本。"""
    tickers = pd.read_parquet(INPUTS["membership"], columns=["ticker"]).ticker.unique()
    paths = [CACHE / kind / f"{ticker}.parquet" for ticker in tickers]
    frames = (pd.read_parquet(path) for path in paths)
    return pd.concat([frame for frame in frames if len(frame)], ignore_index=True).rename(columns={"ts_code": "ticker"})


def market_factor(name, dates):
    """按市场日历构造单个量价特征，再按历史打分样本取值。"""
    bars = pd.read_parquet(INPUTS["prices"], columns=KEY + ["close", "adj_factor", "vol", "amount", "high", "low"])
    bars.trade_date = pd.to_datetime(bars.trade_date)
    bars = bars.set_index(KEY, verify_integrity=True)
    if name == "ABTURN":
        shares = cached("shares")
        shares.trade_date = pd.to_datetime(shares.trade_date)
        capital = shares.set_index(KEY, verify_integrity=True).total_share.reindex(bars.index)
        # 成交量为手，总股本为万股。
        turn = (bars.vol.where(bars.vol.ge(0)) / (capital.where(capital.gt(0)) * 100)).unstack().reindex(dates)
        factor = turn.rolling(20).mean() / turn.rolling(250).mean().replace(0, np.nan)
    elif name == "ILLIQ20":
        close = (bars.close * bars.adj_factor).where(bars.vol.gt(0) & bars.close.gt(0) & bars.adj_factor.gt(0))
        ret = close.unstack().reindex(dates).pct_change(fill_method=None)
        yuan = (bars.amount * 1000).where(bars.amount.gt(0)).unstack().reindex(dates)
        factor = (ret.abs() / yuan).rolling(20, min_periods=1).mean()
        factor.iloc[:19] = np.nan
    else:
        limits = cached("limit")
        limits.trade_date = pd.to_datetime(limits.trade_date)
        b = bars.join(limits.set_index(KEY, verify_integrity=True), validate="one_to_one")
        traded = b.vol.gt(0)
        valid = b.up_limit.gt(0) & b.down_limit.gt(0) & b.up_limit.gt(b.down_limit)
        # 三个月区间为 (T - 3 个自然月, T]。无成交日不记触及。
        def counts(flags):
            cumulative = flags.astype("int8").unstack(fill_value=0).reindex(dates, fill_value=0).cumsum()
            prior = cumulative.reindex(dates - pd.DateOffset(months=3), method="ffill").fillna(0)
            return cumulative - prior.set_axis(dates)
        up = counts(traded & valid & b.high.ge(b.up_limit - 1e-8))
        down = counts(traded & valid & b.low.le(b.down_limit + 1e-8))
        unknown = counts(traded & ~valid)
        factor = ((up - down) / (up + down).replace(0, np.nan)).where(unknown.eq(0))
        factor.loc[dates < dates[0] + pd.DateOffset(months=3)] = np.nan
    return factor.rename_axis(index="trade_date", columns="ticker").stack(future_stack=True).dropna().rename(name)


def statement_events(frame, value):
    """按实际披露日期保留各期版本。同日同版本类型中有冲突的数值留空。"""
    rows = frame.loc[frame.end_date.notna() & (frame.f_ann_date.notna() | frame.ann_date.notna())].reset_index(drop=True)
    rows["published"] = pd.to_datetime(rows.f_ann_date)
    rows["published"] = pd.concat([rows.published, pd.to_datetime(rows.ann_date)], axis=1).max(axis=1)
    quarter = pd.to_datetime(rows.end_date)
    rows = rows.loc[quarter.dt.is_quarter_end & quarter.le(rows.published)].copy()
    rows["quarter"] = quarter.loc[rows.index].dt.to_period("Q").astype("int64")
    keys = ["ticker", "published", "quarter"]
    # 同日优先明确更正的版本，其次普通版，最后调整前报表。
    rows["priority"] = rows.report_type.astype(str).map({"2": 1, "3": 2, "5": 0, "1": 1, "4": 2})
    rows = rows.loc[rows.priority.eq(rows.groupby(keys).priority.transform("max"))]
    stats = rows.groupby(keys)[value].agg(["first", "nunique"])
    return stats["first"].where(stats["nunique"].eq(1)).rename(value).reset_index()


def financial_factor(name, dates):
    """在每次披露时更新当时已知季度，显式重算追溯修订，仅向下个交易日传播。"""
    statements = cached("income")
    if name == "SUE":
        # 按每条指标记录的公告版本匹配实际日期，同一版本有多个披露日时延后使用。
        keys = ["ticker", "end_date", "ann_date"]
        actual = statements.groupby(keys).f_ann_date.max()
        eps = cached("eps").join(actual, on=keys, validate="many_to_one")
        eps["f_ann_date"] = eps.f_ann_date.fillna(eps.ann_date)
        statements = eps.assign(report_type="2")
    field = "q_eps" if name == "SUE" else "n_income_attr_p"
    income = statement_events(statements, field).assign(kind="income").rename(columns={field: "value"})
    if name == "dROE":
        equity = "total_hldr_eqy_exc_min_int"
        balance = statement_events(cached("equity"), equity).assign(kind="equity").rename(columns={equity: "value"})
        income = pd.concat([income, balance], ignore_index=True)
    series = {}
    for ticker, records in income.groupby("ticker"):
        state = {"income": {}, "equity": {}}
        updates = {}
        for day, changes in records.groupby("published", sort=True):
            for row in changes.itertuples():
                state[row.kind][row.quarter] = row.value
            if not state["income"]:
                continue
            q = max(state["income"])
            def metric(period):
                v = state["income"].get(period, np.nan)
                if name == "SUE":
                    return v - state["income"].get(period - 4, np.nan)
                capital = state["equity"].get(period - 1, np.nan)
                return v / capital if pd.notna(capital) and capital != 0 else np.nan
            if name == "SUE":
                past = np.array([metric(q-i) for i in range(1, 9)])
                usable = past[np.isfinite(past)]
                sd = usable.std(ddof=1) if len(usable) >= 6 else np.nan
                value = metric(q) / sd if sd > 0 and np.ptp(usable) > 0 else np.nan
            else:
                value = metric(q) - metric(q-4)
            pos = dates.searchsorted(day, side="right")
            if pos < len(dates):
                updates[dates[pos]] = value
        if updates:
            series[ticker] = pd.Series(updates).sort_index().reindex(dates, method="ffill")
    return pd.DataFrame(series).rename_axis(index="trade_date", columns="ticker").stack(future_stack=True).dropna().rename(name)


def report(out):
    """从基线和已完成的真实账本计算分年与全期相同口径的指标。"""
    summaries = []
    for name, path in [("Alpha158", BASE), *((n, out/n) for n in FACTORS if (out/n/"ledger.parquet").exists())]:
        ledger = pd.read_parquet(path/"ledger.parquet")
        if name == "Alpha158":
            ledger = ledger.loc[ledger.strategy.eq("blend_equal")]
        ledger = ledger.set_index("trade_date", verify_integrity=True).sort_index()
        for period, rows in [("全期", ledger), *ledger.groupby(ledger.index.year)]:
            metrics = qlib_excess_metrics(rows.daily_return, rows.benchmark_return)
            summaries.append(dict(factor=name, period=period, **metrics,
                                  turnover=rows.turnover_two_sided.mean()*QLIB_DAILY_ANNUALIZATION/2))
    summary = pd.DataFrame(summaries)
    table = markdown_table(["第159列", "期间", "扣费年化超额", "IR", "最大超额回撤", "年化单边换手"],
                           [(r.factor,r.period,f"{r.annualized_excess_return:.2%}",f"{r.information_ratio:.3f}",
                             f"{r.max_excess_drawdown:.2%}",f"{r.turnover:.2f} 倍") for r in summary.itertuples()])
    coverage = pd.concat([pd.read_csv(p) for p in sorted(out.glob("*/coverage.csv"))],ignore_index=True)
    coverage.to_csv(out/"coverage.csv",index=False)
    cov = coverage.pivot(index="year",columns="factor",values="coverage")
    cov_table = markdown_table(["年份",*cov.columns], [(year,*(f"{v:.2%}" for v in row)) for year,row in cov.iterrows()])
    baselines = summary.loc[summary.factor.eq("Alpha158")].set_index("period")
    excess = summary.annualized_excess_return - summary.period.map(baselines.annualized_excess_return)
    summary.assign(excess_vs_baseline=excess).to_csv(out/"comparison.csv", index=False)
    totals = summary.loc[summary.period.eq("全期") & summary.factor.ne("Alpha158")]
    leader = totals.loc[totals.annualized_excess_return.idxmax()]
    delta_pp = 100 * (leader.annualized_excess_return - baselines.at["全期", "annualized_excess_return"])
    finding = (f"已完成的 {len(totals)} 个单列实验中，{leader.factor} 的全期扣费年化超额最高，为 "
               f"**{leader.annualized_excess_return:.2%}**，与 Alpha158 相差 **{delta_pp:+.2f} 个百分点**。"
               "2023–2025 年已多次用于模型对照，结果反映该历史区间的表现。")
    note = """## 因子和数据口径

- ABTURN 用成交手数与 TuShare 每日总股本（万股）换算日换手率。20 日与 250 日均值均含 T，按完整市场日历窗口计算。停牌有明确零成交记录时计零，缺记录留空。行情从 2013 年 9 月起，2014 年前段仍在预热。
- SUE 用 TuShare fina_indicator 的单季 q_eps 的同比差除以前八个季度同比差的样本标准差，至少六个有效值。按每条指标公告匹配利润表实际披露日期，只有公告日期的版本也从下一交易日起使用。
- ΔROE 用单季合并归母净利润除以上季末归母股东权益，再取同比差。分母为零留空。
- 财报采用 TuShare 逐股利润表类型 2、3 与资产负债表类型 1、4、5；在各自实际公告日后才更新对应季度。版本公布日取实际披露日与公告日中较晚者，只有公告日期时从公告后的首个交易日起使用，两个日期都缺失时留空。同日同优先级的冲突值留空。该处理依赖供应商保留的版本及时间戳，缺少独立的历史数据库快照认证。
- ILLIQ20 按整段 20 个市场日内的有效日取均值，以相邻市场日复权收盘计算绝对收益，成交额由千元换为元。
- nPLHR 采用 (T−3 个自然月, T]。以原始最高、最低价和当日实际限价计数，每个方向每天最多计一次。零触及留空，有成交却缺有效限价的窗口留空。
- 所有分支从归档的同一张 MAD Alpha158 特征表读入。个别新增列值缺失也保留原有样本。新增一列会改变树模型的列抽样路径，同种子实验的差额衡量整套模型的变化。

[量价因子来源](https://rodneywhitecenter.wharton.upenn.edu/wp-content/uploads/2018/10/01-18.Stambaugh.pdf) · [财报因子附录](https://pubsonline.informs.org/doi/suppl/10.1287/mnsc.2023.4904/suppl_file/mnsc.2023.4904.sm1.pdf) · [限价研究](https://doi.org/10.1016/j.frl.2023.104803)
"""
    (out/"report.md").write_text("# Alpha158 单列增量实验\n\n2014–2020 年拟合，2021 年早停，沿用 2022 年确定的参数。"
        "固定八排序器按逐日排名百分位等权，2023–2025 年历史成分，复权收盘交易，Top50 每日计划替换 5 只，买 5 bp、卖 15 bp。"
        "新增列单独按当日模型样本做 5 MAD 截尾，缺失保留。费用后日收益减价格指数日收益按 Qlib sum 模式计算，年化基数 238。\n\n"
        +finding+"\n\n表 1　回测绩效\n\n"+table+"\n\n表 2　逐年新增列覆盖率（有效值数 / 原 Alpha158 股票日数）\n\n"+cov_table+"\n\n"+note)



def run(name, out, cfg, winners, dates):
    folder = out/name
    folder.mkdir(exist_ok=True)
    source = Path(cfg["feature_source"])
    keys = pd.read_parquet(source/"features.parquet", columns=KEY)
    raw = financial_factor(name, dates) if name in ("SUE","dROE") else market_factor(name,dates)
    values = keys.join(raw,on=KEY,validate="one_to_one")
    values[name] = values[name].replace([np.inf,-np.inf],np.nan)
    coverage = values.groupby(values.trade_date.dt.year)[name].agg(["count","size"]).rename_axis("year")
    coverage.assign(coverage=coverage["count"]/coverage["size"],factor=name).to_csv(folder/"coverage.csv")
    def clip_day(day):
        return mad_clip(day) if day.notna().any() else day
    values[name] = values.groupby("trade_date")[name].transform(clip_day).astype("float32")
    factor_path = folder / "factor.parquet"
    if factor_path.exists():
        pd.testing.assert_frame_equal(values, pd.read_parquet(factor_path), check_exact=True)
    else:
        values.to_parquet(factor_path, index=False)
    train_feature_set(folder, name, name, values.set_index(KEY)[name], alpha158_columns(source), cfg, winners, dates)
    report(out)


def train_feature_set(folder, strategy, extra_name, values, features, cfg, winners, dates):
    """用指定列重训八个排序器。保持原分组标签、列顺序和交易规则。"""
    source = Path(cfg["feature_source"])
    def matrix(rows):
        x = rows[features].copy()
        x[extra_name] = values.reindex(pd.MultiIndex.from_frame(rows[KEY])).to_numpy()
        return x
    blocks = []
    for _, rows, groups in training_periods(source,Path(cfg["reference"]),cfg["segments"]):
        blocks.append((matrix(rows),rows.grade.to_numpy(dtype=int),groups))
        del rows
    if blocks[0][0][extra_name].nunique() < 2:
        raise ValueError(f"{extra_name} 缺少训练数据")
    lo, hi = map(pd.Timestamp,cfg["segments"]["test"])
    test = pd.read_parquet(source/"features.parquet",columns=[*KEY,*features],
                           filters=[("trade_date",">=",lo),("trade_date","<=",hi)]).sort_values(KEY)
    scores = test[KEY].copy()
    xt = matrix(test)
    del test, values
    for family, params in winners.items():
        mats = quantile_matrices(blocks,params) if family=="xgboost" else []
        predictions = train_candidate(folder,family,params,cfg,blocks,mats,xt)
        scores[predictions.add_prefix(family+"_").columns] = predictions.to_numpy()
        scores[family] = equal_ranks(scores,[family+"_"+t for t in BASES])
        del mats
    raw_columns = [family + "_" + target for family in winners for target in BASES]
    if not np.isfinite(scores[raw_columns]).all().all():
        raise ValueError("八个排序器应为每个测试股票日生成有限分数")
    archived = pd.read_parquet(BASE/"predictions.parquet", columns=KEY, filters=[("split", "==", "test")])
    pd.testing.assert_index_equal(pd.MultiIndex.from_frame(scores[KEY]).sort_values(),
                                  pd.MultiIndex.from_frame(archived).sort_values())
    scores[strategy] = (scores.lightgbm + scores.xgboost)/2
    scores.to_parquet(folder/"predictions.parquet",index=False)
    del blocks, xt
    gc.collect()
    panel, test_days = prepare_test_panel(cfg,dates)
    run_backtests(INPUTS,cfg,test_days,scores,panel,folder,strategies=(strategy,))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path)
    parser.add_argument("factors", nargs="*", default=["nPLHR"])
    args = parser.parse_args()
    if set(args.factors)-set(FACTORS):
        parser.error("可选因子 " + ", ".join(FACTORS))
    out = args.out or new_run(1,tag="alpha159")
    out.mkdir(parents=True,exist_ok=True)
    cfg = json.loads((BASE/"config.json").read_text())
    winners = json.loads((BASE/"winners.json").read_text())
    config_path = out / "config.json"
    frozen = {**cfg, **winners}
    if config_path.exists() and json.loads(config_path.read_text()) != frozen:
        raise ValueError("当前参数与该结果目录的存档不同，请使用新的输出目录")
    config_path.write_text(json.dumps(frozen, indent=2))
    dates = load_market_calendar(INPUTS["calendar"])
    for name in args.factors:
        if not (out/name/"ledger.parquet").exists():
            print(name, out, flush=True)
            run(name,out,cfg,winners,dates)
    report(out)


if __name__ == "__main__":
    main()
