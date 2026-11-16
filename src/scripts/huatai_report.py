"""The study's four figures, two reports, and reproducibility checks."""
import hashlib
import importlib.util
import json
import platform
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow
import scipy


def table(frame):
    def cell(value):
        if pd.isna(value):
            return "—"
        return f"{value:.4g}" if isinstance(value,(float,np.floating)) else str(value)
    return "\n".join(["| "+" | ".join(frame.columns)+" |","| "+" | ".join(["---"]*len(frame.columns))+" |"]+
                     ["| "+" | ".join(cell(v) for v in row)+" |" for row in frame.itertuples(index=False,name=None)])


def plot_results(out, benchmark):
    ic=pd.read_parquet(out/"details/daily_ic.parquet")
    ic=ic[ic.specification.eq("primary") & ic.horizon.eq(5)]
    fig,axes=plt.subplots(2,3,figsize=(13,6),constrained_layout=True)
    for row,factor in enumerate(["Alpha13","Alpha44"]):
        s=ic[ic.factor.eq(factor)].set_index("trade_date").rank_ic
        for ax,series,title in zip(axes[row],[s,s.cumsum(),s.rolling(60,min_periods=60).mean()],
                                   ["daily IC","cumulative IC","60-market-day IC"]):
            ax.plot(series.index,series,linewidth=.7);ax.axhline(0,color="gray",linewidth=.4)
            ax.set_title(f"{factor}: {title}")
    fig.savefig(out/"figures/ic_stability.svg");plt.close(fig)
    d=pd.read_csv(out/"diagnostics.csv",dtype={"period":str})
    dist=d[d.analysis.eq("distribution") & d.period.eq("all") & d.horizon.eq(5)]
    fig,axes=plt.subplots(2,2,figsize=(11,6),constrained_layout=True)
    for col,factor in enumerate(["Alpha13","Alpha44"]):
        g=dist[dist.factor.eq(factor)].set_index("quantile")
        for metric in ["mean","median","trimmed_mean"]:
            axes[0,col].plot(g.index,g[metric]*10000,marker="o",label=metric)
        axes[0,col].set_title(f"{factor}: 5d returns (bp)");axes[0,col].legend()
        for metric in ["p90","p99"]:
            axes[1,col].plot(g.index,g[metric]*100,marker="o",label=metric)
        axes[1,col].set_title("Mean daily upper return quantiles (%)");axes[1,col].legend()
    fig.savefig(out/"figures/distributions.svg");plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,6),constrained_layout=True)
    spreads=d[d.analysis.eq("spread") & d["sample"].eq("all") & ~d.period.eq("all")]
    for row,factor in enumerate(["Alpha13","Alpha44"]):
        for col,h in enumerate([5,20]):
            g=spreads[spreads.factor.eq(factor)&spreads.horizon.eq(h)&spreads.metric.isin(["mean","median","trimmed_mean"])]
            (g.pivot(index="period",columns="metric",values="mean")*10000).plot.bar(ax=axes[row,col])
            axes[row,col].set_title(f"{factor}: {h}d Q5-Q1 (bp)")
    fig.savefig(out/"figures/annual_spreads.svg");plt.close(fig)
    ledgers=pd.read_parquet(out/"details/portfolio.parquet")
    fig,ax=plt.subplots(figsize=(11,4),constrained_layout=True)
    for (h,bps),g in ledgers.groupby(["holding_days","fee_bps"]):
        if h==20 or bps==15:
            ax.plot(g.trade_date,g.nav,label=f"{h}d / {bps}bp",linestyle="--" if h==5 else "-")
    dates=pd.DatetimeIndex(sorted(ledgers.trade_date.unique()));b=benchmark.reindex(dates)
    ax.plot(dates,b/b.iloc[0],label="CSI500 price index",color="black")
    ax.set_title("Alpha13: simplified next-open ledger; offset 0");ax.legend(ncol=3)
    fig.savefig(out/"figures/portfolio.svg");plt.close(fig)


def write_reports(root,out):
    ic=pd.read_csv(out/"ic_summary.csv",dtype={"period":str})
    diag=pd.read_csv(out/"diagnostics.csv",dtype={"period":str})
    panel=pd.read_parquet(out/"details/research_panel.parquet")
    portfolio=pd.read_csv(out/"portfolio_summary.csv")
    primary=ic[ic.analysis.eq("primary")].copy()
    main=primary[["factor","horizon","mean","icir","p_value","p_holm"]].rename(columns={"mean":"Mean IC (%)"})
    main["Mean IC (%)"]*=100
    spreads=diag[diag.analysis.eq("spread") & diag["sample"].eq("all")]
    central=spreads[spreads.period.eq("all")].pivot(index=["factor","horizon"],columns="metric",values="mean")
    central=central[["mean","median","trimmed_mean","winsor_mean"]]*10000
    years=spreads[~spreads.period.eq("all") & spreads.horizon.eq(5)].pivot(index=["factor","period"],columns="metric",values="mean")
    years=years[["mean","median","trimmed_mean"]]*10000
    yearly_ic=ic[ic.analysis.eq("yearly")&ic.horizon.eq(5)].set_index(["factor","period"])["mean"]*100
    years["Mean IC (%)"]=yearly_ic
    dist=diag[diag.analysis.eq("distribution")&diag.period.eq("all")&diag.horizon.eq(5)&diag["quantile"].isin([1,5])].copy()
    for c in ["mean","median"]:dist[c]*=10000
    dist["p99"]*=100;dist["median_total_mv"]/=10000
    dist=dist[["factor","quantile","mean","median","p99","median_total_mv"]]
    mean_spread=spreads[spreads.period.eq("all")&spreads.metric.eq("mean")]
    influence=pd.read_csv(out/"influential_stocks.csv")
    top=influence[influence.factor.eq("Alpha13")&influence.horizon.eq(5)].sort_values("contribution").head(3).copy()
    top["contribution"]*=10000
    residual=ic[ic.analysis.eq("incremental") & ic.metric.eq("residual_ic")][["factor","horizon","mean","icir","p_value","p_holm_expanded"]]
    neut=ic[ic.analysis.eq("neutralization")&ic.horizon.eq(5)&ic.factor.str.startswith(("Alpha13","Alpha44"))][["factor","mean","icir"]]
    costs=portfolio[portfolio.offset.eq(0)][["holding_days","fee_bps","annual_return","annual_volatility","sharpe","max_drawdown","relative_total_return"]].copy()
    for c in ["annual_return","annual_volatility","max_drawdown","relative_total_return"]:costs[c]*=100
    p=portfolio[portfolio.holding_days.eq(20)&portfolio.offset.eq(0)&portfolio.fee_bps.eq(15)].iloc[0]
    matched=spreads[spreads.period.eq("all")&spreads.horizon.eq(5)&spreads.metric.isin(["industry_adjusted_mean","industry_matched_mean","tail_contribution"])].copy()
    matched["mean"]*=10000
    ntests=re.search(r"Ran (\d+) tests",(out/"evidence/tests.txt").read_text()).group(1)
    report=f"""# 华泰量价因子：近期有效性、收益分布与经济可实现性

研究题目：*Revisiting Price–Volume Alpha Factors in China's A-Share Market*。诊断与重构日期：2026-11-12（Asia/Shanghai）。

## 1. Research Background

本研究核对并复现[华泰2019年原研报第5、6、32页](https://crm.htsc.com.cn/doc/2019/10750101/dae9cd9e-c2cd-4f05-bb6c-e5ac8d216d3d.pdf#page=6)的Alpha3、13、44，另与[Kakushadze作者预印本](https://arxiv.org/pdf/1601.00991v3)第8、9、11、15—16页对照。原报告是2010—2019年全A，本项目是2023—2025动态中证500的近期验证，不是原组合曲线完整复现。

本轮不改公式方向、窗口和主实验参数，重点判断“正RankIC为何没有带来更高组均值”，并核查成本后组合是否有经济价值。完整阴性结果保留；诊断性截尾不能替代主结果。

## 2. Methodology

原始行情约280万行、1263只历史标识，2015-09-01—2026-02-27。工作网格从2022-12-01起，保留入选前、退出后行情；股票日期唯一、升序，缺行不补为正常成交。复权锚点固定为每股最早历史观测，收益标签和MOM20用复权价格，Alpha主规格用原始OHLC和原vol。

```text
Alpha3  = -correlation(rank(OPEN), rank(VOLUME), 10)
Alpha13 = -rank(covariance(rank(CLOSE), rank(VOLUME), 5))
Alpha44 = -correlation(HIGH, rank(VOLUME), 5)
MOM20   = Close_adj(t) / Close_adj(t-20) - 1
```

rank为平均并列秩/N；相关、协方差沿单股市场日时间轴计算，包含当日，要求完整10/5/5日窗口，协方差ddof=1，零方差相关缺失。Alpha13外层再次截面排序，Alpha44的HIGH不rank。信号日t的成员集合M_t作为窗口内固定排名参照，是自主设计；另完整保存M_u历史日排名、复权价格及共同因子支持集的敏感性结果。

会员快照不晚于当日，保留来源日期，但缺少公告可知时间及完整生效表，仍为近似PIT。名义期727交易日、363,500个成员证券日，其中{panel.observation_status.ne('observed').sum()}个信号日记录无有效行情。本地独立日历止于2025年末：8项主比较统一用2023-01-03—2025-12-03的707个信号日；各期限最大可用结果另存。标签是确切市场日t至t+5/t+20收盘收益，绝不以第h条股票记录替代；也不以未来标签筛选事前资格。

每日Spearman IC至少100个配对，ICIR不年化。主均值推断采用HAC20、双侧检验、95%区间，8项Holm校正；HAC60及固定60日块/500次重采样为预设辅助分析。保留缺失日期的市场日位置。五组在连接标签前按因子确定，日间等权，Q1低值、Q5高值。重叠标签收益从未复利为净值。

## 3. Main Results

{table(main)}

Alpha13、Alpha44的5日Mean IC在8项Holm校正后显著；20日均未达到0.05。正IC是排序证据，不是收益大小或组合超额收益的保证。MOM20保留原始正动量方向，即使其IC为负也没有取反。

![IC稳定性](../results/huatai_alpha/figures/ic_stability.svg)

## 4. Diagnostics

### 4.1 先排除方向、分组和样本错误

三公式与独立标量参考一致；Q5确实是较高因子值，Alpha13负值中接近0的值较高。每日IC使用的有限因子/标签股票集合，等于五组所有有效配对的并集；组内缺失标签只影响可观测均值，不改变事前组别。逐日组内IC贡献加总还原全截面IC，组内有效数加总还原IC配对数。IC覆盖整个截面，Q5−Q1只比较两端，这种统计目标差异不是程序样本错位。

### 4.2 排序较好，但Q1右尾抬高均值

以下均为“逐日先算组统计，再对日期等权”的Q5−Q1，单位bp；median不是全期混合样本的中位数，trimmed_mean按组内1%/99%分位点截尾，winsor_mean将尾部压至该边界。它们仅用于事后诊断，不用于重新选股。

{table(central.reset_index())}

Alpha13的5日原始价差−1.64bp，而中位数差+25.14bp、截尾差+3.15bp、缩尾差+0.85bp。典型股票的排序和少数大幅上涨股票对算术均值的影响可以相反。1%/99%诊断不证明剔除这些股票后存在可交易收益。

下表mean、median为bp，p99为百分数，中位总市值为亿元（原字段万元）；各项都是日内统计的时间均值。

{table(dist)}

Alpha13 Q1的平均日内99%收益分位约15.69%，高于Q5的12.15%；其右尾更强。两端中位市值接近，不能把现象直接归因于“小盘低价彩票”。按绝对5日收益>10%分解，Alpha13的尾部对价差贡献约−20.45bp，Alpha44约−13.34bp；非尾部贡献都约+18.7bp。两因子总价差相差约6.92bp，主要体现尾部贡献差异，而非IC符号或均值计算方向不同。

{table(top[['ticker','quantile','contribution','n_labels']])}

上表是Alpha13/5日最负的三个个股贡献，单位bp，贡献按每期组内等权分母计算、再对707日平均；所有股票贡献的和严格等于原始价差。它们是Q1中上涨带来的负贡献，不是坏数据认定。重叠标签会重复覆盖同一行情事件，n_labels不是独立事件数；未据此排除这些股票或改策略。

![组分布诊断](../results/huatai_alpha/figures/distributions.svg)

### 4.3 年度、调样与行业：结论对环境敏感

{table(years.reset_index())}

Alpha13的5日价差在2023为+7.20bp，2024/2025分别−5.58/−6.96bp；Alpha44为−2.35/+3.49/+15.49bp。Alpha44改善并不等于三个年度均稳定：20日原始价差也主要由2025支撑。

{table(mean_spread[['factor','horizon','positive_ratio','p_value','p_holm_diagnostic4']])}

Alpha13、44的5日价差正值比例约49.1%/51.5%，HAC原始p值约0.81/0.42；两者算术价差均无可靠显著性。表中仅四项原始价差检验采用诊断族Holm4，截尾、分年和稳定成员窗口不作为筛选显著结果的检验。

在过去10日成员集合没有变化的648个日期，Alpha13/5日价差约+0.25bp，原全样本−1.64bp：59个调样邻近日期影响了整体符号。但这种事后划分也对应不同市场时期，不能证明指数调样造成了收益差，更不能只保留“稳定期”冒充主结果。

{table(matched[['factor','metric','mean']])}

行业内收益去均值仅作归因诊断：同一行业数据可用样本的Alpha13原始价差约−1.68bp，去行业后约+4.08bp，说明行业暴露也有影响；不证明因果或显著盈利。行业是历史申万有效区间，不是当前行业回填。

{table(neut)}

同样本MAD→行业及对数市值OLS残差→标准化后，5日ICIR提高而Mean IC下降，说明主要是波动下降，不能称为所有指标全面改善。未知行业、冲突区间和无效市值保留缺失。组收益的中性化/原始同样本比较完整保存在quantile_summary.csv。

![分年价差](../results/huatai_alpha/figures/annual_spreads.svg)

### 4.4 MOM20以外的条件信息

{table(residual)}

在t的共同有效因子样本上，用MOM20秩残差化Alpha秩，再连接未来收益；残差计算不看标签。两个Alpha的5日残差IC仍为正，但比原始IC低。原始p与扩展112项均值检验校正p同时保留；20日证据较弱。标准化收益回归的Alpha系数未给出同等有力证据。残差秩相关、算术回归系数和净收益不是同一问题，也不是因果识别。

## 5. Portfolio Analysis

主策略仍为事先指定Alpha13/Q5、20市场日调仓、偏移0，t收盘信号在t+1开盘执行；5日方案只是对既定5/20日期限的完整诊断。所有0/10/15/20bp情形及5/20日的所有起始偏移均保存，未挑最好参数。组合覆盖727市场日，与707日收盘标签主统计不是同一个收益定义。

{table(costs)}

收益、波动、回撤及相对收益单位为%；Sharpe无风险利率0。15bp/20日主结果年化{p.annual_return:.2%}、最大回撤{p.max_drawdown:.2%}、相对中证500价格指数累计{p.relative_total_return:.2%}。费用按实际买卖金额双边扣除，年化单边换手约9.8倍。0费年化11.17%下降到15bp的7.96%，无法据此宣称稳健净超额收益。

主20日组合应对应20日预测证据，不能用更显著的5日IC为其背书。匹配5日的方案在15bp下年化约−1.74%，年化单边换手约37.7倍；更强的短期排序证据没有抵消更频繁交易的费用。相对累计收益定义为策略净值/同期归一化价格指数−1，非两个累计收益率之差；Sharpe用算术日均收益、年化收益用几何复利，二者符号不必一致。

开盘涨停不买、跌停不卖，无报价或缺官方涨跌停价不成交；受阻退出以后逐日重试，未成交新买单不补买。退出指数的持仓在下次定期调仓处理。每日保留现金、持仓、成交与费用，断言交易前后价值差恰为费用；成交记录保留原信号日期及执行价格/单位变化以独立对账。

本轮修复两个边界：开盘报价从原始开盘价与复权因子单独验证，不再因当日高低收盘字段异常而拒单；持有股票缺收盘价时，先用当日已知开盘价、再用历史价格估值。这些修复没有改变本真实样本的核心结果，原因是实际缺失日整行无报价，而非仅收盘字段异常。合成测试证实修复避免了未来收盘信息干扰开盘决策及忽略最新可得价格。

这是分数复权单位的简化账本，未精确模拟竞价流动性、整手、冲击、公司行动现金流和退市结算。期间存在陈旧估值，期末不强制清仓；价格指数与股票复权收益也非完全同口径总收益比较。15bp结果是模型结果，不是精确可执行收益。

最终验收按原始行情和127,276笔成交独立重建八套账本，净值最大误差约1.9e-14。20日/15bp主组合在2024-09-30、10-08、10-09分别约+9.53%、+5.63%、−6.67%，本地中证500分别+10.41%、+7.18%、−6.82%；对应指数点位或涨跌幅与[首创证券公告](https://www.hunan-bank.com/96599/attachDir/2024/10/2024101009175482122.pdf)、[东吴期货原观点](https://www.cls.cn/detail/1819394)、[银河期货原报告检索表格](https://www.yhqh.com.cn/upload/cn/file/2024-10/col17/1728549751469.pdf)的抽查一致。波动来自持仓收益而非这三日的陈旧估值；主动组合与指数不必同幅。该核对不认证真实成交、全部公司行动或退市清算，细项见[验收证据](../results/huatai_alpha/evidence/final_acceptance.json)。

![组合净值](../results/huatai_alpha/figures/portfolio.svg)

## 6. Conclusions & Limitations

1. 正IC与负组均值价差并非方向或样本错位。Alpha13典型收益排序较好，但Q1的强上涨右尾、部分股票贡献及年度/行业/调样窗口暴露抬高均值。
2. Alpha44尾部拖累较轻，组均值优于Alpha13，但原始价差不显著，年度也非全面稳定；不能把较好点估计当作稳健投资优势。
3. MOM20残差分析提供有限条件排序证据，未同时证明因果信息、强算术收益回归或扣费可实现超额。
4. 当前股票池和行业仍近似PIT、代码可能被回溯统一；标签缺失可能非随机。2024曾用于实验，整个区间不能称完全未接触OOS。原价格输入存在除权台阶，华泰具体调整口径未完全说明。
5. 三年样本、重叠标签、结构变化、文献筛选和多重检验限制推断；保留不显著和负结果。缺报价估值、退市结算及价格指数口径限制组合结论。

### 简短附录：复现与证据

项目根目录运行`python -B -m unittest discover -s tests -v`及`python -B -m src.scripts.run_huatai_research`，后者一次生成所有统计、图表、两份报告与校验。详见[README](../README.md)。所有固定输入和辅助数据均保留，主配置未改。

自动化测试{ntests}项；三真实日期共4,500个独立标量对照；重构前后的IC/ICIR、分组和20日组合核心输出逐项比较。IC与收益指标容差1e-10；旧CSV仅12位有效数字，计数均值的舍入差允许1e-9，不属于研究结果变化。结果和比较详情在[manifest.json](../results/huatai_alpha/manifest.json)。详细面板、每日IC/分组、收益贡献与交易账本集中于details/；输入来源、原始请求日志、旧证据和基线集中于evidence/，未丢弃不可再生的获取记录。

原已提交的事前研究方案已逐字归档到[原始证据](../results/huatai_alpha/evidence/provenance.json)的original_research_design，保留原路径和SHA-256；不再另留过程性Markdown。公式与本项目股票池、排名集合、五组、残差分析及简化组合的边界保持明确。
"""
    (root/"docs/huatai_alpha_research_report.md").write_text(report)
    english=f"""# Research Summary

**Revisiting Price–Volume Alpha Factors in China's A-Share Market**

This project reproduces Alpha3, Alpha13 and Alpha44 from Huatai Securities' 2019 research and examines their predictive properties in recent Chinese equities. The original formulas, signs and lookback windows are preserved. The study emphasizes empirical replication, distributional diagnosis and reproducibility rather than maximizing historical returns.

Approximately 2.8 million daily records cover 1,263 historical CSI 500 securities. Dynamic membership reconstructed from monthly snapshots is explicitly treated as approximate point-in-time information. Signals span 2023–2025; the principal five- and twenty-market-day comparisons use 707 common dates through December 3, 2025. Prices before index entry and after exit remain available, and market-calendar endpoints determine future returns. Missing quotes are not converted into ordinary trading observations or zero-return labels.

Alpha13's five-day mean RankIC is approximately 0.0199, with a nonannualized ICIR of 0.258. Alpha44's corresponding values are approximately 0.0214 and 0.248. Both five-day mean-IC tests survive the prespecified eight-test Holm correction, whereas twenty-day evidence is weaker. However, positive rank correlation does not guarantee increasing arithmetic portfolio returns.

Independent checks find no reversed sign, incorrect quantile order or mismatch between the observable IC sample and the union of quantile samples. Alpha13's five-day Q5-minus-Q1 arithmetic return spread is −1.64 basis points, compared with a positive daily-median spread of 25.14 basis points and a 1st/99th-percentile trimmed spread of 3.15 basis points. Strong positive-return tails in Q1 raise its arithmetic mean. Individual stock contributions, annual variation, industry exposures and index-reconstitution windows also matter. These are diagnostic decompositions, not retrospective stock exclusions. Alpha44's positive 5.28-basis-point spread is less adversely affected by tails, but its spread is not statistically robust and is not positive in every year.

Residualizing Alpha ranks against contemporaneous MOM20 ranks leaves positive five-day conditional ranking evidence. This does not establish causality, nonlinear independence or reliably tradable incremental profits. All negative and nonsignificant findings are retained.

The prespecified Alpha13 twenty-day long-only illustration returns approximately 7.96% annualized under a 15-basis-point fee per side, with −0.49% cumulative relative performance against the CSI 500 price index. Complete five-day and twenty-day diagnostic comparisons are reported without selecting an optimal strategy. The simplified next-open ledger blocks missing quotes and opening price limits, charges actual modeled trades and reconciles cash and holdings. Auction liquidity, exact corporate-action cash flows and delisting settlements remain unresolved.

Tests, independent scalar calculations, preserved input hashes and numerical comparisons before and after refactoring support reproducibility. Approximate information timing, previously explored 2024 data, missing outcomes and the short sample remain important limitations. The project illustrates why formula correctness, statistical ranking ability and economic investment value require separate verification.
"""
    (root/"docs/huatai_alpha_english_summary.md").write_text(english)


def validate_results(root,out):
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    provenance=json.loads((out/"evidence/provenance.json").read_text())
    for name,expected in provenance["original_input_hashes"].items():
        assert digest(root/name)==expected,f"Input changed: {name}"
    assert digest(out/"config.json")==provenance["original_config_hash"]
    for name,expected in provenance["auxiliary_hashes"].items():
        assert digest(out/"auxiliary"/name)==expected
    panel=pd.read_parquet(out/"details/research_panel.parquet")
    assert len(panel)==363500 and not panel.duplicated(["trade_date","ticker"]).any()
    assert panel.groupby("trade_date").size().eq(500).all() and (panel.source_snapshot_date<=panel.trade_date).all()
    ic=pd.read_csv(out/"ic_summary.csv");primary=ic[ic.analysis.eq("primary")]
    assert len(primary)==8 and primary.n_days.eq(707).all()
    q=pd.read_parquet(out/"details/daily_quantiles.parquet");d=pd.read_parquet(out/"details/daily_ic.parquet")
    totals=q.groupby(["specification","trade_date","factor","horizon"])[["n_assigned","n_observed"]].sum()
    joined=d.set_index(["specification","trade_date","factor","horizon"]).join(totals).dropna(subset="n_assigned")
    assert joined.n_assigned.eq(joined.n_factor).all() and joined.n_observed.eq(joined.n_pairs).all()
    diagnostic=pd.read_parquet(out/"details/diagnostics.parquet")
    parts=diagnostic.groupby(["trade_date","factor","horizon"]).ic_contribution.sum()
    raw=d[d.specification.eq("primary")].set_index(["trade_date","factor","horizon"]).rank_ic.reindex(parts.index)
    np.testing.assert_allclose(parts,raw,atol=1e-10)
    baseline=json.loads((out/"evidence/baseline.json").read_text());comparisons=[]
    quantile=pd.read_csv(out/"quantile_summary.csv");quantile=quantile[quantile.specification.eq("primary")]
    portfolio=pd.read_csv(out/"portfolio_summary.csv")
    pairs=[("ic_summary_primary",primary,["factor","horizon"]),
        ("quantile_summary",quantile,["factor","horizon","quantile"]),
        ("quantile_monotonicity",quantile.drop_duplicates(["factor","horizon"]),["factor","horizon"]),
        ("portfolio_summary",portfolio[portfolio.holding_days.eq(20)&portfolio.offset.eq(0)],["fee_bps"]),
        ("portfolio_start_offset_sensitivity",portfolio[portfolio.holding_days.eq(20)&portfolio.fee_bps.eq(15)],["offset"])]
    years=ic[ic.analysis.eq("yearly")].assign(year=lambda x:x.period.astype(int))
    robust=ic[ic.analysis.isin(["robustness","neutralization"])].copy()
    robust.loc[robust.analysis.eq("neutralization"),"specification"]="SW_effective_history_neutralization_and_matched_raw"
    pairs.extend([("ic_summary_yearly",years,["factor","horizon","year"]),
        ("neutralization_summary",ic[ic.analysis.eq("neutralization")],["factor","horizon"]),
        ("robustness_summary",robust,["factor","horizon","specification"])])
    for name,current,keys in pairs:
        old=pd.DataFrame(baseline[name]).set_index(keys).sort_index();new=current.set_index(keys).reindex(old.index)
        numeric=old.select_dtypes("number").columns.intersection(new.columns)
        for column in numeric:
            # The old CSV used only 12 significant digits. Count means near
            # 500 lose ~5e-10 on serialization; IC/return tolerances stay strict.
            tolerance=1e-9 if column in ["mean_n_pairs","mean_assigned","mean_observed"] else 1e-10
            np.testing.assert_allclose(old[column],new[column],atol=tolerance,rtol=0,equal_nan=True)
        comparisons.append(dict(table=name,rows=len(old),numeric_columns=len(numeric),max_abs_difference=float(np.nanmax(abs(old[numeric]-new[numeric])))))
    old=pd.DataFrame(baseline["incremental_summary"]).set_index(["factor","horizon","metric"]).sort_index()
    new=ic[ic.analysis.eq("incremental")].set_index(["factor","horizon","metric"]).reindex(old.index)
    numeric=["mean","hac_se","p_value","ci_low","ci_high"]
    np.testing.assert_allclose(old[numeric],new[numeric],atol=1e-10,rtol=0,equal_nan=True)
    comparisons.append(dict(table="incremental_summary",rows=len(old),numeric_columns=len(numeric),max_abs_difference=float(np.nanmax(abs(old[numeric]-new[numeric])))))
    ledgers=pd.read_parquet(out/"details/portfolio.parquet");trades=pd.read_parquet(out/"details/trades.parquet")
    for keys,g in ledgers.groupby(["holding_days","fee_bps"]):
        assert g.cash.ge(-1e-10).all()
        np.testing.assert_allclose(np.r_[1.,g.nav.to_numpy()[:-1]]*(1+g.daily_return),g.nav,atol=1e-10)
        t=trades[trades.holding_days.eq(keys[0])&trades.fee_bps.eq(keys[1])]
        assert np.isclose(t.fee.sum(),g.cost.sum())
        assert (t.signal_date<t.trade_date).all()
    # Rerun the independent scalar reference retained in the tests on real data.
    spec=importlib.util.spec_from_file_location("reference",root/"tests/test_huatai.py")
    ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
    calendar=pd.DatetimeIndex(pd.to_datetime(pd.read_parquet(root/"data/raw/calender/trade_calender_2016_2025.parquet").cal_date)).sort_values()
    days=pd.to_datetime(["2023-01-03","2024-07-31","2025-12-03"])
    needed=set().union(*(set(calendar[calendar.get_loc(day)-9:calendar.get_loc(day)+1]) for day in days))
    prices=pd.read_parquet(root/"data/processed/daily/csi500_processed_daily.parquet")
    prices.trade_date=pd.to_datetime(prices.trade_date,format="%Y%m%d");prices=prices[prices.trade_date.isin(needed)]
    members=pd.read_parquet(root/"data/processed/universe/csi500_membership.parquet");members.trade_date=pd.to_datetime(members.trade_date)
    for day in days:
        actual=panel[panel.trade_date.eq(day)].set_index("ticker")
        for factor in ["Alpha3","Alpha13","Alpha44"]:
            expected=ref.scalar_reference(prices,members,calendar,day,factor)
            np.testing.assert_allclose(actual[factor],[expected[t] for t in actual.index],atol=1e-10,rtol=0,equal_nan=True)
    words=len((root/"docs/huatai_alpha_english_summary.md").read_text().split());assert 300<=words<=500,words
    files=[out/"config.json",*out.glob("*.csv"),*out.glob("details/*"),*out.glob("figures/*"),
           *out.glob("auxiliary/*.parquet"),out/"auxiliary/benchmark.csv",*out.glob("evidence/*.json"),out/"evidence/tests.txt"]
    files.extend([root/"README.md",root/"requirements-research.txt",root/"docs/huatai_alpha_research_report.md",root/"docs/huatai_alpha_english_summary.md"])
    manifest=dict(validation="passed",tests=int(re.search(r"Ran (\d+) tests",(out/"evidence/tests.txt").read_text()).group(1)),
        baseline_comparison=comparisons,independent_reference_values=4500,english_words=words,
        original_input_hashes=provenance["original_input_hashes"],auxiliary_hashes=provenance["auxiliary_hashes"],
        config_sha256=digest(out/"config.json"),software=dict(python=platform.python_version(),pandas=pd.__version__,
            numpy=np.__version__,scipy=scipy.__version__,pyarrow=pyarrow.__version__,matplotlib=matplotlib.__version__),
        source_hashes={str(p.relative_to(root)):digest(p) for folder in ["src","tests"] for p in (root/folder).rglob("*.py")},
        artifact_hashes={str(p.relative_to(root)):digest(p) for p in files if p.is_file()})
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    # The report links to this manifest: create it before validating that link.
    for doc in [root/"README.md",root/"docs/huatai_alpha_research_report.md",root/"docs/huatai_alpha_english_summary.md"]:
        for target in re.findall(r"\]\(([^)]+)\)",doc.read_text()):
            if not target.startswith("https://"):assert (doc.parent/target.split("#")[0]).exists(),target
