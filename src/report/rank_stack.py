"""汇总固定排名五档的四个选股目标、线性对照和模型堆叠。"""

import json
import os
from pathlib import Path

import pandas as pd

from src.data.research_panel import ROOT
from src.report.charts import plot_portfolio_nav
from src.report.research import markdown_table, save_report


def write_report(out: Path) -> Path:
    """从同一次实验的预测诊断、持仓和组合账本生成研究报告。"""
    cfg = json.loads((out / "config.json").read_text())
    logistic = json.loads((out / "logistic.json").read_text())
    report = ROOT / "docs/results/lightgbm_rank_stack_daily.md"
    links = os.path.relpath(out, report.parent)
    names = dict(top50="A 选前50", top100="B 选前100", avoid50="C 避后50", avoid100="D 避后100",
                 linear="固定五档线性对照", equal="四模型等权", logistic="逻辑回归堆叠")
    selections = pd.read_csv(out / "selection_daily.csv").groupby(["split", "strategy"]).mean(numeric_only=True)
    ic = pd.read_csv(out / "rank_ic_summary.csv").query('year == "all"').set_index(["split", "strategy"])
    summary = selections.join(ic[["mean_rank_ic"]], validate="one_to_one")
    summary.to_csv(out / "selection_summary.csv")
    rows = []
    for split, period in (("validation", "验证期"), ("test", "2023–2025 测试")):
        for name, display in names.items():
            r = summary.loc[(split, name)]
            rows.append([period, display, f"{r.hit50:.2%}", f"{r.hit100:.2%}", f"{r.bottom50:.2%}",
                         f"{r.bottom100:.2%}", f"{r.top_return:.4%}", f"{r.mean_rank_ic:.4f}"])
    signals = markdown_table(["期间", "模型", "命中前50", "命中前100", "落入后50", "落入后100",
                              "入选平均远期收益", "Rank IC"], rows)
    held = pd.read_csv(out / "holding_bottom_daily.csv").groupby(["strategy", "cutoff"]).held_bottom_rate.mean().unstack()
    p = pd.read_csv(out / "portfolio_metrics.csv").set_index("strategy")
    rows = []
    for name, display in dict(**names, benchmark="中证500").items():
        r = p.loc[name]
        tails = [f"{held.at[name, n]:.2%}" for n in (50, 100)] if name in held.index else ["—", "—"]
        excess = [f"{r.annualized_excess_return:.2%}", f"{r.information_ratio:.3f}"] if name in held.index else ["—", "—"]
        rows.append([display, *tails, f"{r.total_return:.2%}", f"{r.annual_return:.2%}", *excess, f"{r.max_drawdown:.2%}"])
    portfolio = markdown_table(["组合", "实持后50", "实持后100", "累计收益", "绝对年化", "年化超额",
                                "信息比率", "最大回撤"], rows)
    old = pd.read_csv(Path(cfg["historical_reference"]) / "portfolio_metrics.csv").set_index("strategy").loc["quintile_gap3"]
    gains = cfg["label_gains"]
    rounds = {name: pd.read_csv(out / "final" / name / "training_ndcg.csv").set_index("iteration").validation.idxmax()
              for name in gains}
    fit = pd.read_csv(out / "final/fit_rank_ic.csv").pivot(index="strategy", columns="period", values="mean_rank_ic")
    design = markdown_table(["基础模型", "标签 0–4 的增益", "最佳轮数", "训练 Rank IC", "早停期 Rank IC"],
                            [[names[name], str(gain), rounds[name], f"{fit.at[name, 'train']:.4f}",
                              f"{fit.at[name, 'early_stop']:.4f}"] for name, gain in gains.items()])
    coefficients = "、".join(f"{names[n]} {w:+.3f}" for n, w in logistic["coefficients"].items())
    plotted = ("linear", "equal", "logistic")
    ledger = pd.read_parquet(out / "ledger.parquet")
    annual = (1 + ledger.daily_return).groupby([ledger.strategy, ledger.trade_date.dt.year]).prod() - 1
    yearly = "；".join(f"{names[n]} " + "、".join(f"{x:.2%}" for x in annual.loc[n]) for n in ("equal", "logistic"))
    plot_portfolio_nav(out, ledger, [], plotted,
                       dict(linear="Linear gains", equal="Equal ranks", logistic="Logistic stack", benchmark="CSI 500"),
                       dict(linear="#6843a6", equal="#24658a", logistic="#00866c"))
    counts = pd.read_csv(out / "final/grade_counts.csv")
    scores = pd.read_parquet(out / "predictions.parquet", columns=["trade_date", "grade", "evaluable"])
    counts = pd.concat([counts.set_index("trade_date")[[str(i) for i in range(5)]].set_axis(range(5), axis=1),
                        scores.loc[scores.evaluable].groupby(["trade_date","grade"]).size().unstack(fill_value=0)])
    ranges = "、".join(f"{counts[i].min():.0f}–{counts[i].max():.0f}" for i in range(5))
    v, t = summary.loc["validation"], summary.loc["test"]
    metrics = dict(top50="hit50", top100="hit100", avoid50="bottom50", avoid100="bottom100")
    deltas = "，".join(f"{names[n]} {100*(v.at[n, m]-v.at['linear', m]):+.2f}" for n, m in metrics.items())
    return save_report(report, f"""# 中证500前 50 名选股 · 固定排名五档与模型堆叠

验证期内，堆叠和等权的真实前 50 命中率分别为 {v.at['logistic','hit50']:.2%} 和 {v.at['equal','hit50']:.2%}。2023–2025 年两者扣费年化分别为 {p.at['logistic','annual_return']:.2%} 和 {p.at['equal','annual_return']:.2%}，最大回撤分别为 {p.at['logistic','max_drawdown']:.2%} 和 {p.at['equal','max_drawdown']:.2%}。测试名单落入真实后 50 的比例，堆叠为 {t.at['logistic','bottom50']:.2%}，等权为 {t.at['equal','bottom50']:.2%}。

**表　相同五档标签下的训练目标**

{design}

每天用 T+1 收盘到 T+{cfg["label_end_days"]} 收盘的复权收益划档。从低到高依次为后 50、后 51–100、中间、前 51–100、前 50。边界并列取相同档位。各期参与评价或训练的有效交易日，五档人数范围依次为 {ranges}，前后 100 名重叠日为 0。

**表　每日预测前 50 名的选股表现**

{signals}

验证期各模型相对线性对照在自身目标上的比例变化（百分点）为 {deltas}。A、B 看命中率升幅，C、D 看误选率降幅。

名单在当日可评价股票内按分数取前 50，同分按股票代码排序，逐日等权。前 100 包含前 50，后 100 包含后 50。平均远期收益是名单中股票的等权收益。回测使用全体有预测分数的股票。

**表　2023–2025 年实际持仓和组合收益**

{portfolio}

按日账本复算，2023、2024、2025 三个自然年的扣费收益依次为{yearly}。

实持比例按只数计算，以 T+1 日收盘实际持有且能与 T 日有效标签匹配的股票为分母，再取日均。组合共用 Qlib 日频 Top-50，每次计划替换 5 只，T+1 收盘成交。初始资金 1 亿元，买入 5 bp、卖出 15 bp。绝对收益已扣费，年化超额按 Qlib 算术口径计算。历史五等分增益 [0,3,4,5,6] 使用 2021–2022 年早停，同期扣费年化 {old.annual_return:.2%}，最大回撤 {old.max_drawdown:.2%}。[历史绩效明细]({os.path.relpath(Path(cfg['historical_reference']) / 'portfolio_metrics.csv', report.parent)})

![测试期线性增益、等权、逻辑回归与指数的净值]({links}/portfolio_nav.svg)

图 1　2023–2025 年日频 offset 0 扣费净值，虚线为中证500价格指数。

**逻辑回归如何利用四个模型。** 2018、2019、2020 年分别生成全年样本外分数，每折用此前一年早停，更早的数据训练。对每个交易日全股票池的 A–D 分数分别取百分位，再以其中 {logistic['n_fit']:,} 个有标签股票日拟合 L2 逻辑回归（C=1），目标为进入真实前 50。系数为 {coefficients}；截距 {logistic['intercept']:+.3f}。系数允许为负，表示其他三列给定时对预测的调整。等权组合使用相同四列百分位求均值。

最终基础模型用 2014–2020 年样本训练，按 {cfg.get("early_stop_bounds", ["2021-01-01", "2021-12-31"])} 早停，在 {cfg["segments"]["validation"]} 验证。每段只使用收益终点落入本段的标签。树参数沿用项目既有实验，当时的选参涉及 2021–2022 年，且 2023–2025 年的结果已被多次查看。独立性限于本轮的训练及堆叠流程。

[实验配置]({links}/config.json) · [样本外预测]({links}/oof_predictions.parquet) · [堆叠系数]({links}/logistic.json) · [每日选股诊断]({links}/selection_daily.csv) · [实际持仓覆盖]({links}/holding_bottom_daily.csv) · [组合账本]({links}/ledger.parquet) · [组合指标]({links}/portfolio_metrics.csv)
""")


def write_blend_report(out: Path) -> Path:
    """比较连续排名、固定等权和原逻辑回归的名单及真实组合。"""
    cfg = json.loads((out / "config.json").read_text())
    model = cfg["rank_blend"]
    report = ROOT / "docs/results/rank_blend_daily.md"
    links = os.path.relpath(out, report.parent)
    names = dict(blend="约束加权", equal="四模型等权", logistic="原逻辑回归")
    stats = pd.read_csv(out / "selection_daily.csv").groupby(["split", "strategy"]).mean(numeric_only=True)
    ic = pd.read_csv(out / "rank_ic_summary.csv").query('year == "all"').set_index(["split", "strategy"])
    held = pd.read_csv(out / "holding_bottom_daily.csv").groupby(["strategy", "cutoff"]).held_bottom_rate.mean().unstack()
    port = pd.read_csv(out / "portfolio_metrics.csv").set_index("strategy")
    rows = []
    for split, year in (("validation", "验证期"), ("test", "2023–2025 测试")):
        for name, display in names.items():
            r = stats.loc[(split, name)]
            i = ic.loc[(split, name)]
            rows.append([year, display, f"{r.hit50:.2%}", f"{r.hit100:.2%}", f"{r.bottom50:.2%}",
                         f"{r.bottom100:.2%}", f"{r.top_return:.4%}", f"{i.mean_rank_ic:.4f}", f"{i.rank_icir:.4f}"])
    signals = markdown_table(["期间", "组合", "命中前50", "命中前100", "落入后50", "落入后100",
                              "名单平均远期收益", "Rank IC", "Rank ICIR"], rows)
    rows = []
    for name, display in dict(**names, benchmark="中证500").items():
        r = port.loc[name]
        ir = f"{r.information_ratio:.3f}" if name in names else "—"
        tail = f"{held.at[name, 50]:.2%}" if name in names else "—"
        icir = f"{ic.at[('test', name), 'icir']:.4f}" if name in names else "—"
        rows.append([display, tail, f"{r.total_return:.2%}", f"{r.annual_return:.2%}", f"{r.annualized_excess_return:.2%}",
                     ir, f"{r.max_drawdown:.2%}", icir])
    performance = markdown_table(["组合", "实持后50", "累计收益", "年化收益", "年化超额", "信息比率", "最大回撤", "ICIR"], rows)
    weights = model["weights"]
    ledger = pd.read_parquet(out / "ledger.parquet")
    plot_portfolio_nav(out, ledger, [], tuple(names), dict(blend="Weighted ranks", equal="Equal ranks",
                       logistic="Logistic stack", benchmark="CSI 500"), dict(blend="#6843a6", equal="#24658a",
                       logistic="#00866c"), monthly_axis=True, benchmark_width=2.7)
    return save_report(report, f"""# 中证500 · 四模型连续排名堆叠

测试期约束加权组合的实持后 50 比例为 {held.at['blend', 50]:.2%}，原逻辑回归为 {held.at['logistic', 50]:.2%}，等权为 {held.at['equal', 50]:.2%}。约束加权与等权的扣费年化收益分别为 {port.at['blend', 'annual_return']:.2%} 和 {port.at['equal', 'annual_return']:.2%}。

**选股方法。** 每日将 A、B、C、D 的打分分别换算成当日全池百分位。训练目标是股票真实远期收益在当日有效截面中的百分位。用 2018–2020 年 {model['fit_stock_days']:,} 个样本外股票日拟合，最小化每天等权的均方误差，将每个模型权重事先限制在 {model['lower']:.0%}–{model['upper']:.0%}，四项合计 100%。冻结后的权重为 A 选前50 {weights['top50']:.2%}、B 选前100 {weights['top100']:.2%}、C 避后50 {weights['avoid50']:.2%}、D 避后100 {weights['avoid100']:.2%}。每天按四项加权分数给股票排序。

**表　每日预测前 50 名的事后表现**

{signals}

按真实标签有效的股票统计每日名单，命中和误选含边界并列。平均远期收益从 T+1 收盘算到 T+{cfg["label_end_days"]} 收盘。正权重保证其他三项相同时，任一模型调高排名都会调高综合得分。

**表　2023–2025 年扣费组合表现**

{performance}

图表沿用 Qlib 日频 Top-50，目标每次换 5 只，T+1 收盘执行，买卖费率分别为 5 bp 和 15 bp。实持后 50 的分母为当日可评价的实际持仓只数。按归档窗口验证，2023–2025 年已经参与多次历史实验比较。

![四模型三种组合方式与中证500净值]({links}/portfolio_nav.svg)

图 1　日频 offset 0 扣费净值，加粗虚线为中证500价格指数。横轴逐月显示。

[权重与配置]({links}/config.json) · [逐日选股]({links}/selection_daily.csv) · [IC 汇总]({links}/rank_ic_summary.csv) · [组合指标]({links}/portfolio_metrics.csv) · [逐日账本]({links}/ledger.parquet)
""")


def write_seed_report(out: Path) -> Path:
    """从等权回测、逐日名单和训练记录汇总一次配对种子实验。"""
    cfg = json.loads((out / "config.json").read_text())
    ref = Path(cfg["equal_reference"])
    old_seed = json.loads((ref / "config.json").read_text())["seed"]
    seed = cfg["seed"]
    report = ROOT / "docs/results" / f"rank_equal_seed_{old_seed}_{seed}_daily.md"
    links = os.path.relpath(out, report.parent)
    portfolio = pd.read_csv(out / "portfolio_metrics.csv").set_index("strategy")
    ic = pd.read_csv(out / "rank_ic_summary.csv").query('year == "all"').set_index(["split", "strategy"])
    selection = pd.read_csv(out / "selection_daily.csv").groupby(["split", "strategy"]).mean(numeric_only=True)
    holding = pd.read_csv(out / "holding_bottom_daily.csv").query("cutoff == 50").groupby("strategy").held_bottom_rate.mean()
    overlap = pd.read_csv(out / "overlap_daily.csv").groupby("split")[["share", "common"]].mean()
    names = dict(reference=f"等权 {old_seed}", equal=f"等权 {seed}", benchmark="中证500")
    rows = []
    for name, display in names.items():
        p = portfolio.loc[name]
        stats = [f"{p.total_return:.2%}", f"{p.annual_return:.2%}", f"{p.annualized_excess_return:.2%}",
                 f"{p.information_ratio:.3f}" if pd.notna(p.information_ratio) else "—", f"{p.max_drawdown:.2%}"]
        signals = ["—"] * 4
        if name != "benchmark":
            i = ic.loc[("test", name)]
            signals = [f"{i.icir:.4f}", f"{i.mean_rank_ic:.4f}", f"{i.rank_icir:.4f}", f"{holding[name]:.2%}"]
        rows.append([display, *stats, *signals])
    performance = markdown_table(["组合", "累计收益", "绝对年化", "年化超额", "信息比率", "最大回撤",
                                  "ICIR", "Rank IC", "Rank ICIR", "实持后50"], rows)
    rows = []
    for split, year in (("validation", "验证期"), ("test", "2023–2025 测试")):
        for name in ("reference", "equal"):
            r = selection.loc[(split, name)]
            rows.append([year, names[name], f"{r.hit50:.2%}", f"{r.bottom50:.2%}", f"{r.top_return:.4%}",
                         f"{ic.at[(split,name), 'mean_rank_ic']:.4f}"])
    predictions = markdown_table(["期间", "组合", "命中前50", "落入后50", "名单平均远期收益", "Rank IC"], rows)
    def iterations(folder):
        return [int(pd.read_csv(folder / "final" / name / "training_ndcg.csv").set_index("iteration").validation.idxmax())
                for name in ("top50", "top100", "avoid50", "avoid100")]
    ledger = pd.read_parquet(out / "ledger.parquet")
    yearly = (1 + ledger.daily_return).groupby([ledger.strategy, ledger.trade_date.dt.year]).prod() - 1
    annual = markdown_table(["组合", *yearly.loc["equal"].index], [
        [names[name], *[f"{r:.2%}" for r in yearly.loc[name]]] for name in ("reference", "equal")])
    plot_portfolio_nav(out, ledger, [], ("reference", "equal"), dict(reference=f"Equal {old_seed}", equal=f"Equal {seed}",
                       benchmark="CSI 500"), dict(reference="#24658a", equal="#00866c"), monthly_axis=True, benchmark_width=2.7)
    delta = portfolio.at["equal","annual_return"] - portfolio.at["reference","annual_return"]
    return save_report(report, f"""# 四模型等权 · 随机种子 {old_seed} 与 {seed}

改用 {seed} 后，测试期扣费年化变化 {delta*100:+.2f} 个百分点。每天按完整预测池取得分最高的 50 只股票，两组在测试期平均重合 {overlap.at['test','common']:.2f} 只（{overlap.at['test','share']:.2%}），验证期重合 {overlap.at['validation','common']:.2f} 只。

**表　2023–2025 年测试期组合表现**

{performance}

**表　按自然年复算的扣费累计收益**

{annual}

**表　每日前 50 名的预测表现**

{predictions}

训练期为 2014–2020 年，按配置的早停区间确定各模型最佳轮数，验证区间为 {cfg["segments"]["validation"]}。两个种子的 A、B、C、D 最优轮数分别为 {iterations(ref)} 和 {iterations(out)}。树参数、截面标签和数据相同，每个模型占 25%。前 50 命中及误选只在有真实标签的股票中计算。实持比例按 T+1 收盘可评价持仓的只数计算。

![两个种子的等权净值与中证500价格指数]({links}/portfolio_nav.svg)

图 1　2023–2025 年日频 offset 0，初始资金 1 亿元，每次计划替换 5 只，T+1 收盘执行，买卖费率分别为 5 bp 和 15 bp。两条实线为扣费后的组合净值，加粗虚线为中证500。此对照仅覆盖两个种子，不能据此估计全部种子的收益分布。

[配置]({links}/config.json) · [逐日名单重合]({links}/overlap_daily.csv) · [训练及早停期 Rank IC]({links}/final/fit_rank_ic.csv) · [测试和验证 IC]({links}/rank_ic_summary.csv) · [回测指标]({links}/portfolio_metrics.csv) · [逐日账本]({links}/ledger.parquet)
""")


def blend_loss_table(out: Path, cfg: dict, keys: list[str]) -> str:
    """从选参时保存的结果展示混合前后损失和相对首组候选的变化。"""
    grid = pd.read_csv(out / "blend_trials.csv")
    start = grid.iloc[0].blend_loss
    rows = []
    for row in grid.to_dict("records"):
        rows.append([row["trial"], *[row[k] for k in keys], f"{row['baseline_loss']:.6f}",
                     f"{row['model_loss']:.6f}", f"{row['blend_loss']:.6f}",
                     f"{row['delta_vs_reference']:+.6f}", f"{row['blend_loss']-start:+.6f}",
                     "选定" if row["trial"] == cfg["selected_trial"] else ""])
    return markdown_table(["候选", *keys, "原组合损失", "新模型四组等权损失", "混合损失",
                           "较原组合", "较候选 0 混合", "选择"], rows)


def write_xgboost_report(out: Path, summary: pd.DataFrame, reference: Path) -> Path:
    """将验证期与测试期的同组相关、前 50 重合和收益写入同一份报告。"""
    names = dict(top50="A 选前50", top100="B 选前100", avoid50="C 避后50", avoid100="D 避后100")
    cfg = json.loads((out / "config.json").read_text())
    weight = cfg.get("lightgbm_xgboost_blend", {}).get("xgboost_weight", 0.5)
    blend = "两类模型各占 50%" if weight == 0.5 else f"LightGBM {(1-weight)*100:g}% + XGBoost {weight*100:g}%"
    ensembles = dict(equal="LightGBM 四组等权", xgb_equal="XGBoost 四组等权", blend_equal=blend)
    tuned = "selected_trial" in cfg
    filename = "xgboost_tuned_daily.md" if tuned else "xgboost_comparison_daily.md"
    mixed_selection = (out / "blend_trials.csv").is_file()
    if mixed_selection:
        filename = "xgboost_blend_tuned_daily.md"
    report = ROOT / "docs/results" / filename
    links = os.path.relpath(out, report.parent)
    rows = []
    for name, title in {**names, "equal": "四组等权"}.items():
        for split, period in (("validation", "验证期"), ("test", "2023–2025 测试")):
            r = summary.loc[(split, name)]
            rows.append([title, period, f"{r.pearson:.3f}", f"{r.spearman:.3f}", f"{r.shared:.1f}",
                         f"{r.overlap:.1%}", f"{int(r.spearman_days)}/{int(r.days)}"])
    correlations = markdown_table(["配对", "区间", "Pearson", "Spearman", "前50重合只数", "前50重合率",
                                    "有效排名相关天数 / 总天数"], rows)
    ic = pd.read_csv(out / "rank_ic_summary.csv").query("year == 'all'").set_index(["split", "strategy"])
    metrics = pd.read_csv(out / "portfolio_metrics.csv").set_index("strategy")
    old = pd.read_csv(reference / "portfolio_metrics.csv").set_index("strategy")
    portfolios = pd.concat([metrics, old.loc[list(names)]], verify_integrity=True)
    order = [(code, method + " " + title) for name, title in names.items()
             for code, method in ((name, "LightGBM"), (f"xgb_{name}", "XGBoost"))]
    rows = []
    for name, title in [*order, *ensembles.items(), ("benchmark", "中证500")]:
        p = portfolios.loc[name]
        vals = ([f"{ic.loc[(split, name), 'mean_rank_ic']:.4f}" for split in ("validation", "test")] +
                [f"{ic.loc[('test', name), 'rank_icir']:.3f}"]) if name != "benchmark" else ["—"] * 3
        rows.append([title, *vals, f"{p.total_return:.2%}", f"{p.annualized_excess_return:.2%}",
                     f"{p.information_ratio:.3f}" if pd.notna(p.information_ratio) else "—", f"{p.max_drawdown:.2%}"])
    performance = markdown_table(["策略", "验证 Rank IC", "测试 Rank IC", "测试 Rank ICIR", "扣费累计收益",
                                  "年化超额", "信息比率", "最大回撤"], rows)
    ledger = pd.read_parquet(out / "ledger.parquet")
    yearly = (1 + ledger.daily_return).groupby([ledger.trade_date.dt.year, ledger.strategy]).prod().sub(1).unstack()
    years = markdown_table(["年份", *ensembles.values()],
                           [[year, *(f"{row[name]:.2%}" for name in ensembles)] for year, row in yearly.iterrows()])
    r = summary.loc[("test", "equal")]
    delta = metrics.loc["blend_equal"] - metrics.loc["equal"]
    rows = []
    for name, title in names.items():
        curve = pd.read_csv(out / f"{name}_early_stop.csv").set_index("iteration").ndcg_at_50
        rows.append([title, str(cfg["label_gains"][name]), curve.idxmax(), f"{curve.max():.5f}"])
    fit = markdown_table(["目标", "由低到高的五档增益", "XGBoost 最佳轮数", "早停期 NDCG@50"], rows)
    tuning = ""
    if tuned and not mixed_selection:
        trials = pd.read_csv(out / "trials.csv")
        keys = list(cfg["xgboost_tuning"]["search_space"])
        rows = []
        for number, group in trials.groupby("trial"):
            scores = group.set_index("target").loc[list(names), "validation_ndcg"]
            rows.append([number, *group.iloc[0][keys], *(f"{x:.6f}" for x in scores),
                         f"{scores.mean():.6f}", "选定" if number == cfg["selected_trial"] else ""])
        grid = markdown_table(["候选", *keys, "A", "B", "C", "D", "四组均值", "选择"], rows)
        tuning = f"""**表　验证集 NDCG@50**

{grid}

候选 0 为本次起点。先按早停窗口为每个模型选择轮数，再对验证窗口的逐日 NDCG@50 取均值。四组等权选择同一套参数，确定后才预测测试期。[候选明细]({links}/trials.csv) · [最优配置]({links}/best_config.json)

"""
    if mixed_selection:
        keys = list(cfg["xgboost_tuning"]["search_space"])
        selected = pd.read_csv(out / "blend_trials.csv").set_index("trial").loc[cfg["selected_trial"]]
        selection_period = "扩展窗等权平均" if "walk_forward" in cfg else "固定验证集"
        if cfg.get("walk_forward", {}).get("selection_windows") == "latest":
            selection_period = "最近验证窗"
        detail = "fold_trials.csv" if "walk_forward" in cfg else "trials.csv"
        tuning = f"""本轮选择深度 **{cfg['xgboost']['max_depth']}**、最小子节点权重 **{cfg['xgboost']['min_child_weight']}**，混合损失 **{selected.blend_loss:.6f}**，相对纯 LightGBM 变化 **{selected.delta_vs_reference:+.6f}**。

**表　{selection_period}混合损失，越低越好**

{blend_loss_table(out, cfg, keys)}

每个候选先生成两模型混合排名。用同一份排名分别对照 A–D 的真实增益，计算逐日 NDCG@50，再对四组和交易日等权取均值。验证损失为 1 减去该均值。原组合为 LightGBM 四组等权。[候选验证明细]({links}/{detail}) · [混合选参明细]({links}/blend_trials.csv) · [所选配置]({links}/best_config.json)

"""
    if not tuned and (out / "validation_losses.csv").is_file():
        losses = pd.read_csv(out / "validation_losses.csv").set_index("strategy").loss
        tuning = (f"验证期损失为 LightGBM {losses.equal:.6f}，XGBoost {losses.xgb_equal:.6f}，"
                  f"两者混合 {losses.blend_equal:.6f}。损失等于 1 减去同一份名单对 A–D 的平均 NDCG@50。\n\n")
    xgb = cfg["xgboost"]
    return save_report(report, f"""# XGBoost 与 LightGBM 的日频选股对照

测试期，两类模型的四组等权分数日均排名相关为 **{r.spearman:.3f}**，前 50 名平均重合 **{r.shared:.1f} 只**。在原 LightGBM 组合中加入 {weight*100:g}% 的 XGBoost 排名后，年化超额收益变动 **{100*delta.annualized_excess_return:+.2f} 个百分点**，信息比率变动 **{delta.information_ratio:+.3f}**。

{tuning}**表　相同选股目标下，两种模型给出的分数有多接近**

{correlations}

每天在相同股票池上分别计算原始分数的 Pearson 和排名的 Spearman，再对有效日等权平均。预测名单在当日全部已打分股票中选取，两份名单的交集除以 50 得到重合率。收益标签缺失时仍可比较两份预测。

**表　各组信号质量与测试期日频 Top-50 组合表现**

{performance}

Rank IC 使用有真实远期收益的股票。四组等权先将每个模型的分数转为当日百分位再平均；混合方案中每个 LightGBM 模型占 {(1-weight)*25:g}%，每个 XGBoost 模型占 {weight*25:g}%。回测沿用 1 亿元初始资金、T+1 收盘执行、每日计划替换 5 只、买入 5 bp 和卖出 15 bp。年化超额为扣费后日超额均值乘 238。2023–2025 年已在此前研究中使用。

**表　等权组合分年扣费收益**

{years}

以当年每日扣费收益连乘。三个组合均从 2023 年初连续持仓到测试期结束。

**表　四个 XGBoost 模型的训练设置与早停结果**

{fit}

两个框架使用归档中的同一份 MAD Alpha158 特征、同一批股票日和五档真实收益标签。2014–2020 年训练，按配置的早停窗口以 NDCG@50 选择轮数，验证区间为 {cfg["segments"]["validation"]}。XGBoost 使用原生 `rank:ndcg`，将五档标签映射为上表中的增益，设置 `ndcg_exp_gain=false`，采用 `{xgb["lambdarank_pair_method"]}` 方法，配对参数为 {xgb["lambdarank_num_pair_per_sample"]}。两种实现的配对、梯度归一化和树生长细节可以不同。XGBoost 种子 {xgb["seed"]}，L1 为 {xgb["reg_alpha"]:.6f}，L2 为 {xgb["reg_lambda"]:.6f}，树深 {xgb["max_depth"]}，最小子节点 Hessian 权重 {xgb["min_child_weight"]}，学习率 {xgb["learning_rate"]:.6f}，行、列采样率分别为 {xgb["subsample"]:.0%}、{xgb["colsample_bytree"]:.0%}。[XGBoost 排序参数](https://xgboost.readthedocs.io/en/release_3.1.0/parameter.html#parameters-for-learning-to-rank-rank-ndcg-rank-map-rank-pairwise)

<details><summary>XGBoost 本次实际训练参数</summary>

```json
{json.dumps(xgb, indent=2, ensure_ascii=False)}
```

最大 {cfg["num_boost_round"]} 轮，连续 {cfg["early_stopping_rounds"]} 轮未改善时早停，各组最佳轮数见上表。
</details>

[逐日配对]({links}/paired_daily.csv) · [配对汇总]({links}/paired_summary.csv) · [逐股预测]({links}/predictions.parquet) · [IC 指标]({links}/rank_ic_summary.csv) · [回测账本]({links}/ledger.parquet) · [训练配置]({links}/config.json)
""")


def update_comparison_section(report: Path, title: str, section: str) -> Path:
    """替换综合报告中同名章节，并按正文顺序更新表号。"""
    body = report.read_text() if report.is_file() else "# 中证500日频排序模型比较\n"
    if title in body:
        before, remainder = body.split(title, 1)
        _, sep, after = remainder.partition("\n## ")
        body = before.rstrip() + "\n\n" + section + sep + after
    else:
        body = body.rstrip() + "\n\n" + section
    return save_report(report, body)


def write_target_weights(out: Path) -> Path:
    """汇总前 50 名与实持股票的尾部占比，并回放核对等权组的历史收益。"""
    from src.report.charts import plot_rank_feature_paths

    cfg = json.loads((out / "config.json").read_text())
    cfg_weights = cfg["four_model_weights"]
    weights = "/".join(f"{cfg_weights[k]:.0%}" for k in ("top50", "top100", "avoid50", "avoid100"))
    methods = {"equal": "LightGBM 等权", "weighted": "LightGBM 调权",
               "xgb_equal": "XGBoost 等权", "xgb_weighted": "XGBoost 调权",
               "blend_equal": "两类各半·等权", "blend_weighted": "两类各半·调权"}
    ledger = pd.read_parquet(out / "ledger.parquet")
    keys = ["strategy", "trade_date", "offset"]
    equal = ["equal", "xgb_equal", "blend_equal"]
    column = "daily_return"
    archived = pd.read_parquet(Path(cfg["prediction_source"]) / "ledger.parquet",
                               columns=[*keys, column], filters=[("strategy", "in", equal)])
    pd.testing.assert_series_equal(ledger.loc[ledger.strategy.isin(equal)].set_index(keys)[column].sort_index(),
                                   archived.set_index(keys)[column].sort_index(), atol=1e-12, rtol=1e-12)
    table = pd.read_csv(out / "portfolio_metrics.csv").set_index("strategy")
    table["validation_loss"] = pd.read_csv(out / "validation_loss_daily.csv").groupby("strategy").loss.mean()
    ic = pd.read_csv(out / "rank_ic_summary.csv").query("split == 'test' and year == 'all'").set_index("strategy")
    table = table.join(ic[["mean_ic", "icir", "mean_rank_ic", "rank_icir"]])
    picks = pd.read_csv(out / "selection_daily.csv").query("split == 'test'").groupby("strategy")
    table = table.join(picks[["hit50", "hit100", "bottom50", "bottom100"]].mean())
    held = pd.read_csv(out / "holding_bottom_daily.csv").groupby(["strategy", "cutoff"]).held_bottom_rate.mean()
    table = table.join(held.unstack().rename(columns={50: "held_bottom50", 100: "held_bottom100"}))
    table.to_csv(out / "comparison.csv")
    rows = []
    for key, name in methods.items():
        r = table.loc[key]
        rows.append([name, f"{r.validation_loss:.6f}", f"{r.mean_rank_ic:.4f}", f"{r.held_bottom50:.2%}",
                     f"{r.held_bottom100:.2%}", f"{r.total_return:.2%}", f"{r.annualized_excess_return:.2%}",
                     f"{r.information_ratio:.3f}", f"{r.max_drawdown:.2%}"])
    r = table.loc["benchmark"]
    rows.append(["中证500", "—", "—", "—", "—", f"{r.total_return:.2%}", "0.00%", "—", f"{r.max_drawdown:.2%}"])
    results = markdown_table(["组合", "验证损失↓", "测试 Rank IC", "实持后50", "实持后100", "累计收益",
                              "年化超额", "信息比率", "最大回撤"], rows)
    report = ROOT / "docs/results/lightgbm_xgboost_nav_daily.md"
    links = os.path.relpath(out, report.parent)
    plot_rank_feature_paths(out, ledger, {"blend_equal": "Equal targets", "blend_weighted": "Weighted targets"},
                            "target_weights.png")
    title = "## 四个排序目标的权重对照"
    old, new = table.loc["blend_equal"], table.loc["blend_weighted"]
    section = f"""{title}

沿用仅做 MAD 的 Alpha158 和已训练的八个排序器。每天在各算法内把 A 选前50、B 选前100、C 避后50、D 避后100 的股票分数分别转为截面排名百分位，以 {weights} 合成，再将 LightGBM 与 XGBoost 各取一半。原等权组每项目标占 25%。验证损失继续对四个目标等权评价，2022 年取值越小越好。

**表　目标权重、测试期实际持仓与扣费收益**

{results}

“实持后50”和“实持后100”按 T+1 收盘实际持仓计算，分母为有真实远期收益的持股数，各日等权。日频组合每次计划换出 5 只，实际持仓可以与当日预测前 50 名不同。

在每日有标签股票中取预测前 50 名，双模型组合命中真实前 50 的比例从 {old.hit50:.2%} 变为 {new.hit50:.2%}，命中前 100 的比例从 {old.hit100:.2%} 变为 {new.hit100:.2%}。落入真实后 50 的比例从 {old.bottom50:.2%} 变为 {new.bottom50:.2%}，落入后 100 的比例从 {old.bottom100:.2%} 变为 {new.bottom100:.2%}。

![两种权重的净值及累计超额收益]({links}/target_weights.png)

图 3　2023–2025 年日频双模型组合。紫线为四目标等权，蓝线为本次权重，黑色虚线为中证500。上图按日复利计算扣费净值，下图逐日累加扣费后超额收益。

[全指标]({links}/comparison.csv) · [实验配置]({links}/config.json) · [逐日持仓尾部比例]({links}/holding_bottom_daily.csv)
"""
    return update_comparison_section(report, title, section)


def write_pair_tuning(out: Path) -> Path:
    """将双模型分阶段调参的验证损失与冻结后的回测写入综合报告。"""
    from src.report.charts import plot_rank_feature_paths

    cfg = json.loads((out / "config.json").read_text())
    ref = Path(cfg["pair_reference"])
    best = json.loads((out / "winners.json").read_text())
    stages = pd.read_csv(out / "stages.csv")
    params = {"depth": ("深度", "max_depth", "max_depth"), "leaves": ("叶子数上限", "num_leaves", "max_leaves"),
              "min_child": ("叶节点约束", "min_data_in_leaf", "min_child_weight"), "l1": ("L1", "lambda_l1", "reg_alpha"),
              "l2": ("L2", "lambda_l2", "reg_lambda"), "row_sampling": ("行采样", "subsample", "subsample"),
              "column_sampling": ("列采样", "colsample_bytree", "colsample_bytree")}
    rows = []
    for _, stage in stages.iterrows():
        label, lk, xk = params[stage["stage"]]
        values = []
        for f, key in (("lightgbm", lk), ("xgboost", xk)):
            source = out / f / stage[f] / "params.json"
            p = json.loads(source.read_text()) if source.exists() else cfg[f]
            values.append(f"{p[key]:g}" if key in p else "0（仅受深度限制）")
        rows.append([label, *values, stage.pairs, f"{stage.before:.6f}", f"{stage.loss:.6f}"])
    stage_table = markdown_table(["阶段", "LightGBM 优胜值", "XGBoost 优胜值", "组合数",
                                  "起点验证损失", "最低验证损失"], rows)
    old_b = pd.read_parquet(ref / "ledger.parquet", filters=[("strategy", "==", "blend_equal")])
    books = pd.read_parquet(out / "ledger.parquet")
    book = books.loc[books.strategy.eq("baseline")]
    if not cfg.get("validation_mode"):
        pd.testing.assert_series_equal(old_b.set_index("trade_date").daily_return.sort_index(),
                                       book.set_index("trade_date").daily_return.sort_index(), atol=1e-12, rtol=1e-12)
    ms = pd.read_csv(out / "portfolio_metrics.csv").set_index("strategy")
    ic = pd.read_csv(out / "rank_ic_summary.csv").query("split=='test' and year=='all'").set_index("strategy")
    comparison = ms.join(ic[["mean_ic","icir","mean_rank_ic","rank_icir"]])
    comparison["blend_validation_loss"] = pd.Series({"baseline":stages.before.iloc[0], "blend_equal":stages.loss.iloc[-1]})
    test = pd.read_parquet(out / "predictions.parquet", filters=[("split", "==", "test")],
                           columns=["trade_date", "ticker", "evaluable", "grade", "label",
                                    "baseline", "equal", "xgb_equal", "blend_equal"])
    for name in ("baseline", "equal", "xgb_equal", "blend_equal"):
        ordered = test.sort_values(["trade_date", name, "ticker"], ascending=[True, False, True])
        picks = ordered.groupby("trade_date", sort=False).head(50)
        valid = picks.loc[picks.evaluable]
        daily = valid.assign(hit50=valid.grade.eq(4), hit100=valid.grade.ge(3),
                             bottom50=valid.grade.eq(0), bottom100=valid.grade.le(1))
        rates = daily.groupby("trade_date")[["hit50", "hit100", "bottom50", "bottom100", "label"]].mean().mean()
        comparison.loc[name, rates.index] = rates
    comparison.to_csv(out / "comparison.csv")
    names = {"baseline":"原双模型", "blend_equal":"调参后双模型", "equal":"调参后 LightGBM",
             "xgb_equal":"调参后 XGBoost", "benchmark":"中证500"}
    rows = []
    for method, label in names.items():
        r = comparison.loc[method]
        ranks = [f"{r.mean_rank_ic:.4f}", f"{r.rank_icir:.3f}"] if method != "benchmark" else ["—","—"]
        ir = f"{r.information_ratio:.3f}" if pd.notna(r.information_ratio) else "—"
        rows.append([label, *ranks, f"{r.total_return:.2%}", f"{r.annual_return:.2%}",
                     f"{r.annualized_excess_return:.2%}", ir, f"{r.max_drawdown:.2%}"])
    performance = markdown_table(["组合", "测试 Rank IC", "Rank ICIR", "扣费累计", "绝对年化", "年化超额",
                                  "信息比率", "最大回撤"], rows)
    plot_rank_feature_paths(out, books, {"baseline":"Original pair","blend_equal":"Tuned pair"}, "pair_tuning.png")
    report = ROOT / "docs/results/lightgbm_xgboost_nav_daily.md"
    links = os.path.relpath(out, report.parent)
    old, new = ms.loc["baseline"], ms.loc["blend_equal"]
    months = pd.read_csv(out / "validation_monthly.csv")
    better_months = int((months.blend_equal < months.baseline).sum())
    title = "## 按等权组合验证损失逐步调参"
    selection_period = ("按逐年扩展的训练窗重拟合，随后两年验证，各折损失等权。"
                        if "walk_forward" in cfg else "2014–2020 年拟合，2021 年早停，2022 年验证。")
    if cfg.get("validation_mode"):
        selection_period = f"早停区间 {cfg['early_stop_bounds']}，选参区间 {cfg['segments']['validation']}。"
    if "early_stop_years" in cfg.get("walk_forward", {}):
        selection_period = "逐折扩展训练区间，在独立早停期确定轮数，再按随后选参期的组合损失跨折等权。清单见 fold_trials.csv。"
    if cfg.get("walk_forward", {}).get("selection_windows") == "latest":
        selection_period = "用最近完整验证窗的日均排序损失选参，训练与早停均在该窗之前。日期见 fold_trials.csv。"
    if cfg.get("rank_pair_tuning", {}).get("selection_loss") == "lambdarank_stable":
        selection_period += " 阶段优胜者按各折平均损失加 0.5 倍总体标准差选出。"
    return update_comparison_section(report, title, f"""{title}

每类模型内部 A–D 等权，LightGBM 与 XGBoost 各占一半。采用原五档、MAD Alpha158、种子 158。{selection_period}在有完整标签的交易日上按四个目标的平均 NDCG@50 评价同一张混合排名，验证损失为 1 减去该均值。

**表　逐阶段交叉比较两类模型的参数**

{stage_table}

每个阶段保留起点，交叉评价两边候选，冻结本阶段最低损失的组合再进入下一阶段。叶子数阶段尝试按树深允许的最大叶数再设较紧上限。LightGBM 用样本数约束叶节点，XGBoost 用最小 Hessian 权重和。行采样在 LightGBM 中启用 bagging_freq=1。本轮学习率固定为 {best["lightgbm"]["learning_rate"]:.6f}。完整网格和每一对候选的分数见实验文件。

**表　冻结后的 2023–2025 年日频 Top-50 回测**

{performance}

原双模型与新双模型的验证损失分别为 **{stages.before.iloc[0]:.6f}**、**{stages.loss.iloc[-1]:.6f}**。按最终模型的验证结果分月，调参后在 {len(months)} 个验证月中的 {better_months} 个月损失更低。测试期扣费累计收益从 **{old.total_return:.2%}** 变为 **{new.total_return:.2%}**，年化超额从 **{old.annualized_excess_return:.2%}** 变为 **{new.annualized_excess_return:.2%}**。历史验证样本被多个阶段用于选参，2023–2025 年也已在本项目的以往研究中查看。

![等权组合调参前后的净值及累计超额收益]({links}/pair_tuning.png)

图 7　紫色为原双模型，蓝色为调参后的双模型，黑色虚线为中证500。上图为扣费复利净值，下图逐日累加扣费后策略收益与指数收益之差，均为日频 offset 0。

[最优配置]({links}/best_config.json) · [逐个候选的实际参数]({links}/candidate_parameters.csv) · [候选配对及验证损失]({links}/trials.csv) · [逐日验证损失]({links}/validation_loss_daily.csv) · [完整测试指标]({links}/comparison.csv)
""")


def write_expanding_pair(out: Path) -> Path:
    """比较年度滚动训练与冻结基线的连续扣费账本、年度收益及选股结果。"""
    from src.report.charts import plot_rank_feature_paths

    cfg = json.loads((out / "config.json").read_text())
    method = cfg["rolling_scheme"].removesuffix("_annual")
    sliding = method == "sliding"
    window = "滑动窗口" if sliding else "扩展窗口"
    rule = "训练起点和终点各向后移一年" if sliding else "训练起点固定，终点每年延后一年"
    report = ROOT / f"docs/results/{method}_pair_daily.md"
    links = os.path.relpath(out, report.parent)
    names = {method: window, "fixed": "固定模型", "benchmark": "中证500"}
    ledger = pd.read_parquet(out / "ledger.parquet")
    plot_rank_feature_paths(out, ledger, {method: f"{method.title()} pair", "fixed": "Fixed pair"}, "nav_excess.png")
    ic = pd.read_csv(out / "rank_ic_summary.csv").query("year == 'all'").set_index("strategy")
    losses = pd.read_csv(out / "test_ndcg_loss.csv").groupby("strategy").loss.mean()
    hits = pd.read_csv(out / "hits_daily.csv").groupby("strategy")[["hit50", "hit100", "bottom50", "bottom100"]].mean()
    summary = pd.read_csv(out / "portfolio_metrics.csv").set_index("strategy").join(
        ic[["mean_ic", "icir", "mean_rank_ic", "rank_icir"]]).join(losses.rename("ndcg_loss")).join(hits)
    summary.to_csv(out / "comparison.csv")
    portfolios, signals = [], []
    for key, display in names.items():
        row = summary.loc[key]
        portfolios.append([display, f"{row.total_return:.2%}", f"{row.annual_return:.2%}",
                           f"{row.annualized_excess_return:.2%}", f"{row.information_ratio:.3f}" if key != "benchmark" else "—",
                           f"{row.max_excess_drawdown:.2%}"])
        if key != "benchmark":
            signals.append([display, f"{row.ndcg_loss:.5f}", f"{row.mean_ic:.4f}", f"{row.icir:.4f}",
                            f"{row.mean_rank_ic:.4f}", f"{row.rank_icir:.4f}",
                            *[f"{row[col]:.2%}" for col in hits.columns]])
    totals = markdown_table(["策略", "累计收益", "绝对年化", "年化超额", "信息比率", "最大超额回撤"], portfolios)
    picks = markdown_table(["策略", "1−NDCG@50 ↓", "IC", "ICIR", "Rank IC", "Rank ICIR", "前50", "前100", "后50", "后100"], signals)
    period_rows = []
    annual = ledger.pivot(index="trade_date", columns="strategy", values="daily_return")
    annual["benchmark"] = ledger.loc[ledger.strategy.eq("fixed")].set_index("trade_date").benchmark_return
    annual = (1 + annual).groupby(annual.index.year).prod() - 1
    annual.to_csv(out / "annual_returns.csv")
    for year, row in annual.iterrows():
        bounds = pd.read_csv(out / str(year) / "periods.csv").set_index("period")
        train, stop = bounds.loc["train"], bounds.loc["early_stop"]
        period_rows.append([year, f"{train.start} 至 {train.end}", str(stop.start)[:4], f"{train.stock_days:,}",
                            *[f"{row[name]:.2%}" for name in names]])
    periods = markdown_table(["预测年", "训练信号区间", "早停年", "有效训练股票日", f"{window}收益", "固定模型收益", "指数收益"], period_rows)
    roll, fixed = summary.loc[method], summary.loc["fixed"]
    n_sessions = ledger.trade_date.nunique()
    n_evaluable = int(ic.loc[method, "n_days"])
    return save_report(report, f"""# 中证500日频选股：{window}滚动训练

{window}的扣费累计收益为 **{roll.total_return:.2%}**，固定模型为 **{fixed.total_return:.2%}**，相差 **{(roll.total_return-fixed.total_return)*100:+.2f} 个百分点**。两者日频选 50 只，沿用同一组已冻结的 LightGBM 和 XGBoost 参数、MAD Alpha158、交易规则及费率。

**表　训练窗口与各自然年扣费收益**

{periods}

每年更新八个模型。{rule}。前一个完整自然年用于 NDCG@50 早停，仅使用在各自区间内已实现的收益标签。固定对照沿用 2021 年早停、2022 年验证选参的正式归档。2022 年也作为首个窗口的早停年。更晚的窗口可使用已经结束的历史测试年份。全期持仓与资金跨年延续。

**表　全期组合绩效**

{totals}

回测覆盖 {n_sessions} 个交易日，初始资金 {cfg['initial_capital_cny']/1e8:g} 亿元，每日计划替换 {cfg['n_drop']} 只股票，买入费率 {cfg['buy_fee_bps']} bp、卖出费率 {cfg['sell_fee_bps']} bp。累计收益和绝对年化由扣费日收益复利计算。年化超额由扣费日收益减当日指数收益后取算术平均，再乘以 238。

**表　全期排序与预测前50的真实收益名次占比**

{picks}

每天将八个排序器各自的分数转成股票排名百分位，同类的四个目标各占 25%，两类树各占 50%。在 {n_evaluable} 个可评价交易日，每天从有真实收益的股票中取预测前 50 只，计算落入相应真实收益排名的比例。真实收益取 T+{cfg["label_end_days"]} 相对 T+1 的复权收盘涨幅，跨年但在测试期内的收益也计入。回测选股使用当天所有有效预测。2023–2025 年已在本项目此前的实验中观察。

![{window}与固定模型的净值和累计超额收益]({links}/nav_excess.png)

图 1　日频 offset 0。上图为策略扣费净值与中证500净值，下图为扣费日收益与指数日收益之差的逐日累加，包含所有测试交易日。

[实验配置]({links}/config.json) · [比较指标]({links}/comparison.csv) · [逐日账本]({links}/ledger.parquet) · [固定模型归档]({os.path.relpath(cfg['fixed_pair_reference'], report.parent)}/comparison.csv)
""")
