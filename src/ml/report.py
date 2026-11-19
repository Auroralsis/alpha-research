"""用已保存的打分、训练日志和回测结果生成中证500比较报告。"""

import json
import os
import platform
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from src.evaluation.portfolio import benchmark_nav, load_benchmark
from src.ml.models import ROLLING_SCORE, SCORE_COLUMNS
from src.ml.rolling_report import rolling_section
from src.reporting import markdown_table, number_table_captions, sha256


def write_results(config: dict, config_path: Path, inputs: dict[str, Path], out: Path, root: Path) -> None:
    """读取 out 中的实验结果，写出 Markdown 报告、配图及复现清单。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    import numpy as np
    from src.scripts.run_factor_study import saved_factor_section

    display = dict(lightgbm="Fixed LightGBM", rolling_lightgbm="Rolling LightGBM", ridge="Ridge",
                   benchmark="CSI 500 index")
    names = dict(display, lightgbm="固定 LightGBM", rolling_lightgbm="滚动 LightGBM", benchmark="中证500价格指数")
    colors = dict(zip(SCORE_COLUMNS, ("#28649a", "#087c74", "#b14c27", "#7e549c", "#946916")))
    colors["rolling_lightgbm"] = "#b52d64"
    metrics = pd.read_csv(out / "portfolio_metrics.csv")
    strategies = ("lightgbm", ROLLING_SCORE, *SCORE_COLUMNS[1:])
    windows_spec = json.loads((out / "windows.json").read_text())
    switch_dates = [pd.Timestamp(window["effective"]["test"][0]) for window in windows_spec["windows"][1:]]
    switch_caption = "浅灰竖线标出滚动模型的切换日。" if switch_dates else ""
    ic = pd.read_csv(out / "rank_ic_summary.csv", dtype={"year": str})
    daily_ic = pd.read_csv(out / "daily_rank_ic.csv", parse_dates=["trade_date"]).query('split == "test"')
    groups = pd.read_csv(out / "quintile_summary.csv", dtype={"year": str})
    ledger = pd.read_parquet(out / "ledger.parquet")
    curve = pd.read_csv(out / "training_mse.csv")
    if "strategy" in curve:
        curve = curve.loc[curve.strategy.eq("lightgbm")]
    training = json.loads((out / "training_log.json").read_text())
    data = json.loads((out / "data_audit.json").read_text())
    audit = json.loads((out / "execution_audit.json").read_text())
    coverage = pd.read_csv(out / "feature_coverage.csv")
    predictions = pd.read_parquet(out / "predictions.parquet", columns=["split", "evaluable", "label_end_date"])
    test = predictions.query('split == "test"')
    endpoint_missing = int((~test.evaluable & test.label_end_date.notna()).sum())
    calendar_tail = int(test.label_end_date.isna().sum())
    most_missing = coverage.loc[coverage.missing_rate.idxmax()]
    fit, valid, held_out = (training["split_audit"][split] for split in ("train", "validation", "test"))
    best, offset = training["best_iteration"], config["main_offset"]
    losses = curve.loc[curve.iteration.eq(best)].iloc[0]
    fees = f"买入 {config['buy_fee_bps']} bp、卖出 {config['sell_fee_bps']} bp"
    comparison = metrics.query('strategy != "benchmark"')
    benchmark_metrics = metrics.query('strategy == "benchmark"').iloc[0]
    primary = comparison.loc[comparison.offset.eq(offset)].set_index("strategy").reindex(strategies)
    average = comparison.groupby("strategy").mean(numeric_only=True).reindex(strategies)
    returns_by_offset = comparison.pivot(index="offset", columns="strategy", values="annual_return")
    all_ic = ic.query('split == "test" and year == "all"').set_index("strategy").reindex(strategies)
    all_groups = groups.query('split == "test" and year == "all"').set_index("strategy").reindex(strategies)
    signal_days = daily_ic.loc[daily_ic.strategy.eq("lightgbm")]
    test_start, test_end = config["segments"]["test"]
    validation_start, validation_end = config["segments"]["validation"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(curve.iteration, curve.train_mse, label="Train")
    ax.plot(curve.iteration, curve.validation_mse, label="Validation")
    ax.scatter(best, losses.validation_mse, color="C1", zorder=3)
    ax.axvline(best, color="0.5", linestyle="--", label=f"Best: round {best}")
    ax.set(xlabel="Boosting round", ylabel="MSE on daily Z-scored labels")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.17), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "training_mse.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for name in strategies:
        days = daily_ic.loc[daily_ic.strategy.eq(name)].sort_values("trade_date")
        ax.plot(days.trade_date, days.rank_ic.rolling(20).mean(), color=colors[name], linewidth=1.2,
                label=display.get(name, name))
    for date in switch_dates:
        ax.axvline(date, color=".8", linewidth=.8, zorder=0)
    ax.axhline(0, color=".55", linewidth=.8)
    ax.set(ylabel="20-session mean RankIC")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.11), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "rank_ic_timeline.svg")
    plt.close(fig)

    buckets = [f"Q{i}" for i in range(1, 6)]
    fig, ax = plt.subplots(figsize=(9, 4))
    width = .8 / len(strategies)
    midpoint = (len(strategies) - 1) / 2
    for i, name in enumerate(strategies):
        ax.bar(np.arange(len(buckets)) + (i - midpoint) * width, all_groups.loc[name, buckets].astype(float),
               width=width, color=colors[name], label=display.get(name, name))
    ax.set(xticks=np.arange(len(buckets)), xticklabels=buckets, ylabel="Mean 5-session return")
    ax.axhline(0, color=".55", linewidth=.8)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "quintile_returns.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    main = ledger.loc[ledger.offset.eq(offset)]
    for name in strategies:
        path = main.loc[main.strategy.eq(name)]
        ax.plot(path.trade_date, path.nav, label=display.get(name, name), color=colors[name], linewidth=1.3)
    dates = pd.DatetimeIndex(main.trade_date.drop_duplicates())
    index_nav = benchmark_nav(load_benchmark(inputs["benchmark"]), dates)
    ax.plot(dates, index_nav, "--", color=".36", label=display["benchmark"])
    for date in switch_dates:
        ax.axvline(date, color=".8", linewidth=.8, zorder=0)
    ax.set(ylabel="NAV")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.11), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "portfolio_nav.svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    offsets = np.arange(len(returns_by_offset))
    for i, name in enumerate(strategies):
        ax.bar(offsets + (i - midpoint) * width, returns_by_offset[name], width=width, color=colors[name],
               label=display.get(name, name))
    ax.axhline(benchmark_metrics.annual_return, color=".36", linestyle="--", label=display["benchmark"])
    ax.axhline(0, color=".55", linewidth=.8)
    ax.set(xticks=offsets, xticklabels=returns_by_offset.index.astype(int), xlabel="Rebalance offset",
           ylabel="Annualized net return")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.2), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "offset_returns.svg")
    plt.close(fig)

    ic_table = markdown_table(["方法", "有效交易日", "Mean RankIC", "ICIR", "正 IC 比例"], [
        [names.get(name, name), str(int(row.n_days)), f"{row['mean']:.4f}", f"{row.icir:.4f}", f"{row.positive_rate:.2%}"]
        for name, row in all_ic.iterrows()
    ])
    yearly_ic = ic.loc[ic.year.ne("all") | ic.split.eq("validation")].copy()
    yearly_ic = yearly_ic.set_index(["strategy", "split", "year"])
    yearly_ic_table = markdown_table(["阶段", "年度", "方法", "有效交易日", "Mean RankIC", "ICIR"], [
        ["验证" if split == "validation" else "测试", "全期" if year == "all" else year, names.get(name, name),
         str(int(row.n_days)), f"{row['mean']:.4f}", f"{row.icir:.4f}"]
        for (name, split, year), row in yearly_ic.sort_index(level=["split", "year", "strategy"]).iterrows()
    ])
    group_table = markdown_table(["方法", *buckets, "Q5−Q1"], [
        [names.get(name, name), *(f"{row[key]:.3%}" for key in [*buckets, "spread"])]
        for name, row in all_groups.iterrows()
    ])

    def metrics_rows(frame: pd.DataFrame) -> list[list[str]]:
        """把逐策略的收益、风险和费用统计转成可显示的表格行。"""
        return [
            [names.get(name, name), f"{row.total_return:.2%}", f"{row.relative_total_return:.2%}",
             f"{row.annual_return:.2%}", f"{row.annual_volatility:.2%}", f"{row.sharpe:.3f}", f"{row.max_drawdown:.2%}",
             f"{row.month_win_rate:.2%}",
             f"{row.annual_turnover_one_way:.1%}" if pd.notna(row.annual_turnover_one_way) else "—",
             f"{row.cost_cny:,.0f}" if pd.notna(row.cost_cny) else "—"]
            for name, row in frame.iterrows()
        ]

    perf_headers = ["方法", "累计收益", "相对指数累计", "年化收益", "年化波动", "夏普率", "最大回撤", "月胜率",
                    "年化单边换手", "费用（元）"]
    metrics_table = markdown_table(
        perf_headers, metrics_rows(primary) + metrics_rows(metrics.loc[metrics.strategy.eq("benchmark")].set_index("strategy")),
    )
    averages_table = markdown_table(perf_headers, metrics_rows(average))
    offset_table = markdown_table(["Offset", *(names.get(name, name) for name in strategies)], [
        [str(int(index)), *(f"{row[name]:.2%}" for name in strategies)]
        for index, row in returns_by_offset.iterrows()
    ])
    report = root / "docs/ml_study.md"
    links = Path(os.path.relpath(out, report.parent)).as_posix()
    ledger_path = Path(os.path.relpath(out / "ledger.parquet", root)).as_posix()
    official = f"https://github.com/microsoft/qlib/blob/v{config['qlib_version']}"
    train_start, train_end = config["segments"]["train"]
    overview = markdown_table(
        ["方法", "Mean RankIC", "ICIR", "主轨道年化", "各偏移平均年化"],
        [[names.get(name, name), f"{all_ic.loc[name, 'mean']:.4f}", f"{all_ic.loc[name, 'icir']:.4f}",
          f"{primary.loc[name, 'annual_return']:.2%}", f"{average.loc[name, 'annual_return']:.2%}"]
         for name in strategies]
        + [[names["benchmark"], "—", "—", f"{benchmark_metrics.annual_return:.2%}", "—"]],
    )
    rolling_details = rolling_section(out, links, windows_spec)
    traditional = saved_factor_section(report)
    report_body = f"""# 中证500选股研究

{test_start} 至 {test_end}，在同一测试期比较预测排序和扣费后的周频组合。主轨道使用 offset {offset}，各偏移平均年化是各条调仓轨道年化收益的算术均值。

**表　测试期各方法的排序能力与年化收益**

{overview}

[对比 RankIC]({links}/rank_ic_summary.csv) · [对比分层收益]({links}/quintile_summary.csv) · [对比组合绩效]({links}/portfolio_metrics.csv)

## 数据与训练

**表　固定模型的样本区间与有效样本数**

| 阶段 | 日期 | 可用于训练或评价的模型样本 |
| --- | --- | ---: |
| 训练 | {train_start} 至 {train_end} | {fit['fitted_rows']:,} |
| 验证 | {validation_start} 至 {validation_end} | {valid['fitted_rows']:,} |
| 测试 | {test_start} 至 {test_end} | {held_out['evaluable_rows']:,} |

固定模型使用同一组动态中证500成员的 158 项量价特征。信号在 T 日收盘生成，标签由 T+6 的复权收盘价除以 T+1 的复权收盘价再减 1。各阶段的收益终点须落在本阶段内。LightGBM 对标签做每日截面 Z-score 并保留缺失特征，Ridge 拟合原始收益，使用训练期拟合的特征中位数填补和标准化，L2 惩罚参数固定为 {config['ridge_alpha']:g}。[Alpha158 特征定义]({links}/alpha158.json) · [配置]({links}/config.json)

LightGBM 以 MSE 为损失，最多训练 {config['num_boost_round']} 轮。扩大后的验证集选出**第 {best} 轮**，实际运行 {training['trained_rounds']} 轮。最优轮的训练 MSE 为 {losses.train_mse:.6f}、验证 MSE 为 {losses.validation_mse:.6f}。Ridge 一次拟合完成，验证原始收益 MSE 为 {training['ridge_mse']['validation']:.6f}。两个模型的 MSE 对应各自的标签尺度。[训练日志]({links}/training_log.json) · [Qlib 官方配置]({official}/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml)

![LightGBM 逐轮训练损失]({links}/training_mse.svg)

**图 1　固定 LightGBM 训练与验证损失。** 横轴是提升轮数，纵轴是标准化标签的 MSE。Train 是训练集，Validation 是 {validation_start[:4]}–{validation_end[:4]} 年验证集，竖线标出第 {best} 轮。

## 预测排序

测试期的 RankIC、五组收益和回测使用同一可选池，保留各方法都有有效分数的股票日。单因子使用复权量价计算的分数，按信号日做行业和总市值中性化。本地市值数据覆盖测试期，因此验证期表中报告固定 LightGBM 和 Ridge。测试期共有 {audit['common_scored_stock_days']:,} 条共同可打分记录，每日候选池为 {audit['min_daily_candidates']} 至 {audit['max_daily_candidates']} 只，其中 {int(signal_days.n_pairs.sum()):,} 条记录有可用于排序评价的收益标签。中证500价格指数参与组合绩效对照。

日 RankIC 衡量分数和真实五日收益的排序相关性。ICIR 为日 RankIC 均值除以日样本标准差，使用未年化口径。

**表　测试期各方法的 RankIC 统计**

{ic_table}

<details>
<summary>展开验证期与逐年 RankIC</summary>

**表　验证期汇总与分年度 RankIC**

{yearly_ic_table}

</details>

![测试期各方法的滚动 RankIC]({links}/rank_ic_timeline.svg)

**图 2　测试期排序能力随时间的变化。** 横轴为信号日，纵轴为最近 20 个交易日的平均 RankIC。每条彩色曲线对应图例中的一种方法，零线以上代表近期平均排序相关性为正。{switch_caption}

每天按各方法分数由低到高分为 Q1–Q5，组内先算原始五日等权收益，再对信号日取平均。Q5−Q1 衡量最高分与最低分组的平均收益差。

**表　测试期五组平均收益及 Q5−Q1 收益差**

{group_table}

![测试期五组平均五日收益]({links}/quintile_returns.svg)

**图 3　测试期按分数分组的原始五日收益。** 横轴从最低分的 Q1 到最高分的 Q5，纵轴为各组的平均五日收益。每种颜色对应一种打分方法，收益根据事后标签计算。

[逐日 RankIC 对比]({links}/daily_rank_ic.csv) · [逐年 RankIC 对比]({links}/rank_ic_summary.csv) · [逐年五组收益对比]({links}/quintile_summary.csv) · [同股同日分数]({links}/predictions.parquet)

## 周频组合

各方法在同一动态候选池上分别生成持仓。每 {config['rebalance_days']} 个交易日调仓，T 日收盘出信号，T+1 日按复权收盘价计划成交。目标持有 {config['topk']} 只，后续每次最多计划替换 {config['n_drop']} 只，原始收盘价和涨跌停价决定交易可行性。组合统一采用{fees}。中证500价格指数以收盘点位归一，年化统一按 252 个交易日折算。

### 主轨道（offset {offset}）

**表　主轨道与中证500指数的绩效对比**

{metrics_table}

![主轨道与中证500净值]({links}/portfolio_nav.svg)

**图 4　各方法的主轨道净值与中证500价格指数。** 横轴是测试交易日，纵轴以 1 为初始净值，1.20 表示累计收益为 20%。实线是 offset {offset} 的扣费策略净值（{fees}），灰色虚线是同期中证500价格指数。{switch_caption}

### 各调仓偏移及均值

**表　各调仓偏移的扣费年化收益**

{offset_table}

![各调仓偏移的扣费年化收益]({links}/offset_returns.svg)

**图 5　不同调仓起始日下的年化收益。** Offset 0 从测试期第一个交易日取信号，Offset 1 推迟一个交易日，其余依此类推。彩色柱是相应策略扣费后的年化收益，灰色虚线是同期中证500价格指数年化收益。

以下为各策略在所有偏移上的指标算术均值。

**表　各策略跨调仓偏移的平均绩效**

{averages_table}

[各方法逐轨道统计]({links}/portfolio_metrics.csv) · [资金账本]({links}/ledger.parquet) · [成交]({links}/trades.parquet) · [委托]({links}/orders.parquet) · [持仓]({links}/holdings.parquet)

{rolling_details}

## 数据限制

- 成分股历史基于已到达的快照。原始公告时刻缺失，预热后有 {data['eligible_members_missing_observation']:,} 条成分股日记录缺少可用信号日行情。
- 训练、测试期特征缺失率分别为 {fit['feature_nan_rate']:.4%} 与 {held_out['feature_nan_rate']:.4%}。缺失率最高的是 {most_missing['split']} 的 `{most_missing.feature}`（{most_missing.missing_rate:.2%}）。测试期有 {calendar_tail:,} 条模型预测的标签终点超出研究日历，另有 {endpoint_missing:,} 条因缺行情无法计算。
- 共同打分池覆盖测试期 {audit['common_scored_stock_days'] / audit['test_stock_days']:.2%} 的模型股票日。回测使用日频成交约束，流动性、价格冲击和退市现金结算的刻画有限。报价存在但缺交易限价的股票日有 {audit['observed_close_missing_limit']:,} 条。

[数据审计]({links}/data_audit.json) · [特征缺失统计]({links}/feature_coverage.csv) · [成交约束统计]({links}/execution_audit.json)

## 结果存放在哪里

本次实验的产物目录为 `{Path(os.path.relpath(out, root)).as_posix()}`，各模型和单因子共用逐股打分、排序评价和回测明细。长表用 `strategy` 区分方法。逐股打分按股票日对齐，每种方法占一列，`window` 标出滚动预测来自哪个测试窗口。完整运行结束后清理用于训练的 Alpha158 中间特征表，可用配置与输入校验值复现。

**表　实验产物及主要字段**

| 文件 | 一行代表什么，先看哪些字段 |
| --- | --- |
| [predictions.parquet]({links}/predictions.parquet) | 一只股票在一个验证或测试信号日的打分。模型和因子各占一列，`window` 是滚动窗口编号，`label` 是事后五日收益，`evaluable` 表示标签能否用于评价。 |
| [training_mse.csv]({links}/training_mse.csv) | 一轮训练，含 `strategy` 和 `window`，以及轮数 `iteration`、训练误差 `train_mse` 和验证误差 `validation_mse`。 |
| [daily_rank_ic.csv]({links}/daily_rank_ic.csv) | 一种方法（`strategy`）在一个信号日的 `rank_ic`，以及同时具有分数和有效收益标签的股票数 `n_pairs`。 |
| [rank_ic_summary.csv]({links}/rank_ic_summary.csv) | 一种方法在某阶段全期或某年度的 RankIC 汇总。重点看均值 `mean`、标准差 `rank_ic_std`、`icir` 和有效天数 `n_days`。 |
| [quintile_daily.csv]({links}/quintile_daily.csv) | 一种方法在一个信号日的一组股票。`group` 取 1 至 5，`mean_return` 是该组等权五日收益。 |
| [quintile_summary.csv]({links}/quintile_summary.csv) | 一种方法在某阶段全期或某年度的五组收益。`Q1` 至 `Q5` 是各组的平均五日收益，`spread` 是同日 Q5 减 Q1 的平均收益差。 |
| [portfolio_metrics.csv]({links}/portfolio_metrics.csv) | 一种方法的一条调仓轨道，用 `strategy` 与 `offset` 区分。也有一行中证500指数统计。重点看年化收益 `annual_return`、累计收益 `total_return`、最大回撤 `max_drawdown` 和手续费 `cost_cny`。收益以小数存储，0.06 表示 6%。 |
| [ledger.parquet]({links}/ledger.parquet) | 一种策略的一条轨道在一个交易日的总账。 `nav` 是归一净值，`daily_return` 是当日收益，`equity_cny` 和 `cash_cny` 是以元计的账户资产与现金。 |
| [orders.parquet]({links}/orders.parquet) | 一条轨道在执行日对一只股票的委托。含 `side`（买卖方向）、计划份额、成交份额和 `status`（成交、部分成交或受阻原因）。 |
| [trades.parquet]({links}/trades.parquet) | 一条轨道在执行日对一只股票的实际成交。看 `buy_cny`、`sell_cny` 和 `fee_cny` 可核对买入、卖出金额和费用。 |
| [holdings.parquet]({links}/holdings.parquet) | 一条轨道某日收盘持有的一只股票。 `holding_value_cny` 是该股票的持仓市值，`raw_share_equivalent` 是换算到原始报价口径的等效股数。 |
| [feature_coverage.csv]({links}/feature_coverage.csv) | 一个阶段的一项特征，记录缺失数量 `n_missing` 和比例 `missing_rate`。 |
| [data_audit.json]({links}/data_audit.json)、[execution_audit.json]({links}/execution_audit.json) | 两份检查记录，分别保存原始数据的覆盖状况和回测行情、限价、基准信息。 |
| [alpha158.json]({links}/alpha158.json)、[config.json]({links}/config.json)、[training_log.json]({links}/training_log.json) | 分别保存特征公式、本次运行参数、训练轮数和样本统计。 |
| [lightgbm.txt]({links}/models/lightgbm.txt)、[manifest.json]({links}/manifest.json) | 保存最佳轮 LightGBM 和实验依赖、输入文件校验值。另有 [ridge.joblib]({links}/models/ridge.joblib) 保存 Ridge 及其预处理。 |

要查看 LightGBM 主轨道前五个交易日的净值，可以在项目根目录运行下面的代码。`columns` 选出三列，`filters` 按策略名和起始偏移过滤，`.head()` 取前五行。

```python
import pandas as pd

ledger = pd.read_parquet(
    {ledger_path!r},
    columns=["trade_date", "nav", "daily_return"],
    filters=[("strategy", "==", "lightgbm"), ("offset", "==", {offset})],
)
print(ledger.head())
```

{traditional}"""
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    sources = [*inputs.values(), config_path.resolve(), root / "requirements.txt",
               root / "src/scripts/run_all.py", root / "src/evaluation/portfolio.py",
               root / "src/evaluation/factor_metrics.py", root / "src/data/research_panel.py",
               root / "src/features/price_volume.py", root / "src/features/neutralization.py", root / "src/reporting.py",
               *sorted((root / "src/ml").glob("*.py")), out / "alpha158.json", out / "predictions.parquet",
               out / "windows.json"]
    dependencies = ["pyqlib", "mlflow-skinny", "lightgbm", "numpy", "pandas", "scipy", "pyarrow", "joblib",
                    "scikit-learn", "matplotlib"]
    checksums = {os.path.relpath(path, root): sha256(path) for path in sources}
    manifest = out / "manifest.json"
    features = out / "features.parquet"
    previous = json.loads(manifest.read_text()) if manifest.exists() else {}
    feature_key = os.path.relpath(features, root)
    feature_checksum = (sha256(features) if features.is_file()
                        else previous.get("features_sha256", previous.get("sha256", {}).get(feature_key)))
    manifest.write_text(json.dumps({
        "python": platform.python_version(),
        "strategies": list(strategies),
        "dependencies": {name: version(name) for name in dependencies},
        "features_sha256": feature_checksum,
        "sha256": checksums,
    }, indent=2) + "\n")
    report.write_text(number_table_captions(report_body), encoding="utf-8")
