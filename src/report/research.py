"""从实验产物生成中证500策略比较报告和 Alpha101 因子报告。"""

import hashlib
import json
import os
import platform
import re
from collections.abc import Iterable
from importlib.metadata import version
from itertools import count
from pathlib import Path

import pandas as pd

from src.config import ZSCORE_LABEL, normalize_training_label, rebalance_frequency
from src.data.research_panel import ROOT

IC_COLUMNS = {"mean_ic": "IC", "icir": "ICIR", "mean_rank_ic": "Rank IC", "rank_icir": "Rank ICIR"}


def markdown_table(columns: Iterable[object], rows: Iterable[Iterable[object]]) -> str:
    """输入列名和逐行数据，返回 Markdown 表格文本。"""
    headers = list(columns)
    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in [headers, ["---"] * len(headers), *rows])


def number_table_captions(report: str) -> str:
    """按整份报告的出现顺序为表题编号，已有编号也一并更新。"""
    numbers = count(1)
    return re.sub(r"^\*\*表(?: [0-9]+)?　", lambda _: f"**表 {next(numbers)}　", report, flags=re.MULTILINE)


def save_report(path: Path, body: str) -> Path:
    """按正文出现顺序编号表格，并将报告写入所属目录。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(number_table_captions(body), encoding="utf-8")
    return path


def write_results(config: dict, config_path: Path, inputs: dict[str, Path], out: Path, root: Path,
                  strategies: tuple[str, ...]) -> Path:
    """读取 out 中的实验结果，写出 Markdown 报告、配图及复现清单。"""
    from src.report.charts import (
        plot_offset_returns, plot_portfolio_nav, plot_quintiles, plot_rank_ic, plot_training_mse,
    )

    standalone = strategies == ("lightgbm",)
    display = dict(lightgbm="Fixed LightGBM", rolling_lightgbm="Rolling LightGBM", ridge="Ridge",
                   benchmark="CSI 500 index")
    names = dict(display, lightgbm="固定 LightGBM", rolling_lightgbm="滚动 LightGBM", benchmark="中证500价格指数")
    colors = dict(zip(strategies, ("#28649a", "#b52d64", "#087c74", "#b14c27", "#7e549c", "#946916")))
    metrics = pd.read_csv(out / "portfolio_metrics.csv")
    windows = json.loads((out / "windows.json").read_text())["windows"] if "rolling_lightgbm" in strategies else []
    switch_dates = [pd.Timestamp(window["effective"]["test"][0]) for window in windows[1:]]
    switch_caption = "浅灰竖线标出滚动模型的切换日。" if switch_dates else ""
    ic = pd.read_csv(out / "rank_ic_summary.csv", dtype={"year": str})
    daily_ic = pd.read_csv(out / "daily_rank_ic.csv", parse_dates=["trade_date"]).query('split == "test"')
    groups = pd.read_csv(out / "quintile_summary.csv", dtype={"year": str})
    ledger = pd.read_parquet(out / "ledger.parquet")
    curve = pd.read_csv(out / "training_mse.csv")
    if "strategy" in curve:
        curve = curve.loc[curve.strategy.eq("lightgbm")]
    training = json.loads((out / "training_log.json").read_text())
    audit = json.loads((out / "execution_audit.json").read_text())
    fit, valid, held_out = (training["split_audit"][split] for split in ("train", "validation", "test"))
    label_start, label_end = config["label_start_days"], config["label_end_days"]
    horizon = label_end - label_start
    training_label = normalize_training_label(training.get("training_label", ZSCORE_LABEL))
    raw_labels = training_label == "raw"
    label_scale = "原始收益标签" if raw_labels else "每日截面 Z-score 标签"
    feature_mode = {"raw": "原始 Alpha158", "mad": "仅做逐日 5 MAD 截尾的 Alpha158",
                    "neutral": "截尾后经行业、市值中性化及残差标准化的 Alpha158"}[training.get("alpha158_processing", "raw")]
    ridge_losses = ""
    if "ridge" in strategies:
        mse = training["ridge_mse"]
        ridge_scale = normalize_training_label(training.get("ridge_training_label", "raw"))
        train_loss = f"训练 {mse['train']:.6f}，" if "train" in mse else ""
        ridge_losses = f" Ridge（{ridge_scale} 标签）MSE 为{train_loss}验证 {mse['validation']:.6f}。"
    fees = f"买入 {config['buy_fee_bps']} bp、卖出 {config['sell_fee_bps']} bp"
    buy_rule = {"vwap": "复权 VWAP 买入、复权收盘卖出", "close": "复权收盘价买卖"}[audit["buy_price"]]
    comparison = metrics.query('strategy != "benchmark"')
    benchmark_metrics = metrics.query('strategy == "benchmark"').set_index("strategy")
    average = comparison.groupby("strategy").mean(numeric_only=True)
    averaged_metrics = pd.concat([average, benchmark_metrics]).reindex([*strategies, "benchmark"])
    returns_by_offset = comparison.pivot(index="offset", columns="strategy", values="annualized_excess_return")
    all_ic = ic.query('split == "test" and year == "all"').set_index("strategy").reindex(strategies)
    all_groups = groups.query('split == "test" and year == "all"').set_index("strategy").reindex(strategies)
    all_ic["n_pairs"] = daily_ic.groupby("strategy").n_pairs.sum()
    test_start, test_end = config["segments"]["test"]
    validation_start, validation_end = config["segments"]["validation"]

    buckets = [f"Q{i}" for i in range(1, 6)]
    training_details = ""
    training_chart = ""
    if "lightgbm" in strategies:
        best = training["best_iteration"]
        losses = curve.loc[curve.iteration.eq(best)].iloc[0]
        plot_training_mse(out, curve, best, losses.validation_mse, raw_labels=raw_labels)
        training_details = f"固定 LightGBM 在第 {best} 轮取得最佳验证结果，训练 / 验证 MSE 为 {losses.train_mse:.6f} / {losses.validation_mse:.6f}。"

    plot_rank_ic(out, daily_ic, switch_dates, strategies, display, colors)
    plot_quintiles(out, all_groups, buckets, horizon, strategies, display, colors)
    plot_portfolio_nav(out, ledger, switch_dates, strategies, display, colors)
    plot_offset_returns(out, returns_by_offset, strategies, display, colors)

    yearly_ic = ic.loc[ic.year.ne("all") | ic.split.eq("validation")].copy()
    yearly_ic = yearly_ic.set_index(["strategy", "split", "year"])
    yearly_ic_table = markdown_table(["阶段", "年度", "方法", "有效交易日", *IC_COLUMNS.values()], [
        ["验证" if split == "validation" else "测试", "全期" if year == "all" else year, names.get(name, name),
         str(int(row.n_days)), *(f"{row[key]:.4f}" for key in IC_COLUMNS)]
        for (name, split, year), row in yearly_ic.sort_index(level=["split", "year", "strategy"]).iterrows()
    ])
    group_table = markdown_table(["方法", *buckets, "Q5−Q1"], [
        [names.get(name, name), *(f"{row[key]:.3%}" for key in [*buckets, "spread"])]
        for name, row in all_groups.iterrows()
    ])

    fields = [
        ("n_days", "IC 有效日", ".0f"), ("n_pairs", "IC 股票日配对数", ",.0f"),
        ("mean_ic", "IC", ".4f"), ("icir", "ICIR", ".4f"),
        ("mean_rank_ic", "Rank IC", ".4f"), ("rank_icir", "Rank ICIR", ".4f"),
        ("positive_rate", "正 IC 比例（Rank IC）", ".2%"), ("total_return", "累计收益", ".2%"),
        ("annual_return", "年化收益", ".2%"), ("annualized_excess_return", "年化超额收益", ".2%"),
        ("information_ratio", "信息比率", ".3f"), ("max_excess_drawdown", "最大超额回撤", ".2%"),
        ("sharpe", "夏普率", ".3f"), ("month_win_rate", "月胜率", ".2%"),
        ("annual_turnover_one_way", "年化单边换手", ".1%"), ("cost_cny", "费用（元）", ",.0f"),
    ]
    overview = markdown_table(["方法", *(label for _, label, _ in fields)], [
        [names.get(name, name), *(format(row[key], spec) if pd.notna(row[key]) else "—" for key, _, spec in fields)]
        for name, row in averaged_metrics.join(all_ic).iterrows()
    ])
    offset_table = markdown_table(["Offset", *(names.get(name, name) for name in strategies)], [
        [str(int(index)), *(f"{row[name]:.2%}" for name in strategies)]
        for index, row in returns_by_offset.iterrows()
    ])
    suffix, frequency = rebalance_frequency(config["rebalance_days"])
    report_name = f"{'lightgbm_study' if standalone else 'ml_study'}_{suffix}.md"
    report = root / "docs" / "results" / report_name
    links = Path(os.path.relpath(out, report.parent)).as_posix()
    if "lightgbm" in strategies:
        training_chart = (f"![训练与验证损失]({links}/training_mse.svg)\n\n"
                          f"**图 1　固定 LightGBM 训练与验证 MSE。** 横轴为提升轮数，纵轴使用{label_scale}，竖线标出验证集选出的轮数。")
    train_start, train_end = config["segments"]["train"]
    rolling_details = rolling_section(out, links, windows, names) if windows else ""
    title = "固定 LightGBM" if standalone else "多方法"
    candidate_rule = ("各方法按自身有效分数选股" if "candidate_coverage" in audit else "各方法使用共同候选池")
    group_rule = "有效 LightGBM 样本" if standalone else "各方法共同有分的股票日"
    tracks = config["rebalance_days"]
    track_rule = "日频使用 Offset 0 一条轨道" if tracks == 1 else f"组合指标取 {tracks} 条调仓轨道的等权平均"
    leader = average.annualized_excess_return.idxmax()
    leader_value = average.loc[leader, "annualized_excess_return"]
    result = (f"固定 LightGBM 的扣费年化超额收益为 {leader_value:.2%}。" if standalone else
              f"{names.get(leader, leader)} 的扣费年化超额收益在本表各方法中最高，为 {leader_value:.2%}。")
    if windows and "lightgbm" not in strategies:
        result += "本档案只保留测试期扣费累计收益跑赢指数的 LightGBM。"
    template = Path(__file__).with_name("strategy_report.md").read_text(encoding="utf-8")
    report_body = template.format_map(locals())
    if not training_chart:
        report_body = re.sub(r"^\*\*图 (\d+)　", lambda m: f"**图 {int(m[1])-1}　", report_body, flags=re.M)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    write_manifest(config_path, inputs, out, root, strategies)
    return save_report(report, report_body)


def rolling_section(out: Path, links: str, windows: list[dict], names: dict[str, str]) -> str:
    """用逐窗汇总展示固定与滚动 LightGBM 的排序及超额表现。"""
    window_ic = pd.read_csv(out / "window_rank_ic.csv")
    dates_by_window = {window["window"]: " 至 ".join(value[:10] for value in window["effective"]["test"])
                       for window in windows}
    test_table = markdown_table(
        ["窗口", "测试日期", "方法", "IC 有效日", *IC_COLUMNS.values(),
         "轨道平均年化超额收益", "正 IC 比例（Rank IC）"],
        [[row.window, dates_by_window[row.window], names[row.strategy], row.n_days, *(f"{getattr(row, key):.4f}" for key in IC_COLUMNS),
          f"{row.annualized_excess_return:.2%}", f"{row.positive_rate:.2%}"]
         for row in window_ic.itertuples(index=False)],
    )
    heading = "固定与滚动 LightGBM 的逐窗对比" if "lightgbm" in set(window_ic.strategy) else "滚动 LightGBM 的逐窗结果"
    return f"""## {heading}

**表　各测试窗口的 IC、Rank IC 与轨道平均年化超额收益**

{test_table}

窗口首日承接已有持仓，各轨道分别计算窗口内年化超额收益后取平均。

[逐窗指标]({links}/window_rank_ic.csv) · [窗口日期]({links}/windows.json)"""


def sha256(path: Path) -> str:
    """分块读取文件，返回 SHA-256 十六进制校验值。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(config_path: Path, inputs: dict[str, Path], out: Path, root: Path,
                   strategies: tuple[str, ...]) -> None:
    """记录当前实验的依赖版本、文件校验值和训练特征校验值。"""
    sources = [*inputs.values(), config_path.resolve(), root / "requirements.txt",
               *sorted((root / "src").rglob("*.py")), root / "src/report/strategy_report.md", out / "alpha158.json", out / "predictions.parquet"]
    if "rolling_lightgbm" in strategies:
        sources.append(out / "windows.json")
    dependencies = ["pyqlib", "cvxpy", "mlflow-skinny", "lightgbm", "numpy", "pandas", "scipy", "pyarrow", "joblib",
                    "scikit-learn", "matplotlib"]
    if (out / "optuna.sqlite3").is_file():
        dependencies.append("optuna")
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
        **({"excluded_from_archive": previous["excluded_from_archive"]} if "excluded_from_archive" in previous else {}),
        "dependencies": {name: version(name) for name in dependencies},
        "features_sha256": feature_checksum,
        "sha256": checksums,
    }, indent=2) + "\n")


def write_alpha101_report(summary: pd.DataFrame, frame: pd.DataFrame, out: Path, config: dict,
                          *, root: Path = ROOT) -> Path:
    """根据原始和中性化因子的评价表生成排名、样本覆盖与计算说明。"""
    def signed(value):
        return f"{value:+.5f}" if pd.notna(value) else "—"

    target = root / "docs" / "factors" / "alpha101_ic_daily.md"
    link = Path(os.path.relpath(out, target.parent)).as_posix()
    raw = summary.query("version == 'raw'").set_index("factor")
    neutral = summary.query("version == 'neutral'").set_index("factor")
    labels = frame.dropna(subset=["label"])
    eligible_days = labels.trade_date.nunique()
    complete = neutral.loc[neutral.n_days.eq(eligible_days) & neutral.n_pairs.ge(0.8 * len(labels))]
    body = ["# 中证500 Alpha101 单因子检验",
            f"{labels.trade_date.min():%Y-%m-%d} 至 {labels.trade_date.max():%Y-%m-%d} 共有 "
            f"{eligible_days} 个可评价信号日、{len(labels):,} 个有收益标签的股票日。"
            f"101 项因子中，{neutral.mean_rank_ic.notna().sum()} 项得到中性化后的 Rank IC。",
            "**表　绝对值榜首，保留符号并列出有效交易日**"]
    rows = []
    for metric, title in IC_COLUMNS.items():
        row = [title]
        for group in (raw, neutral, complete):
            valid = group[metric].dropna()
            if valid.empty:
                row.append("—")
                continue
            name = valid.abs().idxmax()
            row.append(f"{name}　{signed(valid[name])}（{int(group.loc[name,'n_days'])} 天）")
        rows.append(row)
    body += [markdown_table(["指标", "原始全部", "中性化全部", "中性化且覆盖充分¹"], rows),
             "**表　中性化 Rank ICIR 绝对值前十名（全部因子）**"]
    ranked = neutral.sort_values("abs_rank_icir_rank", kind="stable")
    top = ranked.loc[ranked.abs_rank_icir_rank.le(10)]
    body.append(markdown_table(["名次", "因子", "Rank ICIR 绝对值", "Rank ICIR", "Rank IC", "有效日", "有效股票日"], (
        [int(row.abs_rank_icir_rank), name, f"{abs(row.rank_icir):.5f}", signed(row.rank_icir),
         signed(row.mean_rank_ic), int(row.n_days), f"{int(row.n_pairs):,}"] for name, row in top.iterrows())))
    if not complete.empty:
        inverse = complete.mean_rank_ic.idxmin()
        body.append(f"覆盖充分的因子中，{inverse} 的 Rank IC 最低（{signed(complete.at[inverse, 'mean_rank_ic'])}），"
                    f"Rank ICIR 为 {signed(complete.at[inverse, 'rank_icir'])}。")
    body += [f"¹ 覆盖充分指 {eligible_days} 天均有 Rank IC，且有效股票日达到全部有标签股票日的 80%。完整榜保留其余因子。", "## 计算方法与覆盖",
             f"公式取自华泰研报图表 2–4。开高低收和 VWAP 使用项目复权价格，成交量沿用原始成交量。"
             f"ADV 是过去指定交易日的平均成交量。历史窗口的截面排名按信号日成分名单重算，组内行业采用各历史日生效的分类。"
             f"公式内部和最终回归都用本地申万一级行业，研报使用中信一级。"
             f"研报研究全 A 股、5/10/20 日周期，本表使用动态中证500，"
             f"原始收益标签为 T+{config['label_end_days']} 复权收盘价 / T+{config['label_start_days']} 复权收盘价 − 1。"
             " [华泰原文，第 5–8 页](https://crm.htsc.com.cn/doc/2019/10750101/dae9cd9e-c2cd-4f05-bb6c-e5ac8d216d3d.pdf#page=5)",
             "小数窗口向下取整，时间窗口要求完整观测。并列排名取平均，窗口极值位置从最早的交易日计为 1，并列时取首次。"
             "相关窗口缺值或方差为零、分母为零时保留 NaN。两个滚动信号逐元素取最值时要求两者均有效。"
             " [原论文，附录 A](https://arxiv.org/pdf/1601.00991#page=15)",
             "原始分数保留公式自带的行业操作。中性化版对每个因子逐日做 5 倍 MAD 截尾、行业与对数总市值等权回归、残差标准化。"
             "用各因子自己的有效股票日对齐原始未来收益。Qlib 的逐日 Pearson、Spearman 相关系数分别求均值得到 IC、Rank IC。"
             "ICIR、Rank ICIR 用相应均值除以逐日样本标准差，不年化。负值表示低分股票随后的收益倾向更高。"
             "绝对值排名保留这类信号，方向按研究期外验证。"]
    sparse = neutral.loc[neutral.n_days.lt(eligible_days / 2) | neutral.n_pairs.lt(len(labels) / 2)]
    if not sparse.empty:
        sparse = sparse.sort_index(key=lambda names: names.str.removeprefix("Alpha").astype(int))
        body.append("**表　有效样本不足一半的因子**")
        body.append("以下因子的有效交易日或有效股票日少于样本的一半。少量股票的相关系数容易接近 ±1，应结合覆盖一起读。")
        body.append(markdown_table(["因子", "原始有效日", "原始股票日", "中性化有效日", "中性化股票日"], (
            [name, int(raw.at[name, "n_days"]), int(raw.at[name, "n_pairs"]), int(row.n_days), int(row.n_pairs)]
            for name, row in sparse.iterrows())))
    audit = out / "constant_window_audit.csv"
    if audit.is_file():
        changes = pd.read_csv(audit).set_index("factor")
        body.append(f"旧版滚动计算在常数窗口内可能因浮点舍入产生微小波动。本次显式检查取值是否相同，"
                    f"Alpha3 有 {changes.at['Alpha3','now_missing']:,} 个股票日、Alpha44 有 "
                    f"{changes.at['Alpha44','now_missing']:,} 个股票日的相关系数改为缺失。常数窗口的协方差取零后，"
                    f"Alpha13 有 {changes.at['Alpha13','changed_valid_scores']:,} 个股票日因并列排名调整了分数。"
                    f"这三项的新分数和修正后的原有计算入口逐股逐日相同。 [新旧覆盖核对]({link}/constant_window_audit.csv)")
    trace = out / "alpha96_coverage.csv"
    if trace.is_file():
        coverage = pd.read_csv(trace, index_col=0).iloc[:, 0]
        body.append(f"Alpha96 要求两条分支同时有效。第二条分支的短窗相关有 {coverage.branch2_corr:,} 个股票日可算，"
                    f"之后取 12 日极值位置剩 {coverage.branch2_argmax:,} 个，再做 14 日加权平均剩 {coverage.branch2_decay:,} 个，"
                    f"最后做 13 日时序排名剩 {coverage.branch2:,} 个。 [逐层计数]({link}/alpha96_coverage.csv)")
    body += [f"[指标、有效样本数和四项绝对值排名]({link}/factor_metrics.csv) · "
             f"[逐日指标]({link}/daily_ic.parquet) · [原始样本和标签]({link}/alpha101_raw.parquet) · "
             f"[中性化样本]({link}/neutral.parquet)",
             "**表　全部 101 项，按中性化 Rank ICIR 绝对值降序排列**",
             "两版分别用未舍入的绝对值排名，同值并列。指标按原始 → 中性化展示，无定义的因子以 — 表示并列在末尾。"]
    body.append(markdown_table(["中性化名次", "原始名次", "因子", *IC_COLUMNS.values(), "中性化有效日", "中性化股票日"], (
        [*(int(rank) if pd.notna(rank) else "—" for rank in (row.abs_rank_icir_rank, raw.at[name, "abs_rank_icir_rank"])),
         name, *[f"{signed(raw.at[name,m])} → {signed(row[m])}" for m in IC_COLUMNS],
         int(row.n_days), f"{int(row.n_pairs):,}"] for name, row in ranked.iterrows())))
    return save_report(target, "\n\n".join(body) + "\n")


def write_equal_weight(config: dict, source: Path, out: Path, directions: pd.DataFrame) -> Path:
    """对照同一测试期的等权组合、已回测的 LightGBM 和中证500。"""
    from src.report.charts import plot_portfolio_nav

    suffix, frequency = rebalance_frequency(config["rebalance_days"])
    with_ridge = (out / "ridge_trials.csv").is_file()
    experiment = "alpha158_ridge_comparison" if with_ridge else "alpha158_equal_weight"
    report = ROOT / "docs/results" / f"{experiment}_{suffix}.md"
    links = os.path.relpath(out, report.parent)
    labels = dict(equal_weight="方向统一等权", neutral_tuned="中性化 LightGBM（验证最优）",
                  matched_raw="原始 Alpha158 LightGBM（同样本）", benchmark="中证500价格指数")
    ic = pd.read_csv(out / "rank_ic_summary.csv").query('year == "all"').set_index(["split", "strategy"])
    strategies = ["equal_weight"]
    ridge_section = ""
    if with_ridge:
        trials = pd.read_csv(out / "ridge_trials.csv").set_index("alpha")
        criterion = "fold_mean_mse" if "fold_mean_mse" in trials else "validation_mse"
        best = trials[criterion].idxmin()
        labels["ridge_strong"] = f"强正则 Ridge（α={best:,.0f}）"
        strategies = ["ridge_strong", "equal_weight"]
        trial_table = markdown_table(["α", "训练 MSE", "验证 MSE", "训练 Rank IC", "验证 Rank IC", "系数 L2 范数"],
            [[f"{alpha:,.0f}", f"{r.train_mse:.7f}", f"{r.validation_mse:.7f}", f"{r.train_rank_ic:.4f}",
              f"{r.validation_rank_ic:.4f}", f"{r.coefficient_norm:.5f}"] for alpha, r in trials.iterrows()])
        ridge_section = f"""## 岭回归正则化强度

**表　训练及验证结果**

{trial_table}

Ridge 的输入为 158 列中性化特征。训练期拟合缺失值中位数、标准化参数及线性系数，训练和验证沿用来源实验的 `{config['training_label']}` 收益标签并剔除跨期标签。α 越大，平方误差总和中附加的系数惩罚越强。按 {criterion} 选定 α={best:,.0f} 用于测试期回测。测试期沿用项目此前的研究区间。

[各档正则化指标]({links}/ridge_trials.csv) · [最佳配置]({links}/best_config.json) · [最优模型及预处理器]({links}/ridge.joblib)

"""
    portfolio = pd.read_csv(out / "portfolio_metrics.csv").groupby("strategy").mean(numeric_only=True)
    archived = pd.read_csv(source / "comparison.csv").set_index("strategy")
    columns = ["total_return", "annual_return", "annualized_excess_return", "information_ratio", "max_drawdown"]
    if abs(archived.at["benchmark", "total_return"] - portfolio.at["benchmark", "total_return"]) > 1e-10:
        raise ValueError("两次实验的指数收益不一致")
    references = [s for s in ("neutral_tuned", "matched_raw") if s in archived.index]
    summary = pd.concat([portfolio.loc[strategies].join(ic.loc["test", list(IC_COLUMNS)]),
                         archived.loc[[*references, "benchmark"]]])
    summary.rename_axis("strategy").to_csv(out / "comparison.csv")
    fields = [*columns, *IC_COLUMNS]
    formats = [".2%", ".2%", ".2%", ".3f", ".2%", *([".4f"] * 4)]
    table = markdown_table(["方法", "累计收益", "绝对年化", "年化超额", "信息比率", "最大回撤", *IC_COLUMNS.values()],
                           [[labels[name], *(format(row[k], f) if pd.notna(row[k]) else "—" for k, f in zip(fields, formats))]
                            for name, row in summary.iterrows()])
    ledger = pd.read_parquet(out / "ledger.parquet")
    display = dict(equal_weight="Equal weight", neutral_tuned="Neutral LightGBM", matched_raw="Raw LightGBM",
                   benchmark="CSI 500", ridge_strong="Strong Ridge")
    colors = dict(equal_weight="#007763", neutral_tuned="#a83464", matched_raw="#326299",
                  ridge_strong="#6737a2")
    paths = ledger
    if references:
        old = pd.read_parquet(source / "ledger.parquet")
        paths = pd.concat([ledger, old.loc[old.strategy.isin(references)]])
    plot_portfolio_nav(out, paths, [], (*strategies, *references), display, colors)
    positive, negative = (directions.direction.eq(s).sum() for s in (1, -1))
    families = directions.loc[directions.direction.ne(0), "family"].nunique()
    train, valid, test = (config["segments"][k] for k in ("train", "validation", "test"))
    track_caption = ("组合使用日频 offset 0 单轨道。" if config["rebalance_days"] == 1
                     else f"组合净值按 {config['rebalance_days']} 条调仓轨道等权平均。")
    key = strategies[0]
    principal = ledger.loc[ledger.strategy.eq(key)]
    name = labels[key]
    if with_ridge:
        annual = principal.assign(excess=principal.daily_return - principal.benchmark_return).groupby(
            [principal.trade_date.dt.year, "offset"]).excess.mean().groupby(level=0).mean() * 238
        ridge_section += "强正则 Ridge 的分年扣费年化超额为 " + "、".join(
            f"{year} 年 {value:+.2%}" for year, value in annual.items()) + "。\n\n"
    gross = 238 * (principal.daily_return_before_cost - principal.benchmark_return).mean()
    fee = 238 * principal.cost_rate.mean()
    title = "中性化 Alpha158 岭回归与等权选股" if with_ridge else "中性化 Alpha158 方向统一等权选股"
    return save_report(report, f"""# {title}

{test[0]} 至 {test[1]}，{name}组合扣费累计收益 **{summary.at[key, 'total_return']:.2%}**，绝对年化 **{summary.at[key, 'annual_return']:.2%}**，相对中证500的年化超额 **{summary.at[key, 'annualized_excess_return']:.2%}**。

**表　同一测试期的组合绩效和每日截面 IC**

{table}

所有策略沿用来源实验的 Qlib TopkDropoutStrategy，目标 {config['topk']} 只，每次计划替换 {config['n_drop']} 只，{frequency}调仓。买入和卖出费率分别为 {config['buy_fee_bps']}、{config['sell_fee_bps']} bp。年化超额按每日扣费收益与指数收益之差的均值乘以 238 计算。{name}组合扣费前年化超额为 {gross:+.2%}，交易费用消耗 {fee*100:.2f} 个百分点。

![组合与指数净值]({links}/portfolio_nav.svg)

图 1　横轴为测试期交易日，纵轴为从 1 起算的净值。实线表示各选股组合的扣费净值，虚线表示中证500价格指数。各曲线覆盖同一交易日历。{track_caption}

{ridge_section}## 等权因子如何合成

方向只用 {train[0]} 至 {train[1]} 的有效未来收益确定，标签终点也须在训练期内。每列按日计算 Spearman 相关，取跨日均值的符号，并要求超过半数训练年份的日均相关同号。158 列中保留正向 {positive} 列、反向 {negative} 列，跳过 {len(directions)-positive-negative} 列。入选的 {families} 个公式家族等权。

每天先对反向列乘以 −1，再在当日股票截面取百分位排名，并列取平均名次。同一公式的多个窗口先等权，随后按每只股票当日有效家族取平均。缺失值跳过，所有家族都缺失时该股票无分。验证期（{valid[0]} 至 {valid[1]}）的 Rank IC 为 {ic.at[('validation', 'equal_weight'), 'mean_rank_ic']:.4f}，测试期为 {ic.at[('test', 'equal_weight'), 'mean_rank_ic']:.4f}。两期沿用冻结的方向和权重规则。

[逐列方向及分年 Rank IC]({links}/directions.csv) · [全期及分年 IC]({links}/rank_ic_summary.csv) · [分组收益]({links}/quintile_summary.csv) · [测试期总表]({links}/comparison.csv) · [逐日预测]({links}/predictions.parquet) · [交易账本]({links}/ledger.parquet) · [缓存来源]({links}/source.json)
""")


def write_lambdarank(out: Path, k: int, best_iteration: int) -> Path:
    """将头部排序质量与同一组预测的实际组合收益并列展示。"""
    from src.report.charts import plot_portfolio_nav

    cfg = json.loads((out / "config.json").read_text())
    suffix, frequency = rebalance_frequency(cfg["rebalance_days"])
    source = json.loads((out / "source.json").read_text())
    trained = {"daily_topk_relevance": "lambdarank", "daily_graded_relevance": "graded",
               "daily_quintile_relevance": "quintile", "daily_decile_relevance": "decile",
               "daily_quintile_gap3_relevance": "quintile_gap3"}[source["target"]]
    tag = "lambdarank" if trained == "lambdarank" else f"lambdarank_{trained}"
    report = ROOT / "docs/results" / f"lightgbm_{tag}_{suffix}.md"
    links = os.path.relpath(out, report.parent)
    top = pd.read_csv(out / "topk_daily.csv").groupby(["split", "strategy"]).mean(numeric_only=True)
    ic = pd.read_csv(out / "rank_ic_summary.csv").query('year == "all"').set_index(["split", "strategy"])
    names = dict(mse="MSE 基线", lambdarank="原版头部标签", graded="全截面分档标签", quintile="五档标签", quintile_gap3="五档间距 3", decile="十档标签", benchmark="中证500")
    strategies = tuple(name for name in names if name in top.index.get_level_values("strategy"))
    rows = []
    for split, label in (("train", "训练"), ("validation", "验证"), ("test", "测试")):
        for name in strategies:
            r = top.loc[(split, name)]
            rank_ic = f"{ic.at[(split, name), 'mean_rank_ic']:.5f}" if (split, name) in ic.index else "—"
            rows.append([label, names[name], f"{r.ndcg:.5f}", f"{r.hit_rate:.2%}", f"{r.bottom_rate:.2%}",
                         f"{r.top_return:.4%}", rank_ic])
    table = markdown_table(["阶段", "模型", f"NDCG@{k}", f"前 {k} 命中比例", f"落入后 {k} 比例", f"前 {k} 平均标签收益", "Rank IC"], rows)
    portfolio = pd.read_csv(out / "portfolio_metrics.csv").groupby("strategy").mean(numeric_only=True)
    rows = []
    for name in (*strategies, "benchmark"):
        r = portfolio.loc[name]
        ir = f"{r.information_ratio:.3f}" if name != "benchmark" else "—"
        rows.append([names[name], f"{r.total_return:.2%}", f"{r.annual_return:.2%}",
                     f"{r.annualized_excess_return:.2%}", ir, f"{r.max_drawdown:.2%}"])
    performance = markdown_table(["组合", "累计收益", "绝对年化", "年化超额", "信息比率", "最大回撤"], rows)
    plot_portfolio_nav(out, pd.read_parquet(out / "ledger.parquet"), [], strategies,
                       dict(mse="MSE", lambdarank="Head-only ranker", graded="Graded ranker", quintile="Quintile ranker", quintile_gap3="Quintile gap 3", decile="Decile ranker", benchmark="CSI 500"),
                       dict(mse="#24658a", lambdarank="#a54159", graded="#00866c", quintile="#6843a6", quintile_gap3="#00866c", decile="#cb701b"))
    rules = {
        "lambdarank": f"每日按真实远期收益给前 {k} 名赋 {k} 至 1 分，其余为零，并列同分。",
        "graded": f"前 {k} 名逐名计分，第 1 名得 {2*k-1} 分，第 {k} 名得 {k} 分，其余名次分为 {k} 档。",
        "decile": "每日按真实远期收益将股票分为十档，以平均名次处理并列。最低至最高档的标签与增益依次为 0 至 9，每档约占 10%。同档股票在训练中得分相同。",
        "quintile_gap3": "每日按真实远期收益五等分，并列取平均名次。原五档的增益为 [0,1,2,3,4]，本次为 [0,3,4,5,6]。最低档约占当日样本的 20%。",
        "quintile": "每日把有标签的股票按真实远期收益从低到高分成五档，按平均名次处理并列，五档的标签及增益依次为 0、1、2、3、4。每档约占 20%，同档股票的真实收益差别不参与训练。",
    }
    training_rule = rules[trained]
    conclusion = (f"新标签的测试期前 {k} 命中比例为 {top.at[('test', trained), 'hit_rate']:.2%}，"
                  f"落入后 {k} 的比例为 {top.at[('test', trained), 'bottom_rate']:.2%}。"
                  f"扣费累计收益为 {portfolio.at[trained, 'total_return']:.2%}，"
                  f"原版头部标签为 {portfolio.at['lambdarank', 'total_return']:.2%}，"
                  f"MSE 基线为 {portfolio.at['mse', 'total_return']:.2%}。" if "lambdarank" in strategies and trained != "lambdarank" else
                  f"测试期 LambdaRank 的前 {k} 命中比例为 {top.at[('test', trained), 'hit_rate']:.2%}，"
                  f"扣费累计收益为 {portfolio.at[trained, 'total_return']:.2%}。")
    if trained == "decile":
        conclusion = (f"测试期十档模型扣费累计收益为 {portfolio.at[trained, 'total_return']:.2%}，"
                      f"五档为 {portfolio.at['quintile', 'total_return']:.2%}。预测前 {k} 名的真实前 {k} 命中比例"
                      f"分别为 {top.at[('test', trained), 'hit_rate']:.2%} 和 {top.at[('test', 'quintile'), 'hit_rate']:.2%}。")
    if trained == "quintile_gap3":
        held = pd.read_csv(out / "holding_bottom_daily.csv").groupby("strategy").held_bottom_rate.mean()
        conclusion = (f"测试期实际持仓落入真实后 {k} 的日均比例，原五档为 {held['quintile']:.2%}，"
                      f"新五档为 {held[trained]:.2%}。两组扣费年化收益分别为 "
                      f"{portfolio.at['quintile', 'annual_return']:.2%} 和 {portfolio.at[trained, 'annual_return']:.2%}。"
                      f"实持比例用 T 日标签评价 T+1 收盘后有有效标签的持仓，按股票只数统计。"
                      f"[逐日持仓覆盖]({links}/holding_bottom_daily.csv)")
    return save_report(report, f"""# 中证500前 {k} 名选股 · {names[trained]}与基线对照

{conclusion}

{training_rule}使用同一份 MAD 截尾 Alpha158、同一批训练股票日和相同的树参数。MSE 基线拟合每日标准化的未来收益。排序训练使用配置中的增益，截断位置为 {k+3}，以本次训练标签的验证集 NDCG@{k} 早停，选中第 {best_iteration} 轮。[LightGBM 排序参数](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html#lambdarank_truncation_level)

**表　预测排序质量，逐日计算后等权平均**

{table}

表内所有模型的 NDCG 统一按原版头部标签计算，便于横向比较。NDCG 越接近 1，表示模型越能把当天事后收益靠前的股票排在前面，榜首的权重更高。命中比例是预测前 {k} 名中有多少落入真实前 {k}（含边界并列）。落入后 {k} 比例衡量同一批入选股票中，有多少实际落入收益最低的 {k} 名（含边界并列）。排序诊断及平均标签收益均在有有效标签的股票内计算。收益标签从 T+1 收盘到 T+2 收盘。

**表　{cfg["segments"]["test"][0]} 至 {cfg["segments"]["test"][1]} 组合与指数**

{performance}

各模型都在完整测试股票池上打分，使用同一套 Qlib TopkDropoutStrategy，目标持有 {k} 只，每次计划替换 {cfg['n_drop']} 只。买入 {cfg['buy_fee_bps']} bp，卖出 {cfg['sell_fee_bps']} bp，表中组合收益已扣费。

![各模型的组合净值与指数]({links}/portfolio_nav.svg)

图 1　{frequency}测试期各调仓偏移的平均净值。彩线是扣费后的组合净值，虚线是中证500价格指数。日频时只有 offset 0 一条轨道。

[逐日头部排序诊断]({links}/topk_daily.csv) · [验证和测试期预测]({links}/predictions.parquet) · [逐轮 NDCG]({links}/training_ndcg.csv) · [回测指标]({links}/portfolio_metrics.csv) · [逐日账本]({links}/ledger.parquet)
""")


def write_bottom_comparison(out: Path) -> Path:
    """汇总后 k 名独立成档实验的验证选股、测试持仓和扣费收益。"""
    from src.report.charts import plot_portfolio_nav

    cfg = json.loads((out / "config.json").read_text())
    suffix, frequency = rebalance_frequency(cfg["rebalance_days"])
    report = ROOT / "docs/results" / f"lightgbm_lambdarank_bottom_{suffix}.md"
    links = os.path.relpath(out, report.parent)
    k = cfg["topk"]
    gains = json.loads((out / "gain_candidates.json").read_text())
    names = dict(mse="MSE 基线", quintile="原五档", decile="原十档")
    names.update({name: f"增益间距 {gain[1]}" for name, gain in gains.items()})
    top = pd.read_csv(out / "topk_daily.csv").groupby(["split", "strategy"]).mean(numeric_only=True)
    ic = pd.read_csv(out / "rank_ic_summary.csv").query('year == "all"').set_index(["split", "strategy"])
    held = pd.read_csv(out / "holding_bottom_daily.csv")
    held_rate = held.groupby("strategy").held_bottom_rate.mean()
    port = pd.read_csv(out / "portfolio_metrics.csv").groupby("strategy").mean(numeric_only=True)
    valid = top.loc["validation"]
    selection = (pd.read_csv(out / "fold_trials.csv").groupby("strategy").mean(numeric_only=True)
                 if (out / "fold_trials.csv").exists() else valid)
    winner = selection.loc[list(gains)].sort_values(["bottom_rate", "top_return"], ascending=[True, False]).index[0]
    baseline = selection.loc["quintile"]
    candidates = selection.loc[list(gains)]
    improved = candidates.loc[candidates.bottom_rate.lt(baseline.bottom_rate) &
                              candidates.top_return.ge(baseline.top_return)]
    joint_result = ("同时达到这两个目标的方案有" + "、".join(names[n] for n in improved.index) + "。"
                    if len(improved) else "各组候选均未在验证期同时实现比五档更低的误选率和至少相同的入选平均收益。")
    validation_rows, test_rows = [], []
    for name, display in names.items():
        v, t = valid.loc[name], top.loc[("test", name)]
        rounds = "—"
        if name in gains:
            curve = pd.read_csv(out / name / "training_ndcg.csv").set_index("iteration")
            rounds = curve.validation.idxmax()
        validation_rows.append([display, rounds, f"{v.bottom_rate:.2%}", f"{v.hit_rate:.2%}", f"{v.top_return:.4%}"])
        p = port.loc[name]
        test_rows.append([display, f"{t.bottom_rate:.2%}", f"{held_rate[name]:.2%}", f"{t.top_return:.4%}",
                          f"{ic.at[('test', name), 'mean_rank_ic']:.4f}", f"{p.total_return:.2%}",
                          f"{p.annual_return:.2%}", f"{p.annualized_excess_return:.2%}", f"{p.max_drawdown:.2%}"])
    p = port.loc["benchmark"]
    test_rows.append(["中证500", "—", "—", "—", "—", f"{p.total_return:.2%}", f"{p.annual_return:.2%}", "—", f"{p.max_drawdown:.2%}"])
    validation = markdown_table(["模型", "最佳轮数", "误选后 " + str(k), "命中前 " + str(k), "入选平均远期收益"], validation_rows)
    test = markdown_table(["模型", "名单误选", "实持误选", "入选平均远期收益", "Rank IC", "累计收益",
                           "绝对年化", "年化超额", "最大回撤"], test_rows)
    plotted = ("quintile", "decile", *gains)
    palette = ["#24658a", "#00866c", "#a54159"]
    colors = {name: palette[i] if i < len(palette) else f"C{i % 10}" for i, name in enumerate(gains)}
    labels = {name: f"Gap {gain[1]}" for name, gain in gains.items()}
    plot_portfolio_nav(out, pd.read_parquet(out / "ledger.parquet"), [], plotted,
                       dict(quintile="Quintile", decile="Decile", benchmark="CSI 500", **labels),
                       dict(quintile="#6843a6", decile="#cb701b", **colors))
    return save_report(report, f"""# 中证500选股 · 单独标记真实收益后 {k} 名

{len(gains)} 组增益中，{names[winner]}的验证集误选率最低，为 {selection.at[winner, 'bottom_rate']:.2%}。它的入选股票平均远期收益为 {selection.at[winner, 'top_return']:.4%}，原五档为 {selection.at['quintile', 'top_return']:.4%}。{joint_result}测试期该方案扣费累计收益为 {port.at[winner, 'total_return']:.2%}，原五档为 {port.at['quintile', 'total_return']:.2%}。

每天按真实远期收益从低到高排名，后 {k} 名及边界并列股票记为 0，其余按平均名次等分为 1–9 档。增益的第一个间距取 {"、".join(str(g[1]) for g in gains.values())}，其余相邻档间距均为 1。各组共用 MAD 截尾 Alpha158、树参数、训练期及验证期，均按自身验证 NDCG@{k} 早停。比较候选时优先看验证误选率，再看入选平均收益。

**表　验证期选股（{cfg['segments']['validation'][0]} 至 {cfg['segments']['validation'][1]}）**

{validation}

预测名单在有有效标签的股票中按分数取前 {k}，命中和误选分别按真实前、后 {k}（含边界并列）计算，每日等权。远期收益为 T+1 收盘至 T+2 收盘的个股收益。

**表　测试期选股与组合表现（{cfg['segments']['test'][0]} 至 {cfg['segments']['test'][1]}）**

{test}

实持误选率将 T+1 成交后收盘持有的股票与 T 日信号对应的后 {k} 名匹配，以当日有有效标签的持仓只数为分母，逐日等权。缺标签、现金和无可评价持仓的日期均不计入分母。回测使用完整预测股票池，每次计划替换 {cfg['n_drop']} 只，买卖费率分别为 {cfg['buy_fee_bps']} bp 和 {cfg['sell_fee_bps']} bp。

![各增益方案、五档和十档基线的测试期净值]({links}/portfolio_nav.svg)

图 1　{frequency}回测的各偏移平均净值。日频只有 offset 0，实线表示扣费后的组合，虚线为中证500价格指数。

[候选增益]({links}/gain_candidates.json) · [逐日预测诊断]({links}/topk_daily.csv) · [逐日实持误选与覆盖]({links}/holding_bottom_daily.csv) · [组合指标]({links}/portfolio_metrics.csv) · [逐日账本]({links}/ledger.parquet)
""")
