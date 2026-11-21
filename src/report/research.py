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
    best = training["best_iteration"]
    label_start, label_end = config["label_start_days"], config["label_end_days"]
    horizon = label_end - label_start
    losses = curve.loc[curve.iteration.eq(best)].iloc[0]
    training_label = normalize_training_label(training.get("training_label", ZSCORE_LABEL))
    raw_labels = training_label == "raw"
    label_scale = "原始收益标签" if raw_labels else "每日截面 Z-score 标签"
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
    plot_training_mse(out, curve, best, losses.validation_mse, raw_labels=raw_labels)
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
              f"{names.get(leader, leader)} 的扣费年化超额收益在六种方法中最高，为 {leader_value:.2%}。")
    template = Path(__file__).with_name("strategy_report.md").read_text(encoding="utf-8")
    report_body = template.format_map(locals())
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
    return f"""## 固定与滚动 LightGBM 的逐窗对比

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
