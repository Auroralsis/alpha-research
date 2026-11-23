"""在固定八模型中比较仅含 Alpha158 子集和 nPLHR 的输入。"""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import xgboost as xgb
import numpy as np
import pandas as pd

from src.data.research_panel import load_market_calendar, segment_masks
from src.ml.dataset import alpha158_columns
from src.ml.rank_validation import BASES, equal_ranks, read_period, training_periods, validation_loss
from src.scripts.factor_extensions import BASE, INPUTS, KEY, train_feature_set
from src.scripts.run_all import RUNS, new_run
from src.scripts.run_pair_tuning import FAMILIES, candidate_id, check_config, tune_walk_forward
from src.config import load_config, load_tuning
from src.ml.walk_forward import folds, rank_losses
from src.backtest.engine import run_backtests
from src.data.research_panel import prepare_test_panel
from src.backtest.portfolio import QLIB_DAILY_ANNUALIZATION, qlib_excess_metrics
from src.report.research import markdown_table

FULL = RUNS / "daily_alpha159_20261122/nPLHR"
SIZES = (50, 80, 120, 159)


def day_correlations(frame):
    """按日算百分位，对至少 30 只共同有效股票计算两列百分位的 Pearson 相关。"""
    ranks = frame.rank(pct=True).to_numpy(dtype=float)
    mask = np.isfinite(ranks).astype(float)
    # 秩居中能降低常量列附近的数值误差。缺失项只参与掩码计数。
    values = np.nan_to_num(ranks - 0.5)
    counts = mask.T @ mask
    sums = values.T @ mask
    safe_counts = np.maximum(counts, 1)
    variances = np.maximum((values**2).T @ mask - sums**2 / safe_counts, 0)
    denom = np.sqrt(variances * variances.T)
    cross = values.T @ values - sums * sums.T / safe_counts
    return np.clip(np.divide(cross, denom, out=np.full_like(cross, np.nan),
                             where=(counts >= 30) & (variances > 1e-12) & (variances.T > 1e-12)), -1, 1)


def select(source, extra, cfg, out):
    """仅在训练区间计算相关性，每个交易日等权；按增量评分生成嵌套名单。"""
    names = [*alpha158_columns(source), "nPLHR"]
    lo, hi = map(pd.Timestamp, cfg["segments"]["train"])
    train = pd.read_parquet(source/"features.parquet", columns=[*KEY,*names[:-1],"label","label_end_date"],
                            filters=[("trade_date",">=",lo),("trade_date","<=",hi)])
    train = train.loc[segment_masks(train, {"train": [lo, hi]})["train"]].join(
        extra,on=KEY,validate="one_to_one")
    stats = [day_correlations(group[names+["label"]]) for _,group in train.groupby("trade_date")]
    correlations = np.stack(stats)
    with np.errstate(invalid="ignore", divide="ignore"):
        def day_mean(values):
            return np.nansum(values, axis=0) / np.isfinite(values).sum(axis=0)
        redundancy = day_mean(np.abs(correlations[:, :-1, :-1]))
        ic = day_mean(correlations[:, :-1, -1])
    pd.DataFrame(redundancy,index=names,columns=names).to_csv(out/"train_abs_correlation.csv")
    quality = np.abs(np.nan_to_num(ic))
    # 未知相关性不能作为独立信息奖励。常量列和低有效率列按可测天数折减。
    usable = np.isfinite(correlations[:, :-1, -1]).mean(axis=0)
    quality *= usable
    redundancy = np.nan_to_num(redundancy, nan=1.0)
    order = [names.index("nPLHR")]
    steps = [("nPLHR", np.nan, np.nan)]
    while len(order) < len(names):
        max_r = redundancy[:, order].max(axis=1)
        score = quality * (1-max_r)
        score[order] = -np.inf
        choice = int(np.argmax(score))
        order.append(choice)
        steps.append((names[choice], max_r[choice], score[choice]))
    ranking = pd.DataFrame(steps,columns=["feature","max_corr_when_added","selection_score"])
    ranking["train_mean_rank_correlation"] = ranking.feature.map(dict(zip(names,ic)))
    ranking["usable_day_fraction"] = ranking.feature.map(dict(zip(names,usable)))
    ranking.to_csv(out/"feature_order.csv",index=False)
    sets = {size:[n for n in names if n in set(ranking.feature.iloc[:size])] for size in SIZES}
    selection_file = out / "features.json"
    serialized = json.dumps(sets, indent=2)
    if selection_file.exists() and selection_file.read_text() != serialized:
        raise ValueError("本次特征名单和已归档结果不同，请使用新实验目录")
    selection_file.write_text(serialized)
    return sets


def fold_subsets(source, extra, cfg, out):
    """在每折训练期内重新估计特征顺序，选参年只用于打分。"""
    by_size = {size: {} for size in SIZES}
    for fold in folds(cfg):
        start = fold["validation"][0]
        folder = out / "feature_folds" / start
        folder.mkdir(parents=True, exist_ok=True)
        selected = select(source, extra, dict(cfg, segments=fold), folder)
        for size in SIZES:
            by_size[size][start] = selected[size]
    return by_size


def frozen_scores(folder, family, params, x):
    """从归档恢复四个排序器，核对列名和顺序后返回原始分数。"""
    raw = pd.DataFrame(index=x.index)
    for target in BASES:
        model_dir = folder/family/candidate_id(params)/target
        if family == "lightgbm":
            model = lgb.Booster(model_file=str(model_dir/"lambdarank.txt"))
            columns = model.feature_name()
        else:
            model = xgb.Booster()
            model.load_model(model_dir/"model.ubj")
            columns = model.feature_names
        if columns != x.columns.tolist():
            raise ValueError(f"{model_dir} 模型列名或顺序与输入不符")
        raw[target] = (model.predict(x, num_threads=params["num_threads"]) if family == "lightgbm"
                       else model.inplace_predict(x))
    if not np.isfinite(raw).all().all():
        raise ValueError(f"{family} 预测存在无效分数")
    return raw


def validate(folder, names, valid, cfg, winners):
    """用冻结权重对 2022 年打分，计算组合验证损失。"""
    by_family = []
    for family, params in winners.items():
        raw = frozen_scores(folder, family, params, valid[names])
        by_family.append(equal_ranks(raw.assign(trade_date=valid.trade_date),list(BASES)))
    return validation_loss(valid,(by_family[0]+by_family[1])/2,cfg["label_gains"],cfg["topk"])


def write_comparison(out, losses):
    """对不同输入数在相同的日期上复算费用后指标。"""
    rows = []
    for size,loss in losses.items():
        folder = FULL if size == 159 else out/str(size)
        ledger = pd.read_parquet(folder/"ledger.parquet").set_index("trade_date").sort_index()
        for period, data in [("全期",ledger),*ledger.groupby(ledger.index.year)]:
            row = qlib_excess_metrics(data.daily_return,data.benchmark_return)
            rows.append(dict(size=size,period=period,validation_loss=loss,
                             turnover=data.turnover_two_sided.mean()*QLIB_DAILY_ANNUALIZATION/2,**row))
    summary = pd.DataFrame(rows)
    summary.to_csv(out/"comparison.csv",index=False)
    table = markdown_table(["输入列数","期间","五折平均选参损失","扣费年化超额","IR","最大超额回撤","年化单边换手"],
                           [(r.size,r.period,f"{r.validation_loss:.6f}",f"{r.annualized_excess_return:.2%}",
                             f"{r.information_ratio:.3f}",f"{r.max_excess_drawdown:.2%}",f"{r.turnover:.2f} 倍")
                            for r in summary.itertuples(index=False)])
    corr = pd.read_csv(out/"train_abs_correlation.csv",index_col=0)
    sets = json.loads((out/"features.json").read_text())
    redundancy_rows = []
    for size in SIZES:
        names = sets[str(size)]
        pairs = corr.loc[names,names].to_numpy()[np.triu_indices(size, k=1)]
        redundancy_rows.append((size, f"{np.nanmean(pairs):.3f}", int((pairs >= .90).sum())))
    redundancy_table = markdown_table(["输入列数","平均两两冗余度","冗余度 ≥ 0.90 的特征对数"], redundancy_rows)
    (out/"report.md").write_text("# nPLHR 与 Alpha158 删列对照\n\n"
        "固定保留 nPLHR。其余列只按 2014–2020 年训练股票日排序。每日先对特征和标签计算截面百分位，"
        "两列至少有 30 只共同有效股票才计算百分位相关。选择得分为每日因子与标签相关的均值绝对值"
        " × 有效日占比 ×（1 − 与已选列的最大冗余度）。冗余度是两列每日相关绝对值的时间均值。"
        "每次取最高分，得到嵌套子集，模型输入沿用原列顺序。2021 年按原规则早停。"
        "各折按当折训练数据重新筛列，规模按五折平均组合损失最低选择，2023–2025 为反复研究过的历史测试期。\n\n"
        +f"四组中，验证损失最低的是 **{min(losses,key=losses.get)} 列**。\n\n表 1　各组验证及测试结果\n\n"
        +table+"\n\n表 2　训练期特征相似程度\n\n"+redundancy_table+"\n\n[入选列清单](features.json) · [逐列入选依据](feature_order.csv) · [训练期冗余度](train_abs_correlation.csv)\n")


def tune_subset(selection, out):
    """固定上一轮的 120 列和基线，先用历史验证期选参，再评价测试期。"""
    cfg = json.loads((selection/"config.json").read_text())
    names = json.loads((selection/"features.json").read_text())["120"]
    if len(set(names)) != 120 or len(names) != 120 or names[-1] != "nPLHR":
        raise ValueError("120 列特征清单异常")
    cfg["walk_forward"] = load_config()["walk_forward"]
    cfg.update(selection_reference=str(selection), features=names, rank_pair_tuning=load_tuning("rank_pair_tuning", {}))
    check_config(cfg)
    config_path = out/"config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != cfg:
            raise ValueError("归档配置或特征不同，请使用新目录")
    elif any(p.name != "training.log" for p in out.iterdir()):
        raise ValueError("请指定空目录，或接续有相同配置的实验")
    config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    source = Path(cfg["feature_source"])
    baseline = selection/"120"

    def add_factor(frame):
        lo, hi = frame.trade_date.min(), frame.trade_date.max()
        factor = pd.read_parquet(Path(cfg["full_reference"])/"factor.parquet", columns=[*KEY,"nPLHR"],
                                 filters=[("trade_date",">=",lo),("trade_date","<=",hi)])
        return frame.join(factor.set_index(KEY), on=KEY, validate="one_to_one")

    valid = add_factor(read_period(source, cfg["segments"]["validation"]))
    x = valid[names]
    keys = valid[[*KEY,"evaluable","grade"]]
    del valid
    cache = {}
    for family in FAMILIES:
        params = cfg[family]
        raw = frozen_scores(baseline, family, params, x)
        cache[candidate_id(params)] = equal_ranks(raw.assign(trade_date=keys.trade_date), list(BASES))
    gains = {name: cfg["label_gains"][name] for name in BASES}
    original = validation_loss(keys, sum(cache.values())/2, gains, cfg["topk"])
    archived = pd.read_csv(selection/"validation_120.csv", parse_dates=["trade_date"])
    pd.testing.assert_frame_equal(original, archived[original.columns], rtol=0, atol=1e-12)
    print(f"120 列原组合逐日验证损失复核通过，{len(original)} 日均值 {original.loss.mean():.6f}", flush=True)
    blocks = []
    for _, rows, groups in training_periods(source, Path(cfg["reference"]), cfg["segments"]):
        matrix = add_factor(rows[[*KEY,*names[:-1]]])[names]
        blocks.append((matrix, rows.grade.to_numpy(dtype=int), groups))
    del rows, matrix
    extra = pd.read_parquet(Path(cfg["full_reference"]) / "factor.parquet").set_index(KEY, verify_integrity=True)["nPLHR"]
    subsets = fold_subsets(source, extra, cfg, out)
    best = tune_walk_forward(out, cfg, blocks=blocks, fold_features=subsets[120], extra=extra)
    selected_scores = []
    for family, params in best.items():
        folder = baseline if params == cfg[family] else out
        raw = frozen_scores(folder, family, params, x)
        selected_scores.append(equal_ranks(raw.assign(trade_date=keys.trade_date), list(BASES)))
    validation_loss(keys, sum(selected_scores) / 2, gains, cfg["topk"]).to_csv(out / "validation_loss_daily.csv", index=False)
    del blocks, x, keys, cache
    tuned_daily = pd.read_csv(out/"validation_loss_daily.csv", parse_dates=["trade_date"])
    losses = original[["trade_date","loss"]].merge(tuned_daily[["trade_date","loss"]], on="trade_date",
                                                    validate="one_to_one", suffixes=("_baseline","_tuned"))
    losses.to_csv(out/"validation_comparison.csv", index=False)
    lo, hi = map(pd.Timestamp, cfg["segments"]["test"])
    test = add_factor(pd.read_parquet(source/"features.parquet", columns=[*KEY,*names[:-1]],
                                      filters=[("trade_date",">=",lo),("trade_date","<=",hi)]).sort_values(KEY))
    scores = test[KEY].copy()
    control = pd.read_parquet(baseline/"predictions.parquet", columns=KEY).sort_values(KEY)
    pd.testing.assert_frame_equal(scores.reset_index(drop=True), control.reset_index(drop=True))
    for family, params in best.items():
        folder = baseline if params == cfg[family] else out
        raw = frozen_scores(folder, family, params, test[names])
        scores[raw.add_prefix(family+"_").columns] = raw.to_numpy()
        scores[family] = equal_ranks(raw.assign(trade_date=test.trade_date), list(BASES))
    scores["nPLHR"] = (scores.lightgbm + scores.xgboost)/2
    scores.to_parquet(out/"predictions.parquet", index=False)
    panel, days = prepare_test_panel(cfg, load_market_calendar(INPUTS["calendar"]))
    run_backtests(INPUTS, cfg, days, scores, panel, out, strategies=("nPLHR",))
    subset_report(out, baseline, losses, cfg, best)


def subset_report(out, baseline, losses, cfg, best):
    """按相同交易日比较原组合与验证选参组合，汇总参数变化及扣费结果。"""
    rows, bench = [], None
    for label, folder in [("原参数",baseline), ("验证选参",out)]:
        ledger = pd.read_parquet(folder/"ledger.parquet").set_index("trade_date", verify_integrity=True).sort_index()
        if bench is None:
            bench = ledger.benchmark_return
        else:
            pd.testing.assert_series_equal(bench, ledger.benchmark_return)
        for period, part in [("全期",ledger),*ledger.groupby(ledger.index.year)]:
            rows.append(dict(model=label, period=period, days=len(part), total_return=(1+part.daily_return).prod()-1,
                             turnover=part.turnover_two_sided.mean()*QLIB_DAILY_ANNUALIZATION/2,
                             **qlib_excess_metrics(part.daily_return,part.benchmark_return)))
    summary = pd.DataFrame(rows)
    summary.to_csv(out/"comparison.csv", index=False)
    metrics = markdown_table(["参数","期间","累计收益","扣费年化超额","IR","最大超额回撤","年化单边换手"],
               [(r.model,r.period,f"{r.total_return:.2%}",f"{r.annualized_excess_return:.2%}",
                 f"{r.information_ratio:.3f}",f"{r.max_excess_drawdown:.2%}",f"{r.turnover:.2f} 倍")
                for r in summary.itertuples()])
    changes = [(f,k,cfg[f].get(k,"默认"),v) for f in FAMILIES for k,v in best[f].items() if v != cfg[f].get(k)]
    (out/"report.md").write_text(
        "# 120 列八排序器验证集调参\n\n固定 119 个 Alpha158 特征及 nPLHR。2014–2020 年拟合，2021 年早停。"
        f"按五折平均组合损失选参。冻结后的 2022 年 {len(losses)} 个有标签交易日中，"
        f"原参数 {losses.loss_baseline.mean():.6f}，选参后 {losses.loss_tuned.mean():.6f}。"
        "七个阶段依次搜索，每次保留上阶段最优候选，两个模型家族各有四个排序器，各占组合的 1/8。"
        "2023–2025 年测试期曾用于历史研究。本次仅在参数冻结后回测，采用相同的 Top50 日换 5 只及买 5 / 卖 15 bp 费率。\n\n"
        "表 1　最优组合相对起点的参数变化\n\n"+(markdown_table(["模型","参数","原值","新值"], changes) if changes else "两模型均沿用原参数。")
        +"\n\n表 2　测试期扣费表现\n\n"+metrics+"\n\n[阶段结果](stages.csv) · [全部验证比较](trials.csv) · "
        "[逐日验证损失](validation_comparison.csv) · [最优参数](winners.json) · [特征和实验配置](config.json)\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path)
    parser.add_argument("--tune-120", type=Path, metavar="删列实验目录", help="固定归档的120列，用验证集调优八模型组合")
    args = parser.parse_args()
    if args.tune_120:
        out = args.out or new_run(1, tag="nplhr120_tuning")
        out.mkdir(parents=True, exist_ok=True)
        tune_subset(args.tune_120.resolve(), out.resolve())
        return
    out = args.out or new_run(1,tag="nplhr_selection")
    out.mkdir(parents=True,exist_ok=True)
    print(out,flush=True)
    cfg = json.loads((BASE/"config.json").read_text())
    winners = json.loads((BASE/"winners.json").read_text())
    source = Path(cfg["feature_source"])
    extra = pd.read_parquet(FULL/"factor.parquet").set_index(KEY,verify_integrity=True)["nPLHR"]
    cfg["walk_forward"] = load_config()["walk_forward"]
    config_file = out / "config.json"
    frozen = {**cfg, **winners, "full_reference": str(FULL)}
    if config_file.exists() and json.loads(config_file.read_text()) != frozen:
        raise ValueError("该目录已用于不同训练配置，请使用新实验目录")
    config_file.write_text(json.dumps(frozen, indent=2))
    subsets = fold_subsets(source, extra, cfg, out)
    sets = select(source,extra,cfg,out)
    dates = load_market_calendar(INPUTS["calendar"])
    valid = read_period(source,cfg["segments"]["validation"]).join(extra,on=KEY,validate="one_to_one")
    losses = {}
    for size in reversed(SIZES):
        folder = FULL if size == 159 else out/str(size)
        names = sets[size]
        if size != 159:
            folder.mkdir(exist_ok=True)
            if not (folder/"ledger.parquet").is_file():
                print(f"训练 {size} 列",flush=True)
                train_feature_set(folder,"nPLHR","nPLHR",extra,names[:-1],cfg,winners,dates)
        daily = validate(folder,names,valid,cfg,winners)
        daily.to_csv(out/f"validation_{size}.csv",index=False)
        group = rank_losses(out / str(size), dict(cfg, **winners), {f: [p] for f, p in winners.items()},
                            0.5, fold_features=subsets[size], extra=extra)
        group.to_csv(out / f"fold_trials_{size}.csv", index=False)
        losses[size] = group.blend_loss.mean()
        print(f"{size} 列验证损失 {losses[size]:.6f}",flush=True)
    write_comparison(out,losses)


if __name__ == "__main__":
    main()
