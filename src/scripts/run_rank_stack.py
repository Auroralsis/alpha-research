"""四目标排序组合、跨年堆叠与六档标签对照。"""

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from src.backtest.engine import evaluate_signals
from src.backtest.factor_metrics import daily_ic
from src.config import CONFIG_FILE, load_config
from src.data.research_panel import stock_day_index
from src.ml.dataset import alpha158_columns
from src.ml.rank_validation import BASES, METHODS, equal_ranks, fixed_grades, read_period, selection_stats, validation_loss
from src.scripts.run_all import latest_run, new_run
from src.scripts.run_lambdarank import fit_ranker, holding_bottom_rates

def training_diagnostics(source: Path, features: list[str], models: dict[str, lgb.Booster],
                         periods: dict, folder: Path, threads: int) -> None:
    """按有效原始收益复算每折训练、早停两期的日均 Rank IC。"""
    rows = []
    for part, bounds in periods.items():
        frame = read_period(source, bounds)
        frame = frame.loc[frame.evaluable]
        matrix = frame[features]
        index = stock_day_index(frame)
        labels = frame.label.set_axis(index)
        for name, model in models.items():
            prediction = pd.Series(model.predict(matrix, num_threads=threads), index=index)
            _, rank = daily_ic(prediction, labels)
            rows.append(dict(period=part, strategy=name, mean_rank_ic=rank.mean(), trading_days=rank.count()))
    pd.DataFrame(rows).to_csv(folder / "fit_rank_ic.csv", index=False)


def train_fold(source: Path, cfg: dict, features: list[str], stop_year: int, names: Iterable[str],
               folder: Path) -> dict[str, lgb.Booster]:
    """训练截止早停年之前，早停只用该年。记录实际收益末日和每日各档人数。"""
    folder.mkdir()
    periods = {"train": [cfg["segments"]["train"][0], f"{stop_year - 1}-12-31"],
               "early_stop": [f"{stop_year}-01-01", f"{stop_year}-12-31"]}
    datasets, audit = [], []
    summaries = []
    for part, bounds in periods.items():
        split_top25 = cfg.get("split_top25", False)
        frame = read_period(source, bounds, split_top25=split_top25)
        frame = frame.loc[frame.evaluable]
        groups = frame.groupby("trade_date", sort=False)
        sizes = groups.size().to_numpy()
        datasets.append(lgb.Dataset(frame[features], label=frame.grade.to_numpy(dtype=int), group=sizes,
                                    reference=datasets[0] if datasets else None))
        counts = groups.grade.value_counts().unstack(fill_value=0).reindex(columns=range(6 if split_top25 else 5), fill_value=0)
        audit.append(counts.assign(period=part))
        summaries.append(dict(period=part, start=bounds[0], end=bounds[1], stock_days=len(frame),
                              last_label_end=str(frame.label_end_date.max().date())))
        del frame, groups
    pd.concat(audit).to_csv(folder / "grade_counts.csv")
    (folder / "periods.json").write_text(json.dumps(summaries, indent=2))
    models = {}
    for name in names:
        directory = folder / name
        directory.mkdir()
        params = dict(cfg["lightgbm"], label_gain=cfg["label_gains"][name])
        models[name] = fit_ranker(datasets, params, cfg, directory)
    del datasets
    training_diagnostics(source, features, models, periods, folder, cfg["lightgbm"]["num_threads"])
    return models


def score(frame: pd.DataFrame, features: list[str], models: dict[str, lgb.Booster], threads: int) -> pd.DataFrame:
    """对完整信号股票池预测，未知真实收益仍保留模型分数。"""
    scores = frame[["trade_date", "ticker", "label", "label_end_date", "grade", "evaluable"]].copy()
    matrix = frame[features]
    for name, model in models.items():
        scores[name] = model.predict(matrix, num_threads=threads)
    return scores


def rank_inputs(scores: pd.DataFrame) -> pd.DataFrame:
    """在全部已打分股票中计算当日百分位；四列的顺序与逻辑回归系数一致。"""
    return scores.groupby("trade_date")[list(BASES)].rank(method="average", pct=True)


def evaluate_stack(scores: pd.DataFrame, cfg: dict, out: Path, methods: tuple[str, ...]) -> None:
    """对给定堆叠分数使用共同的排序诊断、持仓分析和 Qlib 日频回测。"""
    scores.to_parquet(out / "predictions.parquet", index=False)
    stats = selection_stats(scores, methods)
    stats.to_csv(out / "selection_daily.csv", index=False)
    dates = evaluate_signals(scores, cfg, out, methods)
    test = scores.loc[scores.split.eq("test")]
    holdings = pd.read_parquet(out / "holdings.parquet")
    held = pd.concat([holding_bottom_rates(holdings, test, dates, k).assign(cutoff=k) for k in (50, 100)])
    held.to_csv(out / "holding_bottom_daily.csv", index=False)


def fit_rank_blend(oof: pd.DataFrame, lower: float, upper: float) -> pd.Series:
    """用样本外分数拟合真实收益百分位，每日等权，限制四个正权重的范围与总和。"""
    inputs = rank_inputs(oof).loc[oof.evaluable]
    valid = oof.loc[oof.evaluable]
    by_date = valid.groupby("trade_date").label
    target = by_date.rank(method="average", pct=True).to_numpy()
    x = inputs.to_numpy()
    if not (np.isfinite(x).all() and np.isfinite(target).all() and len(x)):
        raise ValueError("拟合需要有限的股票日预测与收益")
    sample_weight = 1 / (by_date.transform("size").to_numpy() * by_date.ngroups)
    gram = (x.T * sample_weight) @ x
    cross = x.T @ (sample_weight * target)
    result = minimize(lambda w: w @ gram @ w - 2 * w @ cross, np.full(len(BASES), 1 / len(BASES)),
                      jac=lambda w: 2 * (gram @ w - cross), method="SLSQP",
                      bounds=[(lower, upper)] * len(BASES),
                      constraints=[dict(type="eq", fun=lambda w: w.sum() - 1)],
                      options={"ftol": 1e-14})
    if not result.success or not np.isclose(result.x.sum(), 1):
        raise RuntimeError(f"权重拟合失败：{result.message}")
    return pd.Series(result.x, index=BASES, name="weight")


def run_blend(source: Path) -> Path:
    """复用已冻结的四列基础预测，与原等权及逻辑回归比较连续排名堆叠。"""
    cfg = json.loads((source / "config.json").read_text())
    oof = pd.read_parquet(source / "oof_predictions.parquet")
    valid = oof.loc[oof.evaluable]
    if (set(oof.trade_date.dt.year) != set(cfg["oof_years"]) or
            not valid.label_end_date.dt.year.eq(valid.trade_date.dt.year).all() or
            valid.label_end_date.max() >= pd.Timestamp(cfg["segments"]["validation"][0])):
        raise ValueError("样本外预测日期或标签终点越界")
    lower, upper = 0.15, 0.35
    weights = fit_rank_blend(oof, lower, upper)
    cfg["rank_blend"] = dict(source=str(source.resolve()), lower=lower, upper=upper, weights=weights.to_dict(),
                             fit_stock_days=len(valid), fit_dates=valid.trade_date.nunique(),
                             target="daily_return_percentile", loss="day_equal_weighted_mse")
    scores = pd.read_parquet(source / "predictions.parquet")
    ranks = rank_inputs(scores)
    np.testing.assert_allclose(ranks.mean(axis=1), scores.equal, rtol=0, atol=1e-12)
    scores["blend"] = ranks @ weights
    out = new_run(cfg["rebalance_days"], tag="rank_blend")
    out.mkdir()
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(out, weights.to_dict(), flush=True)
    methods = ("equal", "logistic", "blend")
    evaluate_stack(scores, cfg, out, methods)
    old = pd.read_parquet(source / "ledger.parquet")
    new = pd.read_parquet(out / "ledger.parquet")
    keys = ["strategy", "trade_date", "offset"]
    for name in ("equal", "logistic"):
        pd.testing.assert_frame_equal(old.loc[old.strategy.eq(name)].set_index(keys).sort_index(),
                                      new.loc[new.strategy.eq(name)].set_index(keys).sort_index(),
                                      rtol=1e-10, atol=1e-10)
    from src.report.rank_stack import write_blend_report
    return write_blend_report(out)


def run_equal_seed(reference: Path, seed: int) -> Path:
    """只更换基础模型种子，训练四个排名模型并与存档等权预测配对回测。"""
    cfg = json.loads((reference / "config.json").read_text())
    old_seed = cfg["seed"]
    if seed == old_seed:
        raise ValueError("对照实验需要不同的种子")
    cfg["seed"] = cfg["lightgbm"]["seed"] = seed
    cfg["equal_reference"] = str(reference.resolve())
    out = new_run(cfg["rebalance_days"], tag=f"equal_seed_{seed}")
    out.mkdir()
    print(out, flush=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    source = Path(cfg["feature_source"])
    features = alpha158_columns(source)
    models = train_fold(source, cfg, features, cfg["final_early_stop_year"], BASES, out / "final")
    pieces = []
    for split in ("validation", "test"):
        frame = read_period(source, cfg["segments"][split])
        scores = score(frame, features, models, cfg["lightgbm"]["num_threads"])
        scores["equal"] = rank_inputs(scores).mean(axis=1)
        pieces.append(scores.assign(split=split))
        del frame
    keys = ["split", "trade_date", "ticker"]
    scores = pd.concat(pieces).set_index(keys).sort_index()
    archived = pd.read_parquet(reference / "predictions.parquet",
                               columns=keys + ["equal", "evaluable", "label"]).set_index(keys).sort_index()
    pd.testing.assert_frame_equal(scores[["evaluable", "label"]], archived[["evaluable", "label"]])
    scores["reference"] = archived.equal
    scores = scores.reset_index()
    methods = ("reference", "equal")
    evaluate_stack(scores, cfg, out, methods)

    lists = []
    for name in methods:
        picks = scores.sort_values(["trade_date", name, "ticker"], ascending=[True, False, True])
        lists.append(picks.groupby("trade_date").head(cfg["topk"])[keys])
    intersection = lists[0].merge(lists[1], on=keys, validate="one_to_one")
    groups = ["split", "trade_date"]
    overlap = lists[0].groupby(groups).size().to_frame("picks")
    overlap["common"] = intersection.groupby(groups).size().reindex(overlap.index, fill_value=0)
    overlap["share"] = overlap.common / overlap.picks
    overlap.to_csv(out / "overlap_daily.csv")

    old = pd.read_parquet(reference / "ledger.parquet", filters=[("strategy", "==", "equal")])
    replay = pd.read_parquet(out / "ledger.parquet", filters=[("strategy", "==", "reference")])
    pd.testing.assert_frame_equal(old.drop(columns="strategy").reset_index(drop=True),
                                  replay.drop(columns="strategy").reset_index(drop=True), rtol=1e-10, atol=1e-10)
    from src.report.rank_stack import write_seed_report
    return write_seed_report(out)


def run(source: Path, historical: Path, config_path: Path = CONFIG_FILE) -> Path:
    """冻结样本外逻辑回归与最终基础模型，再进行统一诊断与回测。"""
    metadata = json.loads((historical / "source.json").read_text())
    if Path(metadata["baseline"]).resolve() != source.resolve() or metadata["target"] != "daily_quintile_gap3_relevance":
        raise ValueError("历史对照必须来自相同特征和参数来源的五等分间距3实验")
    cfg = load_config(source / "best_config.json")
    if cfg["topk"] != 50 or cfg["rebalance_days"] != 1:
        raise ValueError("本实验按日频 Top-50 设计")
    cfg["segments"]["validation"] = ["2022-01-01", "2022-12-31"]
    settings = load_config(config_path)
    for key in ("seed", "num_boost_round", "early_stopping_rounds"):
        cfg[key] = settings[key]
    cfg["lightgbm"] = dict(settings["lightgbm_four_equal"]["params"], seed=cfg["seed"])
    gains = dict(settings["four_model_targets"], linear=list(range(5)))
    cfg.update(label_gains=gains, oof_years=[2018, 2019, 2020], final_early_stop_year=2021,
               stack_C=1.0, feature_source=str(source.resolve()), historical_reference=str(historical.resolve()))
    out = new_run(1, tag="rank_stack")
    out.mkdir()
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(out, flush=True)
    features = alpha158_columns(source)
    threads = cfg["lightgbm"]["num_threads"]
    pieces = []
    for year in cfg["oof_years"]:
        print(f"开始 {year} 年样本外预测", flush=True)
        models = train_fold(source, cfg, features, year - 1, BASES, out / str(year))
        frame = read_period(source, [f"{year}-01-01", f"{year}-12-31"])
        pieces.append(score(frame, features, models, threads))
        del frame, models
    oof = pd.concat(pieces, ignore_index=True)
    oof.to_parquet(out / "oof_predictions.parquet", index=False)
    model = LogisticRegression(C=cfg["stack_C"], solver="lbfgs", max_iter=1000)
    model.fit(rank_inputs(oof).loc[oof.evaluable], oof.loc[oof.evaluable, "grade"].eq(4).astype(int))
    fitted = dict(intercept=model.intercept_[0], coefficients=dict(zip(BASES, model.coef_[0])),
                  n_fit=int(oof.evaluable.sum()))
    (out / "logistic.json").write_text(json.dumps(fitted, indent=2))
    del oof, pieces
    models = train_fold(source, cfg, features, cfg["final_early_stop_year"], gains, out / "final")
    pieces = []
    for split in ("validation", "test"):
        frame = read_period(source, cfg["segments"][split])
        predicted = score(frame, features, models, threads)
        ranks = rank_inputs(predicted)
        predicted["equal"] = ranks.mean(axis=1)
        predicted["logistic"] = model.predict_proba(ranks)[:, 1]
        pieces.append(predicted.assign(split=split))
        del frame
    scores = pd.concat(pieces, ignore_index=True)
    evaluate_stack(scores, cfg, out, METHODS)
    from src.report.rank_stack import write_report
    return write_report(out)


def main(argv=None):
    parser = argparse.ArgumentParser(description='用固定排名五档比较四种排序目标及跨年度样本外堆叠。')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source", type=Path)
    mode.add_argument("--blend-from", type=Path, help="从已有四模型的样本外预测拟合连续排名堆叠")
    mode.add_argument("--equal-seed-from", type=Path, help="采用四模型等权，与原实验比较一个新种子")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--historical", type=Path)
    parser.add_argument("--config", type=Path, help="新训练读取的四模型等权配置")
    args = parser.parse_args(argv)
    if args.config and not args.source:
        parser.error("--config 适用于 --source 新训练，归档重放使用该次实验的配置")
    if args.equal_seed_from and args.seed is not None and not args.historical:
        print(run_equal_seed(args.equal_seed_from, args.seed))
    elif args.seed is not None or args.equal_seed_from:
        parser.error("--equal-seed-from 需配合 --seed，不使用 --historical")
    elif args.blend_from and not args.historical:
        print(run_blend(args.blend_from))
    elif args.source and args.historical:
        print(run(args.source, args.historical, args.config or CONFIG_FILE))
    else:
        parser.error("--historical 仅与 --source 成对使用")


SIX_GRADE_METHODS = ("equal", "xgb_equal", "cat_equal", "blend_equal", "three_equal")


def train_six_grades(reference: Path, targets: dict) -> Path:
    """按原模型的树参数、股票日、早停年和交易配置训练十二个六档排序器。"""
    from src.scripts.boosting import fit_catboost, train_xgboost

    cfg = json.loads((reference / "config.json").read_text())
    old_final = Path(cfg["reference"]) / "final"
    out = new_run(cfg["rebalance_days"], tag="six_grades")
    if set(targets) != set(BASES) or any(len(g) != 6 or g != sorted(g) or g[0] < 0 or
                                         any(type(v) is not int for v in g) or g[4] == g[5]
                                         for g in targets.values()):
        raise ValueError("六档实验需要四组非负递增的六档增益，并区分前25和第26–50")
    cfg.update(five_grade_source=str(reference), five_grade_targets={k: cfg["label_gains"][k] for k in BASES},
               split_top25=True, reference=str(out), label_gains=targets, four_model_targets=targets)
    cfg["single_catboost"]["params"] = cfg["catboost"]
    cfg.pop("selected_trial", None)
    cfg["catboost_tuned"] = False
    out.mkdir()
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(out, flush=True)
    source = Path(cfg["feature_source"])
    names = alpha158_columns(source)
    train_fold(source, cfg, names, cfg["final_early_stop_year"], BASES, out / "final")
    # 前两期的股票日与原实验逐日一致；合并六档中的 4、5 应能精确还原原五档人数。
    counts = pd.read_csv(out / "final/grade_counts.csv")
    counts["4"] += counts.pop("5")
    pd.testing.assert_frame_equal(counts, pd.read_csv(old_final / "grade_counts.csv"))
    assert json.loads((out / "final/periods.json").read_text()) == json.loads((old_final / "periods.json").read_text())
    train_xgboost(out, cfg.copy(), source, names, out)
    fit_catboost(cfg, source, names, out, tune=False)
    return out


def evaluate_six_grades(out: Path) -> Path:
    """对齐旧五档和新六档信号，分别按五档、六档计算验证损失，回测全部新信号。"""
    import xgboost as xgb
    from catboost import CatBoostRanker

    cfg = json.loads((out / "config.json").read_text())
    source, old = Path(cfg["feature_source"]), Path(cfg["five_grade_source"])
    features = alpha158_columns(source)
    parts = []
    for split in ("validation", "test"):
        frame = read_period(source, cfg["segments"][split], split_top25=True)
        signals = frame[["trade_date", "ticker", "label", "label_end_date", "grade", "evaluable"]].copy()
        matrix = frame[features]
        for target in BASES:
            lg = lgb.Booster(model_file=str(out / "final" / target / "lambdarank.txt"))
            xg = xgb.Booster()
            xg.load_model(out / f"{target}.ubj")
            cat = CatBoostRanker().load_model(str(out / "trial_0" / f"{target}.cbm"))
            signals[target] = lg.predict(matrix, num_threads=cfg["lightgbm"]["num_threads"])
            signals["xgb_" + target] = xg.inplace_predict(matrix)
            signals["cat_" + target] = cat.predict(matrix, thread_count=cfg["catboost"]["thread_count"])
        for prefix in ("", "xgb_", "cat_"):
            signals[prefix + "equal"] = equal_ranks(signals, [prefix + n for n in BASES])
        xgb_weight = cfg["lightgbm_xgboost_blend"]["xgboost_weight"]
        cat_weight = cfg["catboost_four_equal"]["catboost_weight"]
        signals["blend_equal"] = (1 - xgb_weight) * signals.equal + xgb_weight * signals.xgb_equal
        signals["three_equal"] = (1 - cat_weight) * signals.blend_equal + cat_weight * signals.cat_equal
        keys = ["trade_date", "ticker"]
        prior = pd.read_parquet(old / "predictions.parquet", filters=[("split", "==", split)],
                                columns=keys + ["evaluable", "label", "grade", *SIX_GRADE_METHODS])
        prior = prior.set_index(keys, verify_integrity=True).sort_index()
        signals = signals.set_index(keys, verify_integrity=True).sort_index()
        pd.testing.assert_frame_equal(signals[["evaluable", "label"]], prior[["evaluable", "label"]])
        pd.testing.assert_series_equal(signals.grade.clip(upper=4), prior.grade)
        parts.append(signals.join(prior[list(SIX_GRADE_METHODS)].add_prefix("old_")).reset_index().assign(split=split))
    scores = pd.concat(parts, ignore_index=True)
    scores.to_parquet(out / "predictions.parquet", index=False)
    valid = scores.loc[scores.split.eq("validation")]
    methods = (*SIX_GRADE_METHODS, *["old_" + m for m in SIX_GRADE_METHODS])
    losses = []
    for scheme, frame, gains in [(5, valid.assign(grade=valid.grade.clip(upper=4)), cfg["five_grade_targets"]),
                               (6, valid, cfg["label_gains"])]:
        for m in methods:
            losses.append(validation_loss(frame, frame[m], gains, cfg["topk"]).assign(strategy=m, scheme=scheme))
    pd.concat(losses).to_csv(out / "validation_loss_daily.csv", index=False)
    evaluate_signals(scores, cfg, out, methods, portfolio_methods=SIX_GRADE_METHODS)
    from src.report.rank_stack import write_six_grades
    return write_six_grades(out)


def six_grades_main(argv=None):
    parser = argparse.ArgumentParser(description='在已冻结的五档排序基线上，检验把真实前 50 拆分为两个档位的效果。')
    parser.add_argument("--reference", type=Path, help="原五档 CatBoost 混合调参归档")
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    parser.add_argument("--evaluate", type=Path, help="复用该次实验的十二个六档模型")
    args = parser.parse_args(argv)
    if args.evaluate:
        directory = args.evaluate.resolve()
    else:
        ref = args.reference or latest_run("trial_0/top50.cbm", "blend_trials.csv", "validation_losses.csv", "ledger.parquet", rebalance_days=1)
        directory = train_six_grades(ref.resolve(), json.loads(args.config.read_text())["six_grade_targets"])
    print(evaluate_six_grades(directory), flush=True)


if __name__ == "__main__":
    main()
