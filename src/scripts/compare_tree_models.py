"""比较树模型的分数、融合权重与堆叠方法。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.backtest.engine import evaluate_signals
from src.backtest.factor_metrics import daily_ic
from src.config import label_argument, set_label_days, CONFIG_FILE, fixed_validation, load_config, load_tuning, validation_argument, walk_forward_policy, with_model_params, write_best_config
from src.data.research_panel import ROOT, stock_day_index
from src.ml.dataset import alpha158_columns
from src.ml import rank_validation
from src.ml.rank_validation import (
    BASES, aligned_reference, candidates_from_grid, equal_ranks, training_periods, validation_loss,
)
from src.scripts.run_all import RUNS, latest_run, new_run
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from src.ml.rank_validation import METHODS, read_period
from src.scripts.train_lightgbm import evaluate_stack, frozen_lightgbm, rank_inputs, score, train_fold
from src.scripts.train_xgboost import fit_target


def verify_baseline_returns(out, reference, strategy):
    """按交易日和偏移，核对复用策略的逐日扣费收益。"""
    def returns(folder):
        ledger = pd.read_parquet(folder / "ledger.parquet", filters=[("strategy", "==", strategy)])
        return ledger.set_index(["trade_date", "offset"]).daily_return.sort_index()

    pd.testing.assert_series_equal(returns(out), returns(reference), atol=1e-10, rtol=1e-10)


def blend_validation(frame, baseline, weight, gains, k, baseline_loss):
    """对同一批验证股票日计算四模型等权、与基线混合后的排序损失。"""
    own = equal_ranks(frame, list(BASES))
    model_loss = validation_loss(frame, own, gains, k).loss.mean()
    daily = validation_loss(frame, (1 - weight) * baseline + weight * own, gains, k)
    loss = daily.loss.mean()
    return daily, dict(baseline_loss=baseline_loss, model_loss=model_loss, blend_loss=loss,
                       delta_vs_reference=loss - baseline_loss, days=len(daily))


def train_xgboost(reference: Path, cfg: dict, source: Path, names: list[str], out: Path, *, tune: bool = False) -> dict:
    """核对每个训练日的五档人数，按 2021 年 NDCG@50 早停并保存各组最优树。"""
    params = cfg["xgboost"]
    if tune and "walk_forward" in cfg:
        from src.ml.walk_forward import rank_losses

        space = cfg["xgboost_tuning"]["search_space"]
        candidates = candidates_from_grid(params, space)
        options = {"lightgbm": [cfg["lightgbm"]], "xgboost": candidates}
        weight = cfg.get("lightgbm_xgboost_blend", {}).get("xgboost_weight", 0.5)
        folds = rank_losses(out, cfg, options, weight).rename(columns={"xgb_index": "trial"})
        folds.to_csv(out / "fold_trials.csv", index=False)
        summary = folds.groupby("trial")[["model_loss", "baseline_loss", "blend_loss"]].mean()
        summary["delta_vs_reference"] = summary.blend_loss - summary.baseline_loss
        for key in space:
            summary[key] = [candidate[key] for candidate in candidates]
        summary.to_csv(out / "blend_trials.csv")
        selected = int(summary.blend_loss.idxmin())
        cfg.update(xgboost=candidates[selected], selected_trial=selected)
        return train_xgboost(reference, cfg, source, names, out)
    matrices, grades = [], []
    for _, rows, sizes in training_periods(source, reference, cfg["segments"], cfg.get("early_stop_bounds"), cfg):
        matrix = xgb.QuantileDMatrix(rows[names], group=sizes,
                                     ref=matrices[0] if matrices else None, max_bin=params.get("max_bin", 256),
                                     nthread=params["nthread"])
        matrices.append(matrix)
        grades.append(rows.grade.to_numpy(dtype=int))
        del rows
    train_set, early_stop = matrices
    candidates = [params]
    if tune:
        space = cfg["xgboost_tuning"]["search_space"]
        candidates = candidates_from_grid(params, space)
        frame = rank_validation.read_period(source, cfg["segments"]["validation"], cfg=cfg)
        rows = frame.loc[frame.evaluable]
        validation = xgb.DMatrix(rows[names], group=rows.groupby("trade_date", sort=False).size().to_numpy(),
                                 nthread=params["nthread"])
        matrices.append(validation)
        grades.append(rows.grade.to_numpy(dtype=int))
        validation_keys = frame[["trade_date", "ticker", "grade", "evaluable"]].copy()
        validation_keys["reference"] = (frozen_lightgbm(reference, frame, cfg).equal if cfg.get("early_stop_bounds") or cfg.get("label_end_days", 2) != 2
                                        else aligned_reference(reference, frame, "validation", ["equal"]).equal)
        validation_features = frame[names]
        weight = cfg.get("lightgbm_xgboost_blend", {}).get("xgboost_weight", 0.5)
        gains = {name: cfg["label_gains"][name] for name in BASES}
        baseline_loss = validation_loss(validation_keys, validation_keys.reference, gains, cfg["topk"]).loss.mean()
        del rows, frame

    trials = []
    mixtures = []
    best_loss = np.inf
    best_models = {}
    for number, candidate in enumerate(candidates):
        models, curves = {}, {}
        for name in BASES:
            gain = np.array(cfg["label_gains"][name])
            for matrix, grade in zip(matrices, grades):
                matrix.set_label(gain[grade])
            model, curve = fit_target(candidate, cfg, train_set, early_stop)
            models[name], curves[name] = model, curve
            if tune:
                value = model.eval(validation, name="validation").rsplit(":", 1)[1]
                trials.append(dict(trial=number, target=name, rounds=model.num_boosted_rounds(),
                                   early_stop_ndcg=curve.max(), validation_ndcg=float(value),
                                   **{key: candidate[key] for key in space}))
                pd.DataFrame(trials).to_csv(out / "trials.csv", index=False)
        if tune:
            for name, model in models.items():
                validation_keys[name] = model.inplace_predict(validation_features)
            daily, metrics = blend_validation(validation_keys, validation_keys.reference, weight, gains,
                                              cfg["topk"], baseline_loss)
            loss = metrics["blend_loss"]
            daily.to_csv(out / f"validation_loss_{number}.csv", index=False)
            mixtures.append(dict(trial=number, **metrics, **{key: candidate[key] for key in space}))
            pd.DataFrame(mixtures).to_csv(out / "blend_trials.csv", index=False)
            print(f"候选 {number + 1}/{len(candidates)} 完成，混合验证损失 {loss:.6f}", flush=True)
            if not loss < best_loss:
                continue
            best_loss = loss
            cfg["xgboost"] = candidate
            cfg["selected_trial"] = number
        best_models = models
        for name, model in models.items():
            model.save_model(out / f"{name}.ubj")
            curves[name].to_csv(out / f"{name}_early_stop.csv")
    return best_models


def predict_xgboost(reference: Path, cfg: dict, source: Path, names: list[str], models: dict, out: Path) -> None:
    """逐股票日核对归档预测及标签，保留没有未来收益的股票分数。"""
    keys = ["trade_date", "ticker"]
    chunks = []
    for split in ("validation", "test"):
        frame = rank_validation.read_period(source, cfg["segments"][split], cfg=cfg)
        scored = frame[keys + ["label", "label_end_date", "grade", "evaluable"]].copy()
        matrix = frame[names]
        for name in BASES:
            scored[f"xgb_{name}"] = models[name].inplace_predict(matrix)
        if cfg.get("early_stop_bounds") or cfg.get("label_end_days", 2) != 2:
            scored = scored.join(frozen_lightgbm(reference, frame, cfg))
        chunks.append(scored.assign(split=split))
        del frame, matrix
    scores = pd.concat(chunks).set_index(["split", *keys], verify_integrity=True).sort_index()
    if not cfg.get("early_stop_bounds") and cfg.get("label_end_days", 2) == 2:
        old = pd.read_parquet(reference / "predictions.parquet").set_index(scores.index.names, verify_integrity=True).sort_index()
        pd.testing.assert_frame_equal(old[["label", "label_end_date", "grade", "evaluable"]],
                                      scores[["label", "label_end_date", "grade", "evaluable"]])
        scores = scores.join(old[[*BASES, "equal"]])
    scores = scores.reset_index()
    xcols = [f"xgb_{name}" for name in BASES]
    if not np.isfinite(scores[[*BASES, *xcols]].to_numpy()).all():
        raise ValueError("配对模型存在无效分数")
    scores["xgb_equal"] = equal_ranks(scores, xcols)
    weight = cfg.get("lightgbm_xgboost_blend", {}).get("xgboost_weight", 0.5)
    scores["blend_equal"] = scores.equal * (1 - weight) + scores.xgb_equal * weight
    scores.to_parquet(out / "predictions.parquet", index=False)


def paired_signals(scores: pd.DataFrame, k: int, left: str = "", right: str = "xgb_") -> pd.DataFrame:
    """逐日比较同目标的两个完整打分截面，分数并列时按股票代码确定前 k 名。"""
    pieces = []
    for split, frame in scores.groupby("split"):
        frame = frame.sort_values(["trade_date", "ticker"])
        index = stock_day_index(frame)
        daily = frame.groupby("trade_date", sort=False)
        counts = daily.size()
        if counts.min() < k:
            raise ValueError("共同股票池小于 Top-K")
        for name in (*BASES, "equal"):
            first, other = left + name, right + name
            ranks = daily[[first, other]].rank(method="first", ascending=False)
            common = ranks.le(k).all(axis=1).groupby(frame.trade_date).sum()
            pearson, spearman = daily_ic(frame[first].set_axis(index), frame[other].set_axis(index))
            part = pd.DataFrame(dict(pearson=pearson, spearman=spearman, shared_picks=common,
                                     overlap=common / k, n_scored=counts))
            pieces.append(part.rename_axis("trade_date").reset_index().assign(split=split, target=name))
    return pd.concat(pieces, ignore_index=True)


def evaluate_xgboost(out: Path) -> Path:
    """在共同股票池中评价相关性、信号有效性以及加入新分数后的组合收益。"""
    cfg = json.loads((out / "config.json").read_text())
    reference = Path(cfg["reference"])
    scores = pd.read_parquet(out / "predictions.parquet")
    valid = scores.loc[scores.split.eq("validation")]
    gains = {name: cfg["label_gains"][name] for name in BASES}
    losses = {name: validation_loss(valid, valid[name], gains, cfg["topk"]).loss.mean()
              for name in ("equal", "xgb_equal", "blend_equal")}
    pd.Series(losses, name="loss").rename_axis("strategy").to_csv(out / "validation_losses.csv")
    if (out / "blend_trials.csv").is_file():
        selected = pd.read_csv(out / "blend_trials.csv").set_index("trial").loc[cfg["selected_trial"]]
        if "walk_forward" in cfg:
            folds = pd.read_csv(out / "fold_trials.csv")
            expected = folds.loc[folds.trial.eq(cfg["selected_trial"]), "blend_loss"].mean()
        else:
            expected = losses["blend_equal"]
        np.testing.assert_allclose(expected, selected.blend_loss, rtol=0, atol=1e-12)
    daily = paired_signals(scores, cfg["topk"])
    daily.to_csv(out / "paired_daily.csv", index=False)
    summary = daily.groupby(["split", "target"]).agg(
        pearson=("pearson", "mean"), spearman=("spearman", "mean"), shared=("shared_picks", "mean"),
        overlap=("overlap", "mean"), days=("n_scored", "size"), spearman_days=("spearman", "count"),
        stock_days=("n_scored", "sum"))
    summary.to_csv(out / "paired_summary.csv")
    xcols = tuple(f"xgb_{name}" for name in BASES)
    ensembles = ("equal", "xgb_equal", "blend_equal")
    methods = (*BASES, *xcols, *ensembles)
    evaluate_signals(scores, cfg, out, methods, portfolio_methods=(*xcols, *ensembles))
    # 同一归档信号重放后，逐日扣费收益应与原账本一致。
    verify_baseline_returns(out, reference, "equal")
    from src.report.rank_stack import write_xgboost_report
    return write_xgboost_report(out, summary, reference)


def compare_target_weights(reference: Path, weights: dict) -> Path:
    """复用冻结的八组预测，比较同一交易规则下等权和指定权重的结果。"""
    if set(weights) != set(BASES) or any(type(w) not in (int, float) or not 0 <= w <= 1 for w in weights.values()):
        raise ValueError("四项目标各需一个 0 到 1 之间的权重")
    if not np.isclose(sum(weights.values()), 1, rtol=0, atol=1e-12):
        raise ValueError("四项权重之和须为 1")
    cfg = json.loads((reference / "config.json").read_text())
    scores = pd.read_parquet(reference / "predictions.parquet")
    for prefix in ("", "xgb_"):
        cols = [prefix + name for name in BASES]
        if not np.isfinite(scores[cols]).all(axis=None):
            raise ValueError("基础模型预测缺分")
        ranks = scores.groupby("trade_date")[cols].rank(pct=True)
        np.testing.assert_allclose(ranks.mean(axis=1), scores[prefix + "equal"], rtol=0, atol=1e-12)
        scores[prefix + "weighted"] = ranks @ [weights[name] for name in BASES]
    w = cfg["lightgbm_xgboost_blend"]["xgboost_weight"]
    scores["blend_weighted"] = (1 - w) * scores.weighted + w * scores.xgb_weighted
    methods = ("equal", "weighted", "xgb_equal", "xgb_weighted", "blend_equal", "blend_weighted")
    valid = scores.loc[scores.split.eq("validation")]
    gains = {name: cfg["label_gains"][name] for name in BASES}
    daily = pd.concat([validation_loss(valid, valid[name], gains, cfg["topk"]).assign(strategy=name)
                       for name in methods], ignore_index=True)
    out = new_run(cfg["rebalance_days"], tag="target_weights")
    out.mkdir()
    print(out, flush=True)
    cfg.update(four_model_weights=weights, prediction_source=str(reference.resolve()))
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    daily.to_csv(out / "validation_loss_daily.csv", index=False)
    evaluate_stack(scores, cfg, out, methods)
    from src.report.rank_stack import write_target_weights
    return write_target_weights(out)


def xgboost_main(argv=None) -> None:
    parser = argparse.ArgumentParser(description='按归档 LightGBM 的四组增益训练 XGBoost，并比较同日选股分数。')
    parser.add_argument("--reference", type=Path, help="训练时指定四组 LightGBM 归档，比较权重时指定已调参的双模型归档")
    parser.add_argument("--config", type=Path, default=CONFIG_FILE, help="XGBoost 固定参数和训练设置")
    parser.add_argument("--tuning-config", type=Path, help="配合 --tune 指定候选网格，默认 configs/tuning.json")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--tune", action="store_true", help="以最终混合分数的验证损失选择共享参数")
    mode.add_argument("--reweight", action="store_true", help="用归档预测比较 four_model_weights 与原等权配置")
    label_argument(parser)
    validation_argument(parser)
    args = parser.parse_args(argv)
    if args.tuning_config and not args.tune:
        parser.error("--tuning-config 需要同时指定 --tune")
    settings = load_config(args.config)
    if args.reweight and args.label_days is not None:
        parser.error("归档重配权重沿用原训练标签")
    if args.reweight:
        if args.validation_mode:
            parser.error("归档重配权重沿用原验证窗口")
        reference = args.reference or latest_run("top50.ubj", "blend_trials.csv", "ledger.parquet", rebalance_days=1)
        print(compare_target_weights(reference, settings["four_model_weights"]))
        return
    requested = json.loads(args.config.read_text())
    baseline = json.loads((RUNS / "baselines/nav_comparison_daily/baseline.json").read_text())
    reference = (args.reference or ROOT / baseline["rank_run"]).resolve()
    cfg = json.loads((reference / "config.json").read_text())
    source = Path(cfg["feature_source"])
    names = alpha158_columns(source)
    set_label_days(cfg, args.label_days if args.label_days is not None else settings.get("label_days"))
    for key in ("num_boost_round", "early_stopping_rounds", "seed"):
        cfg[key] = settings[key]
    if "four_model_targets" in settings and any(
            settings["four_model_targets"][name] != cfg["label_gains"][name] for name in BASES):
        raise ValueError("四组增益与 LightGBM 归档不同。请先用新增益训练 LightGBM，再用 --reference 指向该实验")
    if "lightgbm_xgboost_blend" in settings:
        cfg["lightgbm_xgboost_blend"] = settings["lightgbm_xgboost_blend"]
    params = dict(seed=cfg["seed"])
    params.update(settings["xgboost"])
    cfg["xgboost"] = params
    if (params["eval_metric"] != f"ndcg@{cfg['topk']}" or params["objective"] != "rank:ndcg"
            or params["ndcg_exp_gain"] is not False):
        raise ValueError("配对实验需要 rank:ndcg、线性增益和与持股数相应的 NDCG 指标")
    cfg.pop("xgboost_tuning", None)
    cfg.pop("selected_trial", None)
    cfg.pop("walk_forward", None)
    if args.tune:
        cfg["walk_forward"] = walk_forward_policy(settings)
        cfg["xgboost_tuning"] = load_tuning("xgboost_tuning", settings, args.tuning_config)
        space = cfg["xgboost_tuning"]["search_space"]
        tunable = {"max_depth", "min_child_weight", "reg_alpha", "reg_lambda", "subsample", "colsample_bytree", "learning_rate"}
        if not space or not space.keys() <= tunable & cfg["xgboost"].keys() or not all(space.values()):
            raise ValueError("xgboost_tuning.search_space 需要有效的树结构、正则化、采样率或学习率候选")
    fixed_validation(cfg, args.validation_mode or settings.get("validation_mode") or cfg.get("validation_mode"))
    cfg.update(xgboost_version=xgb.__version__, reference=str(reference))
    cfg.pop("neutral_search", None)
    out = new_run(cfg["rebalance_days"], tag="xgboost_tuned" if args.tune else "xgboost")
    out.mkdir()
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(out, flush=True)
    models = train_xgboost(reference, cfg, source, names, out, tune=args.tune)
    if args.tune:
        selected = with_model_params(requested, "xgboost", cfg["xgboost"])
        set_label_days(selected, cfg.get("label_days"))
        fixed_validation(selected, cfg.get("validation_mode"))
        write_best_config(out, selected)
        (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    predict_xgboost(reference, cfg, source, names, models, out)
    print(evaluate_xgboost(out), flush=True)


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


def run_equal_seed(reference: Path, seed: int, validation_mode=None, label_days=None) -> Path:
    """只更换基础模型种子，训练四个排名模型并与存档等权预测配对回测。"""
    cfg = json.loads((reference / "config.json").read_text())
    fixed_validation(cfg, validation_mode)
    set_label_days(cfg, label_days)
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
        frame = read_period(source, cfg["segments"][split], cfg=cfg)
        scores = score(frame, features, models, cfg["lightgbm"]["num_threads"])
        scores["equal"] = rank_inputs(scores).mean(axis=1)
        if validation_mode:
            scores["reference"] = frozen_lightgbm(reference, frame, cfg).equal.to_numpy()
        pieces.append(scores.assign(split=split))
        del frame
    keys = ["split", "trade_date", "ticker"]
    scores = pd.concat(pieces).set_index(keys).sort_index()
    archived = pd.read_parquet(reference / "predictions.parquet",
                               columns=keys + ["equal", "evaluable", "label"]).set_index(keys).sort_index()
    if not validation_mode:
        if label_days is None:
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


def run(source: Path, historical: Path, config_path: Path = CONFIG_FILE, validation_mode=None, label_days=None) -> Path:
    """冻结样本外逻辑回归与最终基础模型，再进行统一诊断与回测。"""
    metadata = json.loads((historical / "source.json").read_text())
    if Path(metadata["baseline"]).resolve() != source.resolve() or metadata["target"] != "daily_quintile_gap3_relevance":
        raise ValueError("历史对照必须来自相同特征和参数来源的五等分间距3实验")
    cfg = load_config(source / "best_config.json")
    if cfg["topk"] != 50 or cfg["rebalance_days"] != 1:
        raise ValueError("本实验按日频 Top-50 设计")
    cfg["segments"]["validation"] = ["2022-01-01", "2022-12-31"]
    fixed_validation(cfg, validation_mode)
    settings = load_config(config_path)
    set_label_days(cfg, label_days if label_days is not None else settings.get("label_days"))
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
        frame = read_period(source, [f"{year}-01-01", f"{year}-12-31"], cfg=cfg)
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
        frame = read_period(source, cfg["segments"][split], cfg=cfg)
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


def stack_main(argv=None):
    parser = argparse.ArgumentParser(description='用固定排名五档比较四种排序目标及跨年度样本外堆叠。')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source", type=Path)
    mode.add_argument("--blend-from", type=Path, help="从已有四模型的样本外预测拟合连续排名堆叠")
    mode.add_argument("--equal-seed-from", type=Path, help="采用四模型等权，与原实验比较一个新种子")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--historical", type=Path)
    parser.add_argument("--config", type=Path, help="新训练读取的四模型等权配置")
    label_argument(parser)
    validation_argument(parser)
    args = parser.parse_args(argv)
    if args.config and not args.source:
        parser.error("--config 适用于 --source 新训练，归档重放使用该次实验的配置")
    if args.equal_seed_from and args.seed is not None and not args.historical:
        print(run_equal_seed(args.equal_seed_from, args.seed, args.validation_mode, args.label_days))
    elif args.seed is not None or args.equal_seed_from:
        parser.error("--equal-seed-from 需配合 --seed，不使用 --historical")
    elif args.blend_from and not args.historical:
        if args.validation_mode or args.label_days is not None:
            parser.error("归档融合沿用原训练窗口")
        print(run_blend(args.blend_from))
    elif args.source and args.historical:
        print(run(args.source, args.historical, args.config or CONFIG_FILE, args.validation_mode, args.label_days))
    else:
        parser.error("--historical 仅与 --source 成对使用")
