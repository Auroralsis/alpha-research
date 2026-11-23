"""XGBoost、CatBoost 四目标排序和树模型组合对照。"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src.backtest.engine import evaluate_signals
from src.backtest.factor_metrics import daily_ic
from src.config import walk_forward_policy, CONFIG_FILE, load_config, load_tuning, with_model_params, write_best_config
from src.data.research_panel import ROOT, stock_day_index
from src.ml.dataset import alpha158_columns
from src.ml import rank_validation
from src.ml.rank_validation import (
    BASES, aligned_reference, candidates_from_grid, equal_ranks, training_periods, validation_loss,
)
from src.scripts.run_all import RUNS, latest_run, new_run
from src.scripts.run_rank_stack import evaluate_stack


def verify_baseline_returns(out, reference, strategy):
    """按交易日和偏移，核对复用策略的逐日扣费收益。"""
    def returns(folder):
        ledger = pd.read_parquet(folder / "ledger.parquet", filters=[("strategy", "==", strategy)])
        return ledger.set_index(["trade_date", "offset"]).daily_return.sort_index()

    pd.testing.assert_series_equal(returns(out), returns(reference), atol=1e-10, rtol=1e-10)


def fit_target(params: dict, cfg: dict, train_set: xgb.DMatrix, early_stop: xgb.DMatrix):
    """在本组标签上训练，用早停期选出的轮数截取模型和记录曲线。"""
    history = {}
    model = xgb.train(params, train_set, cfg["num_boost_round"], evals=[(early_stop, "early_stop")],
                      early_stopping_rounds=cfg["early_stopping_rounds"], maximize=True,
                      evals_result=history, verbose_eval=False)
    rounds = model.best_iteration + 1
    curve = pd.Series(history["early_stop"][params["eval_metric"]], name="ndcg_at_50")
    curve.index = pd.RangeIndex(1, len(curve) + 1, name="iteration")
    return model[:rounds], curve


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
    for _, rows, sizes in training_periods(source, reference, cfg["segments"], split_top25=cfg.get("split_top25", False)):
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
        frame = rank_validation.read_period(source, cfg["segments"]["validation"], split_top25=cfg.get("split_top25", False))
        rows = frame.loc[frame.evaluable]
        validation = xgb.DMatrix(rows[names], group=rows.groupby("trade_date", sort=False).size().to_numpy(),
                                 nthread=params["nthread"])
        matrices.append(validation)
        grades.append(rows.grade.to_numpy(dtype=int))
        validation_keys = frame[["trade_date", "ticker", "grade", "evaluable"]].copy()
        validation_keys["reference"] = aligned_reference(reference, frame, "validation", ["equal"])["equal"]
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
        frame = rank_validation.read_period(source, cfg["segments"][split])
        scored = frame[keys + ["label", "label_end_date", "grade", "evaluable"]].copy()
        matrix = frame[names]
        for name in BASES:
            scored[f"xgb_{name}"] = models[name].inplace_predict(matrix)
        chunks.append(scored.assign(split=split))
        del frame, matrix
    scores = pd.concat(chunks).set_index(["split", *keys], verify_integrity=True).sort_index()
    old = pd.read_parquet(reference / "predictions.parquet").set_index(scores.index.names, verify_integrity=True).sort_index()
    pd.testing.assert_frame_equal(old[["label", "label_end_date", "grade", "evaluable"]],
                                  scores[["label", "label_end_date", "grade", "evaluable"]])
    scores = scores.join(old[[*BASES, "equal"]]).reset_index()
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
    args = parser.parse_args(argv)
    if args.tuning_config and not args.tune:
        parser.error("--tuning-config 需要同时指定 --tune")
    settings = load_config(args.config)
    if args.reweight:
        reference = args.reference or latest_run("top50.ubj", "blend_trials.csv", "ledger.parquet", rebalance_days=1)
        print(compare_target_weights(reference, settings["four_model_weights"]))
        return
    requested = json.loads(args.config.read_text())
    baseline = json.loads((RUNS / "baselines/nav_comparison_daily/baseline.json").read_text())
    reference = (args.reference or ROOT / baseline["rank_run"]).resolve()
    cfg = json.loads((reference / "config.json").read_text())
    source = Path(cfg["feature_source"])
    names = alpha158_columns(source)
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
    cfg.update(xgboost_version=xgb.__version__, reference=str(reference))
    cfg.pop("neutral_search", None)
    out = new_run(cfg["rebalance_days"], tag="xgboost_tuned" if args.tune else "xgboost")
    out.mkdir()
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    print(out, flush=True)
    models = train_xgboost(reference, cfg, source, names, out, tune=args.tune)
    if args.tune:
        selected = with_model_params(requested, "xgboost", cfg["xgboost"])
        write_best_config(out, selected)
        (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    predict_xgboost(reference, cfg, source, names, models, out)
    print(evaluate_xgboost(out), flush=True)


def fit_catboost(cfg, source, names, out, tune):
    """训练候选并保存各组的验证分数。选参之前只读训练、早停和验证样本。"""
    from catboost import CatBoostRanker, Pool

    params = cfg["single_catboost"]["params"]
    space = cfg["catboost_four_equal"]["tuning"]["search_space"] if tune else {}
    candidates = candidates_from_grid(params, space)
    if tune and cfg.get("walk_forward", {}).get("early_stop_years"):
        from src.ml.walk_forward import rank_losses

        options = {"lightgbm": [cfg["lightgbm"]], "xgboost": [cfg["xgboost"]], "catboost": candidates}
        table = rank_losses(out, cfg, options, cfg["lightgbm_xgboost_blend"]["xgboost_weight"])
        table.rename(columns={"cat_index": "trial"}).to_csv(out / "fold_trials.csv", index=False)
    split_data = []
    for _, rows, _sizes in training_periods(source, Path(cfg["reference"]), cfg["segments"], split_top25=cfg.get("split_top25", False)):
        split_data.append((rows[names], rows.grade.to_numpy(dtype=int), pd.factorize(rows.trade_date)[0]))
        del rows
    valid = rank_validation.read_period(source, cfg["segments"]["validation"], split_top25=cfg.get("split_top25", False))
    logs = []
    for number, candidate in enumerate(candidates):
        directory = out / f"trial_{number}"
        directory.mkdir(exist_ok=True)
        (directory / "params.json").write_text(json.dumps(candidate, indent=2))
        scores = valid[["trade_date", "ticker", "grade", "evaluable"]].copy()
        for name in BASES:
            gains = np.array(cfg["label_gains"][name])
            train, stop = [Pool(data, label=gains[grade], group_id=groups, thread_count=params["thread_count"])
                           for data, grade, groups in split_data]
            model = CatBoostRanker(**candidate, iterations=cfg["num_boost_round"], random_seed=cfg["seed"],
                                   allow_writing_files=False)
            model.fit(train, eval_set=stop, use_best_model=True, early_stopping_rounds=cfg["early_stopping_rounds"],
                      verbose=False)
            model.save_model(str(directory / f"{name}.cbm"))
            scores[name] = model.predict(valid[names], thread_count=params["thread_count"])
            curves = model.get_evals_result()["validation"]
            metric = next(key for key in curves if key.startswith("NDCG:") and "top=50" in key.split(":")[1].split(";"))
            curve = pd.Series(curves[metric], index=pd.RangeIndex(1, len(curves[metric])+1, name="iteration"), name="ndcg_at_50")
            curve.to_csv(directory / f"{name}_early_stop.csv")
            logs.append(dict(trial=number, target=name, rounds=model.tree_count_,
                             early_stop_ndcg=curve.loc[model.tree_count_],
                             **{key: candidate[key] for key in space}))
            pd.DataFrame(logs).to_csv(out / "training.csv", index=False)
            print(f"CatBoost 候选 {number} {name} 最佳轮数 {model.tree_count_}", flush=True)
            del train, stop
        scores.to_parquet(directory / "validation.parquet", index=False)


def score_candidate(cfg, directory):
    """用指定归档的四个模型打分，并与冻结的双模型分数合并。"""
    from catboost import CatBoostRanker

    source = Path(cfg["feature_source"])
    ref = Path(cfg["xgboost_reference"])
    names = alpha158_columns(source)
    w = cfg["catboost_four_equal"]["catboost_weight"]
    scored = []
    for split in ("validation", "test"):
        frame = rank_validation.read_period(source, cfg["segments"][split])
        scores = frame[["trade_date", "ticker", "grade", "label", "evaluable"]].copy()
        refs = aligned_reference(ref, frame, split, [*BASES, "equal", "xgb_equal", "blend_equal",
                                                      *["xgb_"+n for n in BASES]])
        scores = scores.join(refs)
        cat_names = []
        for name in BASES:
            key = "cat_"+name
            model = CatBoostRanker().load_model(str(directory / f"{name}.cbm"))
            scores[key] = model.predict(frame[names], thread_count=cfg["catboost"]["thread_count"])
            cat_names.append(key)
        scores["cat_equal"] = equal_ranks(scores, cat_names)
        scores["three_equal"] = scores.blend_equal*(1-w) + scores.cat_equal*w
        scored.append(scores.assign(split=split))
    return pd.concat(scored, ignore_index=True)


def evaluate_catboost(out):
    """先以验证集冻结候选，再预测测试集，并按相同 Qlib 规则回测。"""
    cfg = json.loads((out / "config.json").read_text())
    ref = Path(cfg["xgboost_reference"])
    w = cfg["catboost_four_equal"]["catboost_weight"]
    gains = {name: cfg["label_gains"][name] for name in BASES}
    k = cfg["topk"]
    best_loss = np.inf
    trials = []
    space = cfg["catboost_four_equal"]["tuning"]["search_space"] if cfg["catboost_tuned"] else {}
    candidates = candidates_from_grid(cfg["single_catboost"]["params"], space)
    for number, params in enumerate(candidates):
        folder = out / f"trial_{number}"
        if json.loads((folder / "params.json").read_text()) != params:
            raise ValueError("候选的实际参数与实验配置不一致")
        frame = pd.read_parquet(folder / "validation.parquet")
        base = aligned_reference(ref, frame, "validation", ["blend_equal"]).blend_equal
        base_loss = validation_loss(frame, base, gains, k).loss.mean()
        daily, metrics = blend_validation(frame, base, w, gains, k, base_loss)
        daily.to_csv(folder / "blend_loss_daily.csv", index=False)
        loss = metrics["blend_loss"]
        trials.append(dict(trial=number, **metrics, **params))
        if loss < best_loss:
            best_loss, best = loss, number
    table = pd.DataFrame(trials)
    fixed_validation = table.set_index("trial")
    if cfg["catboost_tuned"] and cfg.get("walk_forward", {}).get("early_stop_years"):
        detail = pd.read_csv(out / "fold_trials.csv")
        summary = detail.groupby("trial")[["baseline_loss", "model_loss", "blend_loss"]].mean()
        summary["days"] = detail.groupby("trial").days.sum()
        summary["delta_vs_reference"] = summary.blend_loss - summary.baseline_loss
        table = fixed_validation.drop(columns=summary.columns).join(summary, validate="one_to_one").reset_index()
        best = int(summary.blend_loss.idxmin())
    table.to_csv(out / "blend_trials.csv", index=False)
    directory = out / f"trial_{best}"
    xgb_config = json.loads((ref / "config.json").read_text())
    cfg["xgboost"] = xgb_config["xgboost"]
    cfg["xgboost_selected_trial"] = xgb_config.get("selected_trial")
    cfg.update(selected_trial=best, catboost=json.loads((directory / "params.json").read_text()))
    cfg["four_model_targets"] = gains
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    base_config = json.loads((ref / "best_config.json").read_text())
    base_config.update({key: cfg[key] for key in ("seed", "num_boost_round", "early_stopping_rounds",
                                                  "single_catboost", "catboost_four_equal", "four_model_targets")})
    selected = with_model_params(base_config, "catboost", cfg["catboost"])
    write_best_config(out, selected)
    scores = score_candidate(cfg, directory)
    valid = scores.loc[scores.split.eq("validation")]
    check = validation_loss(valid, valid.three_equal, gains, k)
    np.testing.assert_allclose(check.loss.mean(), fixed_validation.at[best, "blend_loss"], rtol=0, atol=1e-12)
    losses = {name: validation_loss(valid, valid[name], gains, k).loss.mean()
              for name in ("blend_equal", "cat_equal")}
    losses["three_equal"] = check.loss.mean()
    np.testing.assert_allclose(list(losses.values()), fixed_validation.loc[best, ["baseline_loss", "model_loss", "blend_loss"]]
                               .to_numpy(dtype=float), rtol=0, atol=1e-12)
    pd.Series(losses, name="loss").rename_axis("strategy").to_csv(out / "validation_losses.csv")
    scores.to_parquet(out / "predictions.parquet", index=False)
    pairs = pd.concat([paired_signals(scores, k, prefix, "cat_").assign(reference=family)
                       for prefix, family in (("", "LightGBM"), ("xgb_", "XGBoost"))])
    pairs.to_csv(out / "paired_daily.csv", index=False)
    methods = (*["cat_" + name for name in BASES], "equal", "xgb_equal", "blend_equal", "cat_equal", "three_equal")
    evaluate_signals(scores, cfg, out, methods)
    verify_baseline_returns(out, ref, "blend_equal")
    from src.report.rank_stack import write_catboost_report
    return write_catboost_report(out)


def replay_candidate(archive: Path, trial: int):
    """复用已训练的指定候选，用相同账本比较单独与混合的测试表现。"""
    cfg = json.loads((archive / "config.json").read_text())
    directory = archive / f"trial_{trial}"
    cfg["catboost"] = json.loads((directory / "params.json").read_text())
    if cfg["selected_trial"] == trial:
        from src.report.rank_stack import write_catboost_report
        return write_catboost_report(archive)
    cfg["validation_selected_trial"] = cfg.pop("selected_trial")
    cfg["single_catboost"]["params"] = cfg["catboost"]
    chosen = pd.read_csv(archive / "blend_trials.csv").set_index("trial").loc[trial]
    out = new_run(cfg["rebalance_days"], tag="catboost_replay")
    out.mkdir()
    print(out, flush=True)
    cfg.update(replayed_trial=trial, model_archive=str(archive))
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    scores = score_candidate(cfg, directory)
    methods = ("blend_equal", "cat_equal", "three_equal")
    valid = scores.loc[scores.split.eq("validation")]
    gains = {name: cfg["label_gains"][name] for name in BASES}
    if cfg.get("walk_forward", {}).get("early_stop_years") and cfg["catboost_tuned"]:
        chosen = pd.read_csv(directory / "blend_loss_daily.csv").mean(numeric_only=True)
        np.testing.assert_allclose(validation_loss(valid, valid.three_equal, gains, cfg["topk"]).loss.mean(), chosen.loss)
    else:
        for name, col in zip(methods, ("baseline_loss", "model_loss", "blend_loss")):
            loss = validation_loss(valid, valid[name], gains, cfg["topk"])
            np.testing.assert_allclose(loss.loss.mean(), chosen[col], rtol=0, atol=1e-12)
    scores.to_parquet(out / "predictions.parquet", index=False)
    evaluate_signals(scores, cfg, out, methods)
    from src.report.rank_stack import write_catboost_replay
    return write_catboost_replay(out)


def catboost_main(argv=None):
    parser = argparse.ArgumentParser(description='比较 CatBoost 四组排序信号及其加入 LightGBM/XGBoost 后的验证损失。')
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    parser.add_argument("--xgboost-run", type=Path, help="按混合验证损失选参的 XGBoost 实验")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--tuning-config", type=Path, help="配合 --tune 指定候选网格，默认 configs/tuning.json")
    parser.add_argument("--fit-only", action="store_true", help="保存全部候选，可在两模型基线完成后评价")
    parser.add_argument("--evaluate", type=Path, help="使用已训练的 CatBoost 候选继续选参与评价")
    parser.add_argument("--trial", type=int, help="按候选编号单独回放，和 --evaluate 一起使用")
    args = parser.parse_args(argv)
    if args.tuning_config and (not args.tune or args.evaluate):
        parser.error("--tuning-config 仅用于 --tune 新实验")
    if args.evaluate:
        archive = args.evaluate.resolve()
        print(evaluate_catboost(archive) if args.trial is None else replay_candidate(archive, args.trial))
        return
    if args.trial is not None:
        parser.error("--trial 需要 --evaluate 指定原调参实验")
    import catboost

    settings = load_config(args.config)
    ref = (args.xgboost_run or latest_run("top50.ubj", "blend_trials.csv", "predictions.parquet", "ledger.parquet", rebalance_days=1)).resolve()
    cfg = json.loads((ref / "config.json").read_text())
    if any(settings["four_model_targets"][name] != cfg["label_gains"][name] for name in BASES):
        raise ValueError("CatBoost 与基线的四组增益不同")
    cfg.pop("selected_trial", None)
    for key in ("single_catboost", "catboost_four_equal", "seed", "num_boost_round", "early_stopping_rounds"):
        cfg[key] = settings[key]
    cfg.pop("walk_forward", None)
    if args.tune:
        cfg["walk_forward"] = walk_forward_policy(settings)
        cfg["catboost_four_equal"]["tuning"] = load_tuning("catboost_tuning", settings, args.tuning_config)
    weight = cfg["catboost_four_equal"]["catboost_weight"]
    if not 0 <= weight <= 1:
        raise ValueError("catboost_weight 须在 0 至 1 之间")
    cfg.update(xgboost_reference=str(ref), catboost_version=catboost.__version__, catboost_tuned=args.tune)
    out = new_run(cfg["rebalance_days"], tag="catboost")
    out.mkdir()
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(out, flush=True)
    source = Path(cfg["feature_source"])
    fit_catboost(cfg, source, alpha158_columns(source), out, args.tune)
    if not args.fit_only:
        print(evaluate_catboost(out))
