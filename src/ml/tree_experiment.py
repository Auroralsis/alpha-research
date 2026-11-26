"""用验证期最终等权排名，逐步调优 LightGBM 和 XGBoost 的树参数。"""
import argparse
import fcntl
import json
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from src.backtest.engine import evaluate_signals, run_backtests
from src.data.research_panel import CSI500_INPUTS, load_market_calendar, prepare_test_panel
from src.config import label_argument, set_label_days, CONFIG_FILE, fixed_validation, load_config, load_tuning, validation_argument, walk_forward_policy, write_best_config
from src.ml.tuning import STABILITY_PENALTY, STAGES, TUNING_LOSSES, candidate_id, candidates, frozen_scores, select_rank_params
from src.ml.dataset import alpha158_columns, read_feature_split
from src.ml.rank_validation import (
    BASES, aligned_reference, equal_ranks, fuse_rankers, fixed_grades, read_period, selection_stats, training_periods, validation_loss,
)
from src.scripts.run_all import RUNS, latest_run, new_run
from src.scripts.train_lightgbm import fit_ranker
from src.scripts.train_xgboost import fit_target
from src.data.research_panel import segment_masks

FAMILIES = ("lightgbm", "xgboost")


def check_config(cfg):
    """核对搜索阶段和组合口径，树参数由对应的训练器校验。"""
    grid = cfg["rank_pair_tuning"]
    stages = grid.get("stages", STAGES)
    if not stages or set(stages) - set(STAGES) or len(stages) != len(set(stages)):
        raise ValueError("rank_pair_tuning.stages 需要有效且不重复的搜索阶段")
    fields = {"leaves": ("leaf_cap_fractions",), "min_child": ("min_data_in_leaf", "min_child_weight")}
    for stage in stages:
        for key in fields.get(stage, (stage,)):
            if not grid[key]:
                raise ValueError(f"rank_pair_tuning.{key} 需要至少一个候选值")
    if "leaves" in stages and not all(0 < f <= 1 for f in grid["leaf_cap_fractions"]):
        raise ValueError("leaf_cap_fractions 须在 (0, 1] 内")
    if "depth" in stages and any(type(d) is not int or d < 1 for d in grid["depth"]):
        raise ValueError("depth 候选须为正整数")
    if "row_sampling" in stages and cfg["lightgbm"].get("bagging_freq", cfg["lightgbm"].get("subsample_freq", 0)) <= 0:
        raise ValueError("LightGBM 行采样需要正的 bagging_freq")
    if cfg["lightgbm_xgboost_blend"]["xgboost_weight"] != 0.5 or any(len(cfg["label_gains"][n]) != 5 for n in BASES):
        raise ValueError("基准须为使用五档标签的等权组合")


def scoring_inputs(cfg):
    """核对股票日、逐日标签人数与收益终点，仅装入训练和早停两段的矩阵。"""
    source = Path(cfg["feature_source"])
    features = alpha158_columns(source)
    blocks = []
    for _, rows, sizes in training_periods(source, Path(cfg["reference"]) if "reference" in cfg else None, cfg["segments"], cfg.get("early_stop_bounds"), cfg):
        blocks.append((rows[features], rows.grade.to_numpy(dtype=int), sizes))
    return blocks


def train_candidate(out, family, params, cfg, blocks, xmat, valid_x=None, *, targets=BASES):
    """同一树参数训练 A–D 四个目标；完整写出一个目标后即可断点接续。"""
    directory = out / family / candidate_id(params)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "params.json").write_text(json.dumps(params, indent=2))
    matrices = []
    if family == "lightgbm":
        for x, y, groups in blocks:
            matrices.append(lgb.Dataset(x, label=y, group=groups, reference=matrices[0] if matrices else None))
    predictions = {}
    for target in targets:
        folder = directory / target
        folder.mkdir(exist_ok=True)
        gains = cfg["label_gains"][target]
        if family == "lightgbm":
            path = folder / "lambdarank.txt"
            if path.exists():
                model = lgb.Booster(model_file=str(path))
            else:
                with TemporaryDirectory(dir=folder) as work:
                    temp = Path(work)
                    model = fit_ranker(matrices, dict(params, label_gain=gains), cfg, temp)
                    (temp / "training_ndcg.csv").replace(folder / "training_ndcg.csv")
                    (temp / "lambdarank.txt").replace(path)
            if valid_x is not None:
                predictions[target] = model.predict(valid_x, num_threads=params["num_threads"])
        else:
            path = folder / "model.ubj"
            if path.exists():
                model = xgb.Booster()
                model.load_model(path)
            else:
                for matrix, (_, grade, _) in zip(xmat, blocks):
                    matrix.set_label(np.asarray(gains)[grade])
                model, curve = fit_target(params, cfg, *xmat)
                curve.to_csv(folder / "training_ndcg.csv")
                tmp = folder / "model.tmp.ubj"
                model.save_model(tmp)
                tmp.replace(path)
            if valid_x is not None:
                predictions[target] = model.inplace_predict(valid_x)
    if valid_x is not None:
        return pd.DataFrame(predictions, index=valid_x.index)


def quantile_matrices(blocks, params):
    """早停集沿用训练集的特征分箱。"""
    matrices = []
    for x, _, groups in blocks:
        matrices.append(xgb.QuantileDMatrix(x, group=groups, ref=matrices[0] if matrices else None,
                                             nthread=params["nthread"], max_bin=params.get("max_bin", 256)))
    return matrices


def tune_walk_forward(out, cfg, *, blocks=None, fold_features=None, extra=None):
    best = select_rank_params(out, cfg, FAMILIES, fold_features=fold_features, extra=extra)
    # 正式模型按原归档的训练和早停区间拟合，测试期始终留给 evaluate。
    if blocks is None:
        blocks = scoring_inputs(cfg)
    for family, params in best.items():
        refit = [cfg[family]] if cfg.get("label_end_days", 2) != 2 else []
        if params != cfg[family]:
            refit.append(params)
        for candidate in refit:
            matrices = quantile_matrices(blocks, candidate) if family == "xgboost" else []
            train_candidate(out, family, candidate, cfg, blocks, matrices)
    temp = out / "winners.tmp"
    temp.write_text(json.dumps(best, indent=2))
    temp.replace(out / "winners.json")
    return best


COMPOSITE_RETURN_SCALE = 0.20


def composite_loss(annual_return, sharpe, max_drawdown, return_scale=COMPOSITE_RETURN_SCALE):
    """按固定收益尺度计算验证期综合损失。"""
    r, sr, d = float(annual_return), float(sharpe), abs(float(max_drawdown))
    if not np.isfinite([r, sr, d]).all():
        raise ValueError("验证账本的收益、Sharpe 和回撤须为有限数值")
    pr = 1.5 * (max(0, -r) / return_scale) ** 2
    ps = 1.5 * (max(0, -sr) / 1.5) ** 2
    pd_ = (max(0, d - 0.15) / 0.15) ** 2
    floor = 10.0 if r < -0.20 else 0.0
    loss = -0.4 * r / return_scale - 0.4 * sr / 1.5 + 0.2 * d / 0.10 + pr + ps + pd_ + floor
    return dict(R=r, SR=sr, D=d, P_r=pr, P_s=ps, P_d=pd_, P_floor=floor, composite_loss=loss)


def validation_benchmark(dates):
    """复用验证期指数行情缓存，首次回测新窗口时按交易日补齐。"""
    from src.backtest.portfolio import load_benchmark_returns
    from src.data.download import pending_file, tushare_client

    start, end = dates[0].strftime("%Y%m%d"), dates[-1].strftime("%Y%m%d")
    path = CSI500_INPUTS["benchmark"].with_name(f"benchmark_{start}_{end}.csv")
    if not path.exists():
        frame = tushare_client().index_daily(ts_code="000905.SH", start_date=start, end_date=end,
                                            fields="trade_date,close,pre_close")
        with pending_file(path) as pending:
            frame.to_csv(pending, index=False)
            load_benchmark_returns(pending, dates)
    return path


def validation_backtest(cfg, keys, score, market):
    panel, dates = market
    inputs = dict(CSI500_INPUTS, benchmark=validation_benchmark(dates))
    signals = keys[["trade_date", "ticker"]].assign(portfolio=score)
    with TemporaryDirectory() as tmp:
        run_backtests(inputs, cfg, dates, signals, panel, Path(tmp), strategies=("portfolio",))
        row = pd.read_csv(Path(tmp) / "portfolio_metrics.csv").set_index("strategy").loc["portfolio"]
    return composite_loss(row.annual_return, row.sharpe, row.max_drawdown,
                          cfg["rank_pair_tpe"].get("return_scale", 0.30))  # 旧归档沿用原尺度。


def suggest_rank_params(trial, family, baseline, space):
    params = baseline.copy()
    for key, (kind, low, high) in space.items():
        name = f"{family}.{key}"
        if kind == "choice":
            params[key] = trial.suggest_categorical(name, [low, high])
        elif kind in ("int", "int_log"):
            params[key] = trial.suggest_int(name, low, high, log=kind == "int_log")
        else:
            params[key] = trial.suggest_float(name, low, high, log=kind == "log")
    return params


def tune_tpe(out, cfg):
    """每个 Trial 联合提出两套树参数，按八个排序器融合后的固定验证年损失选参。"""
    import optuna
    from optuna.trial import TrialState

    settings = cfg["rank_pair_tpe"]
    source = Path(cfg["feature_source"])
    valid = read_period(source, cfg["segments"]["validation"], cfg)
    matrix = valid[alpha158_columns(source)]
    keys = valid[["trade_date", "ticker", "grade", "evaluable"]]
    gains = {name: cfg["label_gains"][name] for name in BASES}
    model = settings.get("models", "pair")
    families = FAMILIES if model == "pair" else (model,)
    joint = settings.get("joint_validation", False)
    if joint:
        if cfg["segments"]["validation"] != cfg.get("early_stop_bounds", ["2021-01-01", "2022-12-31"]):
            raise ValueError("联合验证的早停和选参区间必须一致")
        blocks = []
        for frame in (read_period(source, cfg["segments"]["train"], cfg), valid):
            rows = frame.loc[frame.evaluable]
            blocks.append((rows[matrix.columns], rows.grade.to_numpy(dtype=int),
                           rows.groupby("trade_date", sort=False).size().to_numpy()))
    else:
        if not cfg.get("validation_mode"):
            base = aligned_reference(Path(cfg["pair_reference"]), keys, "validation", ["equal", "xgb_equal"])
        blocks = scoring_inputs(cfg)
    xmat = quantile_matrices(blocks, cfg["xgboost"]) if "xgboost" in families else []
    criterion = settings.get("criterion", "ndcg")
    if criterion not in ("ndcg", "composite"):
        raise ValueError(f"未知 TPE 目标 {criterion}")
    if criterion == "composite":
        start_year = int(cfg["segments"]["train"][1][:4]) + 1
        if (cfg["segments"]["validation"] != [f"{start_year if joint else start_year+1}-01-01", f"{start_year+1}-12-31"] or
                cfg["segments"]["train"][0][:4] != "2014" or cfg["segments"]["train"][1] != f"{start_year-1}-12-31" or
                cfg["final_early_stop_year"] != (start_year+1 if joint else start_year) or
                (cfg["topk"], cfg["n_drop"], cfg["rebalance_days"]) != (50, 5, 1)):
            raise ValueError("综合目标要求固定验证区间与早停设置一致，且使用每日 Top50/Drop5")
        market = prepare_test_panel(cfg, load_market_calendar(CSI500_INPUTS["calendar"]), split="validation")
    storage = f"sqlite:///{out.resolve()}/optuna.sqlite3"
    study = optuna.create_study(study_name="pair_tpe", direction="minimize", storage=storage, load_if_exists=True)
    fusion = cfg.get("rank_fusion", "percentile")
    objective_spec = [criterion, list(families)]
    if fusion != "percentile":
        objective_spec.append({"rank_fusion": fusion})
    if cfg.get("label_end_days", 2) != 2:
        objective_spec.append({"label_end_days": cfg["label_end_days"]})
    if joint:
        objective_spec.append(cfg["segments"]["validation"])
    if criterion == "composite" and "return_scale" in settings:
        objective_spec.append({"return_scale": settings["return_scale"]})
    previous = study.user_attrs.get("objective_spec", ["ndcg", list(FAMILIES)] if study.trials else objective_spec)
    if previous != objective_spec:
        raise ValueError("已有 Study 的选参目标或模型组合与本次配置不同")
    study.set_user_attr("objective_spec", objective_spec)
    for trial in study.get_trials(states=(TrialState.RUNNING,)):
        study.tell(trial.number, state=TrialState.FAIL)
    anchor = {f"{family}.{key}": cfg[family][key] for family in families for key in settings[family]}
    anchors = [anchor]
    if "lightgbm" in families:
        anchors.append({**anchor, "lightgbm.num_leaves": 64, "lightgbm.lambda_l1": 1, "lightgbm.lambda_l2": 200})
    for params in anchors:
        existing = study.get_trials(states=(TrialState.COMPLETE, TrialState.WAITING))
        if not any(t.system_attrs.get("fixed_params", t.params) == params for t in existing):
            study.enqueue_trial(params)

    def objective(trial):
        ranks, configs = [], {}
        for family in families:
            col = "equal" if family == "lightgbm" else "xgb_equal"
            params = suggest_rank_params(trial, family, cfg[family], settings[family])
            configs[family] = params
            if fusion == "percentile" and not (joint or cfg.get("validation_mode")) and params == cfg[family]:
                ranks.append(base[col])
            else:
                pred = train_candidate(out, family, params, cfg, blocks, xmat, matrix)
                ranks.append(fuse_rankers(pred.assign(trade_date=keys.trade_date), list(BASES), fusion))
        score = sum(ranks) / len(ranks)
        daily = validation_loss(keys, score, gains, cfg["topk"])
        trial.set_user_attr("configurations", configs)
        daily.to_csv(out / f"trial_{trial.number}_daily.csv", index=False)
        ndcg_loss = float(daily.loss.mean())
        trial.set_user_attr("ndcg_loss", ndcg_loss)
        if criterion == "composite":
            metrics = validation_backtest(cfg, keys, score, market)
            for key, value in metrics.items():
                trial.set_user_attr(key, value)
            return metrics["composite_loss"]
        return ndcg_loss

    for i in range(len(study.get_trials(states=(TrialState.COMPLETE,))), settings["trials"]):
        sampler = optuna.samplers.TPESampler(seed=cfg["seed"] + i, n_startup_trials=settings["startup_trials"])
        study = optuna.load_study(study_name=study.study_name, storage=storage, sampler=sampler)
        study.optimize(objective, n_trials=1)
        last = study.get_trials(states=(TrialState.COMPLETE,))[-1]
        print(f"Trial {last.number} loss={last.value:.6f} best={study.best_value:.6f}", flush=True)
    study.trials_dataframe().to_csv(out / "trials.csv", index=False)
    winner = study.best_trial
    (out / "validation_loss_daily.csv").write_bytes((out / f"trial_{winner.number}_daily.csv").read_bytes())
    best = winner.user_attrs["configurations"]
    tmp = out / "winners.tmp"
    tmp.write_text(json.dumps(best, indent=2))
    tmp.replace(out / "winners.json")
    return best


def tune(out, cfg):
    """冻结每一阶段的最优组合，下一阶段以此为起点；只使用验证期最终排名选参。"""
    if "rank_pair_tpe" in cfg:
        return tune_tpe(out, cfg)
    if cfg["rank_pair_tuning"].get("selection_loss") == "lambdarank_stable":
        if not ("walk_forward" in cfg or cfg.get("validation_mode")) or cfg["topk"] != 50:
            raise ValueError("稳定性目标需要 walk-forward 验证和 NDCG@50")
    if "walk_forward" in cfg:
        return tune_walk_forward(out, cfg)
    source, ref = Path(cfg["feature_source"]), Path(cfg["pair_reference"])
    valid = read_period(source, cfg["segments"]["validation"], cfg)
    valid_x = valid[alpha158_columns(source)]
    keys = valid[["trade_date", "ticker", "evaluable", "grade"]].copy()
    if cfg.get("validation_mode") or cfg.get("label_end_days", 2) != 2:
        return search_pair(out, cfg, valid_x, keys, {}, scoring_inputs(cfg))
    old = aligned_reference(ref, keys, "validation", ["equal", "xgb_equal", "blend_equal"])
    np.testing.assert_allclose((old.equal + old.xgb_equal) / 2, old.blend_equal, rtol=0, atol=1e-14)
    del valid
    gains = {n: cfg["label_gains"][n] for n in BASES}
    baseline = {f: cfg[f] for f in FAMILIES}
    base_ids = [candidate_id(baseline[f]) for f in FAMILIES]
    cache = {base_ids[0]: old.equal, base_ids[1]: old.xgb_equal}
    base_daily = validation_loss(keys, old.blend_equal, gains, cfg["topk"])
    base_loss = base_daily.loss.mean()
    expected = pd.read_csv(ref / "validation_losses.csv").set_index("strategy").at["blend_equal", "loss"]
    np.testing.assert_allclose(base_loss, expected, rtol=0, atol=1e-12)
    return search_pair(out, cfg, valid_x, keys, cache, scoring_inputs(cfg))


def search_pair(out, cfg, valid_x, keys, cache, blocks):
    """在给定特征上搜索，以八个排名等权后的验证损失选出阶段最优参数。"""
    best = {family: cfg[family] for family in FAMILIES}
    gains = {name: cfg["label_gains"][name] for name in BASES}
    xmat = quantile_matrices(blocks, cfg["xgboost"])
    trials, winners = [], []
    for stage in cfg["rank_pair_tuning"].get("stages", STAGES):
        options = [candidates(stage, f, best[f], cfg["rank_pair_tuning"]) for f in FAMILIES]
        for family, choices in zip(FAMILIES, options):
            for p in choices:
                key = candidate_id(p)
                if key in cache:
                    continue
                file = out / family / key / "validation.parquet"
                if file.exists():
                    saved = pd.read_parquet(file)
                    pd.testing.assert_frame_equal(saved[["trade_date","ticker"]], keys[["trade_date","ticker"]])
                    cache[key] = saved["score"]
                else:
                    print(f"{stage} {family} {key} 开始", flush=True)
                    preds = train_candidate(out, family, p, cfg, blocks, xmat, valid_x)
                    cache[key] = equal_ranks(preds.assign(trade_date=keys.trade_date), list(BASES))
                    saved = keys[["trade_date","ticker"]].assign(score=cache[key])
                    tmp = file.with_suffix(".tmp.parquet")
                    saved.to_parquet(tmp)
                    tmp.replace(file)
        stage_losses = []
        for lp, xp in product(*options):
            li, xi = candidate_id(lp), candidate_id(xp)
            daily = validation_loss(keys, (cache[li] + cache[xi]) / 2, gains, cfg["topk"])
            loss = daily.loss.mean()
            metrics = {}
            if cfg["rank_pair_tuning"].get("selection_loss") == "lambdarank_stable":
                quarterly = daily.groupby(daily.trade_date.dt.to_period("Q")).loss.mean()
                mean, std = quarterly.mean(), quarterly.std(ddof=0)
                loss = mean + STABILITY_PENALTY * std
                metrics = dict(mean_loss=mean, std_loss=std, stability_penalty=STABILITY_PENALTY, L_opt=loss,
                               **{str(period): value for period, value in quarterly.items()})
            trials.append(dict(stage=stage, lightgbm=li, xgboost=xi, loss=loss, days=len(daily), **metrics))
            stage_losses.append((loss, lp, xp))
        winner, *params = min(stage_losses, key=lambda r: r[0])
        best = dict(zip(FAMILIES, params))
        winners.append(dict(stage=stage, before=stage_losses[0][0], loss=winner, pairs=len(stage_losses),
                            **{f: candidate_id(best[f]) for f in FAMILIES}))
        pd.DataFrame(trials).to_csv(out / "trials.csv", index=False)
        pd.DataFrame(winners).to_csv(out / "stages.csv", index=False)
        print(f"{stage} 完成，组合验证损失 {winner:.6f}（起点 {stage_losses[0][0]:.6f}）", flush=True)
    daily = validation_loss(keys, sum(cache[candidate_id(best[f])] for f in FAMILIES)/2, gains, cfg["topk"])
    daily.to_csv(out / "validation_loss_daily.csv", index=False)
    tmp = out / "winners.tmp"
    tmp.write_text(json.dumps(best, indent=2))
    tmp.replace(out / "winners.json")
    return best


def evaluate(out, cfg, best):
    """冻结参数后读取测试期，仅回测最终组合及其对照。"""
    source = Path(cfg["feature_source"])
    template = json.loads((out / "config_template.json").read_text())
    for family, section in (("lightgbm", "lightgbm_four_equal"), ("xgboost", "single_xgboost")):
        if family not in best:
            continue
        template[section]["params"] = {k: v for k, v in best[family].items() if k != "seed"}
    template["four_model_targets"] = {n: cfg["label_gains"][n] for n in BASES}
    template["seed"] = cfg["seed"]
    fixed_validation(template, cfg.get("validation_mode"))
    write_best_config(out, template)
    parameter_rows = []
    for family in best:
        all_params = {candidate_id(cfg[family]): cfg[family]}
        for path in (out / family).glob("*/params.json"):
            all_params[path.parent.name] = json.loads(path.read_text())
        for key, p in all_params.items():
            parameter_rows.append(dict(family=family, candidate=key, selected=p == best[family], **p))
    pd.DataFrame(parameter_rows).to_csv(out / "candidate_parameters.csv", index=False)
    features = alpha158_columns(source)
    scored = []
    for split in ("validation", "test"):
        frame = read_period(source, cfg["segments"][split], cfg)
        matrix = frame[features]
        s = frame[["trade_date", "ticker", "grade", "label", "label_end_date", "evaluable"]].copy()
        fusion = cfg.get("rank_fusion", "percentile")
        refitted = (fusion != "percentile" or cfg.get("validation_mode") or cfg.get("label_end_days", 2) != 2
                    or cfg.get("rank_pair_tpe", {}).get("joint_validation", False))
        base = None if refitted else aligned_reference(Path(cfg["pair_reference"]), s, split, ["equal", "xgb_equal", "blend_equal"])
        columns, original = [], []
        for family, selected in best.items():
            column, prefix = ("equal", "") if family == "lightgbm" else ("xgb_equal", "xgb_")
            columns.append(column)
            raw = None
            if refitted:
                raw = frozen_scores(out, family, cfg[family], matrix)
                baseline = fuse_rankers(raw.assign(trade_date=s.trade_date), list(BASES), fusion)
            else:
                baseline = base[column]
            original.append(baseline)
            if selected != cfg[family]:
                raw = frozen_scores(out, family, selected, matrix)
            if raw is None:
                s[column] = baseline
            else:
                s = s.join(raw.add_prefix(prefix))
                s[column] = fuse_rankers(raw.assign(trade_date=s.trade_date), list(BASES), fusion)
        s["baseline"] = sum(original) / len(original)
        signal = columns[0]
        if len(columns) == 2:
            signal = "blend_equal"
            s[signal] = s[columns].mean(axis=1)
        methods = tuple(dict.fromkeys([*columns, "baseline", signal]))
        if split == "validation":
            gains = {n: cfg["label_gains"][n] for n in BASES}
            daily = validation_loss(s, s[signal], gains, cfg["topk"])
            if "walk_forward" in cfg:
                daily.to_csv(out / "validation_loss_daily.csv", index=False)
            else:
                archived = pd.read_csv(out / "validation_loss_daily.csv", parse_dates=["trade_date"])
                pd.testing.assert_frame_equal(daily.reset_index(drop=True), archived, atol=1e-12, rtol=1e-12)
            monthly = pd.DataFrame(index=pd.DatetimeIndex(daily.trade_date))
            for name in methods:
                result = validation_loss(s, s[name], gains, cfg["topk"]).set_index("trade_date")
                monthly[name] = result.loss
            monthly.groupby(monthly.index.to_period("M")).mean().to_csv(out / "validation_monthly.csv")
        scored.append(s.assign(split=split))
    scores = pd.concat(scored, ignore_index=True)
    scores.to_parquet(out / "predictions.parquet", index=False)
    evaluate_signals(scores, cfg, out, methods)
    if "rank_pair_tpe" in cfg:
        return
    from src.report.rank_stack import write_pair_tuning
    print(write_pair_tuning(out), flush=True)


def expanding_years(segments, *, sliding=False):
    """每年用前一自然年早停；滑动模式保留首窗的训练年数。"""
    first = pd.Timestamp(segments["train"][0])
    start, end = map(pd.Timestamp, segments["test"])
    if start > end or first.year >= start.year - 1:
        raise ValueError("滚动训练至少需要一个训练年度和一个独立早停年度")
    if "validation" in segments and pd.Timestamp(segments["validation"][1]) >= start:
        raise ValueError("冻结树参数所用的验证集必须结束于首个预测日之前")
    for year in range(start.year, end.year + 1):
        train_start = first + pd.DateOffset(years=year - start.year) if sliding else first
        yield year, {
            "train": [str(train_start.date()), f"{year - 2}-12-31"],
            "early_stop": [f"{year - 1}-01-01", f"{year - 1}-12-31"],
            "test": [max(str(start.date()), f"{year}-01-01"), min(str(end.date()), f"{year}-12-31")],
        }


def expanding_window(out, cfg, year, bounds, names):
    """只向训练器传入历史标签；用本年全部信号日的特征打分。"""
    source = Path(cfg["feature_source"])
    folder = out / str(year)
    folder.mkdir(exist_ok=True)
    blocks, audit = [], []
    for period in ("train", "early_stop"):
        frame = read_period(source, bounds[period], cfg)
        rows = frame.loc[frame.evaluable]
        groups = rows.groupby("trade_date", sort=False).size().to_numpy()
        if rows.empty or not rows.label_end_date.between(*map(pd.Timestamp, bounds[period])).all():
            raise ValueError(f"{year} 年 {period} 收益标签越界或无有效样本")
        audit.append(dict(year=year, period=period, start=bounds[period][0], end=bounds[period][1],
                          first_signal=rows.trade_date.min(), last_signal=rows.trade_date.max(),
                          last_label_end=rows.label_end_date.max(), days=len(groups), stock_days=len(rows)))
        blocks.append((rows[names], rows.grade.to_numpy(dtype=int), groups))
        del frame, rows
    pd.DataFrame(audit).to_csv(folder / "periods.csv", index=False)
    test = read_feature_split(source / "features.parquet", bounds["test"], "滚动预测", cfg).sort_values(["trade_date", "ticker"])
    scores = test[["trade_date", "ticker", "label", "label_end_date"]].copy()
    for family, prefix in (("lightgbm", "lgb_"), ("xgboost", "xgb_")):
        print(f"{year} 年 {family} 训练四个排序器", flush=True)
        params = cfg[family]
        matrices = quantile_matrices(blocks, params) if family == "xgboost" else []
        preds = train_candidate(folder, family, params, cfg, blocks, matrices, test[names])
        scores[preds.add_prefix(prefix).columns] = preds.to_numpy()
        scores[prefix + "equal"] = equal_ranks(scores, [prefix + n for n in BASES])
    method = "sliding" if cfg.get("rolling_scheme") == "sliding_annual" else "expanding"
    scores[method] = (scores.lgb_equal + scores.xgb_equal) / 2
    return scores.assign(window=year)


def rolling_pair_main(argv=None):
    """按年滚动训练，用冻结的混合模型参数做连续样本外回测。"""
    parser = argparse.ArgumentParser(description=rolling_pair_main.__doc__)
    label_argument(parser)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--reference", type=Path, help="已完成联合调参的固定模型归档")
    choice.add_argument("--resume", type=Path, help="从本次滚动窗口归档的已完成模型继续")
    parser.add_argument("--window", choices=("expanding", "sliding"), help="固定起点扩展（默认）或固定年数滑动")
    args = parser.parse_args(argv)
    if args.resume:
        out = args.resume.resolve()
        cfg = json.loads((out / "config.json").read_text())
        if args.label_days is not None:
            parser.error("续跑沿用原实验标签")
        reference = Path(cfg["fixed_pair_reference"])
        if args.window and cfg["rolling_scheme"] != args.window + "_annual":
            parser.error("--window 与接续实验的配置不一致")
    else:
        reference = (args.reference or latest_run("winners.json", "candidate_parameters.csv",
                     "ledger.parquet", "predictions.parquet", rebalance_days=1, winner_families=FAMILIES)).resolve()
        cfg = json.loads((reference / "config.json").read_text())
        set_label_days(cfg, args.label_days)
        cfg.update(json.loads((reference / "winners.json").read_text()), fixed_pair_reference=str(reference),
                   rolling_scheme=(args.window or "expanding") + "_annual", early_stop_years=1)
        if cfg["lightgbm_xgboost_blend"]["xgboost_weight"] != 0.5:
            raise ValueError("本实验要求两类模型各占一半")
        out = new_run(cfg["rebalance_days"], tag=(args.window or "expanding") + "_pair")
        out.mkdir()
        (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    print(out, flush=True)
    names = alpha158_columns(Path(cfg["feature_source"]))
    method = cfg["rolling_scheme"].removesuffix("_annual")
    windows = dict(expanding_years(cfg["segments"], sliding=method == "sliding"))
    scores = pd.concat([expanding_window(out, cfg, year, bounds, names) for year, bounds in windows.items()],
                       ignore_index=True).assign(split="test")
    # 整段测试期统一核验收益标签，保留跨年但仍在测试期内的收益与信号。
    scores["evaluable"] = segment_masks(scores, {"test": cfg["segments"]["test"]})["test"]
    scores["grade"] = fixed_grades(scores.loc[scores.evaluable]).reindex(scores.index).astype("Int8")
    if cfg.get("label_days") is None:
        scores["fixed"] = aligned_reference(reference, scores, "test", ["blend_equal"], check_labels=True).blend_equal
    else:
        keys = ["trade_date", "ticker"]
        archived = pd.read_parquet(reference / "predictions.parquet", columns=[*keys, "blend_equal"],
                                   filters=[("split", "==", "test")]).rename(columns={"blend_equal": "fixed"})
        scores = scores.merge(archived, on=keys, how="left", validate="one_to_one")
    if not np.isfinite(scores[[method, "fixed"]]).all(axis=None):
        raise ValueError("固定或滚动预测含无效分数")
    scores.to_parquet(out / "predictions.parquet", index=False)
    methods = (method, "fixed")
    gains = {name: cfg["label_gains"][name] for name in BASES}
    ndcg = pd.concat([validation_loss(scores, scores[name], gains, cfg["topk"]).assign(strategy=name)
                      for name in methods])
    ndcg.to_csv(out / "test_ndcg_loss.csv", index=False)
    # 命中率沿用精确名次，真实收益并列时按股票代码决定顺序。
    valid = scores.loc[scores.evaluable].groupby("trade_date").label
    rank = valid.rank(method="first", ascending=False)
    low = valid.transform("size") + 1 - rank
    hits = scores.copy()
    hits.loc[rank.index, "grade"] = np.select([rank.le(50), rank.le(100), low.le(50), low.le(100)],
                                              [4, 3, 0, 1], default=2).astype("int8")
    selection_stats(hits, methods).to_csv(out / "hits_daily.csv", index=False)
    evaluate_signals(scores, cfg, out, methods)
    old = pd.read_parquet(reference / "ledger.parquet", filters=[("strategy", "==", "blend_equal")])
    ledger = pd.read_parquet(out / "ledger.parquet")
    columns = ["daily_return", "cost_cny", "nav"]
    pd.testing.assert_frame_equal(ledger.loc[ledger.strategy.eq("fixed")].set_index("trade_date")[columns],
                                  old.set_index("trade_date")[columns], rtol=1e-10, atol=1e-10)
    from src.report.rank_stack import write_expanding_pair
    print(write_expanding_pair(out), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    parser.add_argument("--tpe-objective", choices=("ndcg", "composite"), help="验证集选参依据")
    parser.add_argument("--tpe-models", choices=("pair", "lightgbm"), help="八模型或 LightGBM ABCD 四等权")
    parser.add_argument("--tpe", action="store_true", help="固定验证期的排序器 Optuna TPE")
    parser.add_argument("--tuning-config", type=Path, help="双模型候选网格，默认 configs/tuning.json")
    parser.add_argument("--reference", type=Path, help="已冻结的五档 LightGBM/XGBoost 混合归档")
    parser.add_argument("--resume", type=Path, help="按该实验配置接续训练与回测")
    parser.add_argument("--loss", choices=TUNING_LOSSES,
                        help="mse 使用原回归网格；排序组合默认 lambdarank，lambdarank_stable 加跨期稳定性惩罚")
    label_argument(parser)
    validation_argument(parser)
    args = parser.parse_args(argv)
    if (args.tpe_objective or args.tpe_models) and not args.tpe:
        parser.error("目标与模型选项仅适用于新 --tpe 实验，续跑读取原配置")
    if args.tpe and (args.resume or args.loss not in (None, "lambdarank")):
        parser.error("--tpe 使用 lambdarank；续跑时仅指定 --resume")
    if args.loss == "mse":
        if args.resume or args.reference:
            parser.error("MSE 回归调参使用 --config 和 --tuning-config，排序归档仅适用于排序模式")
        from src.scripts.run_all import main as study_main
        command = ["--stage", "tune", "--config", str(args.config)]
        if args.label_days is not None:
            command.extend(["--label-days", str(args.label_days)])
        if args.tuning_config:
            command.extend(["--tuning-config", str(args.tuning_config)])
        if args.validation_mode:
            command.extend(["--validation-mode", args.validation_mode])
        return study_main(command)
    if args.resume:
        if args.validation_mode or args.label_days is not None:
            parser.error("续跑使用归档验证模式")
        out = args.resume.resolve()
        cfg = json.loads((out / "config.json").read_text())
        if args.loss and args.loss != cfg["rank_pair_tuning"].get("selection_loss", "lambdarank"):
            parser.error("--loss 与接续实验的选参目标不一致")
    else:
        ref = args.reference or RUNS / "daily_xgboost_tuned_20261121_172019_402981"
        if args.tpe and args.reference is None:
            ref = RUNS / "daily_pair_tuning_20261122_100328_550844"
        cfg = json.loads((ref / "config.json").read_text())
        cfg.pop("rank_pair_tpe", None)
        settings = json.loads(args.config.read_text())
        cfg.update(rank_pair_tuning=load_tuning("rank_pair_tuning", settings, None if args.tpe else args.tuning_config),
                   pair_reference=str(ref.resolve()))
        if args.loss:
            cfg["rank_pair_tuning"]["selection_loss"] = args.loss
        if cfg["rank_pair_tuning"].get("selection_loss", "lambdarank") not in TUNING_LOSSES[1:]:
            parser.error("rank_pair_tuning.selection_loss 需要使用排序目标")
        cfg.pop("walk_forward", None)
        set_label_days(cfg, args.label_days if args.label_days is not None else settings.get("label_days"))
        set_label_days(settings, cfg.get("label_days"))
        mode = args.validation_mode or settings.get("validation_mode") or cfg.get("validation_mode")
        fixed_validation(cfg, mode)
        policy = walk_forward_policy(settings)
        if args.tpe:
            if (ref / "winners.json").exists():
                cfg.update(json.loads((ref / "winners.json").read_text()))
            params = cfg["lightgbm"]
            for alias, canonical in (("colsample_bytree", "feature_fraction"), ("subsample", "bagging_fraction")):
                if alias in params:
                    params.setdefault(canonical, params.pop(alias))
            tpe = dict(load_tuning("rank_pair_tpe", settings, args.tuning_config))
            tpe["criterion"] = args.tpe_objective or tpe.get("criterion", "ndcg")
            tpe["models"] = args.tpe_models or tpe.get("models", "pair")
            mode = mode or ("joint" if tpe["models"] == "lightgbm" else "split")
            fixed_validation(cfg, mode)
            tpe["joint_validation"] = mode != "split"
            cfg["final_early_stop_year"] = int(cfg["early_stop_bounds"][1][:4])
            if tpe["criterion"] == "composite":
                tpe["return_scale"] = COMPOSITE_RETURN_SCALE
            cfg["rank_pair_tpe"] = tpe
        elif not mode and (policy.get("selection_windows") == "all" or cfg["rank_pair_tuning"].get("selection_loss") == "lambdarank_stable"):
            cfg["walk_forward"] = policy
        check_config(cfg)
        out = new_run(cfg["rebalance_days"], tag="pair_tpe" if args.tpe else "pair_tuning")
        out.mkdir()
        (out / "config.json").write_text(json.dumps(cfg, indent=2))
        (out / "config_template.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2))
    print(out, flush=True)
    with (out / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        best = json.loads((out / "winners.json").read_text()) if (out / "winners.json").is_file() else tune(out, cfg)
        evaluate(out, cfg, best)


def model_run(args, families):
    """复用同一套拟合与评价函数，按调用入口只训练所选模型家族。"""
    model = families[0] if len(families) == 1 else "pair"
    if args.resume:
        saved = json.loads((args.resume / "config.json").read_text())
        if saved.get("rank_pair_tpe", {}).get("models") != model:
            raise ValueError("续跑目录的 Optuna 模型家族与当前训练入口不同")
        if any((args.tpe, args.tpe_objective, args.tuning_config, args.source, args.out,
                args.validation_mode, args.label_days, getattr(args, "xgboost_loss", None), args.config is not None)):
            raise ValueError("续跑读取归档参数，请只指定 --resume 实验目录")
        return main(["--resume", str(args.resume)])
    if not args.tpe and (args.tpe_objective or args.tuning_config):
        raise ValueError("--tpe-objective 和 --tuning-config 需要 --tpe")
    default_config = RUNS / "daily_pair_tuning_20261122_100328_550844/best_config.json" if model == "pair" or args.tpe else CONFIG_FILE
    config_path = args.config or default_config
    settings = load_config(config_path)
    set_label_days(settings, args.label_days)
    cfg = dict(settings)
    if getattr(args, "fusion", None):
        cfg["rank_fusion"] = args.fusion
    source = args.source or Path(json.loads((RUNS / "baselines/nav_comparison_daily/rank_config.json").read_text())["feature_source"])
    cfg.update(feature_source=str(source), parameter_config=str(config_path.resolve()),
               label_gains={key: settings["four_model_targets"][key] for key in BASES},
               lightgbm=dict(settings["lightgbm_four_equal"]["params"], seed=settings["seed"]),
               xgboost=dict(settings["xgboost"], seed=settings["seed"]))
    if "xgboost" in families:
        cfg["xgboost"]["objective"] = args.xgboost_loss or cfg["xgboost"]["objective"]
    mode = args.validation_mode or settings.get("validation_mode") or ("joint" if args.tpe and families == ("lightgbm",) else "split")
    fixed_validation(cfg, mode)
    cfg["final_early_stop_year"] = int(cfg["early_stop_bounds"][1][:4])
    template = json.loads(config_path.read_text())
    set_label_days(template, cfg.get("label_days"))
    if "rank_fusion" in cfg:
        template["rank_fusion"] = cfg["rank_fusion"]
    if args.tpe:
        params = cfg["lightgbm"]
        for alias, canonical in (("colsample_bytree", "feature_fraction"), ("subsample", "bagging_fraction")):
            if alias in params:
                params.setdefault(canonical, params.pop(alias))
        cfg["rank_pair_tpe"] = dict(load_tuning("rank_pair_tpe", settings, args.tuning_config),
                                    models=model, criterion=args.tpe_objective or "ndcg", joint_validation=mode != "split")
        if cfg["rank_pair_tpe"]["criterion"] == "composite":
            cfg["rank_pair_tpe"]["return_scale"] = COMPOSITE_RETURN_SCALE
    out = args.out or new_run(cfg["rebalance_days"], tag=model + ("_tpe" if args.tpe else "_equal"))
    out.mkdir(parents=True, exist_ok=False)
    print(out, flush=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    if args.tpe:
        (out / "config_template.json").write_text(json.dumps(template, indent=2))
        with (out / ".run.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            evaluate(out, cfg, tune_tpe(out, cfg))
        return
    blocks = scoring_inputs(cfg)
    for family in families:
        matrices = quantile_matrices(blocks, cfg[family]) if family == "xgboost" else []
        train_candidate(out, family, cfg[family], cfg, blocks, matrices)
    del blocks, matrices
    features = alpha158_columns(source)
    parts = []
    signal = "blend_equal" if len(families) == 2 else "xgb_equal"
    for split in ("validation", "test"):
        frame = read_period(source, cfg["segments"][split], cfg)
        scores = frame[["trade_date", "ticker", "label", "label_end_date", "evaluable", "grade"]].copy()
        for family in families:
            raw = frozen_scores(out, family, cfg[family], frame[features])
            prefix = "" if family == "lightgbm" else "xgb_"
            names = [prefix + target for target in BASES]
            scores = scores.join(raw.add_prefix(prefix))
            scores["equal" if family == "lightgbm" else "xgb_equal"] = equal_ranks(scores, names)
        if len(families) == 2:
            scores[signal] = (scores.equal + scores.xgb_equal) / 2
        validation_loss(scores, scores[signal], cfg["label_gains"], cfg["topk"]).to_csv(
            out / f"{split}_loss_daily.csv", index=False)
        parts.append(scores.assign(split=split))
    scores = pd.concat(parts, ignore_index=True)
    scores.to_parquet(out / "predictions.parquet", index=False)
    evaluate_signals(scores, cfg, out, (signal,))


def model_arguments(parser):
    parser.add_argument("--config", type=Path, help="覆盖入口默认的模型参数配置")
    parser.add_argument("--source", type=Path, help="Alpha158 特征缓存目录")
    parser.add_argument("--resume", type=Path, help="继续本模型的 Optuna 实验")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--tpe", action="store_true", help="在验证集上用 Optuna 选参")
    parser.add_argument("--tpe-objective", choices=("ndcg", "composite"), help="Optuna 的选参指标")
    parser.add_argument("--tuning-config", type=Path)
    label_argument(parser)
    validation_argument(parser)
