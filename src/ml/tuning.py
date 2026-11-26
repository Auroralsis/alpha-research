"""使用 Optuna 网格搜索参数，并通过 Qlib 的 LightGBM API 训练。"""

import hashlib
import json
from math import prod
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import optuna
import pandas as pd
from qlib.workflow import R

from src.config import write_best_config
from src.data.qlib_provider import register_provider
from src.data.research_panel import segment_masks
from src.ml.dataset import alpha158_columns, fitting_segment, make_dataset, read_feature_split
from src.ml.models import fit_lightgbm, training_settings


TUNING_LOSSES = ("mse", "lambdarank", "lambdarank_stable")
STABILITY_PENALTY = 0.5


def rank_stage_trials(table, stage, loss_name, families=("lightgbm", "xgboost")):
    """以一组候选参数为一个 Trial，复用各折预测并按验证窗口等权计分。"""
    if loss_name not in TUNING_LOSSES[1:]:
        raise ValueError(f"未知排序选参目标 {loss_name}")
    stable = loss_name == "lambdarank_stable"
    rows = []
    for _, group in table.groupby(list(families), sort=False):
        group = group.sort_values("valid_start")
        losses = group["blend_loss"].to_numpy()
        if stable and len(losses) < 2:
            raise ValueError("稳定性目标需要至少两个验证窗口")
        mean, std = float(np.mean(losses)), float(np.std(losses, ddof=0))
        penalty = STABILITY_PENALTY if stable else 0.0
        value = mean + penalty * std
        params = group.iloc[0][list(families)].to_dict()
        summary = dict(mean_loss=mean, std_loss=std, stability_penalty=penalty, L_opt=value)
        rows.append(dict(stage=stage, trial=len(rows), **params, folds=len(group),
                         selection_loss=loss_name, loss=value, **summary))
    trials = pd.DataFrame(rows)
    return trials, trials.loc[trials.loss.idxmin(), list(families)].to_dict()


STAGES = ("depth", "leaves", "min_child", "l1", "l2", "row_sampling", "column_sampling")


def candidate_id(params):
    """相同完整参数共用磁盘模型与验证分数。"""
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]


def candidates(stage, family, incumbent, grid):
    """各阶段先保留上一组优胜参数，再生成只改当前维度的候选。"""
    lg = family == "lightgbm"
    if stage == "leaves":
        key = "num_leaves" if lg else "max_leaves"
        cap = 2 ** incumbent["max_depth"]
        current = min(incumbent.get(key) or cap, cap)
        values = [max(2, int(cap * f)) for f in grid["leaf_cap_fractions"]]
        values = [n for n in values if n < current]
    else:
        key = {"depth": "max_depth", "min_child": "min_data_in_leaf" if lg else "min_child_weight",
               "l1": "lambda_l1" if lg else "reg_alpha", "l2": "lambda_l2" if lg else "reg_lambda",
               "row_sampling": "subsample", "column_sampling": "colsample_bytree"}[stage]
        values = grid[key if stage == "min_child" else stage]
    options = [incumbent]
    for value in values:
        p = {**incumbent, key: value}
        if p not in options:
            options.append(p)
    return options


def select_rank_params(out, cfg, families, *, fold_features=None, extra=None):
    """按各历史验证窗的平均组合损失推进搜索，最终模型单独拟合。"""
    from src.ml.walk_forward import folds, rank_losses

    loss_name = cfg["rank_pair_tuning"].get("selection_loss", "lambdarank")
    if loss_name == "lambdarank_stable" and len(folds(cfg)) < 2:
        raise ValueError("单个验证期请使用 lambdarank；lambdarank_stable 需显式配置多个验证期（selection_windows=all）")

    best = {family: cfg[family] for family in families}
    fold_tables, summaries, stages = [], [], []
    for stage in cfg["rank_pair_tuning"].get("stages", STAGES):
        options = {f: candidates(stage, f, best[f], cfg["rank_pair_tuning"]) for f in families}
        table = rank_losses(out, cfg, options, fold_features=fold_features, extra=extra)
        for family, column in zip(families, ("light_index", "xgb_index")):
            ids = []
            for params in options[family]:
                key = candidate_id(params)
                folder = out / family / key
                folder.mkdir(parents=True, exist_ok=True)
                (folder / "params.json").write_text(json.dumps(params, indent=2))
                ids.append(key)
            table[family] = table[column].map(dict(enumerate(ids)))
        table["stage"] = stage
        pairs, winner = rank_stage_trials(table, stage, loss_name, families)
        best = {family: next(p for p in options[family] if candidate_id(p) == winner[family])
                for family in families}
        fold_tables.append(table.merge(pairs[[*families, "trial"]], on=list(families),
                                       validate="many_to_one"))
        summaries.append(pairs)
        stages.append(dict(stage=stage, before=pairs.loss.iloc[0], loss=pairs.loss.min(), pairs=len(pairs), **winner))
        pd.concat(fold_tables).to_csv(out / "fold_trials.csv", index=False)
        pd.concat(summaries).to_csv(out / "trials.csv", index=False)
        pd.DataFrame(stages).to_csv(out / "stages.csv", index=False)

    return best


def tuning_dataset(config: dict, out: Path):
    """只读取训练和验证样本，复用现有标签筛选与早停逻辑。"""
    names = alpha158_columns(out)
    frames = {}
    for split, segment in (("train", "train"), ("validation", "valid")):
        bounds = config.get("early_stop_bounds", config["segments"][split]) if split == "validation" else config["segments"][split]
        rows = read_feature_split(out / "features.parquet", bounds, split)
        valid = segment_masks(rows, {split: bounds})[split]
        frame, _ = fitting_segment(rows, names, valid, config["training_label"], split)
        if np.isinf(frame["feature"].to_numpy(dtype=np.float32)).any():
            raise ValueError(f"Infinite Alpha158 inputs in {split}")
        frames[segment] = frame
    del rows, frame
    return make_dataset(**frames)


def tune_lightgbm(config: dict, out: Path, *, candidate_models: Path | None = None) -> None:
    """用验证集 MSE 选参；对照实验可指定临时目录保存各组最佳轮的模型。"""
    settings = training_settings(config)
    space = {key: list(dict.fromkeys(values)) for key, values in config["lightgbm_tuning"]["search_space"].items()}
    params = settings["lightgbm_params"]
    if "subsample" in space and min(space["subsample"]) < 1 and not params.get("bagging_freq", params.get("subsample_freq", 0)):
        raise ValueError("搜索行采样率时，请在 lightgbm 中设置 bagging_freq 为正整数，例如 1")
    for key, aliases in (("subsample", ("bagging_fraction", "sub_row", "bagging")),
                         ("colsample_bytree", ("feature_fraction", "sub_feature"))):
        if key in space and set(aliases).intersection(params):
            raise ValueError(f"搜索 {key} 时，请只保留这个参数名，移除 lightgbm 中的同义参数")
    from src.ml.walk_forward import folds, regression_loss
    windows = folds(config) if "walk_forward" in config or config.get("validation_mode") else []
    fold_results = []
    dataset = None if windows else tuning_dataset(config, out)
    grid_size = prod(map(len, space.values()))
    trials = []
    best_loss = float("inf")

    def evaluate(trial: optuna.Trial | None = None) -> float:
        """先跑默认参数，再逐一训练网格候选；平分时保留先完成的模型。"""
        nonlocal best_loss
        model_params = settings["lightgbm_params"].copy()
        if trial is not None:
            model_params.update({key: trial.suggest_categorical(key, values) for key, values in space.items()})
        number = len(trials)
        if windows:
            losses = []
            for window in windows:
                loss, rounds = regression_loss(out, config, window, model_params)
                if not np.isfinite(loss):
                    raise ValueError("逐折验证 MSE 不是有限数值")
                losses.append(loss)
                fold_results.append(dict(trial=number, train_start=window["train"][0], train_end=window["train"][1],
                                         early_stop_start=window["early_stop"][0], early_stop_end=window["early_stop"][1],
                                         valid_start=window["validation"][0], valid_end=window["validation"][1],
                                         validation_mse=loss, best_iteration=rounds))
            mean = float(np.mean(losses))
            trials.append(dict(trial=number, validation_mse=mean, **{key: model_params[key] for key in space}))
            pd.DataFrame(trials).to_csv(out / "trials.csv", index=False)
            pd.DataFrame(fold_results).to_csv(out / "fold_trials.csv", index=False)
            return mean
        with R.start(experiment_name="lightgbm_tuning", recorder_name=f"trial_{number:02d}"):
            model, curve, rank_metrics = fit_lightgbm(
                dataset, model_params, settings["training_control"], title=f"LightGBM trial {number}/{grid_size}",
            )
        booster = model.model
        best = booster.best_iteration
        losses = curve.iloc[best - 1]
        loss = float(losses.validation_mse)
        if not np.isfinite(loss):
            raise ValueError("验证 MSE 不是有限数值")
        if candidate_models is not None:
            booster.save_model(str(candidate_models / f"{number}.txt"), num_iteration=best)
        search_params = {name: model_params[name] for name in space}
        trials.append(dict(trial=number, best_iteration=best, train_mse=float(losses.train_mse),
                           validation_mse=loss, **rank_metrics, **search_params))
        pd.DataFrame(trials).to_csv(out / "trials.csv", index=False)
        if loss < best_loss:
            best_loss = loss
            booster.save_model(str(out / "lightgbm.txt"), num_iteration=best)
            curve.to_csv(out / "best_training_mse.csv", index=False)
        return loss

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.GridSampler(space, seed=config["seed"])
    study = optuna.create_study(study_name="alpha158_lightgbm", direction="minimize", sampler=sampler,
                                storage=f"sqlite:///{out.resolve() / 'optuna.sqlite3'}")
    with TemporaryDirectory(prefix="qlib-tuning-") as work:
        register_provider(Path(work), config["qlib_version"])
        evaluate()
        study.optimize(evaluate, n_trials=grid_size)
        best = min(trials, key=lambda row: row["validation_mse"])
        if windows:
            dataset = tuning_dataset(config, out)
            for row in trials if candidate_models is not None else [best]:
                fitted_params = {**params, **{key: row[key] for key in space}}
                with R.start(experiment_name="lightgbm_refit", recorder_name=f"trial_{row['trial']:02d}"):
                    model, curve, _ = fit_lightgbm(dataset, fitted_params, settings["training_control"])
                if candidate_models is not None:
                    model.model.save_model(str(candidate_models / f"{row['trial']}.txt"))
                if row["trial"] == best["trial"]:
                    model.model.save_model(str(out / "lightgbm.txt"))
                    curve.to_csv(out / "best_training_mse.csv", index=False)
    best_config = dict(config, lightgbm={**config["lightgbm"], **{key: best[key] for key in space}})
    write_best_config(out, best_config)


def frozen_scores(folder, family, params, x, *, targets=None):
    """从归档恢复四个排序器，核对列名和顺序后返回原始分数。"""
    import lightgbm as lgb
    import xgboost as xgb
    from src.ml.rank_validation import BASES

    raw = pd.DataFrame(index=x.index)
    for target in BASES if targets is None else targets:
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
