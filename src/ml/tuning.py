"""使用 Optuna 网格搜索参数，并通过 Qlib 的 LightGBM API 训练。"""

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


def tuning_dataset(config: dict, out: Path):
    """只读取训练和验证样本，复用现有标签筛选与早停逻辑。"""
    names = alpha158_columns(out)
    frames = {}
    for split, segment in (("train", "train"), ("validation", "valid")):
        rows = read_feature_split(out / "features.parquet", config["segments"][split], split)
        valid = segment_masks(rows, config["segments"])[split]
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
    windows = folds(config) if "walk_forward" in config else []
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
