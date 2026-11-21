"""使用 Optuna 网格搜索参数，并通过 Qlib 的 LightGBM API 训练。"""

import json
from math import prod
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import optuna
import pandas as pd
from qlib.workflow import R

from src.data.qlib_provider import register_provider
from src.data.research_panel import segment_masks
from src.ml.dataset import alpha158_columns, fitting_segment, make_dataset, read_feature_split
from src.ml.models import fit_lightgbm, training_settings


def tune_lightgbm(config: dict, out: Path) -> None:
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
    dataset = make_dataset(**frames)
    del frames
    settings = training_settings(config)
    space = {key: list(dict.fromkeys(values)) for key, values in config["lightgbm_tuning"]["search_space"].items()}
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
    best_config = dict(config, lightgbm={**config["lightgbm"], **{key: best[key] for key in space}})
    (out / "best_config.json").write_text(json.dumps(best_config, indent=2) + "\n")
