"""训练固定窗口的 Alpha158 模型，并生成逐股分数。"""

import json
from contextlib import nullcontext, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd
from qlib.contrib.model.gbdt import LGBModel
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord
from scipy.stats import ConstantInputWarning

from src.config import RIDGE_LOSS, ZSCORE_LABEL, normalize_training_label, write_best_config
from src.data.qlib_provider import register_provider
from src.backtest.factor_metrics import daily_ic, ic_statistics
from src.data.research_panel import segment_masks, stock_day_index
from src.ml.dataset import alpha158_columns, fitting_targets, fitting_segment, make_dataset, qlib_segment, read_feature_split


MODEL_SCORES = ("lightgbm", "ridge")
ROLLING_SCORE = "rolling_lightgbm"


def training_settings(config: dict, *, include_ridge: bool = True) -> dict:
    """从研究配置提取训练参数，供模型拟合和回测核对训练日志使用。"""
    controls = {key: config[key] for key in ("num_boost_round", "early_stopping_rounds")}
    if "log_evaluation_period" in config:
        controls["log_evaluation_period"] = config["log_evaluation_period"]
    settings = {
        "segments": config["segments"],
        "alpha158_processing": config.get("alpha158_processing", "raw"),
        "training_label": normalize_training_label(config["training_label"]),
        "lightgbm_params": dict(config["lightgbm"], seed=config["seed"]),
        "training_control": controls,
    }
    if include_ridge:
        settings.update(ridge_loss=config.get("ridge_loss", RIDGE_LOSS), ridge_alpha=config["ridge_alpha"])
        if "ridge_solver" in config:
            settings["ridge_solver"] = config["ridge_solver"]
    return settings


def fit_lightgbm(dataset: DatasetH, params: dict, controls: dict, *, title: str = "LightGBM") -> tuple[
    LGBModel, pd.DataFrame, dict,
]:
    """拟合模型，用验证 MSE 早停，再评价最佳轮在两段拟合样本上的排序。"""
    model = LGBModel(**params, num_boost_round=controls["num_boost_round"],
                     early_stopping_rounds=controls["early_stopping_rounds"])
    evaluation = {}
    period = controls.get("log_evaluation_period", 0)
    # Qlib 的早停回调直接打印，即使关闭了逐轮进度也会输出。
    with (redirect_stdout(StringIO()) if period <= 0 else nullcontext()):
        model.fit(dataset, evals_result=evaluation, verbose_eval=period)
    curve = pd.DataFrame(dict(train_mse=evaluation["train"]["l2"], validation_mse=evaluation["valid"]["l2"]))
    curve.insert(0, "iteration", range(1, len(curve) + 1))
    ranks = {}
    with catch_warnings():
        # 常数截面的 Rank IC 无定义，跨日汇总时会跳过。
        simplefilter("ignore", ConstantInputWarning)
        for segment in ("train", "valid"):
            label = dataset.prepare(segment, col_set="label", data_key=DataHandlerLP.DK_L)["label"]
            _, ranks[segment] = daily_ic(model.predict(dataset, segment=segment), label)
    train_mean = ranks["train"].mean()
    valid_mean, _, valid_icir = ic_statistics(ranks["valid"])
    best = model.model.best_iteration
    losses = curve.iloc[best - 1]
    print(f"{title} | Train Mean Rank IC={train_mean:.6f} | Valid Mean Rank IC={valid_mean:.6f} | "
          f"Train MSE={losses.train_mse:.8g} | Valid MSE={losses.validation_mse:.8g} | Best iteration={best}", flush=True)
    metrics = {key: float(value) if np.isfinite(value) else None for key, value in (
        ("train_rank_ic", train_mean), ("validation_rank_ic", valid_mean), ("validation_rank_icir", valid_icir),
    )}
    return model, curve, dict(**metrics, rank_ic_days=int(ranks["valid"].count()))


def record_test_signal(model: LGBModel, frame: pd.DataFrame, names: list[str], evaluable: pd.Series) -> None:
    """在当前 Recorder 记录全部测试分数，事后标签保留评价区间内的收益。"""
    dataset = make_dataset(test=qlib_segment(frame, names, frame.label.where(evaluable)))
    with redirect_stdout(StringIO()):
        SignalRecord(model=model, dataset=dataset, recorder=R.get_recorder()).generate()


def ridge_pipeline(config: dict, alpha: float):
    """训练集拟合中位数填补、特征标准化及带截距的岭回归。"""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                         Ridge(alpha=alpha, solver=config.get("ridge_solver", "cholesky")))


def train_and_predict(config: dict, out: Path, *, include_ridge: bool = True) -> pd.DataFrame:
    """读取 out 中的 Alpha158 特征，保存模型与训练日志，返回验证和测试预测长表。

    可选的 Ridge 与 LightGBM 使用同一批训练股票日和标签。
    """
    import joblib
    import lightgbm as lgb
    from threadpoolctl import threadpool_limits

    names = alpha158_columns(out)
    settings = training_settings(config, include_ridge=include_ridge)
    params = settings["lightgbm_params"]
    controls = settings["training_control"]
    feature_audit, split_audit = [], {}
    ridge = ridge_pipeline(config, config["ridge_alpha"]) if include_ridge else None

    def load(split: str) -> pd.DataFrame:
        """读取一个阶段的股票日特征，并记录缺失值与标签覆盖情况。"""
        bounds = config["segments"][split]
        end = pd.Timestamp(bounds[1])
        rows = read_feature_split(out / "features.parquet", bounds, split)
        values = rows[names].to_numpy(dtype=np.float32)
        if np.isinf(values).any():
            raise ValueError(f"Infinite Alpha158 inputs in {split}")
        missing = np.isnan(values)
        feature_audit.extend(dict(split=split, feature=name, n_missing=int(n), missing_rate=n / len(rows))
                             for name, n in zip(names, missing.sum(axis=0)))
        split_audit[split] = dict(rows=len(rows), market_dates=rows.trade_date.nunique(),
                                  label_nonfinite_rows=int((~np.isfinite(rows.label)).sum()),
                                  label_after_split_rows=int(rows.label_end_date.gt(end).sum()),
                                  feature_nan_rate=float(missing.mean()),
                                  rows_all_features_nan=int(missing.all(axis=1).sum()))
        return rows

    fitting, ridge_mse = {}, {}
    for split, qlib_segment_name in (("train", "train"), ("validation", "valid")):
        frame = load(split)
        mask = segment_masks(frame, config["segments"])[split]
        fitting[qlib_segment_name], target = fitting_segment(frame, names, mask, settings["training_label"], split)
        dropped_days = frame.loc[mask & ~frame.index.isin(target.index), "trade_date"]
        split_audit[split].update(purged_label_rows=int(mask.sum()), fitted_rows=len(target),
                                  degenerate_label_days=dropped_days.nunique(), degenerate_label_rows=len(dropped_days))
        if split == "train" and ridge is not None:
            ridge_features = fitting["train"]["feature"].to_numpy(dtype=np.float32)
            with threadpool_limits(limits=config.get("ridge_num_threads", 4)):
                ridge.fit(ridge_features, target.to_numpy(dtype=np.float32))
                ridge_mse["train"] = float(np.mean((ridge.predict(ridge_features) - target) ** 2))
            del ridge_features
    validation_target = target
    models = out / "models"
    models.mkdir(exist_ok=True)
    if ridge is not None:
        joblib.dump(ridge, models / "ridge.joblib")
    validation = frame
    del frame

    predictions = []
    with TemporaryDirectory(prefix="qlib-fixed-") as work:
        register_provider(Path(work), config.get("qlib_version", "0.9.7"))
        with R.start(experiment_name="fixed_lightgbm"):
            dataset = make_dataset(**fitting)
            fitting.clear()
            model, curve, rank_metrics = fit_lightgbm(dataset, params, controls, title="固定 LightGBM")
            del dataset
            best = model.model.best_iteration
            model.model.save_model(str(models / "lightgbm.txt"), num_iteration=best)
            for split in ("validation", "test"):
                frame = validation if split == "validation" else load(split)
                mask = segment_masks(frame, config["segments"])[split]
                if split == "validation":
                    val = make_dataset(valid=qlib_segment(frame, names, frame.label))
                    score = model.predict(val, segment="valid")
                    del val
                else:
                    record_test_signal(model, frame, names, mask)
                    score = R.load_object("pred.pkl")["score"]
                score = score.reindex(stock_day_index(frame)).to_numpy()
                if not np.isfinite(score).all():
                    raise ValueError(f"Invalid model scores in {split}")
                split_audit[split]["evaluable_rows"] = int(mask.sum())
                scored = frame.drop(columns=names).assign(split=split, evaluable=mask, lightgbm=score)
                if ridge is not None:
                    with threadpool_limits(limits=config.get("ridge_num_threads", 4)):
                        ridge_score = ridge.predict(frame[names].to_numpy(dtype=np.float32))
                    if not np.isfinite(ridge_score).all():
                        raise ValueError(f"Invalid Ridge scores in {split}")
                    if split == "validation":
                        fitted_scores = pd.Series(ridge_score, index=frame.index).loc[validation_target.index]
                        ridge_mse["validation"] = float(np.mean((fitted_scores - validation_target) ** 2))
                    scored["ridge"] = ridge_score
                predictions.append(scored)
    curve.assign(strategy="lightgbm", window=pd.NA).to_csv(out / "training_mse.csv", index=False)
    predictions = pd.concat(predictions, ignore_index=True).sort_values(["split", "trade_date", "ticker"])
    predictions.to_parquet(out / "predictions.parquet", index=False)
    pd.DataFrame(feature_audit).to_csv(out / "feature_coverage.csv", index=False)
    log = dict(**settings, **rank_metrics, best_iteration=best, trained_rounds=len(curve), lightgbm_version=lgb.__version__,
               split_audit=split_audit)
    if ridge is not None:
        log.update(ridge_training_label=settings["training_label"], ridge_mse=ridge_mse)
    (out / "training_log.json").write_text(json.dumps(log, indent=2) + "\n")
    audit_file = out / "data_audit.json"
    if audit_file.exists():
        audit = json.loads(audit_file.read_text())
        audit["usable_labels_by_split"] = {name: values.get("purged_label_rows", values.get("evaluable_rows"))
                                           for name, values in split_audit.items()}
        audit_file.write_text(json.dumps(audit, indent=2) + "\n")
    return predictions


def load_predictions(config: dict, out: Path) -> pd.DataFrame:
    """读取当前批次的预测并确认训练参数匹配。"""
    log = json.loads((out / "training_log.json").read_text())
    log.setdefault("ridge_loss", RIDGE_LOSS)
    log.setdefault("alpha158_processing", "raw")
    log["training_label"] = normalize_training_label(log.get("training_label", ZSCORE_LABEL))
    settings = training_settings(config)
    log.get("training_control", {}).pop("log_evaluation_period", None)
    settings["training_control"].pop("log_evaluation_period", None)
    if any(log.get(key) != value for key, value in settings.items()):
        raise ValueError("Saved predictions use a different training configuration")
    return pd.read_parquet(out / "predictions.parquet")


def compare_ridge(config: dict, cache: Path, out: Path) -> pd.DataFrame:
    """用缓存的中性化特征比较 L2 惩罚，冻结验证 MSE 最低的强正则后再打分。"""
    import joblib
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    from threadpoolctl import threadpool_limits

    strong = config["ridge_strong_alphas"]
    if not strong or not all(np.isfinite(a) and a > 0 for a in strong):
        raise ValueError("ridge_strong_alphas 须为有限正数")
    names = alpha158_columns(cache)
    path = cache / "features.parquet"
    segments = config["segments"]

    def fitting_data(split, bounds=None):
        bounds = segments[split] if bounds is None else bounds
        rows = read_feature_split(path, bounds, split)
        eligible = rows.loc[segment_masks(rows, {split: bounds})[split]]
        target = fitting_targets(eligible, config["training_label"])
        rows = rows.loc[target.index]
        return rows[names], target.set_axis(stock_day_index(rows))

    prototype = ridge_pipeline(config, strong[0])
    transform = prototype[:-1]
    x, y = fitting_data("train")
    x = transform.fit_transform(x)
    v, z = fitting_data("validation")
    v = transform.transform(v)
    results, models = [], {}
    with threadpool_limits(limits=config.get("ridge_num_threads", 4)):
        for alpha in sorted(set(strong)):
            ridge = clone(prototype[-1]).set_params(alpha=alpha).fit(x, y)
            row = dict(alpha=alpha, coefficient_norm=float(np.linalg.norm(ridge.coef_)))
            for split, features, target in (("train", x, y), ("validation", v, z)):
                score = pd.Series(ridge.predict(features), index=target.index)
                row[split + "_mse"] = ((score - target) ** 2).mean()
                row[split + "_rank_ic"] = daily_ic(score, target)[1].mean()
            results.append(row)
            models[alpha] = Pipeline([*transform.steps, ("ridge", ridge)])
            print(f"Ridge α={alpha:g} 训练 MSE={row['train_mse']:.7f} 验证 MSE={row['validation_mse']:.7f} "
                  f"训练 Rank IC={row['train_rank_ic']:.5f} 验证 Rank IC={row['validation_rank_ic']:.5f}", flush=True)
        del x, v
        trials = pd.DataFrame(results).set_index("alpha")
        if not np.isfinite(trials.validation_mse).all():
            raise ValueError("Ridge 验证 MSE 无效")
        if "walk_forward" in config:
            from src.ml.walk_forward import folds

            records = []
            for fold in folds(config):
                train_x, train_y = fitting_data("train", fold["train"])
                valid_x, valid_y = fitting_data("validation", fold["validation"])
                preprocess = clone(prototype[:-1])
                train_x = preprocess.fit_transform(train_x)
                valid_x = preprocess.transform(valid_x)
                for alpha in trials.index:
                    estimator = clone(prototype[-1]).set_params(alpha=alpha).fit(train_x, train_y)
                    loss = np.mean((estimator.predict(valid_x) - valid_y.to_numpy()) ** 2)
                    records.append(dict(alpha=alpha, train_end=fold["train"][1], valid_start=fold["validation"][0],
                                        valid_end=fold["validation"][1], mse=loss))
            detail = pd.DataFrame(records)
            if not np.isfinite(detail.mse).all():
                raise ValueError("Ridge 逐折 MSE 无效")
            detail.to_csv(out / "fold_trials.csv", index=False)
            trials["fold_mean_mse"] = detail.groupby("alpha").mse.mean()
            best = trials.fold_mean_mse.idxmin()
        else:
            best = trials.validation_mse.idxmin()
        trials.to_csv(out / "ridge_trials.csv")
        joblib.dump(models[best], out / "ridge.joblib")
        write_best_config(out, dict(config, ridge_alpha=float(best)))
        parts = []
        for split in ("validation", "test"):
            rows = read_feature_split(path, segments[split], split)
            result = rows[["trade_date", "ticker", "label"]].copy()
            features = transform.transform(rows[names])
            result["ridge_strong"] = models[best][-1].predict(features)
            result["evaluable"] = segment_masks(rows, segments)[split]
            parts.append(result.assign(split=split))
    return pd.concat(parts, ignore_index=True)
