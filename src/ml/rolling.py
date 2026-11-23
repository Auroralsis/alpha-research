"""用 Qlib 滑动任务和记录器训练独立的 LightGBM，并拼接测试预测。"""

import json
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.config import RIDGE_LOSS, normalize_training_label
from src.data.qlib_provider import register_provider
from src.ml.dataset import alpha158_columns, fitting_segment, make_dataset, read_feature_split
from src.ml.models import ROLLING_SCORE, fit_lightgbm, record_test_signal, training_settings


def init_rolling_qlib(dates: pd.DatetimeIndex, version: str, out: Path) -> None:
    """把项目交易日历和本地记录器交给 Qlib。"""
    calendar_dir = out / "provider" / "calendars"
    calendar_dir.mkdir(parents=True, exist_ok=True)
    text = "\n".join(dates.strftime("%Y-%m-%d")) + "\n"
    for name in ("day.txt", "day_future.txt"):
        (calendar_dir / name).write_text(text)
    register_provider(calendar_dir.parent, version)


def rolling_windows(dates: pd.DatetimeIndex, config: dict, feature_start: pd.Timestamp) -> tuple[list[dict], int]:
    """调用官方滑动生成器，返回原始与有效区间、交易日步长。"""
    from qlib.workflow.task.gen import RollingGen, task_generator
    from qlib.workflow.task.utils import TimeAdjuster

    class LocalTimeAdjuster(TimeAdjuster):
        def shift(self, seg, step, rtype=TimeAdjuster.SHIFT_SD):
            """将 Qlib 0.9.7 的越界空起点交给生成器已有的结束分支。"""
            result = super().shift(seg, step, rtype)
            if seg[0] is not None and result[0] is None:
                raise KeyError("Shifted start is outside the available market calendar")
            return result

    adjuster = LocalTimeAdjuster(future=True)
    if not pd.DatetimeIndex(adjuster.cals).as_unit("ns").equals(dates.as_unit("ns")):
        raise ValueError("Qlib calendar differs from the project market calendar")
    segments = {name: tuple(bounds) for name, bounds in config["segments"].items()}
    nominal = adjuster.align_seg(segments)
    first = max(nominal["train"][0], feature_start)
    segments["train"] = (first, segments["train"][1])
    test_start, test_end = nominal["test"]
    target_dates = dates[(dates >= test_start) & (dates <= test_end)]
    if target_dates.empty or not first <= nominal["train"][1] < nominal["validation"][0] <= nominal["validation"][1] < test_start:
        raise ValueError("Rolling study requires nonempty and ordered train, validation and test segments")
    step = ceil(len(target_dates) / config.get("rolling_test_windows", 3))
    distance = config["label_end_days"] + 1
    template = {"dataset": {"kwargs": {"segments": segments}}}
    raw_generator = RollingGen(step=step, rtype=RollingGen.ROLL_SD, ds_extra_mod_func=None)
    safe_generator = RollingGen(step=step, rtype=RollingGen.ROLL_SD, trunc_days=distance, ds_extra_mod_func=None)
    raw_generator.ta = safe_generator.ta = adjuster
    raw_tasks = task_generator(template, raw_generator)
    safe_tasks = task_generator(template, safe_generator)
    windows, predicted_dates = [], []
    for i, (raw_task, safe_task) in enumerate(zip(raw_tasks, safe_tasks, strict=True), 1):
        raw = raw_task["dataset"]["kwargs"]["segments"]
        segments = safe_task["dataset"]["kwargs"]["segments"].copy()
        segments["train"] = adjuster.truncate(segments["train"], raw["validation"][0], distance)
        begin, end = raw["test"]
        segments["test"] = (begin, min(end, test_end) if end is not None else test_end)
        train_begin, train_end = segments["train"]
        valid_begin, valid_end = segments["validation"]
        if not train_begin <= train_end < valid_begin <= valid_end < begin:
            raise ValueError(f"Insufficient history to truncate rolling window {i}")
        window_dates = target_dates[(target_dates >= begin) & (target_dates <= segments["test"][1])]
        predicted_dates.extend(window_dates)
        windows.append(dict(window=i, raw=raw, effective=segments, test_sessions=len(window_dates)))
    if not pd.DatetimeIndex(predicted_dates).as_unit("ns").equals(target_dates.as_unit("ns")):
        raise ValueError("Rolling tests must cover the full test calendar exactly once")
    return windows, step


def read_window(features: Path, bounds: tuple, dates: pd.DatetimeIndex, config: dict) -> pd.DataFrame:
    """按信号日读取股票特征，并核对已保存的特征日和收益日期。"""
    rows = read_feature_split(features, bounds, "rolling")
    if (dates.get_indexer(rows.trade_date) < 0).any():
        raise ValueError(f"Invalid feature keys for {bounds}")
    if not rows.feature_date.le(rows.trade_date).all():
        raise ValueError("Feature date follows the signal")
    if "source_snapshot_date" in rows and not rows.source_snapshot_date.le(rows.trade_date).all():
        raise ValueError("Membership snapshot follows the signal")
    for column, key in (("label_start_date", "label_start_days"), ("label_end_date", "label_end_days")):
        session = pd.Series(dates, index=dates).shift(-config[key])
        expected = pd.DatetimeIndex(rows.trade_date.map(session)).as_unit("ns")
        if not expected.equals(pd.DatetimeIndex(rows[column]).as_unit("ns")):
            raise ValueError(f"Stored endpoints in {column} differ from the market calendar")
    return rows


def fit_window(window: dict, config: dict, dates: pd.DatetimeIndex, features: Path, names: list[str],
               out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """拟合本窗口模型，写入 Qlib 信号记录，返回测试股票日和损失。"""
    out.mkdir(parents=True, exist_ok=True)
    settings = training_settings(config)
    bounds, raw = window["effective"], window["raw"]
    fitting, audit = {}, {}
    for split, next_start in (("train", raw["validation"][0]), ("validation", raw["test"][0])):
        rows = read_window(features, bounds[split], dates, config)
        valid = rows.label_end_date.lt(next_start) & rows.label_end_date.le(raw[split][1]) & np.isfinite(rows.label)
        frame, target = fitting_segment(rows, names, valid, settings["training_label"], split)
        if np.isinf(frame["feature"].to_numpy(dtype=np.float32)).any():
            raise ValueError(f"Infinite Alpha158 inputs in {split}")
        fitted = rows.loc[target.index, ["trade_date", "label_end_date"]]
        audit[split] = dict(signal_rows=len(rows), fitted_rows=len(target), signal_days=rows.trade_date.nunique(),
                            valid_label_rows=int(valid.sum()), fitted_signal_start=fitted.trade_date.min(),
                            fitted_signal_end=fitted.trade_date.max(), latest_label_end=fitted.label_end_date.max())
        fitting["train" if split == "train" else "valid"] = frame
        del rows, frame

    dataset = make_dataset(**fitting)
    fitting.clear()
    model, curve, rank_metrics = fit_lightgbm(
        dataset, settings["lightgbm_params"], settings["training_control"], title=f'滚动 LightGBM 窗口 {window["window"]}',
    )
    del dataset
    best = model.model.best_iteration
    model.model.save_model(str(out / "lightgbm.txt"), num_iteration=best)
    test = read_window(features, bounds["test"], dates, config)
    if np.isinf(test[names].to_numpy(dtype=np.float32)).any():
        raise ValueError("Infinite test features")
    test_end = pd.Timestamp(config["segments"]["test"][1])
    evaluable = test.label_end_date.le(test_end) & np.isfinite(test.label)
    record_test_signal(model, test, names, evaluable)
    predictions = test.drop(columns=names).assign(
        split="test", window=window["window"], evaluable=evaluable,
    ).sort_values(["trade_date", "ticker"]).reset_index(drop=True)
    log = dict(**window, **rank_metrics, lightgbm_params=settings["lightgbm_params"], training_control=settings["training_control"],
               training_label=settings["training_label"], best_iteration=best, trained_rounds=len(curve),
               train_mse=curve.iloc[best-1].train_mse, validation_mse=curve.iloc[best-1].validation_mse,
               fit_audit=audit, test_rows=len(predictions), test_labeled_rows=int(predictions.evaluable.sum()),
               fitted_preprocessors=[])
    (out / "training_log.json").write_text(json.dumps(log, default=str, indent=2) + "\n")
    return predictions, curve


def check_prediction_windows(predictions: pd.DataFrame, windows: list[dict], expected: pd.DataFrame) -> None:
    """核对模型归属、交易日覆盖和股票日期键，拒绝重叠与漏项。"""
    keys = ["trade_date", "ticker"]
    if predictions.duplicated(keys).any():
        raise ValueError("Duplicate stock-day predictions before or after rolling merge")
    left = pd.MultiIndex.from_frame(predictions[keys]).sort_values()
    right = pd.MultiIndex.from_frame(expected[keys]).sort_values()
    if not left.equals(right):
        raise ValueError("Rolling prediction coverage differs from the expected stock-days")
    for window in windows:
        begin, end = map(pd.Timestamp, window["effective"]["test"])
        assigned = predictions.window.eq(window["window"]).to_numpy(dtype=bool, na_value=False)
        within = predictions.trade_date.between(begin, end)
        if not np.array_equal(assigned, within.to_numpy()) or predictions.loc[assigned, "trade_date"].nunique() != window["test_sessions"]:
            raise ValueError(f"Incorrect predictions for window {window['window']}")


def merge_rolling_predictions(fixed: pd.DataFrame, rolling: pd.DataFrame, windows: list[dict]) -> pd.DataFrame:
    """将滚动分数与固定分数按股票日对齐，保留一份收益标签。"""
    keys = ["trade_date", "ticker"]
    test = fixed.loc[fixed.split.eq("test")]
    check_prediction_windows(rolling, windows, test)
    paired = test[keys + ["label", "evaluable"]].merge(
        rolling[keys + ["label", "evaluable"]], on=keys, suffixes=("_fixed", "_rolling"), validate="one_to_one",
    )
    for field in ("label", "evaluable"):
        pd.testing.assert_series_equal(paired[f"{field}_fixed"], paired[f"{field}_rolling"], check_names=False)
    merged = fixed.drop(columns=[ROLLING_SCORE, "window"], errors="ignore").merge(
        rolling[keys + [ROLLING_SCORE, "window"]], on=keys, how="left", validate="one_to_one",
    )
    return merged.astype({"window": "Int64"})


def train_rolling(config: dict, dates: pd.DatetimeIndex, out: Path) -> pd.DataFrame:
    """用 TrainerR 逐窗训练，收集记录器中的预测后用 RollingGroup 拼接。"""
    from qlib.model.ens.group import RollingGroup
    from qlib.model.trainer import TrainerR
    from qlib.workflow import R
    from qlib.workflow.task.collect import RecorderCollector

    features = out / "features.parquet"
    names = alpha158_columns(out)
    models = out / "models"
    with TemporaryDirectory(prefix="qlib-window-") as work:
        init_rolling_qlib(dates, config["qlib_version"], Path(work))
        feature_start = pd.Timestamp(json.loads((out / "data_audit.json").read_text())["first_feature_date"])
        windows, step = rolling_windows(dates, config, feature_start)
        chunks, curves = [], []

        def train_task(task: dict, experiment_name: str, recorder_name=None):
            """适配 TrainerR 的本地任务，保存训练产物和日期索引预测。"""
            folder = models / f'window_{task["window"]:02d}'
            with R.start(experiment_name=experiment_name, recorder_name=f'window_{task["window"]:02d}'):
                recorder = R.get_recorder()
                pred, curve = fit_window(task, config, dates, features, names, folder)
                chunks.append(pred)
                curves.append(curve.assign(strategy=ROLLING_SCORE, window=task["window"]))
                recorder.log_params(window=task["window"])
            return recorder

        trainer = TrainerR(experiment_name="rolling_lightgbm", train_func=train_task)
        recorders = trainer(windows)
        keys_by_record = {recorder.id: window["window"] for recorder, window in zip(recorders, windows, strict=True)}
        expected = pd.read_parquet(features, columns=["trade_date", "ticker"],
                                   filters=[("trade_date", ">=", windows[0]["effective"]["test"][0]),
                                            ("trade_date", "<=", windows[-1]["effective"]["test"][1])])
        collector = RecorderCollector(lambda: recorders, process_list=[RollingGroup()],
                                       rec_key_func=lambda rec: ("lightgbm", keys_by_record[rec.id]))
        scores = collector()["pred"][("lightgbm",)].reset_index().rename(
            columns={"datetime": "trade_date", "instrument": "ticker", "score": ROLLING_SCORE})
        merged = pd.concat(chunks, ignore_index=True).merge(
            scores, on=["trade_date", "ticker"], how="outer", validate="one_to_one",
        )
        if not np.isfinite(merged[ROLLING_SCORE]).all() or merged.window.isna().any():
            raise ValueError("Missing or nonfinite Qlib rolling predictions")
        check_prediction_windows(merged, windows, expected)
    curve_path = out / "training_mse.csv"
    if curve_path.is_file():
        fixed_curve = pd.read_csv(curve_path)
        if "strategy" in fixed_curve:
            fixed_curve = fixed_curve.loc[fixed_curve.strategy.eq("lightgbm")]
        curves.insert(0, fixed_curve.assign(strategy="lightgbm", window=pd.NA))
    pd.concat(curves, ignore_index=True).astype({"window": "Int64"}).to_csv(curve_path, index=False)
    specs = dict(step=step, trunc_days=config["label_end_days"] + 1, windows=windows, recorders=keys_by_record,
                 config=config)
    (out / "windows.json").write_text(json.dumps(specs, default=str, indent=2) + "\n")
    return merged


def study_windows(config: dict, out: Path, predictions: pd.DataFrame) -> list[dict]:
    """核对存档预测的训练设置、滚动窗口与测试期覆盖。"""
    if ROLLING_SCORE not in predictions:
        raise ValueError("实验缺少滚动模型预测，请用 run_all 完整运行")
    specs = json.loads((out / "windows.json").read_text())
    saved = specs["config"]
    saved.setdefault("ridge_loss", RIDGE_LOSS)
    saved["training_label"] = normalize_training_label(saved["training_label"])
    # layers 和 main_offset 是旧实验的回测展示参数，可在复用预测时省略。
    evaluation_keys = {"walk_forward", "layers", "min_regression_pairs", "topk", "n_drop", "rebalance_days", "main_offset",
                       "buy_price", "buy_fee_bps", "sell_fee_bps", "initial_capital_cny", "limit_threshold",
                       "single_catboost", "catboost_four_equal", "lightgbm_four_equal", "four_model_targets", "lightgbm_xgboost_blend",
                       "xgboost", "xgboost_tuning", "trade_unit", "min_trade_cost_cny", "lightgbm_tuning", "neutral_search",
                       "listfold_mlp", "rank_pair_tuning", "ridge_strong_alphas", "log_evaluation_period"}
    if any(config.get(key) != saved.get(key) for key in (config.keys() | saved.keys()) - evaluation_keys):
        raise ValueError("配置与存档的滚动训练不匹配")
    test = predictions.loc[predictions.split.eq("test")]
    if not np.isfinite(test[ROLLING_SCORE]).all() or test.window.isna().any():
        raise ValueError("滚动模型缺少测试期分数或窗口编号")
    check_prediction_windows(test, specs["windows"], test)
    return specs["windows"]
