"""用 Qlib 滑动任务和记录器训练独立的 LightGBM，并拼接测试预测。"""

import json
from math import ceil
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.ml.models import ROLLING_SCORE, daily_zscore, fit_lightgbm, training_settings


def init_rolling_qlib(dates: pd.DatetimeIndex, version: str, out: Path) -> None:
    """把项目交易日历交给 Qlib，并把实验记录写入 out/mlruns。"""
    import qlib

    if qlib.__version__ != version:
        raise RuntimeError(f"Expected Qlib {version}, found {qlib.__version__}")
    calendar_dir = out / "provider" / "calendars"
    calendar_dir.mkdir(parents=True, exist_ok=True)
    text = "\n".join(dates.strftime("%Y-%m-%d")) + "\n"
    for name in ("day.txt", "day_future.txt"):
        (calendar_dir / name).write_text(text)
    qlib.init(provider_uri=str(calendar_dir.parent), region="cn", expression_cache=None, dataset_cache=None,
              exp_manager={"class": "MLflowExpManager", "module_path": "qlib.workflow.expm",
                           "kwargs": {"uri": (out.resolve() / "mlruns").as_uri(), "default_exp_name": "rolling_lightgbm"}})


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
    step = ceil(len(target_dates) / 3)
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
    begin, end = bounds
    rows = pd.read_parquet(features, filters=[("trade_date", ">=", begin), ("trade_date", "<=", end)])
    positions = dates.get_indexer(rows.trade_date)
    if rows.empty or rows.duplicated(["trade_date", "ticker"]).any() or (positions < 0).any():
        raise ValueError(f"Invalid feature keys for {begin} to {end}")
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
    """用这一轮合法的训练和验证收益拟合模型，只给其测试区间打分。"""
    import lightgbm as lgb

    out.mkdir(parents=True, exist_ok=True)
    settings = training_settings(config)
    bounds, raw = window["effective"], window["raw"]
    datasets, audit = [], {}
    for split, next_start in (("train", raw["validation"][0]), ("validation", raw["test"][0])):
        rows = read_window(features, bounds[split], dates, config)
        valid = rows.label_end_date.lt(next_start) & rows.label_end_date.le(raw[split][1]) & np.isfinite(rows.label)
        eligible = rows.loc[valid]
        z = daily_zscore(eligible.label, eligible.trade_date)
        z = z.loc[np.isfinite(z)]
        values = rows.loc[z.index, names]
        if z.empty or np.isinf(values.to_numpy(dtype=np.float32)).any():
            raise ValueError(f"No usable fitting labels or infinite features in {split}")
        audit[split] = dict(signal_rows=len(rows), fitted_rows=len(z), signal_days=rows.trade_date.nunique(),
                            valid_label_rows=int(valid.sum()), fitted_signal_start=rows.loc[z.index, "trade_date"].min(),
                            fitted_signal_end=rows.loc[z.index, "trade_date"].max(),
                            latest_label_end=rows.loc[z.index, "label_end_date"].max())
        datasets.append(lgb.Dataset(values, label=z))
        del rows, eligible

    model, curve = fit_lightgbm(datasets, settings["lightgbm_params"], settings["training_control"])
    best = model.best_iteration
    model.save_model(str(out / "lightgbm.txt"), num_iteration=best)
    test = read_window(features, bounds["test"], dates, config)
    x = test[names]
    if np.isinf(x.to_numpy(dtype=np.float32)).any():
        raise ValueError("Infinite test features")
    score = model.predict(x, num_iteration=best, num_threads=settings["lightgbm_params"]["num_threads"])
    if not np.isfinite(score).all():
        raise ValueError("Nonfinite rolling predictions")
    test_end = pd.Timestamp(config["segments"]["test"][1])
    predictions = test.drop(columns=names).assign(
        split="test", window=window["window"], rolling_lightgbm=score,
        evaluable=test.label_end_date.le(test_end) & np.isfinite(test.label),
    ).sort_values(["trade_date", "ticker"]).reset_index(drop=True)
    log = dict(**window, lightgbm_params=settings["lightgbm_params"], training_control=settings["training_control"],
               training_label="daily_sample_zscore", best_iteration=best, trained_rounds=len(curve),
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
    names = [item["name"] for item in json.loads((out / "alpha158.json").read_text())]
    if len(names) != 158 or len(set(names)) != 158 or config["training_label"] != "daily_sample_zscore":
        raise ValueError("Expected the existing Alpha158 and daily label standardization")
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
                indexed = pred.rename(columns={"trade_date": "datetime", "ticker": "instrument"})
                recorder.save_objects(**{"pred.pkl": indexed.set_index(["datetime", "instrument"])})
                recorder.log_params(window=task["window"])
            return recorder

        trainer = TrainerR(experiment_name="rolling_lightgbm", train_func=train_task)
        recorders = trainer(windows)
        keys_by_record = {recorder.id: window["window"] for recorder, window in zip(recorders, windows, strict=True)}
        expected = pd.read_parquet(features, columns=["trade_date", "ticker"],
                                   filters=[("trade_date", ">=", windows[0]["effective"]["test"][0]),
                                            ("trade_date", "<=", windows[-1]["effective"]["test"][1])])
        check_prediction_windows(pd.concat(chunks, ignore_index=True), windows, expected)
        collector = RecorderCollector(lambda: recorders, process_list=[RollingGroup()],
                                       rec_key_func=lambda rec: ("lightgbm", keys_by_record[rec.id]))
        merged = collector()["pred"][("lightgbm",)].reset_index().rename(
            columns={"datetime": "trade_date", "instrument": "ticker"})
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
