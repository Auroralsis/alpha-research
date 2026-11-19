"""训练 Alpha158 模型，并评价各打分方法的排序和分组收益。"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation.factor_metrics import calc_ic_rank
from src.ml.alpha158 import segment_masks


MODEL_SCORES = ("lightgbm", "ridge")
FACTOR_SCORES = ("Alpha13", "Alpha3", "Alpha44")
SCORE_COLUMNS = (*MODEL_SCORES, *FACTOR_SCORES)
ROLLING_SCORE = "rolling_lightgbm"


def training_settings(config: dict) -> dict:
    """从研究配置提取训练参数，供模型拟合和回测核对训练日志使用。"""
    return {
        "segments": config["segments"],
        "ridge_alpha": config["ridge_alpha"],
        "lightgbm_params": dict(config["lightgbm"], seed=config["seed"]),
        "training_control": {key: config[key] for key in ("num_boost_round", "early_stopping_rounds")},
    }


def common_scores(frame: pd.DataFrame, names: tuple[str, ...]) -> pd.DataFrame:
    """输入逐股打分表，只保留指定方法都有有限分数的股票日。"""
    return frame.loc[np.isfinite(frame[list(names)]).all(axis=1)]


def daily_zscore(values: pd.Series, dates: pd.Series) -> pd.Series:
    """按信号日分别计算样本 Z-score，返回与 values 同索引的一列。

    有效样本不足两个或标准差为零的日期返回 NaN。
    """
    groups = values.groupby(dates)
    sd = groups.transform("std", ddof=1)
    return (values - groups.transform("mean")) / sd.where(sd.gt(0) & np.isfinite(sd))


def quintile_assignments(scores: pd.DataFrame, score: str = "lightgbm") -> pd.DataFrame:
    """按指定分数分成 Q1 至 Q5，返回 trade_date、ticker 和 group 三列。

    每天从低分排到高分，同分时按股票代码排序。
    """
    ranked = scores[["trade_date", "ticker", score]].sort_values(["trade_date", score, "ticker"]).copy()
    groups = ranked.groupby("trade_date", sort=False)[score]
    sizes = groups.transform("size").to_numpy()
    positions = groups.cumcount().to_numpy()
    # 与 np.array_split 一致，多出来的股票分给靠前的组。
    width, remainder = sizes // 5, sizes % 5
    wide_end = (width + 1) * remainder
    ranked["group"] = np.where(positions < wide_end, positions // (width + 1),
                                remainder + (positions - wide_end) // np.maximum(width, 1)) + 1
    return ranked[["trade_date", "ticker", "group"]]


def prediction_diagnostics(
    predictions: pd.DataFrame, config: dict, dates: pd.DatetimeIndex, out: Path, *,
    strategies: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """在每个阶段的共同有分样本上计算逐日及汇总 RankIC 和五组收益，写入 CSV。"""
    daily_ic, daily_groups = [], []
    for split, frame in predictions.groupby("split", sort=False):
        names = MODEL_SCORES
        if split == "test":
            names = strategies or (SCORE_COLUMNS if all(factor in frame for factor in FACTOR_SCORES) else MODEL_SCORES)
        frame = common_scores(frame, names)
        if frame.empty:
            raise ValueError(f"No common scored observations in {split}")
        labels = frame[["trade_date", "ticker", "label"]].copy()
        labels["label"] = labels.label.where(frame.evaluable)
        for name in names:
            groups = quintile_assignments(frame, name).merge(labels, on=["trade_date", "ticker"], validate="one_to_one")
            daily_groups.append(groups.groupby(["trade_date", "group"])["label"].agg(
                n_assigned="size", n_labeled="count", mean_return="mean").reset_index().assign(split=split, strategy=name))
            for day, stocks in frame.groupby("trade_date"):
                paired = stocks.loc[stocks.evaluable]
                rho = calc_ic_rank(paired[name], paired.label, config["min_ic_pairs"])
                daily_ic.append(dict(split=split, strategy=name, trade_date=day, n_pairs=len(paired), rank_ic=rho))

    ic = pd.DataFrame(daily_ic)
    q = pd.concat(daily_groups, ignore_index=True).sort_values(["split", "strategy", "group", "trade_date"])
    ic_summary, group_summary = summarize_predictions(ic, q, config, dates)
    for name, table in (("daily_rank_ic", ic), ("rank_ic_summary", ic_summary),
                        ("quintile_daily", q), ("quintile_summary", group_summary)):
        table.to_csv(out / f"{name}.csv", index=False)
    return ic, q


def summarize_predictions(ic: pd.DataFrame, q: pd.DataFrame, config: dict, dates: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """从逐日 RankIC 和分组收益计算给定日期区间的汇总。"""
    ic_rows, group_rows = [], []
    for (split, name), group in ic.groupby(["split", "strategy"]):
        begin, end = config["segments"][split]
        calendar = dates[dates.to_series().between(begin, end).to_numpy()]
        for year in ["all", *sorted(set(calendar.year))]:
            segment = calendar if year == "all" else calendar[calendar.year == year]
            valid = group.set_index("trade_date").rank_ic.reindex(segment).dropna()
            mean_ic, sd = valid.mean(), valid.std(ddof=1)
            ic_rows.append(dict(split=split, strategy=name, year=year, rank_ic_std=sd,
                                icir=mean_ic/sd if sd > 0 else np.nan, positive_rate=valid.gt(0).mean(),
                                mean=mean_ic, n_days=len(valid), calendar_days=len(segment)))
            partition = q.loc[q.split.eq(split) & q.strategy.eq(name) & q.trade_date.isin(segment)]
            wide = partition.pivot(index="trade_date", columns="group", values="mean_return").reindex(columns=range(1, 6))
            complete = wide.dropna()
            mean = wide.mean()
            coverage = partition.groupby("group")[["n_assigned", "n_labeled"]].sum()
            group_rows.append(dict(split=split, strategy=name, year=year,
                                   **{f"Q{i}": mean[i] for i in range(1, 6)},
                                   spread=(wide[5] - wide[1]).mean(), monotonic_means=bool(mean.is_monotonic_increasing),
                                   n_complete_days=len(complete),
                                   daily_monotonic_rate=complete.diff(axis=1).iloc[:, 1:].ge(0).all(axis=1).mean(),
                                   **{f"Q{i}_coverage": coverage.n_labeled.get(i, 0) / max(coverage.n_assigned.get(i, 0), 1)
                                      for i in range(1, 6)}))
    return pd.DataFrame(ic_rows), pd.DataFrame(group_rows)


def fit_lightgbm(datasets: list, params: dict, controls: dict):
    """用训练与验证数据集拟合一个独立的 LightGBM，返回模型和逐轮 MSE。"""
    import lightgbm as lgb

    evaluation = {}
    names = ["train", "validation"]
    model = lgb.train(
        params, datasets[0], num_boost_round=controls["num_boost_round"],
        valid_sets=datasets, valid_names=names,
        callbacks=[lgb.early_stopping(controls["early_stopping_rounds"], first_metric_only=True),
                   lgb.record_evaluation(evaluation), lgb.log_evaluation(period=50)],
    )
    curve = pd.DataFrame({f"{name}_mse": evaluation[name]["l2"] for name in names})
    curve.insert(0, "iteration", range(1, len(curve) + 1))
    if not 1 <= model.best_iteration <= len(curve) or model.best_iteration != int(curve.validation_mse.idxmin()) + 1:
        raise RuntimeError("LightGBM did not select the validation MSE minimum")
    return model, curve


def train_and_predict(config: dict, out: Path) -> pd.DataFrame:
    """读取 out 中的 Alpha158 特征，保存模型与训练日志，返回验证和测试预测长表。

    LightGBM 用按日标准化的标签训练，Ridge 用同一批股票日的原始收益训练。
    """
    import joblib
    import lightgbm as lgb
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    names = [entry["name"] for entry in json.loads((out / "alpha158.json").read_text())]
    if len(names) != 158 or len(set(names)) != 158:
        raise ValueError("Expected 158 unique Alpha158 columns")
    if config["training_label"] != "daily_sample_zscore":
        raise ValueError("Expected daily sample Z-score training labels")
    settings = training_settings(config)
    params = settings["lightgbm_params"]
    threads = params["num_threads"]
    controls = settings["training_control"]
    splits = ("train", "validation")
    feature_audit, split_audit = [], {}
    ridge = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                          Ridge(alpha=config["ridge_alpha"], solver="cholesky"))

    def load(split: str) -> pd.DataFrame:
        """读取一个阶段的股票日特征，并记录缺失值与标签覆盖情况。"""
        begin, end = map(pd.Timestamp, config["segments"][split])
        rows = pd.read_parquet(out / "features.parquet", filters=[("trade_date", ">=", begin), ("trade_date", "<=", end)])
        if rows.empty or rows.duplicated(["trade_date", "ticker"]).any():
            raise ValueError(f"Empty or duplicated feature rows in {split}")
        values = rows[names].to_numpy(dtype=np.float32)
        inf_counts = np.isinf(values).sum(axis=0)
        if inf_counts.any():
            bad = dict(zip(np.array(names)[inf_counts > 0], inf_counts[inf_counts > 0].tolist()))
            raise ValueError(f"Infinite Alpha158 inputs in {split}: {bad}")
        missing = np.isnan(values)
        feature_audit.extend(dict(split=split, feature=name, n_missing=int(n), missing_rate=n / len(rows))
                             for name, n in zip(names, missing.sum(axis=0)))
        split_audit[split] = dict(rows=len(rows), market_dates=rows.trade_date.nunique(),
                                  label_nonfinite_rows=int((~np.isfinite(rows.label)).sum()),
                                  label_after_split_rows=int(rows.label_end_date.gt(end).sum()),
                                  feature_nan_rate=float(missing.mean()),
                                  rows_all_features_nan=int(missing.all(axis=1).sum()))
        return rows

    datasets = []
    for split in splits:
        frame = load(split)
        mask = segment_masks(frame, config["segments"])[split]
        labels = frame.loc[mask, ["trade_date", "label"]]
        z = daily_zscore(labels.label, labels.trade_date)
        z = z.loc[np.isfinite(z)]
        dropped = labels.loc[~labels.index.isin(z.index)]
        split_audit[split].update(purged_label_rows=len(labels), fitted_rows=len(z),
                                  degenerate_label_days=dropped.trade_date.nunique(), degenerate_label_rows=len(dropped))
        if z.empty:
            raise ValueError(f"No nondegenerate labeled rows in {split}")
        features = frame.loc[z.index, names]
        datasets.append(lgb.Dataset(features, label=z))
        if split == "train":
            with threadpool_limits(limits=4):
                ridge.fit(features.to_numpy(dtype=np.float32), labels.loc[z.index, "label"].to_numpy(dtype=np.float32))
    models = out / "models"
    models.mkdir(exist_ok=True)
    joblib.dump(ridge, models / "ridge.joblib")
    validation = frame
    del frame

    print(f"Fit LightGBM: {split_audit['train']['fitted_rows']:,} training rows, "
          f"{split_audit['validation']['fitted_rows']:,} validation rows, {threads} threads", flush=True)
    model, curve = fit_lightgbm(datasets, params, controls)
    best = model.best_iteration
    model.save_model(str(models / "lightgbm.txt"), num_iteration=best)
    curve.assign(strategy="lightgbm", window=pd.NA).to_csv(out / "training_mse.csv", index=False)

    predictions, ridge_mse = [], {}
    for split in ("validation", "test"):
        frame = validation if split == "validation" else load(split)
        features = frame[names]
        score = model.predict(features, num_iteration=best, num_threads=threads)
        with threadpool_limits(limits=4):
            ridge_score = ridge.predict(features.to_numpy(dtype=np.float32))
        if not np.isfinite(score).all() or not np.isfinite(ridge_score).all():
            raise ValueError(f"Invalid model scores in {split}")
        mask = segment_masks(frame, config["segments"])[split]
        split_audit[split]["evaluable_rows"] = int(mask.sum())
        if split == "validation":
            ridge_mse["validation"] = float(np.mean((ridge_score[mask] - frame.loc[mask, "label"]) ** 2))
        predictions.append(frame.drop(columns=names).assign(
            split=split, evaluable=mask, lightgbm=score, ridge=ridge_score,
        ))
    predictions = pd.concat(predictions, ignore_index=True).sort_values(["split", "trade_date", "ticker"])
    predictions.to_parquet(out / "predictions.parquet", index=False)
    pd.DataFrame(feature_audit).to_csv(out / "feature_coverage.csv", index=False)
    log = dict(**settings, best_iteration=best, trained_rounds=len(curve), lightgbm_version=lgb.__version__,
               split_audit=split_audit, ridge_mse=ridge_mse)
    (out / "training_log.json").write_text(json.dumps(log, indent=2) + "\n")
    audit_file = out / "data_audit.json"
    if audit_file.exists():
        audit = json.loads(audit_file.read_text())
        audit["usable_labels_by_split"] = {name: values.get("purged_label_rows", values.get("evaluable_rows"))
                                           for name, values in split_audit.items()}
        audit_file.write_text(json.dumps(audit, indent=2) + "\n")
    return predictions
