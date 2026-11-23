"""扩展训练窗口，按年度推进。每折依次使用互不重叠的训练、早停和选参区间。"""
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from src.data.research_panel import segment_masks
from src.ml.dataset import alpha158_columns, fitting_targets, read_feature_split
from src.ml.rank_validation import BASES, equal_ranks, read_period, validation_loss


def folds(cfg):
    start = pd.Timestamp(cfg["segments"]["train"][0])
    end = pd.Timestamp(cfg["segments"]["validation"][1])
    if end >= pd.Timestamp(cfg["segments"]["test"][0]):
        raise ValueError("验证区间与测试期重叠")
    policy = cfg["walk_forward"]
    initial = policy["initial_train_years"]
    stop_years = policy.get("early_stop_years")
    horizon = policy["validation_years"]
    if any(type(value) is not int or value < 1 for value in (initial, stop_years, horizon)):
        raise ValueError("walk_forward 需要正整数 initial_train_years、early_stop_years、validation_years")
    result = []
    for stop in range(start.year + initial, end.year + 1):
        valid = stop + stop_years
        valid_end = f"{valid + horizon - 1}-12-31"
        if pd.Timestamp(valid_end) > end:
            break
        result.append(dict(train=[str(start.date()), f"{stop - 1}-12-31"],
                           early_stop=[f"{stop}-01-01", f"{valid - 1}-12-31"],
                           validation=[f"{valid}-01-01", valid_end]))
    if not result:
        raise ValueError("配置的历史范围容纳不下完整的首个验证窗口")
    return result


def period(source, bounds, names, cfg, ranking, extra=None):
    if ranking:
        frame = read_period(source, bounds, split_top25=cfg.get("split_top25", False))
        rows = frame.loc[frame.evaluable]
        target = rows.grade.to_numpy(dtype=int)
    else:
        frame = read_feature_split(source / "features.parquet", bounds, "walk-forward")
        mask = segment_masks(frame, {"period": bounds})["period"]
        target = fitting_targets(frame.loc[mask], cfg["training_label"])
        rows = frame.loc[target.index]
    if set(rows.trade_date.dt.year) != set(range(int(bounds[0][:4]), int(bounds[1][:4]) + 1)):
        raise ValueError(f"窗口内缺少有效年度数据 {bounds}")
    groups = rows.groupby("trade_date", sort=False).size().to_numpy() if ranking else None
    if extra is not None:
        rows = rows.join(extra, on=["trade_date", "ticker"], validate="one_to_one")
    return rows[names], np.asarray(target), groups


def fold_models(source, cfg, fold, family, params, names, ranking, extra=None):
    blocks = [period(source, fold[part], names, cfg, ranking, extra) for part in ("train", "early_stop")]
    if family == "catboost":
        from catboost import CatBoostRanker, Pool

        for name in BASES:
            gains = np.asarray(cfg["label_gains"][name])
            train, stop = [Pool(x, label=gains[y], group_id=np.repeat(np.arange(len(groups)), groups),
                                thread_count=params["thread_count"]) for x, y, groups in blocks]
            model = CatBoostRanker(**params, iterations=cfg["num_boost_round"], random_seed=cfg["seed"],
                                   allow_writing_files=False)
            model.fit(train, eval_set=stop, use_best_model=True, early_stopping_rounds=cfg["early_stopping_rounds"],
                      verbose=False)
            yield name, model, model.tree_count_
        return
    matrices = []
    for features, labels, groups in blocks:
        reference = matrices[0] if matrices else None
        if family == "xgboost":
            matrix = xgb.QuantileDMatrix(features, group=groups, ref=reference,
                                         nthread=params["nthread"], max_bin=params.get("max_bin", 256))
        else:
            matrix = lgb.Dataset(features, label=labels, group=groups, reference=reference)
        matrices.append(matrix)
    rounds = cfg["num_boost_round"]
    patience = cfg["early_stopping_rounds"]
    for name in BASES if ranking else ("regression",):
        gains = cfg["label_gains"][name] if ranking else None
        if family == "xgboost":
            for matrix, (_, grades, _) in zip(matrices, blocks):
                matrix.set_label(np.asarray(gains)[grades])
            model = xgb.train(params, matrices[0], rounds, evals=[(matrices[1], "early_stop")], maximize=True,
                              early_stopping_rounds=patience, verbose_eval=False)
            best = model.best_iteration + 1
            model = model[:best]
        else:
            args = dict(params, label_gain=gains) if ranking else params
            model = lgb.train(args, matrices[0], rounds, valid_sets=[matrices[1]],
                              callbacks=[lgb.early_stopping(patience, verbose=False)])
            best = model.best_iteration
        yield name, model, best


def regression_loss(source, cfg, fold, params):
    names = alpha158_columns(source)
    x, y, _ = period(source, fold["validation"], names, cfg, False)
    _, model, rounds = next(fold_models(source, cfg, fold, "lightgbm", params, names, False))
    return float(np.mean((model.predict(x) - y) ** 2)), rounds


def rank_predictions(out, cfg, options, *, names=None, fold_features=None, extra=None):
    """按历史窗口重训并缓存每个候选的四目标等权排名。"""
    source = Path(cfg["feature_source"])
    names = alpha158_columns(source) if names is None else names
    for fold in folds(cfg):
        start, end = fold["validation"]
        valid = read_period(source, fold["validation"], split_top25=cfg.get("split_top25", False))
        if set(valid.loc[valid.evaluable, "trade_date"].dt.year) != set(range(int(start[:4]), int(end[:4]) + 1)):
            raise ValueError(f"验证窗口缺少有效年度数据 {start} 至 {end}")
        if extra is not None:
            valid = valid.join(extra, on=["trade_date", "ticker"], validate="one_to_one")
        if fold_features is not None:
            names = fold_features[fold["validation"][0]]
        features = valid[names]
        keys = valid[["trade_date", "ticker", "evaluable", "grade"]]
        del valid
        ranks = {}
        for family, choices in options.items():
            ranks[family] = []
            for params in choices:
                inputs = [params, fold, names, cfg["label_gains"], cfg["seed"], cfg["num_boost_round"],
                          cfg["early_stopping_rounds"], cfg.get("split_top25", False), str(source.resolve())]
                key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()[:16]
                directory = out / "folds" / f"{fold['early_stop'][0]}_{start}_{end}" / family
                directory.mkdir(parents=True, exist_ok=True)
                path = directory / f"{key}.parquet"
                if path.is_file():
                    scores = pd.read_parquet(path)
                    pd.testing.assert_frame_equal(scores[["trade_date", "ticker"]], keys[["trade_date", "ticker"]])
                else:
                    scores = keys[["trade_date", "ticker"]].copy()
                    for name, model, _ in fold_models(source, cfg, fold, family, params, names, True, extra):
                        if family == "xgboost":
                            scores[name] = model.inplace_predict(features)
                        elif family == "catboost":
                            scores[name] = model.predict(features, thread_count=params["thread_count"])
                        else:
                            scores[name] = model.predict(features, num_threads=params.get("num_threads"))
                    scores["score"] = equal_ranks(scores, list(BASES))
                    scores = scores[["trade_date", "ticker", "score"]]
                    tmp = path.with_suffix(".tmp.parquet")
                    scores.to_parquet(tmp)
                    tmp.replace(path)
                ranks[family].append(scores.score)
        yield fold, keys, ranks


def rank_losses(out, cfg, options, weight, *, names=None, fold_features=None, extra=None):
    gains = {name: cfg["label_gains"][name] for name in BASES}
    records = []
    for fold, keys, ranks in rank_predictions(out, cfg, options, names=names, fold_features=fold_features, extra=extra):
        start, end = fold["validation"]
        if "catboost" in ranks:
            lightgbm, = ranks["lightgbm"]
            xgboost, = ranks["xgboost"]
            ranks["lightgbm"] = [(1 - weight) * lightgbm + weight * xgboost]
            candidates = ranks["catboost"]
            blend_weight = cfg["catboost_four_equal"]["catboost_weight"]
            candidate_col = "cat_index"
        else:
            candidates = ranks["xgboost"]
            blend_weight = weight
            candidate_col = "xgb_index"
        own_losses = [validation_loss(keys, score, gains, cfg["topk"]).loss.mean() for score in candidates]
        for i, left in enumerate(ranks["lightgbm"]):
            baseline = validation_loss(keys, left, gains, cfg["topk"]).loss.mean()
            for j, right in enumerate(candidates):
                daily = validation_loss(keys, (1 - blend_weight) * left + blend_weight * right, gains, cfg["topk"])
                records.append(dict(light_index=i, **{candidate_col: j}, train_start=fold["train"][0], train_end=fold["train"][1],
                                    early_stop_start=fold["early_stop"][0], early_stop_end=fold["early_stop"][1],
                                    valid_start=start, valid_end=end, days=len(daily), baseline_loss=baseline,
                                    model_loss=own_losses[j], blend_loss=daily.loss.mean()))
    table = pd.DataFrame(records)
    if not np.isfinite(table[["baseline_loss", "model_loss", "blend_loss"]]).all().all():
        raise ValueError("逐折排序损失不是有限数值")
    return table
