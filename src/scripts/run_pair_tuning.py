"""用验证期最终等权排名，逐步调优 LightGBM 和 XGBoost 的树参数。"""
import argparse
import hashlib
import json
import math
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb

from src.backtest.engine import evaluate_signals
from src.config import CONFIG_FILE, load_tuning, write_best_config
from src.ml.dataset import alpha158_columns, read_feature_split
from src.ml.rank_validation import (
    BASES, aligned_reference, equal_ranks, fixed_grades, read_period, selection_stats, training_periods, validation_loss,
)
from src.scripts.run_all import latest_run, new_run
from src.scripts.run_lambdarank import fit_ranker
from src.scripts.boosting import fit_target
from src.data.research_panel import segment_masks

FAMILIES = ("lightgbm", "xgboost")
STAGES = ("depth", "leaves", "min_child", "l1", "l2", "row_sampling", "column_sampling")


def candidate_id(params):
    """相同完整参数共用磁盘模型与验证分数。"""
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]


def check_config(cfg):
    """校验逐阶段搜索的范围及实际会启用的采样设置。"""
    grid = cfg["rank_pair_tuning"]
    stages = grid.get("stages", STAGES)
    if not isinstance(stages, (list, tuple)) or not stages or any(s not in STAGES for s in stages) or len(stages) != len(set(stages)):
        raise ValueError("rank_pair_tuning.stages 需要有效且不重复的搜索阶段")
    fields = {"leaves": ["leaf_cap_fractions"], "min_child": ["min_data_in_leaf", "min_child_weight"]}
    for name in (key for stage in stages for key in fields.get(stage, [stage])):
        values = grid.get(name)
        if not isinstance(values, list) or not values:
            raise ValueError(f"rank_pair_tuning.{name} 须给出非空候选列表")
        for value in values:
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} 候选须为非负有限数")
            if name in ("depth", "min_data_in_leaf") and (type(value) is not int or value < 1):
                raise ValueError(f"{name} 候选须为正整数")
            if name in ("leaf_cap_fractions", "row_sampling", "column_sampling") and not 0 < value <= 1:
                raise ValueError(f"{name} 候选须在 (0, 1] 内")
    if "row_sampling" in stages and cfg["lightgbm"].get("bagging_freq", cfg["lightgbm"].get("subsample_freq", 0)) <= 0:
        raise ValueError("LightGBM 行采样需要正的 bagging_freq")
    if cfg["lightgbm_xgboost_blend"]["xgboost_weight"] != 0.5 or any(len(cfg["label_gains"][n]) != 5 for n in BASES):
        raise ValueError("基准须为使用五档标签的等权组合")


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


def scoring_inputs(cfg):
    """核对股票日、逐日标签人数与收益终点，仅装入训练和早停两段的矩阵。"""
    source = Path(cfg["feature_source"])
    features = alpha158_columns(source)
    blocks = []
    for _, rows, sizes in training_periods(source, Path(cfg["reference"]), cfg["segments"]):
        blocks.append((rows[features], rows.grade.to_numpy(dtype=int), sizes))
    return blocks


def train_candidate(out, family, params, cfg, blocks, xmat, valid_x):
    """同一树参数训练 A–D 四个目标；完整写出一个目标后即可断点接续。"""
    directory = out / family / candidate_id(params)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "params.json").write_text(json.dumps(params, indent=2))
    matrices = []
    if family == "lightgbm":
        for x, y, groups in blocks:
            matrices.append(lgb.Dataset(x, label=y, group=groups, reference=matrices[0] if matrices else None))
    predictions = {}
    for target in BASES:
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
            predictions[target] = model.inplace_predict(valid_x)
    return pd.DataFrame(predictions, index=valid_x.index)


def quantile_matrices(blocks, params):
    """早停集沿用训练集的特征分箱。"""
    matrices = []
    for x, _, groups in blocks:
        matrices.append(xgb.QuantileDMatrix(x, group=groups, ref=matrices[0] if matrices else None,
                                             nthread=params["nthread"], max_bin=params.get("max_bin", 256)))
    return matrices


def tune(out, cfg):
    """冻结每一阶段的最优组合，下一阶段以此为起点；只使用验证期最终排名选参。"""
    source, ref = Path(cfg["feature_source"]), Path(cfg["pair_reference"])
    valid = read_period(source, cfg["segments"]["validation"])
    valid_x = valid[alpha158_columns(source)]
    keys = valid[["trade_date", "ticker", "evaluable", "grade"]].copy()
    old = aligned_reference(ref, keys, "validation", ["equal", "xgb_equal", "blend_equal"])
    np.testing.assert_allclose((old.equal + old.xgb_equal) / 2, old.blend_equal, rtol=0, atol=1e-14)
    del valid
    gains = {n: cfg["label_gains"][n] for n in BASES}
    baseline = {f: cfg[f] for f in FAMILIES}
    best = baseline.copy()
    base_ids = [candidate_id(baseline[f]) for f in FAMILIES]
    cache = {base_ids[0]: old.equal, base_ids[1]: old.xgb_equal}
    base_daily = validation_loss(keys, old.blend_equal, gains, cfg["topk"])
    base_loss = base_daily.loss.mean()
    expected = pd.read_csv(ref / "validation_losses.csv").set_index("strategy").at["blend_equal", "loss"]
    np.testing.assert_allclose(base_loss, expected, rtol=0, atol=1e-12)
    blocks = scoring_inputs(cfg)
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
            trials.append(dict(stage=stage, lightgbm=li, xgboost=xi, loss=loss, days=len(daily)))
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
    source, ref = Path(cfg["feature_source"]), Path(cfg["pair_reference"])
    template = json.loads((out / "config_template.json").read_text())
    for family, section in (("lightgbm", "lightgbm_four_equal"), ("xgboost", "single_xgboost")):
        template[section]["params"] = {k: v for k, v in best[family].items() if k != "seed"}
    template["four_model_targets"] = {n: cfg["label_gains"][n] for n in BASES}
    template["seed"] = cfg["seed"]
    write_best_config(out, template)
    parameter_rows = []
    for family in FAMILIES:
        all_params = {candidate_id(cfg[family]): cfg[family]}
        for path in (out / family).glob("*/params.json"):
            all_params[path.parent.name] = json.loads(path.read_text())
        for key, p in all_params.items():
            parameter_rows.append(dict(family=family, candidate=key, selected=p == best[family], **p))
    pd.DataFrame(parameter_rows).to_csv(out / "candidate_parameters.csv", index=False)
    features = alpha158_columns(source)
    scored = []
    for split in ("validation", "test"):
        frame = read_period(source, cfg["segments"][split])
        matrix = frame[features]
        s = frame[["trade_date", "ticker", "grade", "label", "label_end_date", "evaluable"]].copy()
        base = aligned_reference(ref, s, split, ["equal", "xgb_equal", "blend_equal"])
        s["baseline"] = base.blend_equal
        for family, prefix, ref_col in (("lightgbm", "", "equal"), ("xgboost", "xgb_", "xgb_equal")):
            p = best[family]
            if p == cfg[family]:
                s[ref_col] = base[ref_col]
                continue
            folder = out / family / candidate_id(p)
            for name in BASES:
                if family == "lightgbm":
                    model = lgb.Booster(model_file=str(folder / name / "lambdarank.txt"))
                    s[prefix + name] = model.predict(matrix, num_threads=p["num_threads"])
                else:
                    model = xgb.Booster()
                    model.load_model(folder / name / "model.ubj")
                    s[prefix + name] = model.inplace_predict(matrix)
            s[ref_col] = equal_ranks(s, [prefix + name for name in BASES])
        s["blend_equal"] = (s.equal + s.xgb_equal) / 2
        if split == "validation":
            gains = {n: cfg["label_gains"][n] for n in BASES}
            daily = validation_loss(s, s.blend_equal, gains, cfg["topk"])
            archived = pd.read_csv(out / "validation_loss_daily.csv", parse_dates=["trade_date"])
            pd.testing.assert_frame_equal(daily.reset_index(drop=True), archived, atol=1e-12, rtol=1e-12)
            monthly = pd.DataFrame(index=pd.DatetimeIndex(daily.trade_date))
            for name in ("baseline", "equal", "xgb_equal", "blend_equal"):
                result = validation_loss(s, s[name], gains, cfg["topk"]).set_index("trade_date")
                monthly[name] = result.loss
            monthly.groupby(monthly.index.to_period("M")).mean().to_csv(out / "validation_monthly.csv")
        scored.append(s.assign(split=split))
    scores = pd.concat(scored, ignore_index=True)
    scores.to_parquet(out / "predictions.parquet", index=False)
    methods = ("equal", "xgb_equal", "baseline", "blend_equal")
    evaluate_signals(scores, cfg, out, methods)
    from src.report.rank_stack import write_pair_tuning
    print(write_pair_tuning(out), flush=True)


def expanding_years(segments):
    """每年年初定一次模型，训练起点不动，前一自然年专用于早停。"""
    first = pd.Timestamp(segments["train"][0])
    start, end = map(pd.Timestamp, segments["test"])
    if start > end or first.year >= start.year - 1:
        raise ValueError("扩展训练至少需要一个训练年度和一个独立早停年度")
    if "validation" in segments and pd.Timestamp(segments["validation"][1]) >= start:
        raise ValueError("冻结树参数所用的验证集必须结束于首个预测日之前")
    for year in range(start.year, end.year + 1):
        yield year, {
            "train": [str(first.date()), f"{year - 2}-12-31"],
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
        frame = read_period(source, bounds[period])
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
    test = read_feature_split(source / "features.parquet", bounds["test"], "扩展窗口").sort_values(["trade_date", "ticker"])
    scores = test[["trade_date", "ticker", "label", "label_end_date"]].copy()
    for family, prefix in (("lightgbm", "lgb_"), ("xgboost", "xgb_")):
        print(f"{year} 年 {family} 训练四个排序器", flush=True)
        params = cfg[family]
        matrices = quantile_matrices(blocks, params) if family == "xgboost" else []
        preds = train_candidate(folder, family, params, cfg, blocks, matrices, test[names])
        scores[preds.add_prefix(prefix).columns] = preds.to_numpy()
        scores[prefix + "equal"] = equal_ranks(scores, [prefix + n for n in BASES])
    scores["expanding"] = (scores.lgb_equal + scores.xgb_equal) / 2
    return scores.assign(window=year)


def rolling_pair_main(argv=None):
    """按年扩展训练集，用冻结的混合模型参数做连续样本外回测。"""
    parser = argparse.ArgumentParser(description=rolling_pair_main.__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--reference", type=Path, help="已完成联合调参的固定模型归档")
    choice.add_argument("--resume", type=Path, help="从本次扩展窗口归档的已完成模型继续")
    args = parser.parse_args(argv)
    if args.resume:
        out = args.resume.resolve()
        cfg = json.loads((out / "config.json").read_text())
        reference = Path(cfg["fixed_pair_reference"])
    else:
        reference = (args.reference or latest_run("winners.json", "candidate_parameters.csv",
                     "ledger.parquet", "predictions.parquet", rebalance_days=1)).resolve()
        cfg = json.loads((reference / "config.json").read_text())
        cfg.update(json.loads((reference / "winners.json").read_text()), fixed_pair_reference=str(reference),
                   rolling_scheme="expanding_annual", early_stop_years=1)
        if cfg["lightgbm_xgboost_blend"]["xgboost_weight"] != 0.5:
            raise ValueError("本实验要求两类模型各占一半")
        out = new_run(cfg["rebalance_days"], tag="expanding_pair")
        out.mkdir()
        (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    print(out, flush=True)
    names = alpha158_columns(Path(cfg["feature_source"]))
    windows = dict(expanding_years(cfg["segments"]))
    scores = pd.concat([expanding_window(out, cfg, year, bounds, names) for year, bounds in windows.items()],
                       ignore_index=True).assign(split="test")
    # 整段测试期统一核验收益标签，保留跨年但仍在测试期内的收益与信号。
    scores["evaluable"] = segment_masks(scores, {"test": cfg["segments"]["test"]})["test"]
    scores["grade"] = fixed_grades(scores.loc[scores.evaluable]).reindex(scores.index).astype("Int8")
    scores["fixed"] = aligned_reference(reference, scores, "test", ["blend_equal"], check_labels=True).blend_equal
    if not np.isfinite(scores[["expanding", "fixed"]]).all(axis=None):
        raise ValueError("固定或滚动预测含无效分数")
    scores.to_parquet(out / "predictions.parquet", index=False)
    methods = ("expanding", "fixed")
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
    parser.add_argument("--tuning-config", type=Path, help="双模型候选网格，默认 configs/tuning.json")
    parser.add_argument("--reference", type=Path, help="已冻结的五档 LightGBM/XGBoost 混合归档")
    parser.add_argument("--resume", type=Path, help="按该实验配置接续训练与回测")
    args = parser.parse_args(argv)
    if args.resume:
        out = args.resume.resolve()
        cfg = json.loads((out / "config.json").read_text())
    else:
        ref = args.reference or latest_run("top50.ubj", "blend_trials.csv", "validation_losses.csv", rebalance_days=1)
        cfg = json.loads((ref / "config.json").read_text())
        settings = json.loads(args.config.read_text())
        cfg.update(rank_pair_tuning=load_tuning("rank_pair_tuning", settings, args.tuning_config),
                   pair_reference=str(ref.resolve()))
        check_config(cfg)
        out = new_run(cfg["rebalance_days"], tag="pair_tuning")
        out.mkdir()
        (out / "config.json").write_text(json.dumps(cfg, indent=2))
        (out / "config_template.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2))
    print(out, flush=True)
    best = json.loads((out / "winners.json").read_text()) if (out / "winners.json").is_file() else tune(out, cfg)
    evaluate(out, cfg, best)


if __name__ == "__main__":
    main()
