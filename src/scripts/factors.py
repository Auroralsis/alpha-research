"""Alpha101 单因子研究、排序特征对照和 Alpha158 等权组合。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.engine import evaluate_signals
from src.backtest.factor_metrics import evaluate_factors
from src.config import CONFIG_FILE, load_config, load_tuning
from src.data.download import save_parquet, write_parquet_batches
from src.data.research_panel import (
    CSI500_INPUTS, ROOT, execution_labels, load_market_calendar, prepare_membership,
    prepare_panel, segment_masks,
)
from src.factors import factor_batches
from src.factors.alpha101 import HISTORY_DAYS
from src.factors.alpha158 import equal_weight_scores, factor_directions
from src.factors.neutralization import clip_mad_columns
from src.ml.dataset import alpha158_columns, read_feature_split
from src.ml.rank_validation import BASES, aligned_reference, equal_ranks, read_period, validation_loss
from src.report.research import sha256, write_alpha101_report, write_equal_weight
from src.scripts.run_all import FACTOR_RUNS, latest_run, new_run


FACTORS = [f"Alpha{i}" for i in range(1, 102)]
KEYS = ["trade_date", "ticker"]
ENSEMBLES = ["equal", "xgb_equal", "blend_equal"]


def build_raw(out: Path, config: dict):
    """保留完整历史复权锚点，按测试期动态股票池和项目交易日标签写原始样本。"""
    market = load_market_calendar(CSI500_INPUTS["calendar"])
    start, end = config["segments"]["test"]
    dates = market[(market >= start) & (market <= end)]
    if dates.empty:
        raise ValueError("测试期没有交易日")
    working = market[max(0, market.get_loc(dates[0]) - HISTORY_DAYS):market.get_loc(dates[-1]) + 1]
    members = prepare_membership(pd.read_parquet(CSI500_INPUTS["membership"], filters=[
        ("trade_date", ">=", dates[0]), ("trade_date", "<=", dates[-1])]))
    prices = pd.read_parquet(CSI500_INPUTS["prices"], filters=[("ticker", "in", members.ticker.unique().tolist())])
    panel = prepare_panel(prices, working)
    keys = members[["trade_date", "ticker"]].merge(
        panel.daily.loc[panel.daily.observation_status.eq("observed"), ["trade_date", "ticker"]], validate="one_to_one")
    labels = execution_labels(panel, keys, config["label_start_days"], config["label_end_days"])
    labels["label"] = labels.label.where(segment_masks(labels, config["segments"])["test"])
    batches = factor_batches("alpha101", panel, members, dates, config)
    write_parquet_batches(
        (scores.merge(labels, on=["trade_date", "ticker"], validate="one_to_one") for scores in batches),
        out / "alpha101_raw.parquet",
    )


def evaluate(out: Path, config: dict):
    """生成研究指标和带完整因子对照的报告。"""
    frame = pd.read_parquet(out / "alpha101_raw.parquet")
    summary = evaluate_factors(frame, FACTORS, out, config["feature_workers"])
    return write_alpha101_report(summary, frame, out, config)


def alpha101_main(argv=None):
    parser = argparse.ArgumentParser(description='计算中证500的 Alpha101 原值、中性化值和逐日 IC 指标。')
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    out = new_run(tag="daily_alpha101", root=FACTOR_RUNS)
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    build_raw(out, config)
    print(evaluate(out, config))


def build_inputs(out, cfg, original, raw_test):
    """逐年构造相同股票日的特征矩阵；不按因子缺失情况删除样本。"""
    dates = load_market_calendar(CSI500_INPUTS["calendar"])
    members = prepare_membership(pd.read_parquet(CSI500_INPUTS["membership"]))
    prices = pd.read_parquet(CSI500_INPUTS["prices"])
    start = pd.Timestamp(cfg["segments"]["train"][0])
    end = pd.Timestamp(cfg["segments"]["test"][1])
    test_start = pd.Timestamp(cfg["segments"]["test"][0])
    coverage = []

    def batches():
        for year in range(start.year, end.year + 1):
            days = dates[(dates.year == year) & (dates >= start) & (dates <= end)]
            lo, hi = days[0], days[-1]
            filters = [("trade_date", ">=", lo), ("trade_date", "<=", hi)]
            base = pd.read_parquet(original / "features.parquet", filters=filters)
            base = base.set_index(KEYS, verify_integrity=True).sort_index()
            if lo >= test_start:
                cached = pd.read_parquet(raw_test, filters=filters)
                cached = cached.set_index(KEYS, verify_integrity=True).sort_index()
                pd.testing.assert_index_equal(cached.index, base.index)
                pd.testing.assert_frame_equal(cached[["label_end_date", "label"]], base[["label_end_date", "label"]])
                raw = cached[FACTORS]
            else:
                working = dates[max(0, dates.get_loc(lo) - HISTORY_DAYS):dates.get_loc(hi) + 1]
                pool = members.loc[members.trade_date.between(lo, hi)]
                panel = prepare_panel(prices[prices.ticker.isin(pool.ticker.unique())], working)
                raw = pd.concat(factor_batches("alpha101", panel, pool, days, cfg), ignore_index=True)
                raw = raw.set_index(KEYS, verify_integrity=True)
                if not base.index.isin(raw.index).all():
                    raise ValueError(f"{year} 年的原始因子缺少基线股票日")
                raw = raw.reindex(base.index)[FACTORS]
                del panel
            values = raw.to_numpy(dtype=np.float64, copy=True)
            if np.isinf(values).any():
                raise ValueError(f"{year} 年因子含有无穷值")
            for positions in base.groupby(level="trade_date", sort=False).indices.values():
                values[positions] = clip_mad_columns(values[positions])
            counts = np.isfinite(values).sum(axis=0)
            coverage.extend(dict(year=year, factor=name, valid=int(n), rows=len(base)) for name, n in zip(FACTORS, counts))
            print(f"{year} 年 MAD 特征完成，{len(base):,} 个股票日", flush=True)
            clipped = pd.DataFrame(values.astype("float32"), index=base.index, columns=FACTORS)
            yield base.join(clipped).reset_index()
    write_parquet_batches(batches(), out / "features.parquet")
    pd.DataFrame(coverage).to_csv(out / "coverage.csv", index=False)


def fit_arm(out, work, cfg, features, reference):
    """按冻结参数训练八个模型，保存两种算法及各半混合分数。"""
    from src.scripts.run_rank_stack import score, train_fold
    from src.scripts.boosting import train_xgboost

    work.mkdir()
    lg = train_fold(out, cfg, features, cfg["final_early_stop_year"], BASES, work / "final")
    xg = train_xgboost(Path(cfg["reference"]), cfg.copy(), out, features, work)
    results = []
    for split in ("validation", "test"):
        frame = read_period(out, cfg["segments"][split])
        scores = score(frame, features, lg, cfg["lightgbm"]["num_threads"])
        matrix = frame[features]
        for name in BASES:
            scores["xgb_" + name] = xg[name].inplace_predict(matrix)
        del matrix
        baseline = aligned_reference(reference, scores, split, ENSEMBLES)
        scores["equal"] = equal_ranks(scores, list(BASES))
        scores["xgb_equal"] = equal_ranks(scores, ["xgb_" + n for n in BASES])
        w = cfg["lightgbm_xgboost_blend"]["xgboost_weight"]
        scores["blend_equal"] = (1 - w) * scores.equal + w * scores.xgb_equal
        results.append(scores.assign(split=split).join(baseline.add_prefix("base_")))
    return pd.concat(results, ignore_index=True)


def run_ranking(reference: Path, raw_test: Path) -> Path:
    """从已完成的四目标基线读取冻结参数，在新目录构建对照。"""
    cfg = json.loads((reference / "config.json").read_text())
    original = Path(cfg["feature_source"])
    out = new_run(cfg["rebalance_days"], tag="alpha101_ranking")
    out.mkdir()
    cfg.update(comparison_reference=str(reference), raw_alpha101_test=str(raw_test), additional_processing="daily_5mad")
    cfg["reference_xgboost_trial"] = cfg.pop("selected_trial", None)
    cfg["feature_sets"] = {"base": "Alpha158", "alpha101": "Alpha101", "combined": "Alpha158 + Alpha101"}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(out, flush=True)
    build_inputs(out, cfg, original, raw_test)
    return train_and_compare(out)


def train_and_compare(out: Path):
    """按同一特征表逐组训练、验证，最后复用日频 Top-50 回测。"""
    cfg = json.loads((out / "config.json").read_text())
    original = Path(cfg["feature_source"])
    reference = Path(cfg["comparison_reference"])
    metrics = []
    predictions = None
    gains = {n: cfg["label_gains"][n] for n in BASES}
    keys = ["split", *KEYS]
    for arm, features in [("alpha101", FACTORS), ("combined", alpha158_columns(original) + FACTORS)]:
        work = out / arm
        scores = fit_arm(out, work, cfg, features, reference)
        (work / "feature_names.json").write_text(json.dumps(features, indent=2))
        scores.to_parquet(work / "predictions.parquet", index=False)
        wide = scores.set_index(keys, verify_integrity=True)
        if predictions is None:
            base_cols = ["base_" + col for col in ENSEMBLES]
            predictions = wide[["label", "evaluable", "grade", *base_cols]].copy()
        pd.testing.assert_index_equal(predictions.index, wide.index)
        predictions = predictions.join(wide[ENSEMBLES].add_prefix(arm + "_"))
    predictions = predictions.reset_index()
    methods = [f"{arm}_{col}" for arm in ("base", "alpha101", "combined") for col in ENSEMBLES]
    valid = predictions.loc[predictions.split.eq("validation")]
    for name in methods:
        loss = validation_loss(valid, valid[name], gains, cfg["topk"])
        metrics.append(dict(strategy=name, validation_loss=loss.loss.mean(), days=len(loss)))
    pd.DataFrame(metrics).to_csv(out / "validation_loss.csv", index=False)
    evaluate_signals(predictions, cfg, out, methods)
    from src.report.rank_stack import write_feature_comparison
    print(write_feature_comparison(out), flush=True)
    return out


def ranking_main(argv=None):
    parser = argparse.ArgumentParser(description='用同一套四目标排序模型比较 Alpha158、华泰 Alpha101 及二者合并。')
    parser.add_argument("--reference", type=Path, help="已冻结的两模型混合基线")
    parser.add_argument("--raw-test", type=Path, help="日频单因子研究归档的 alpha101_raw.parquet")
    args = parser.parse_args(argv)
    reference = args.reference or latest_run("top50.ubj", "blend_trials.csv", "ledger.parquet", rebalance_days=1)
    raw_test = args.raw_test
    if raw_test is None:
        cached = sorted(FACTOR_RUNS.glob("daily_alpha101_*/alpha101_raw.parquet"))
        if not cached:
            parser.error("缺少 Alpha101 原值，请先运行 python -m src.scripts alpha101")
        raw_test = cached[-1]
    run_ranking(reference.resolve(), raw_test.resolve())


def equal_weight_main(argv=None) -> None:
    parser = argparse.ArgumentParser(description='复用已缓存的中性化 Alpha158，以训练期方向合成等权信号并回测。')
    parser.add_argument("--source", type=Path, help="已有中性化特征及 LightGBM 回测的实验目录")
    parser.add_argument("--ridge", action="store_true", help="加入验证集选出的强正则岭回归")
    parser.add_argument("--tuning-config", type=Path, help="配合 --ridge 指定正则候选，默认 configs/tuning.json")
    args = parser.parse_args(argv)
    if args.tuning_config and not args.ridge:
        parser.error("--tuning-config 需要同时指定 --ridge")
    settings = load_config()
    source = (args.source or latest_run("neutral/features.parquet", "comparison.csv", "manifest.json",
                                        rebalance_days=settings["rebalance_days"])).resolve()
    config = load_config(source / "config.json")
    cache = source / "neutral"
    features = cache / "features.parquet"
    hashes = json.loads((source / "manifest.json").read_text())["sha256"]
    for file in [features, source / "config.json", *[CSI500_INPUTS[k] for k in ("prices", "calendar", "benchmark", "limits")]]:
        if sha256(file) != hashes[file.relative_to(ROOT).as_posix()]:
            raise ValueError(f"回测数据或缓存与来源实验不一致：{file}")
    names = alpha158_columns(cache)
    if args.ridge:
        config.update({key: settings[key] for key in ("ridge_alpha", "ridge_solver", "ridge_num_threads")})
        config["ridge_strong_alphas"] = load_tuning("ridge_strong_alphas", settings, args.tuning_config)
        config["alpha158_processing"] = "neutral"
    out = new_run(config["rebalance_days"], tag="ridge_comparison" if args.ridge else "equal_weight")
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    segments = config["segments"]
    train = read_feature_split(features, segments["train"], "train")
    eligible = segment_masks(train, segments)["train"]
    directions = factor_directions(train.loc[eligible], names)
    del train
    directions.to_csv(out / "directions.csv")
    used = directions.direction.ne(0)
    print(f"训练期定向完成，采用 {used.sum()} 列、{directions.loc[used, 'family'].nunique()} 个家族。", flush=True)
    splits = []
    for split in ("validation", "test"):
        rows = read_feature_split(features, segments[split], split)
        frame = rows[["trade_date", "ticker", "label"]].copy()
        frame["equal_weight"] = equal_weight_scores(rows, directions)
        frame["evaluable"] = segment_masks(rows, segments)[split]
        splits.append(frame.assign(split=split))
    del rows
    scores = pd.concat(splits, ignore_index=True)
    strategies = ("equal_weight",)
    if args.ridge:
        from src.ml.models import compare_ridge

        ridge = compare_ridge(config, cache, out)
        keys = ["trade_date", "ticker"]
        ridge = ridge.set_index(keys).sort_index()
        scores = scores.set_index(keys).sort_index()
        pd.testing.assert_frame_equal(scores[["split", "label", "evaluable"]], ridge[["split", "label", "evaluable"]])
        scores = scores.join(ridge["ridge_strong"]).reset_index()
        strategies += ("ridge_strong",)
    save_parquet(scores, out / "predictions.parquet")
    evaluate_signals(scores, config, out, strategies)
    (out / "source.json").write_text(json.dumps(dict(run=source.relative_to(ROOT).as_posix(),
        features_sha256=hashes[features.relative_to(ROOT).as_posix()], directions_sha256=sha256(out / "directions.csv")),
        indent=2) + "\n")
    print(write_equal_weight(config, source, out, directions), flush=True)
