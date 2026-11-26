"""在固定 MSE 基线的同一份特征缓存上比较 LambdaRank 与原模型。"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.backtest.engine import evaluate_signals
from src.config import label_argument, set_label_days, fixed_validation, load_config, load_tuning, validation_argument
from src.data.research_panel import segment_masks
from src.ml.rank_validation import bottom_members, holding_bottom_rates
from src.ml.dataset import alpha158_columns, fitting_targets, read_feature_split
from src.report.research import write_bottom_comparison, write_lambdarank
from src.scripts.run_all import new_run
from src.scripts.train_lightgbm import fit_ranker

MODELS = ("mse", "lambdarank")


def relevance(frame: pd.DataFrame, k: int, *, graded: bool = False) -> pd.Series:
    """按日收益排名赋分，并列同分。分档版保留头部名次，并将其余名次分为 k 档。"""
    rank = frame.groupby("trade_date")["label"].rank(method="min", ascending=False)
    if graded:
        size = frame.groupby("trade_date")["label"].transform("count")
        tail = np.floor(k * (size - rank) / (size - k).clip(lower=1))
        return (2 * k - rank).where(rank.le(k), tail).astype("int32")
    return (k + 1 - rank).clip(lower=0).astype("int32")


def quantile_relevance(frame: pd.DataFrame, grades: int) -> pd.Series:
    """按日将收益的截面排名等分为指定档数，并列收益使用平均名次。"""
    labels = frame.groupby("trade_date")["label"]
    position = (labels.rank(method="average") - 0.5) / labels.transform("count")
    return np.floor(grades * position).astype("int32")


def bottom_relevance(frame: pd.DataFrame, k: int) -> pd.Series:
    """后 k 名赋零分，其余股票按当日平均名次划入 1–9 档。输入仅含有效收益。"""
    above = ~bottom_members(frame, k)
    labels = pd.Series(0, index=frame.index, dtype="int32")
    labels.loc[above] = 1 + quantile_relevance(frame.loc[above], 9)
    return labels


def ranking_diagnostics(scored: pd.DataFrame, k: int, models: tuple[str, ...] = MODELS) -> pd.DataFrame:
    """在同一批有标签股票上，比较前 k 名的折损累计增益、命中比例和平均远期收益。"""
    rows = []
    for (split, day), group in scored.loc[scored.evaluable].groupby(["split", "trade_date"]):
        frame = group.copy()
        frame["gain"] = relevance(frame, k)
        frame["bottom"] = bottom_members(frame, k)
        n = min(k, len(frame))
        discount = 1 / np.log2(np.arange(2, n + 2))
        ideal = frame.gain.nlargest(n).to_numpy() @ discount
        for name in models:
            top = frame.sort_values([name, "ticker"], ascending=[False, True]).head(n)
            rows.append(dict(split=split, trade_date=day, strategy=name, n_scored=len(frame),
                             ndcg=(top.gain.to_numpy() @ discount) / ideal,
                             hit_rate=top.gain.gt(0).mean(), bottom_rate=top.bottom.mean(),
                             top_return=top.label.mean()))
    return pd.DataFrame(rows)


def compare_gains(source, out, cfg, names, params, candidates):
    """在每折用独立早停期拟合，按选参年的尾部误选率和收益比较增益。"""
    from src.ml.walk_forward import folds

    rows = []
    for fold in folds(cfg):
        frames = {}
        for split, bounds in fold.items():
            raw = read_feature_split(source / "features.parquet", bounds, split, cfg)
            mask = segment_masks(raw, {split: bounds})[split]
            frame = raw.loc[mask].sort_values(["trade_date", "ticker"])
            if split != "validation":
                frame = frame.loc[fitting_targets(frame, cfg["training_label"]).index]
            frames[split] = frame
        valid = frames["validation"]
        scored = valid[["trade_date", "ticker", "label"]].assign(evaluable=True, split="validation")
        for name, gains in {**candidates, "quintile": list(range(5))}.items():
            mats = []
            for split in ("train", "early_stop"):
                frame = frames[split]
                labels = quantile_relevance(frame, 5) if name == "quintile" else bottom_relevance(frame, cfg["topk"])
                mats.append(lgb.Dataset(frame[names], label=labels, group=frame.groupby("trade_date", sort=False).size()))
            folder = out / "folds" / fold["validation"][0] / name
            folder.mkdir(parents=True)
            model = fit_ranker(mats, dict(params, label_gain=gains), cfg, folder)
            scored[name] = model.predict(valid[names], num_threads=params["num_threads"])
        metrics = ranking_diagnostics(scored, cfg["topk"], (*candidates, "quintile"))
        rows.append(metrics.groupby("strategy")[["bottom_rate", "top_return", "hit_rate"]].mean().reset_index().assign(
            train_end=fold["train"][1], early_stop_end=fold["early_stop"][1], valid_start=fold["validation"][0]))
    pd.concat(rows).to_csv(out / "fold_trials.csv", index=False)


def run(source: Path, previous: Path | None = None, *, target: str | None = None,
        tuning_config: Path | None = None, validation_mode=None, label_days=None) -> Path:
    """读取原基线配置、模型和特征，按交易日分组训练，冻结后再生成测试预测与回测。"""
    cfg = load_config(source / "best_config.json")
    set_label_days(cfg, label_days)
    fixed_validation(cfg, validation_mode)
    if target is None:
        target = "graded" if previous else "topk"
    controls = {}
    parent = previous
    expected = {"topk": (), "graded": ("topk",), "quintile": ("graded", "topk"),
                "decile": ("quintile", "graded", "topk"), "bottom": ("decile", "quintile"),
                "quintile_gap3": ("quintile",)}[target]
    for mode in expected:
        if parent is None:
            raise ValueError(f"缺少 {mode} 对照实验")
        meta = json.loads((parent / "source.json").read_text())
        if Path(meta["baseline"]).resolve() != source.resolve() or meta["target"] != f"daily_{mode}_relevance":
            raise ValueError(f"对照实验必须是同一基线的 {mode} 模型")
        controls["lambdarank" if mode == "topk" else mode] = parent
        parent = Path(meta["previous"]) if meta.get("previous") else None
    tag = "lambdarank" if target == "topk" else f"lambdarank_{target}"
    out = new_run(cfg["rebalance_days"], tag=tag)
    out.mkdir(parents=True)
    names = alpha158_columns(source)
    k = cfg["topk"]
    gains = {"topk": k + 1, "graded": 2 * k, "quintile": 5, "decile": 10, "bottom": 10, "quintile_gap3": 5}
    params = dict(cfg["lightgbm"], objective="lambdarank", metric="ndcg", eval_at=[k],
                  lambdarank_truncation_level=k + 3, label_gain=list(range(gains[target])), seed=cfg["seed"])
    if target == "quintile_gap3":
        params["label_gain"] = [0, 3, 4, 5, 6]
    candidates = {target: params["label_gain"]}
    if target == "bottom":
        gaps = load_tuning("lambdarank_bottom_gaps", cfg, tuning_config)
        if not isinstance(gaps, list) or not gaps or any(type(gap) is not int or gap < 1 for gap in gaps):
            raise ValueError("lambdarank_bottom_gaps 需要非空的正整数列表")
        candidates = {f"gap_{gap}": [0, *range(gap, gap + 9)] for gap in gaps}
        if not cfg.get("validation_mode"):
            cfg["walk_forward"] = load_config()["walk_forward"]
        compare_gains(source, out, cfg, names, params, candidates)
        (out / "gain_candidates.json").write_text(json.dumps(candidates, indent=2) + "\n")
    (out / "config.json").write_text(json.dumps(dict(cfg, lightgbm=params), indent=2) + "\n")
    (out / "source.json").write_text(json.dumps(dict(baseline=str(source.resolve()), target=f"daily_{target}_relevance",
                                                     previous=str(previous.resolve()) if previous else None,
                                                     cutoff=k), indent=2) + "\n")
    print(f"实验目录 {out}", flush=True)

    def read(split):
        return read_feature_split(source / "features.parquet", cfg["segments"][split], split, cfg)

    datasets = []
    for split in ("train", "validation"):
        bounds = cfg.get("early_stop_bounds", cfg["segments"][split]) if split == "validation" else cfg["segments"][split]
        raw = read_feature_split(source / "features.parquet", bounds, split, cfg)
        eligible = raw.loc[segment_masks(raw, {split: bounds})[split]]
        index = fitting_targets(eligible, cfg["training_label"]).index
        frame = eligible.loc[index].sort_values(["trade_date", "ticker"])
        groups = frame.groupby("trade_date", sort=False).size()
        if target == "bottom":
            labels = bottom_relevance(frame, k)
        elif target in ("quintile", "quintile_gap3", "decile"):
            labels = quantile_relevance(frame, gains[target])
        else:
            labels = relevance(frame, k, graded=target == "graded")
        datasets.append(lgb.Dataset(frame[names], label=labels, group=groups.to_numpy(),
                                    reference=datasets[0] if datasets else None))
        del raw, frame, eligible

    models = {"mse": lgb.Booster(model_file=str(source / "lightgbm.txt"))}
    for name, directory in reversed(controls.items()):
        models[name] = lgb.Booster(model_file=str(directory / "lambdarank.txt"))
    for name, gain in candidates.items():
        directory = out / name if target == "bottom" else out
        directory.mkdir(exist_ok=True)
        ranker = fit_ranker(datasets, dict(params, label_gain=gain), cfg, directory)
        models["lambdarank" if target == "topk" else name] = ranker
    del datasets
    pieces = []
    for split in ("train", "validation", "test"):
        raw = read(split)
        scores = raw[["trade_date", "ticker", "label"]].copy()
        scores["split"] = split
        scores["evaluable"] = segment_masks(raw, cfg["segments"])[split]
        for name, model in models.items():
            scores[name] = model.predict(raw[names], num_threads=params["num_threads"])
        pieces.append(scores)
        del raw
    scores = pd.concat(pieces, ignore_index=True)
    daily = ranking_diagnostics(scores, k, tuple(models))
    daily.to_csv(out / "topk_daily.csv", index=False)
    scored = scores.loc[scores.split.ne("train")]
    baseline_tag = json.loads((source / "baseline_check.json").read_text())["best_validation_trial"]
    keys = ["split", "trade_date", "ticker"]
    archived = pd.read_parquet(source / "predictions.parquet", columns=keys + [baseline_tag]).set_index(keys)
    replay = scored.set_index(keys)["mse"].sort_index()
    if validation_mode:
        archived = archived.reindex(replay.index).dropna()
        replay = replay.loc[archived.index]
    pd.testing.assert_index_equal(replay.index, archived.sort_index().index)
    np.testing.assert_allclose(replay, archived.sort_index()[baseline_tag], rtol=0, atol=1e-12)
    if previous is not None:
        columns = list(controls)
        original = pd.read_parquet(previous / "predictions.parquet", columns=keys + columns)
        actual = scored.set_index(keys)[columns].sort_index()
        if validation_mode:
            original = original.set_index(keys).reindex(actual.index).dropna().reset_index()
            actual = actual.loc[original.set_index(keys).index]
        pd.testing.assert_frame_equal(actual,
                                      original.set_index(keys)[columns].sort_index(), atol=1e-12, rtol=0)
    scored.to_parquet(out / "predictions.parquet", index=False)
    test_dates = evaluate_signals(scored, cfg, out, tuple(models))
    portfolio = pd.read_csv(out / "portfolio_metrics.csv").set_index("strategy")
    old = pd.read_csv(source / "portfolio_metrics.csv").set_index("strategy")
    np.testing.assert_allclose(portfolio.at["mse", "total_return"], old.at[baseline_tag, "total_return"], atol=1e-10)
    if previous is not None:
        old = pd.read_csv(previous / "portfolio_metrics.csv").set_index("strategy")
        np.testing.assert_allclose(portfolio.loc[list(controls), "total_return"],
                                   old.loc[list(controls), "total_return"], atol=1e-10)
    if target in ("bottom", "quintile_gap3"):
        held = pd.read_parquet(out / "holdings.parquet")
        holding_bottom_rates(held, scored.loc[scored.split.eq("test")], test_dates, k).to_csv(
            out / "holding_bottom_daily.csv", index=False)
    if target == "bottom":
        return write_bottom_comparison(out)
    return write_lambdarank(out, k, ranker.best_iteration)


def main(argv=None):
    parser = argparse.ArgumentParser(description='在固定 MSE 基线的同一份特征缓存上比较 LambdaRank 与原模型。')
    parser.add_argument("--source", required=True, type=Path, help="含已选 MSE 模型和同样本特征的采样实验目录")
    compare = parser.add_mutually_exclusive_group()
    compare.add_argument("--compare-graded", type=Path, metavar="原版排序实验目录",
                        help="使用全截面分档标签，并与指定的原版排序模型和 MSE 基线对照")
    compare.add_argument("--compare-quintile", type=Path, metavar="全截面分档实验目录",
                         help="训练五档标签，与该目录及其来源模型对照")
    compare.add_argument("--compare-decile", type=Path, metavar="五档实验目录",
                         help="训练十档标签，与五档及其来源模型对照")
    compare.add_argument("--compare-bottom", type=Path, metavar="十档实验目录", help="比较后 k 名独立成档的候选增益")
    compare.add_argument("--compare-quintile-gap3", type=Path, metavar="五档实验目录", help="五等分标签使用增益 0、3、4、5、6")
    parser.add_argument("--tuning-config", type=Path, help="配合 --compare-bottom 指定增益候选，默认 configs/tuning.json")
    label_argument(parser)
    validation_argument(parser)
    args = parser.parse_args(argv)
    if args.tuning_config and not args.compare_bottom:
        parser.error("--tuning-config 需要同时指定 --compare-bottom")
    from functools import partial
    launch = partial(run, validation_mode=args.validation_mode, label_days=args.label_days)
    if args.compare_quintile_gap3:
        report = launch(args.source, args.compare_quintile_gap3, target="quintile_gap3")
    elif args.compare_bottom:
        report = launch(args.source, args.compare_bottom, target="bottom", tuning_config=args.tuning_config)
    elif args.compare_decile:
        report = launch(args.source, args.compare_decile, target="decile")
    elif args.compare_quintile:
        report = launch(args.source, args.compare_quintile, target="quintile")
    else:
        report = launch(args.source, args.compare_graded)
    print(report)


if __name__ == "__main__":
    main()
