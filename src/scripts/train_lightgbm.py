"""独立训练 LightGBM，可选择四目标排序或收益回归。"""

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.backtest.engine import evaluate_signals
from src.backtest.factor_metrics import daily_ic
from src.config import label_argument, set_label_days, CONFIG_FILE, fixed_validation, load_config, validation_argument
from src.data.research_panel import stock_day_index
from src.ml.dataset import alpha158_columns
from src.ml.rank_validation import BASES, fuse_rankers, read_period, selection_stats, validation_loss
from src.scripts.run_all import RUNS, new_run
from src.ml.rank_validation import holding_bottom_rates


def fit_ranker(datasets: list[lgb.Dataset], params: dict, cfg: dict, directory: Path) -> lgb.Booster:
    """按独立验证期 NDCG 早停，保存最优模型和逐轮曲线。"""
    history = {}
    ranker = lgb.train(params, datasets[0], num_boost_round=cfg["num_boost_round"],
                       valid_sets=datasets, valid_names=["train", "validation"],
                       callbacks=[lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False),
                                  lgb.record_evaluation(history)])
    metric = f"ndcg@{cfg['topk']}"
    curve = pd.DataFrame({split: values[metric] for split, values in history.items()})
    curve.index = pd.RangeIndex(1, len(curve) + 1, name="iteration")
    curve.to_csv(directory / "training_ndcg.csv")
    ranker.save_model(str(directory / "lambdarank.txt"), num_iteration=ranker.best_iteration)
    print(f"{directory.name} 最佳轮数 {ranker.best_iteration}，验证 {metric}={curve.validation.max():.6f}", flush=True)

    return ranker


def rank_period(source, bounds, extra, cfg=None):
    frame = read_period(source, bounds, cfg)
    if extra is not None:
        frame = frame.join(extra, on=["trade_date", "ticker"], validate="one_to_one")
    return frame


def training_diagnostics(source: Path, features: list[str], models: dict[str, lgb.Booster],
                         periods: dict, folder: Path, threads: int, extra=None, cfg=None) -> None:
    """按有效原始收益复算每折训练、早停两期的日均 Rank IC。"""
    rows = []
    for part, bounds in periods.items():
        frame = rank_period(source, bounds, extra, cfg)
        frame = frame.loc[frame.evaluable]
        matrix = frame[features]
        index = stock_day_index(frame)
        labels = frame.label.set_axis(index)
        for name, model in models.items():
            prediction = pd.Series(model.predict(matrix, num_threads=threads), index=index)
            _, rank = daily_ic(prediction, labels)
            rows.append(dict(period=part, strategy=name, mean_rank_ic=rank.mean(), trading_days=rank.count()))
    pd.DataFrame(rows).to_csv(folder / "fit_rank_ic.csv", index=False)


def train_fold(source: Path, cfg: dict, features: list[str], stop_year: int, names: Iterable[str],
               folder: Path, extra=None) -> dict[str, lgb.Booster]:
    """训练截止早停年之前，早停只用该年。记录实际收益末日和每日各档人数。"""
    folder.mkdir()
    periods = {"train": [cfg["segments"]["train"][0], f"{stop_year - 1}-12-31"],
               "early_stop": [f"{stop_year}-01-01", f"{stop_year}-12-31"]}
    if folder.name == "final" and cfg.get("early_stop_bounds"):
        periods = dict(train=cfg["segments"]["train"], early_stop=cfg["early_stop_bounds"])
    blocks, audit = [], []
    summaries = []
    for part, bounds in periods.items():
        frame = rank_period(source, bounds, extra, cfg)
        frame = frame.loc[frame.evaluable]
        groups = frame.groupby("trade_date", sort=False)
        sizes = groups.size().to_numpy()
        blocks.append((frame[features], frame.grade.to_numpy(dtype=int), sizes))
        counts = groups.grade.value_counts().unstack(fill_value=0).reindex(columns=range(5), fill_value=0)
        audit.append(counts.assign(period=part))
        summaries.append(dict(period=part, start=bounds[0], end=bounds[1], stock_days=len(frame),
                              last_label_end=str(frame.label_end_date.max().date())))
        del frame, groups
    pd.concat(audit).to_csv(folder / "grade_counts.csv")
    (folder / "periods.json").write_text(json.dumps(summaries, indent=2))
    models = {}
    datasets = []
    for name in names:
        directory = folder / name
        directory.mkdir()
        separate = cfg.get("lightgbm_by_target")
        params = dict(separate[name] if separate else cfg["lightgbm"], label_gain=cfg["label_gains"][name])
        if separate or not datasets:
            datasets = []
            for x, y, groups in blocks:
                datasets.append(lgb.Dataset(x, label=y, group=groups, reference=datasets[0] if datasets else None))
        models[name] = fit_ranker(datasets, params, cfg, directory)
    del datasets, blocks
    training_diagnostics(source, features, models, periods, folder, cfg["lightgbm"]["num_threads"], extra, cfg)
    return models


def score(frame: pd.DataFrame, features: list[str], models: dict[str, lgb.Booster], threads: int) -> pd.DataFrame:
    """对完整信号股票池预测，未知真实收益仍保留模型分数。"""
    scores = frame[["trade_date", "ticker", "label", "label_end_date", "grade", "evaluable"]].copy()
    matrix = frame[features]
    for name, model in models.items():
        scores[name] = model.predict(matrix, num_threads=threads)
    return scores


def rank_inputs(scores: pd.DataFrame) -> pd.DataFrame:
    """在全部已打分股票中计算当日百分位；四列的顺序与逻辑回归系数一致。"""
    return scores.groupby("trade_date")[list(BASES)].rank(method="average", pct=True)


def evaluate_stack(scores: pd.DataFrame, cfg: dict, out: Path, methods: tuple[str, ...]) -> None:
    """对给定堆叠分数使用共同的排序诊断、持仓分析和 Qlib 日频回测。"""
    scores.to_parquet(out / "predictions.parquet", index=False)
    stats = selection_stats(scores, methods)
    stats.to_csv(out / "selection_daily.csv", index=False)
    dates = evaluate_signals(scores, cfg, out, methods)
    test = scores.loc[scores.split.eq("test")]
    holdings = pd.read_parquet(out / "holdings.parquet")
    held = pd.concat([holding_bottom_rates(holdings, test, dates, k).assign(cutoff=k) for k in (50, 100)])
    held.to_csv(out / "holding_bottom_daily.csv", index=False)


def frozen_lightgbm(reference, frame, cfg):
    """在指定股票日用已冻结的四个排序器补算对照分数。"""
    raw = frame[["trade_date"]].copy()
    for target in BASES:
        model = lgb.Booster(model_file=str(reference / "final" / target / "lambdarank.txt"))
        raw[target] = model.predict(frame[model.feature_name()], num_threads=cfg["lightgbm"]["num_threads"])
    raw["equal"] = raw.groupby("trade_date")[list(BASES)].rank(pct=True).mean(axis=1)
    return raw[[*BASES, "equal"]]


def equal_main(argv=None):
    """独立训练四个 LightGBM，以每日等权排名验证并回测。"""

    parser = argparse.ArgumentParser(description=equal_main.__doc__)
    parser.add_argument("--config", type=Path,
                        help="四目标排序配置，默认使用原 LightGBM 四等权基线")
    parser.add_argument("--source", type=Path, help="覆盖配置中的特征缓存目录")
    parser.add_argument("--out", type=Path, help="新实验目录")
    parser.add_argument("--nplhr", type=Path, help="含股票日键及 nPLHR 列的因子归档")
    label_argument(parser)
    validation_argument(parser)
    return train_equal(parser.parse_args(argv))


def train_equal(args):
    config = args.config or RUNS / "baselines/nav_comparison_daily/rank_config.json"
    cfg = json.loads(config.read_text())
    if "lightgbm_four_equal" in cfg and "label_gains" not in cfg:
        cfg = load_config(config)
        cfg.update(lightgbm=dict(cfg["lightgbm_four_equal"]["params"], seed=cfg["seed"]),
                   label_gains=cfg["four_model_targets"], final_early_stop_year=2021)
        fixed_validation(cfg, args.validation_mode or cfg.get("validation_mode") or "split")
    if getattr(args, "fusion", None):
        cfg["rank_fusion"] = args.fusion
    set_label_days(cfg, args.label_days)
    source = args.source or cfg.get("feature_source")
    if source is None:
        archive = RUNS / "baselines/nav_comparison_daily/rank_config.json"
        source = json.loads(archive.read_text())["feature_source"]
    source = Path(source).resolve()
    features = alpha158_columns(source)
    extra = None
    factor_archive = args.nplhr or cfg.get("factor_archive")
    if factor_archive:
        extra = pd.read_parquet(factor_archive, columns=["trade_date", "ticker", "nPLHR"]).set_index(
            ["trade_date", "ticker"], verify_integrity=True)
        features.append("nPLHR")
        cfg["factor_archive"] = str(Path(factor_archive).resolve())
    cfg["feature_source"] = str(source)
    cfg["label_gains"] = {name: cfg["label_gains"][name] for name in BASES}
    fixed_validation(cfg, args.validation_mode)
    segments = cfg["segments"]
    stop = cfg["final_early_stop_year"]
    if not cfg.get("early_stop_bounds") and not (segments["train"][0] <= segments["train"][1] == f"{stop-1}-12-31"
            and f"{stop}-12-31" < segments["validation"][0] <= segments["validation"][1] < segments["test"][0]):
        raise ValueError("请按训练、独立早停年、验证、测试顺序设置区间")
    if cfg["lightgbm"]["objective"] != "lambdarank" or (cfg["topk"], cfg["n_drop"]) != (50, 5):
        raise ValueError("四等权实验使用 LambdaRank 和 Top50/Drop5")
    out = args.out or new_run(cfg["rebalance_days"], tag="lightgbm_equal")
    out.mkdir(parents=True, exist_ok=False)
    print(out, flush=True)
    for filename in ("config.json", "best_config.json"):
        (out / filename).write_text(json.dumps(cfg, indent=2))
    models = train_fold(source, cfg, features, stop, BASES, out / "final", extra)
    evaluate_equal(source, cfg, features, extra, models, out)
    (out / "completed").touch()


def evaluate_equal(source, cfg, features, extra, models, out):
    segments = cfg["segments"]
    pieces = []
    for split in ("validation", "test"):
        frame = rank_period(source, segments[split], extra, cfg)
        scores = score(frame, features, models, cfg["lightgbm"]["num_threads"])
        scores["equal"] = fuse_rankers(scores, list(BASES), cfg.get("rank_fusion", "percentile"))
        loss = validation_loss(scores, scores.equal, cfg["label_gains"], cfg["topk"])
        loss.to_csv(out / f"{split}_loss_daily.csv", index=False)
        pieces.append(scores.assign(split=split))
        del frame
    evaluate_stack(pd.concat(pieces, ignore_index=True), cfg, out, ("equal",))


DEFAULT_LOSS = "lambdarank"


def main(argv=None):
    """训练 LightGBM，选择四目标排序或单模型收益回归。"""
    from src.ml.tree_experiment import model_arguments, model_run
    from src.scripts.run_all import lightgbm_main

    parser = argparse.ArgumentParser(description=main.__doc__)
    model_arguments(parser)
    parser.add_argument("--fusion", choices=("percentile", "zscore"), help="四个排序器的日截面等权融合方式")
    parser.add_argument("--loss", choices=("mse", "lambdarank"), default=DEFAULT_LOSS)
    parser.add_argument("--workers", type=int, help="MSE 路线的特征计算进程数")
    parser.add_argument("--nplhr", type=Path, help="固定四排序器的附加因子归档")
    args = parser.parse_args(argv)
    if args.fusion and (args.resume or args.loss == "mse"):
        parser.error("--fusion 适用于新建的四排序器实验，续跑沿用实验归档")
    if args.loss == "mse":
        if args.resume or args.tpe or args.nplhr or args.source or args.tpe_objective or args.tuning_config:
            parser.error("MSE 固定训练从行情准备特征，MSE 网格调参使用 study --stage tune")
        command = ["--config", str(args.config or CONFIG_FILE)]
        for key in ("validation_mode", "label_days", "out", "workers"):
            value = getattr(args, key)
            if value is not None:
                command.extend(["--" + key.replace("_", "-"), str(value)])
        return lightgbm_main(command)
    if args.workers:
        parser.error("--workers 适用于 MSE 特征计算")
    if args.tpe or args.resume:
        if args.nplhr:
            parser.error("Optuna 使用完整 Alpha158")
        return model_run(args, ("lightgbm",))
    if args.tpe_objective or args.tuning_config:
        parser.error("选参选项需要 --tpe")
    return train_equal(args)


if __name__ == "__main__":
    main()
