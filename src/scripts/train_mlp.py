"""固定或逐月重训 MLP，按日输出股票分数并回测。"""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from src.backtest.engine import evaluate_signals
from src.config import label_argument, set_label_days, CONFIG_FILE, fixed_validation, load_config, validation_argument
from src.data.research_panel import segment_masks
from src.ml.dataset import alpha158_columns, read_feature_split
from src.ml.mlp import fit_mlp, regression_output
from src.ml.rank_validation import BASES, equal_ranks, fixed_grades, read_period, training_periods, validation_loss
from src.scripts.run_all import RUNS, new_run

DEFAULT_LOSS = "lambdarank"


def predict_score(predict, frame, names, cfg):
    scores = predict(frame[names])
    if isinstance(scores, pd.DataFrame):
        return equal_ranks(scores.assign(trade_date=frame.trade_date), list(BASES))
    return regression_output(scores, frame.trade_date, cfg["mlp"].get("regression_output", "raw"))


def mlp_windows(test_bounds, *, train_months, stop_months, test_months):
    """每个预测窗口前留出连续的训练和早停区间，首尾按研究测试日期裁剪。"""
    first, last = map(pd.Timestamp, test_bounds)
    start = first.to_period("M").start_time
    day = pd.Timedelta(days=1)
    while start <= last:
        stop = start - pd.DateOffset(months=stop_months)
        train = stop - pd.DateOffset(months=train_months)
        following = start + pd.DateOffset(months=test_months)
        yield {"train": [train, stop - day], "early_stop": [stop, start - day],
               "test": [max(start, first), min(following - day, last)]}
        start = following


def run_walk_forward_mlp(source, names, cfg, out):
    """使用相同的首窗网络作为冻结对照，逐窗预测后只做一次连续账本回测。"""
    path = source / "features.parquet"
    scored, audit = [], []
    fixed = None
    for bounds in mlp_windows(cfg["segments"]["test"], **cfg["refit_window"]):
        window = str(bounds["test"][0].date())
        folder = out / window
        folder.mkdir()
        print(f"{window} 重新拟合 MLP", flush=True)

        def sections():
            for split in ("train", "early_stop"):
                rows = read_feature_split(path, bounds[split], split, cfg)
                rows = rows.loc[segment_masks(rows, {split: bounds[split]})[split]].sort_values(["trade_date", "ticker"])
                if cfg["objective"] == "lambdarank":
                    rows["grade"] = fixed_grades(rows)
                sizes = rows.groupby("trade_date", sort=False).size().to_numpy()
                audit.append(dict(window=window, period=split, start=bounds[split][0], end=bounds[split][1],
                                  days=len(sizes), stock_days=len(rows), last_label_end=rows.label_end_date.max()))
                yield split, rows, sizes

        predict = fit_mlp(sections(), names, cfg, folder)
        test = read_feature_split(path, bounds["test"], "test", cfg).sort_values(["trade_date", "ticker"])
        scores = test[["trade_date", "ticker", "label", "label_end_date"]].copy()
        scores["rolling_mlp"] = predict_score(predict, test, names, cfg)
        if fixed is None:
            fixed = predict
            scores["fixed_mlp"] = scores.rolling_mlp
        else:
            scores["fixed_mlp"] = predict_score(fixed, test, names, cfg)
        scored.append(scores.assign(window=window, split="test"))
        audit.append(dict(window=window, period="test", start=bounds["test"][0], end=bounds["test"][1],
                          days=test.trade_date.nunique(), stock_days=len(test)))
        pd.DataFrame(audit).to_csv(out / "periods.csv", index=False)
    pred = pd.concat(scored, ignore_index=True)
    pred["evaluable"] = segment_masks(pred, {"test": cfg["segments"]["test"]})["test"]
    pred["grade"] = fixed_grades(pred.loc[pred.evaluable]).reindex(pred.index).astype("Int8")
    methods = ("rolling_mlp", "fixed_mlp")
    gains = {name: cfg["label_gains"][name] for name in BASES}
    losses = [validation_loss(pred, pred[name], gains, cfg["topk"]).assign(strategy=name) for name in methods]
    pd.concat(losses, ignore_index=True).to_csv(out / "test_ndcg_loss.csv", index=False)
    pred.to_parquet(out / "predictions.parquet", index=False)
    evaluate_signals(pred, cfg, out, methods)
    daily = pd.read_csv(out / "daily_rank_ic.csv", parse_dates=["trade_date"])
    windows = pred[["trade_date", "window"]].drop_duplicates()
    daily = daily.merge(windows, on="trade_date", validate="many_to_one")
    month = daily.trade_date.dt.to_period("M").astype("int64")
    refit = pd.to_datetime(daily.window).dt.to_period("M").astype("int64")
    daily["month_in_window"] = month - refit + 1
    daily.groupby(["strategy", "month_in_window"]).rank_ic.agg(["mean", "count"]).to_csv(
        out / "rank_ic_by_window_month.csv")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss", choices=("lambdarank", "mse"), default=DEFAULT_LOSS)
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    parser.add_argument("--source", type=Path, help="Alpha158 特征缓存目录")
    parser.add_argument("--input-scaling", choices=("none", "minmax", "daily_zscore"), help="训练与预测共同使用的因子缩放方式")
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--walk-forward", type=int, metavar="月数")
    parser.add_argument("--train-months", type=int, default=36)
    parser.add_argument("--stop-months", type=int, default=1)
    parser.add_argument("--out", type=Path)
    label_argument(parser)
    validation_argument(parser)
    args = parser.parse_args(argv)
    if args.walk_forward is not None and min(args.walk_forward, args.train_months, args.stop_months) < 1:
        parser.error("滚动窗口月数必须为正整数")
    cfg = load_config(args.config)
    cfg["mlp_input_scaling"] = args.input_scaling or cfg.get("mlp_input_scaling", cfg["mlp"].get("input_scaling", "minmax"))
    set_label_days(cfg, args.label_days)
    source = args.source or Path(json.loads((RUNS / "baselines/nav_comparison_daily/rank_config.json").read_text())["feature_source"])
    names = alpha158_columns(source)
    device = args.device or cfg["mlp"]["device"]
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("当前环境无法使用 CUDA，可用 --device cpu")
    cfg.update(feature_source=str(source), device=device, objective=args.loss,
               architecture=[len(names), *cfg["mlp"]["hidden_sizes"], 1],
               label_gains={k: cfg["four_model_targets"][k] for k in BASES},
               ndcg_at=cfg["topk"], truncation=cfg["lightgbm_four_equal"]["params"]["lambdarank_truncation_level"])
    if args.walk_forward is None:
        fixed_validation(cfg, args.validation_mode or cfg.get("validation_mode") or "split")
    else:
        cfg["refit_window"] = dict(train_months=args.train_months, stop_months=args.stop_months,
                                   test_months=args.walk_forward)
    out = args.out or new_run(cfg["rebalance_days"], tag="mlp")
    out.mkdir(parents=True, exist_ok=False)
    print(out, flush=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    if args.walk_forward is not None:
        return run_walk_forward_mlp(source, names, cfg, out)
    sections = training_periods(source, None, cfg["segments"], cfg["early_stop_bounds"], cfg)
    predict = fit_mlp(sections, names, cfg, out)
    parts = []
    for split in ("validation", "test"):
        frame = read_period(source, cfg["segments"][split], cfg)
        scores = frame[["trade_date", "ticker", "label", "label_end_date", "grade", "evaluable"]].copy()
        raw = predict(frame[names])
        if isinstance(raw, pd.DataFrame):
            scores = scores.join(raw)
            scores["mlp"] = equal_ranks(scores, list(BASES))
        else:
            scores["mlp"] = regression_output(raw, frame.trade_date, cfg["mlp"].get("regression_output", "raw"))
        validation_loss(scores, scores.mlp, cfg["label_gains"], cfg["topk"]).to_csv(
            out / f"{split}_loss_daily.csv", index=False)
        parts.append(scores.assign(split=split))
    predictions = pd.concat(parts, ignore_index=True)
    predictions.to_parquet(out / "predictions.parquet", index=False)
    evaluate_signals(predictions, cfg, out, ("mlp",))


if __name__ == "__main__":
    main()
