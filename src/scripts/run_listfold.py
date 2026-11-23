"""在相同训练日和回测规则下，运行 ListFold 或四目标 LambdaRank 实验。"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.engine import evaluate_signals
from src.config import load_config
from src.data.research_panel import ROOT
from src.ml.dataset import alpha158_columns
from src.ml.listfold import fit_mlp, fit_tree
from src.ml.rank_validation import BASES, aligned_reference, equal_ranks, read_period, selection_stats, training_periods, validation_loss
from src.report.charts import plot_rank_feature_paths
from src.scripts.run_all import RUNS, latest_run, new_run


def run(model):
    """确定对照归档，再共用训练样本、验证和测试预测及账本校验。"""
    settings = load_config()
    ranking = model == "mlp-ndcg"
    neural = model in ("mlp", "mlp-ndcg")
    if neural:
        import torch

        device = torch.device(settings["listfold_mlp"].get("device", "cuda"))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("当前 PyTorch 无法使用 CUDA，请检查 quant 环境和显卡驱动，或将 listfold_mlp.device 设为 cpu。")
        device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        print(f"MLP 训练设备 {device}（{device_name}）", flush=True)
        tree = latest_run("listfold.txt", "ledger.parquet", "predictions.parquet", rebalance_days=1)
        cfg = json.loads((tree / "config.json").read_text())
        baseline = Path(cfg["listfold_reference"])
        cfg.update(listfold_mlp=settings["listfold_mlp"], tree_reference=str(tree), seed=settings["seed"],
                   torch_version=str(torch.__version__), optimizer="Adam", activations="ReLU including output",
                   preprocessing="train median imputation, train min-max scaling and clipping",
                   objective="ListFold-exp", device=str(device), device_name=device_name)
        titles = {"mlp":"MLP ListFold", "listfold":"LightGBM ListFold", "rank_equal":"Four-ranker average"}
        references = {"listfold":(tree,"listfold"), "rank_equal":(tree,"rank_equal")}
        if ranking:
            pair = latest_run("candidate_parameters.csv", "winners.json", "predictions.parquet", rebalance_days=1)
            previous = latest_run("mlp.pt", "predictions.parquet", "ledger.parquet", rebalance_days=1)
            params = settings["lightgbm_four_equal"]["params"]
            cfg.update(objective="lambdarank", label_gains={k: settings["four_model_targets"][k] for k in BASES},
                       ndcg_at=params["eval_at"][0], truncation=params["lambdarank_truncation_level"],
                       lambdarank_norm=True, sigmoid=1., pair_reference=str(pair), mlp_reference=str(previous))
    else:
        import lightgbm as lgb

        record = json.loads((RUNS / "baselines/nav_comparison_daily/baseline.json").read_text())
        baseline = ROOT / record["rank_run"]
        cfg = json.loads((baseline / "config.json").read_text())
        params = dict(settings["lightgbm_four_equal"]["params"], seed=settings["seed"])
        for key in ("objective", "metric", "eval_at", "lambdarank_truncation_level"):
            params.pop(key, None)
        cfg.update(num_boost_round=settings["num_boost_round"], early_stopping_rounds=settings["early_stopping_rounds"],
                   lightgbm=dict(params, objective="ListFold-exp", metric="listfold_nll_per_pair"),
                   listfold_reference=str(baseline), lightgbm_version=lgb.__version__,
                   listfold_spec="完整截面，真实收益降序、同收益按股票代码；奇数时中间股票留在分母直至结束")
        titles = {"listfold":"ListFold-exp", "rank_equal":"Four-ranker average"}
        references = {"rank_equal": (baseline, "equal")}
    source = Path(cfg["feature_source"])
    names = alpha158_columns(source)
    out = new_run(cfg["rebalance_days"], tag="mlp_lambdarank" if ranking else f"{model}_listfold")
    out.mkdir()
    print(out, flush=True)
    if neural:
        cfg["architecture"] = [len(names), *cfg["listfold_mlp"]["hidden_sizes"], 1]
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    periods = training_periods(source, baseline, cfg["segments"])
    fit = fit_mlp if neural else fit_tree
    predict = fit(periods, names, cfg, out)
    if ranking:
        evaluate_rank_mlps(out, predict)
        return
    parts = []
    for split in ("validation", "test"):
        frame = read_period(source, cfg["segments"][split])
        if neural:
            keys = ["trade_date", "ticker"]
            frame = frame.set_index(keys, verify_integrity=True).sort_index()
            scores = pd.read_parquet(tree / "predictions.parquet", filters=[("split", "==", split)])
            scores = scores.set_index(keys, verify_integrity=True).sort_index()
            pd.testing.assert_frame_equal(frame[["label", "evaluable"]], scores[["label", "evaluable"]])
            scores = scores.reset_index()
            name = "mlp"
        else:
            scores = frame[["trade_date", "ticker", "label", "grade", "evaluable"]].copy()
            scores["rank_equal"] = aligned_reference(baseline, frame, split, ["equal"]).equal
            # 命中率沿用按代码打破并列的口径，真实前、后50均恰好50只。
            groups = scores.loc[scores.evaluable].groupby("trade_date", sort=False).label
            rank = groups.rank(method="first", ascending=False)
            low = groups.transform("size") + 1 - rank
            scores.loc[rank.index, "grade"] = np.select([rank.le(50), rank.le(100), low.le(50), low.le(100)],
                                                        [4, 3, 0, 1], default=2).astype("int8")
            name = "listfold"
        scores[name] = predict(frame[names])
        assert np.isfinite(scores[name]).all()
        parts.append(scores.assign(split=split))
    evaluate_scores(pd.concat(parts, ignore_index=True), cfg, out, titles, references)


def evaluate_rank_mlps(out, predict=None):
    """使用已冻结的四份权重复算验证、测试指标和回测。"""
    cfg = json.loads((out / "config.json").read_text())
    pair, previous = Path(cfg["pair_reference"]), Path(cfg["mlp_reference"])
    source = Path(cfg["feature_source"])
    names = alpha158_columns(source)
    if predict is None:
        import torch
        from src.ml.listfold import mlp_predictor, network

        models = {}
        for target in BASES:
            checkpoint = torch.load(out / target / "mlp.pt", map_location=cfg["device"], weights_only=True)
            model = network(checkpoint["widths"]).to(cfg["device"])
            model.load_state_dict(checkpoint["state_dict"])
            models[target] = model.eval()
        predict = mlp_predictor(models, names, cfg, out)
    titles = {"mlp_equal": "MLP LambdaRank equal", "mlp": "MLP ListFold",
              "lgb_equal": "LightGBM equal", "blend_equal": "LightGBM/XGBoost equal"}
    references = {"mlp": (previous, "mlp"), "lgb_equal": (pair, "equal"), "blend_equal": (pair, "blend_equal")}
    parts, diagnostics = [], []
    for split in ("validation", "test"):
        frame = read_period(source, cfg["segments"][split])
        scores, losses = score_rank_mlps(frame, names, predict, split, cfg, titles)
        parts.append(scores)
        losses.to_csv(out / f"{split}_ndcg_loss.csv", index=False)
        diagnostics.append(losses.assign(split=split))
    summary = pd.concat(diagnostics).groupby(["split", "strategy"]).loss.agg(["mean", "count"])
    summary.rename(columns={"mean": "one_minus_ndcg", "count": "days"}, inplace=True)
    summary["ndcg"] = 1 - summary.one_minus_ndcg
    summary.to_csv(out / "ndcg_summary.csv")
    evaluate_scores(pd.concat(parts, ignore_index=True), cfg, out, titles, references, extra=BASES)


def score_rank_mlps(frame, names, predict, split, cfg, titles):
    """核对外部归档，按训练分档计算 NDCG，按精确名次计算前后 50/100 命中率。"""
    pair, previous = Path(cfg["pair_reference"]), Path(cfg["mlp_reference"])
    base = aligned_reference(pair, frame, split, ["equal", "blend_equal"], check_labels=True)
    predictions = predict(frame[names])
    if not np.isfinite(predictions.to_numpy()).all():
        raise ValueError("MLP 预测缺分")
    scores = frame[["trade_date", "ticker", "grade", "evaluable", "label"]].join(predictions)
    scores["mlp_equal"] = equal_ranks(scores, list(BASES))
    scores[["lgb_equal", "blend_equal"]] = base[["equal", "blend_equal"]]
    old = pd.read_parquet(previous / "predictions.parquet", filters=[("split", "==", split)],
                          columns=["trade_date", "ticker", "mlp", "grade", "evaluable", "label"])
    keys = ["trade_date", "ticker"]
    scores = scores.set_index(keys, verify_integrity=True).sort_index()
    old = old.set_index(keys, verify_integrity=True).sort_index()
    pd.testing.assert_frame_equal(scores[["evaluable", "label"]], old[["evaluable", "label"]])
    scores["mlp"] = old.mlp
    scores = scores.reset_index()
    # NDCG 与树模型共用并列同档的训练标签，命中率沿用 ListFold 精确 50/100 名口径。
    diagnostics = []
    for name in (*titles, *BASES):
        gains = {name: cfg["label_gains"][name]} if name in BASES else cfg["label_gains"]
        diagnostics.append(validation_loss(scores, scores[name], gains, cfg["ndcg_at"]).assign(strategy=name))
    scores["grade"] = old.grade.to_numpy()
    return scores.assign(split=split), pd.concat(diagnostics, ignore_index=True)


def evaluate_scores(pred, cfg, out, titles, references, *, extra=()):
    """用同一批股票日计算命中率、IC 和回测，并逐日核对复用的基线。"""
    pred.to_parquet(out / "predictions.parquet", index=False)
    methods = (*titles, *extra)
    daily = selection_stats(pred, methods)
    daily.to_csv(out / "hits_daily.csv", index=False)
    daily.groupby(["split","strategy"])[["hit50","hit100","bottom50","bottom100"]].mean().to_csv(out / "hits_summary.csv")
    evaluate_signals(pred, cfg, out, methods)
    ledger = pd.read_parquet(out / "ledger.parquet")
    for name, (folder, archived_name) in references.items():
        old = pd.read_parquet(folder / "ledger.parquet", filters=[("strategy","==",archived_name)])
        cols = ["daily_return", "cost_cny", "nav"]
        pd.testing.assert_frame_equal(ledger.loc[ledger.strategy.eq(name)].set_index("trade_date")[cols],
                                      old.set_index("trade_date")[cols], atol=1e-10, rtol=1e-10)
    plot_rank_feature_paths(out, ledger, titles, "nav_excess.png")
    print((out / "portfolio_metrics.csv").read_text(), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("lightgbm", "mlp", "mlp-ndcg"), default="lightgbm")
    run(parser.parse_args(argv).model)


def mlp_main(argv=None):
    """旧 MLP 入口默认选择 ListFold 神经网络。"""
    import sys

    main(["--model", "mlp", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    main()
