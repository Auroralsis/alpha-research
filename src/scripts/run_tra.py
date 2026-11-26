"""用完整 Alpha158 训练 TRA，与冻结的树模型预测进行五组中证500策略对照。"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import label_argument, set_label_days, fixed_validation, validation_argument
from src.backtest.engine import run_backtests
from src.backtest.portfolio import performance
from src.data.download import write_parquet_batches
from src.data.research_panel import (CSI500_INPUTS, execution_labels, load_market_calendar, parse_dates,
                                     prepare_membership, prepare_panel, prepare_test_panel)
from src.factors.alpha158 import alpha158_batches, alpha158_spec
from src.ml.tra import BATCH, HORIZON, KEYS, LENGTH, MODEL, ROUTER, TRAINING, CausalTRA, Panel, replay
from src.scripts.run_all import RUNS, new_run

REFERENCE = RUNS / "daily_pair_tuning_20261122_100328_550844"
SPLITS = dict(train=["2014-01-02", "2020-12-31"], early=["2021-01-01", "2021-12-31"],
              validation=["2022-01-01", "2022-12-31"], test=["2023-01-01", "2025-12-31"])
ARMS = ("lightgbm", "xgboost", "tra", "tree_pair", "tree_tra")


def history_keys(observed, samples, dates):
    """对每只股票合并各信号日的 60 日窗口，纳入入指前已有的行情。"""
    stocks = pd.Index(samples.ticker.unique())
    obs = observed.loc[observed.ticker.isin(stocks), KEYS]
    end = dates.get_indexer(samples.trade_date)
    col = stocks.get_indexer(samples.ticker)
    changes = np.zeros((len(dates) + 1, len(stocks)), dtype=np.int32)
    np.add.at(changes, (np.maximum(0, end - LENGTH + 1), col), 1)
    np.add.at(changes, (end + 1, col), -1)
    active = changes.cumsum(axis=0, dtype=np.int32)
    return obs.loc[active[dates.get_indexer(obs.trade_date), stocks.get_indexer(obs.ticker)] > 0]


def prepare_history(cfg, dates, out):
    bars = pd.read_parquet(CSI500_INPUTS["prices"])
    bars = bars.loc[parse_dates(bars.trade_date).le(dates[-1]).to_numpy()]
    panel = prepare_panel(bars, dates)
    observed = panel.daily.loc[panel.daily.observation_status.eq("observed"), KEYS]
    members = prepare_membership(pd.read_parquet(CSI500_INPUTS["membership"]))
    members = members.loc[members.trade_date.between(SPLITS["train"][0], dates[-1]),
                          [*KEYS, "source_snapshot_date"]]
    samples = members.merge(observed, on=KEYS, validate="one_to_one")
    execution_labels(panel, samples, 1, cfg["label_end_days"]).to_parquet(out / "samples.parquet", index=False)
    keys = history_keys(observed, samples, dates)
    batches = alpha158_batches(panel, keys, cfg, cfg["feature_workers"])
    write_parquet_batches(batches, out / "history.parquet")


def load_panel(out, calendar, through, scaler=None, reference=None, validation_tree=None, horizon=HORIZON):
    filters = [("trade_date", "<=", pd.Timestamp(through))]
    history = pd.read_parquet(out / "history.parquet", filters=filters)
    samples = pd.read_parquet(out / "samples.parquet", filters=filters)
    if reference is not None:
        # 先固定对照股票池，再计算用于路由误差记忆的每日标签排名。
        tree_keys = pd.read_parquet(reference / "predictions.parquet", columns=KEYS)
        start = SPLITS["validation"][0]
        if validation_tree is not None:
            test_keys = pd.read_parquet(reference / "predictions.parquet", columns=KEYS, filters=[("split", "==", "test")])
            tree_keys = pd.concat([validation_tree[KEYS], test_keys], ignore_index=True)
            start = validation_tree.trade_date.min()
        past = samples.trade_date.lt(start)
        matched = tree_keys.merge(samples.loc[~past], on=KEYS, how="left", validate="one_to_one")
        if matched.source_snapshot_date.isna().any():
            raise ValueError("A frozen tree stock-day has no historical membership/observation")
        pd.DataFrame([dict(reference_rows=len(matched), observed_member_rows=int((~past).sum()),
                           excluded_extra_rows=int((~past).sum()) - len(matched))]).to_csv(
            out / "universe_audit.csv", index=False)
        samples = pd.concat([samples.loc[past], matched], ignore_index=True)
    # 即使归档已有标签，也要等收益终点收盘后才能使用。
    samples.loc[samples.label_end_date.gt(through), "label"] = np.nan
    names = [entry["name"] for entry in alpha158_spec()]
    return Panel(history, samples, names, calendar, SPLITS["train"], scaler, horizon)


def fusion(tree, predictions):
    frame = tree.merge(predictions, on=KEYS, how="left", validate="one_to_one")
    cols = ["equal", "xgb_equal", "tra_raw"]
    if not np.isfinite(frame[[*cols, "blend_equal"]]).all(axis=None):
        raise ValueError("Incomplete frozen tree/TRA scores on the reference universe")
    frame[list(ARMS[:3])] = frame.groupby("trade_date")[cols].rank(pct=True)
    frame["tree_pair"] = frame.blend_equal  # 直接使用原八排序器归档分数，保留原有的并列关系。
    frame["tree_tra"] = frame[list(ARMS[:3])].mean(axis=1)
    return frame


def diagnostics(frame, topk):
    pairs, ics = [], []
    for day, block in frame.groupby("trade_date", sort=True):
        scores = block.set_index("ticker")[list(ARMS)].sort_index()
        corr = scores.corr(method="spearman")
        picks = {arm: set(scores[arm].nlargest(topk).index) for arm in ARMS}
        for left, right in combinations(ARMS, 2):
            pairs.append(dict(trade_date=day, left=left, right=right, spearman=corr.loc[left, right],
                              top50_overlap=len(picks[left] & picks[right]) / min(topk, len(scores))))
        labeled = block.loc[block.evaluable & np.isfinite(block.label)]
        rank_ic = labeled[list(ARMS)].rank().corrwith(labeled.label.rank())
        ics.append(dict(trade_date=day, **rank_ic))
    return pd.DataFrame(pairs), pd.DataFrame(ics).set_index("trade_date")


def frozen_validation(reference, cfg):
    """补算冻结对照在新增验证日期的预测，整段标签按同一个收益截止日判断。"""
    from src.ml.dataset import alpha158_columns
    from src.ml.rank_validation import BASES, equal_ranks, read_period
    from src.ml.tuning import candidate_id
    from src.ml.tuning import frozen_scores
    from src.scripts.train_lightgbm import frozen_lightgbm
    import xgboost as xgb

    source = Path(cfg["feature_source"])
    frame = read_period(source, cfg["segments"]["validation"])
    columns = ["equal", "xgb_equal", "blend_equal"]
    archived = pd.read_parquet(reference / "predictions.parquet", columns=[*KEYS, *columns],
                               filters=[("split", "==", "validation")])
    tree = frame[[*KEYS, "label", "label_end_date", "evaluable"]].join(archived.set_index(KEYS), on=KEYS)
    missing = tree.equal.isna()
    if missing.any():
        rows = frame.loc[missing]
        matrix = rows[alpha158_columns(source)]
        winners = json.loads((reference / "winners.json").read_text())
        for family, column in (("lightgbm", "equal"), ("xgboost", "xgb_equal")):
            params = winners[family]
            folder = reference / family / candidate_id(params) / BASES[0]
            if folder.is_dir():
                scores = frozen_scores(reference, family, params, matrix)
            elif params != cfg[family]:
                raise ValueError(f"归档缺少所选 {family} 模型 {folder}")
            elif family == "lightgbm":
                scores = frozen_lightgbm(Path(cfg["reference"]), rows, cfg)
            else:
                scores = pd.DataFrame({target: xgb.Booster(model_file=Path(cfg["pair_reference"]) /
                                                           f"{target}.ubj").inplace_predict(matrix)
                                       for target in BASES}, index=rows.index)
            tree.loc[missing, column] = equal_ranks(scores.assign(trade_date=rows.trade_date), list(BASES))
        tree.loc[missing, "blend_equal"] = tree.loc[missing, ["equal", "xgb_equal"]].mean(axis=1)
    return tree


def compare(reference, cfg, predictions, calendar, out, validation_tree=None):
    metrics, agreement = [], []
    for split in ("validation", "test"):
        folder = out / split
        folder.mkdir()
        cols = [*KEYS, "label", "label_end_date", "evaluable", "equal", "xgb_equal", "blend_equal"]
        tree = validation_tree if split == "validation" and validation_tree is not None else pd.read_parquet(
            reference / "predictions.parquet", columns=cols, filters=[("split", "==", split)])
        start, end = cfg["segments"][split]
        pred = predictions.loc[predictions.trade_date.between(start, end)]
        if cfg.get("label_days") or cfg["label_end_days"] != 2:
            labels = pd.read_parquet(out / "samples.parquet", columns=[*KEYS, "label", "label_end_date"],
                                     filters=[("trade_date", ">=", pd.Timestamp(start)), ("trade_date", "<=", pd.Timestamp(end))])
            tree = tree.drop(columns=["label", "label_end_date", "evaluable"]).merge(labels, on=KEYS, how="left", validate="one_to_one")
            tree["evaluable"] = np.isfinite(tree.label)
        frame = fusion(tree, pred)
        frame["evaluable"] &= frame.label_end_date.le(end)
        frame.to_parquet(folder / "predictions.parquet", index=False)
        pairs, ic = diagnostics(frame, cfg["topk"])
        pairs.to_csv(folder / "daily_agreement.csv", index=False)
        ic.to_csv(folder / "daily_rank_ic.csv")
        panel, dates = prepare_test_panel(cfg, calendar, split=split)
        inputs = CSI500_INPUTS.copy()
        if split == "validation":
            inputs["benchmark"] = inputs["benchmark"].with_name(f"benchmark_{dates[0]:%Y%m%d}_{dates[-1]:%Y%m%d}.csv")
        run_backtests(inputs, cfg, dates, frame, panel, folder, strategies=ARMS)
        book = pd.read_parquet(folder / "ledger.parquet")
        periods = {start[:4] if start[:4] == end[:4] else f"{start[:4]}-{end[:4]}": dates} if split == "validation" else {
            "2023-2025": dates, **{str(y): dates[dates.year == y] for y in dates.year.unique()}}
        for period, sessions in periods.items():
            daily = pairs.loc[pairs.trade_date.isin(sessions)]
            agreement.append(daily.groupby(["left", "right"])[["spearman", "top50_overlap"]].mean()
                             .reset_index().assign(period=period))
            for arm, g in book.loc[book.trade_date.isin(sessions)].groupby("strategy"):
                g = g.sort_values("trade_date")
                stats = performance(g.assign(nav=(1 + g.daily_return).cumprod()), g.set_index("trade_date").benchmark_return)
                daily_ic = ic.reindex(sessions)[arm]
                mean_ic, std_ic = daily_ic.mean(), daily_ic.std()
                metrics.append(dict(period=period, strategy=arm, rank_ic=mean_ic,
                                    rank_icir=mean_ic / std_ic if std_ic > 0 else np.nan, **stats))
    pd.concat(agreement, ignore_index=True).to_csv(out / "agreement.csv", index=False)
    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=REFERENCE, help="含固定验证期和测试期八模型分数的归档")
    parser.add_argument("--out", type=Path, help="新建实验目录")
    label_argument(parser)
    validation_argument(parser)
    args = parser.parse_args(argv)
    tree_config = json.loads((args.reference / "config.json").read_text())
    cfg = dict(tree_config, segments=tree_config["segments"].copy())
    set_label_days(cfg, args.label_days)
    horizon = cfg["label_end_days"]
    expected = dict(topk=50, n_drop=5, rebalance_days=1, buy_price="close", label_start_days=1)
    if any(cfg[k] != v for k, v in expected.items()) or any(cfg["segments"][s] != SPLITS[s] for s in ("train", "test")):
        raise ValueError("Archived tree protocol differs from the fixed TRA comparison")
    fixed_validation(cfg, args.validation_mode)
    early_bounds = cfg.get("early_stop_bounds", SPLITS["early"])
    out = args.out or new_run(1, tag="tra_alpha158")
    out.mkdir(parents=True, exist_ok=False)
    calendar = load_market_calendar(CSI500_INPUTS["calendar"])
    calendar = calendar[calendar <= SPLITS["test"][1]]
    spec = alpha158_spec()
    manifest = dict(reference=str(args.reference.resolve()), splits=dict(cfg["segments"], early=early_bounds), tree_config=tree_config, label_days=horizon - 1, model=MODEL,
                    router=ROUTER, training=TRAINING, sequence=LENGTH, horizon=horizon, batch_size=BATCH,
                    features=spec, scaler_fit=SPLITS["train"], checkpoint_metric="早停期每日路由 RankIC",
                    memory=f"按个股在 T+{horizon} 收盘时更新专家预测误差")
    (out / "config.json").write_text(json.dumps(manifest, indent=2))
    print(f"TRA experiment: {out}", flush=True)
    prepare_history(cfg, calendar, out)
    fit = load_panel(out, calendar, early_bounds[1], horizon=horizon)
    scaler = fit.scaler
    scaler.to_csv(out / "scaler.csv")
    train = fit.segment(SPLITS["train"], supervised=True)
    early = fit.segment(early_bounds, supervised=True)
    audit = {}
    for name, seg in (("train", train), ("early", early)):
        s = fit.samples.iloc[seg.rows]
        audit[name] = dict(rows=len(s), sessions=len(seg.daily), first_signal=str(s.trade_date.min().date()),
                           last_signal=str(s.trade_date.max().date()), last_label_end=str(s.label_end_date.max().date()))
    (out / "fit_audit.json").write_text(json.dumps(audit, indent=2))
    model = CausalTRA()

    # 这里开始模型训练，train训练，early早停
    model.fit_windows(train, early).to_csv(out / "epochs.csv", index=False)
    torch.save(dict(model=model.model.state_dict(), tra=model.tra.state_dict()), out / "model.pt")
    del train, early, fit
    # 用冻结模型按时间顺序重建误差记忆，跨年兑现的标签也在到期日结算。
    validation_tree = frozen_validation(args.reference, cfg) if cfg.get("validation_mode") else None
    full = load_panel(out, calendar, SPLITS["test"][1], scaler, reference=args.reference, validation_tree=validation_tree, horizon=horizon)
    segment = full.segment([SPLITS["train"][0], SPLITS["test"][1]], supervised=False)
    # 输出预测结果
    predictions = replay(model, segment, output_start=pd.Timestamp(cfg["segments"]["validation"][0]))
    del segment, full
    compare(args.reference, cfg, predictions, calendar, out, validation_tree)


if __name__ == "__main__":
    main()
