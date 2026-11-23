"""逐年生成 CatRank，比较八树基线、九票等权和第 159 列堆叠。"""
import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.backtest.engine import run_backtests
from src.backtest.factor_metrics import daily_score_metrics
from src.backtest.portfolio import QLIB_DAILY_ANNUALIZATION, load_benchmark_returns, qlib_excess_metrics
from src.config import CONFIG_FILE
from src.data.research_panel import (ROOT, CSI500_INPUTS as INPUTS, load_market_calendar, prepare_membership,
                                     prepare_panel, prepare_test_panel, segment_masks, stock_day_index)
from src.ml.dataset import alpha158_columns, build_features
from src.ml.walk_forward import folds
from src.ml.rank_validation import BASES, equal_ranks, read_period, validation_loss
from src.report.research import markdown_table
from src.scripts.factor_extensions import BASE, KEY
from src.scripts.feature_selection import frozen_scores
from src.scripts.run_all import new_run
from src.scripts.run_pair_tuning import FAMILIES, quantile_matrices, train_candidate

START = "2013-09-01"
METHODS = ("A", "B", "C")


def bootstrap(out, cfg, dates):
    """用同期成员快照和可观测行情重建启动期，给年末信号保留两天标签行情。"""
    folder = out/"bootstrap"
    folder.mkdir(exist_ok=True)
    sessions = dates[(dates >= START) & (dates < "2016")]
    working = dates[(dates >= START) & (dates <= dates[dates.get_loc(sessions[-1])+2])]
    raw = pd.read_parquet(INPUTS["prices"], filters=[("trade_date","<=",working[-1].strftime("%Y%m%d"))])
    panel = prepare_panel(raw, working)
    members = prepare_membership(pd.read_parquet(INPUTS["membership"]))
    members = members.loc[members.trade_date.isin(sessions)]
    if not (folder/"data_audit.json").exists():
        build_features(panel, members, cfg, folder, cfg["feature_workers"])
    names = alpha158_columns(folder)
    frame = pd.read_parquet(folder/"features.parquet")
    days = members.groupby("trade_date").size().reindex(sessions,fill_value=0).rename("members").to_frame()
    eligible = members.merge(panel.daily.loc[panel.daily.observation_status.eq("observed"), KEY],
                             on=KEY, validate="one_to_one")
    days["observed"] = eligible.groupby("trade_date").size()
    expected = eligible.loc[eligible.trade_date.ge(working[cfg["feature_warmup_days"]]),KEY]
    pd.testing.assert_index_equal(stock_day_index(expected).sort_values(), stock_day_index(frame).sort_values())
    days["alpha158_rows"] = frame.groupby("trade_date").size()
    days["all_158_valid"] = frame[names].notna().all(axis=1).groupby(frame.trade_date).sum()
    days["realized_by_2015"] = segment_masks(frame, {"boot":[START,"2015-12-31"]})["boot"].groupby(frame.trade_date).sum()
    days.fillna(0).astype(int).rename_axis("trade_date").to_csv(folder/"coverage_daily.csv")
    print(f"启动期 {sessions[0].date()} 至 {sessions[-1].date()}，首个特征日 {frame.trade_date.min().date()}", flush=True)
    return frame


def validation_benchmark(out, bounds, dates):
    """补齐验证期指数日线，核对与项目已有行情重叠的点位和每个验证交易日。"""
    source = INPUTS["benchmark"]
    original = pd.read_csv(source, dtype={"trade_date":str})
    needed = dates[(dates >= bounds[0]) & (dates <= bounds[1])]
    if set(needed.strftime("%Y%m%d")) <= set(original.trade_date):
        return source
    path = out/"validation_benchmark.csv"
    cache = source.with_name(f"benchmark_{needed[0]:%Y%m%d}_{needed[-1]:%Y%m%d}.csv")
    saved = next((p for p in (path, cache) if p.is_file()), None)
    if saved:
        data = pd.read_csv(saved, dtype={"trade_date":str})
    else:
        from src.data.download import tushare_client
        client = tushare_client()
        client._DataApi__http_url = "https://api.waditu.com/dataapi"
        data = client.index_daily(ts_code="000905.SH", start_date=needed[0].strftime("%Y%m%d"),
                                  end_date=needed[-1].strftime("%Y%m%d"), fields="ts_code,trade_date,close,pre_close")
    data = data.set_index("trade_date", verify_integrity=True).sort_index()
    overlap = original.set_index("trade_date").index.intersection(data.index)
    if len(overlap) == 0 or set(data.ts_code) != {"000905.SH"}:
        raise ValueError("验证基准缺少重叠行情或指数代码不符")
    cols = ["close","pre_close"]
    np.testing.assert_allclose(data.loc[overlap,cols], original.set_index("trade_date").loc[overlap,cols], atol=1e-4, rtol=0)
    verified = data.loc[needed.strftime("%Y%m%d")].rename_axis("trade_date")
    if not (np.isfinite(verified[cols]) & verified[cols].gt(0)).all(axis=None):
        raise ValueError("指数点位无效")
    from src.data.download import pending_file
    for target in (path, cache):
        if not target.is_file():
            with pending_file(target) as tmp:
                verified.to_csv(tmp)
    load_benchmark_returns(path, needed)
    return path


def historical_features(out, cfg, dates):
    """重建 2016 至验证期的完整可观测截面，供第一层拟合和日内排名。"""
    folder = out/"historical"
    folder.mkdir(exist_ok=True)
    if not (folder/"data_audit.json").exists():
        end = pd.Timestamp(cfg["segments"]["validation"][1])
        raw = pd.read_parquet(INPUTS["prices"], filters=[("trade_date","<=",end.strftime("%Y%m%d"))])
        panel = prepare_panel(raw, dates[(dates >= START) & (dates <= end)])
        members = prepare_membership(pd.read_parquet(INPUTS["membership"]))
        members = members.loc[members.trade_date.between("2016-01-01",end)]
        build_features(panel, members, cfg, folder, cfg["feature_workers"])
    return folder


def cat_scores(model, frame, features):
    """在打分日全体有效股票中算平均秩，端点缩放到 [0,1]。"""
    scores = frame[KEY].copy()
    scores["prediction"] = model.predict(frame[features], thread_count=model.get_param("thread_count") or 16)
    group = scores.prediction.groupby(scores.trade_date)
    counts = group.transform("count")
    scores["CatRank"] = (group.rank(method="average")-1)/(counts-1).where(counts.gt(1))
    # 项目八张原始选票均为 rank(pct=True)。第九票换成相同刻度。
    scores["vote"] = group.rank(method="average", pct=True)
    if not np.isfinite(scores[["prediction","CatRank","vote"]]).all(axis=None):
        raise ValueError("CatRank 截面人数不足或模型预测缺失")
    return scores


def fit_first_layer(out, cfg, dates, names):
    """逐年扩展拟合，模型和兑现边界分别留档，2020 年底冻结最后一个模型。"""
    folder = historical_features(out, cfg, dates)
    boot = bootstrap(out, cfg, dates)
    parts = [boot, pd.read_parquet(folder/"features.parquet",
             filters=[("trade_date","<=",pd.Timestamp(cfg["segments"]["train"][1]))])]
    history = pd.concat(parts, ignore_index=True).sort_values(KEY)
    del parts, boot
    if history.duplicated(KEY).any() or (history.source_snapshot_date > history.trade_date).any():
        raise ValueError("第一层股票日重复或成员快照越界")
    train_end = pd.Timestamp(cfg["segments"]["train"][1])
    directory = out/"first_layer"
    directory.mkdir(exist_ok=True)
    audit, ranks = [], []
    for year in range(pd.Timestamp(cfg["segments"]["train"][0]).year, train_end.year+2):
        cutoff = pd.Timestamp(year, 1, 1)-pd.Timedelta(days=1)
        fitted = history.loc[segment_masks(history, {"fit":[START,cutoff]})["fit"]]
        if not (fitted.label_end_date.gt(fitted.trade_date) & fitted.label_end_date.le(cutoff)).all():
            raise ValueError("第一层使用了尚未兑现的标签")
        target = fitted.groupby("trade_date").label.rank(pct=True,method="average")
        model = CatBoostRegressor(**cfg["catrank_regressor"], random_seed=cfg["seed"],
                                  allow_writing_files=False, verbose=False)
        file = directory/f"through_{cutoff.year}.cbm"
        if file.exists():
            model.load_model(str(file))
        else:
            model.fit(fitted[names], target)
            tmp = file.with_suffix(".tmp")
            model.save_model(str(tmp))
            tmp.replace(file)
        audit.append(dict(fit_start=fitted.trade_date.min(), last_signal=fitted.trade_date.max(),
                          last_label=fitted.label_end_date.max(), cutoff=cutoff, train_rows=len(fitted),
                          train_days=fitted.trade_date.nunique(), predict_from=year, model=file.name))
        del fitted, target
        if year <= train_end.year:
            predict = history.loc[history.trade_date.dt.year.eq(year)]
            ranks.append(cat_scores(model, predict, names).assign(model_cutoff=cutoff))
        print(f"第一层拟合至 {cutoff.date()}，股票日 {audit[-1]['train_rows']:,}", flush=True)
    del history
    for split in ("early_stop","validation"):
        lo, hi = map(pd.Timestamp,cfg["segments"][split])
        frame = pd.read_parquet(folder/"features.parquet", columns=[*KEY,*names],
                                filters=[("trade_date",">=",lo),("trade_date","<=",hi)])
        ranks.append(cat_scores(model,frame,names).assign(model_cutoff=train_end))
    result = pd.concat(ranks,ignore_index=True)
    if result.duplicated(KEY).any() or not result.model_cutoff.lt(result.trade_date).all():
        raise ValueError("CatRank 模型训练边界与打分日期交叉")
    result.to_parquet(directory/"historical_scores.parquet", index=False)
    pd.DataFrame(audit).to_csv(directory/"fits.csv",index=False)
    return result, model


def merge_scores(frame, rankings):
    """所有原实验股票日均须有 CatRank，保留未来标签缺失的打分行。"""
    combined = frame.join(rankings.set_index(KEY, verify_integrity=True)[["CatRank","vote"]],on=KEY,validate="one_to_one")
    if combined.CatRank.isna().any():
        raise ValueError("原实验股票日缺 CatRank")
    return combined


def layer_two(out, cfg, names, train, early, frame, *, fit):
    """训练两套八排序器，并将基线的八票直接用于九票组。"""
    scores = frame[[*KEY,"label","label_end_date","evaluable","grade","CatRank","vote"]].copy()
    for method, features in [("A",names),("C",[*names,"CatRank"])]:
        blocks = []
        if fit:
            for rows in (train,early):
                blocks.append((rows[features], rows.grade.to_numpy(dtype=int),
                               rows.groupby("trade_date",sort=False).size().to_numpy()))
        families = []
        for family in FAMILIES:
            params = cfg[family]
            if fit:
                print(f"第二层 {method} 组 {family}，{len(features)} 列", flush=True)
                mats = quantile_matrices(blocks,params) if family=="xgboost" else []
                raw = train_candidate(out/method,family,params,cfg,blocks,mats,frame[features])
                del mats
            else:
                raw = frozen_scores(out/method,family,params,frame[features])
            if not np.isfinite(raw).all(axis=None):
                raise ValueError("八排序器存在无效分数")
            families.append(equal_ranks(raw.assign(trade_date=frame.trade_date),list(BASES)))
            raw = raw.add_prefix(f"{method}_{family}_")
            scores[raw.columns] = raw
        scores[method] = (families[0]+families[1])/2
    scores["vote"] = scores.groupby("trade_date").CatRank.rank(method="average", pct=True)
    scores["B"] = (8*scores.A + scores.vote)/9
    return scores


def select_method(out, cfg, names, historical):
    """每折重训第二层，早停和选参使用同一个截至该折训练末日的第一层模型。"""
    source = Path(cfg["feature_source"])
    rows = []
    gains = {name: cfg["label_gains"][name] for name in BASES}
    for fold in folds(cfg):
        folder = out / "folds" / fold["validation"][0]
        folder.mkdir(parents=True, exist_ok=True)
        cutoff = pd.Timestamp(fold["train"][1]).year
        first = CatBoostRegressor().load_model(str(out / "first_layer" / f"through_{cutoff}.cbm"))
        periods = {}
        for split, bounds in fold.items():
            frame = read_period(source, bounds)
            if split == "train":
                scored = merge_scores(frame, historical)
            else:
                full = pd.read_parquet(out / "historical/features.parquet", columns=[*KEY, *names],
                                       filters=[("trade_date", ">=", pd.Timestamp(bounds[0])),
                                                ("trade_date", "<=", pd.Timestamp(bounds[1]))])
                scored = merge_scores(frame, cat_scores(first, full, names))
            periods[split] = scored if split == "validation" else scored.loc[scored.evaluable]
        scores = layer_two(folder, cfg, names, periods["train"], periods["early_stop"], periods["validation"], fit=True)
        for method in METHODS:
            loss = validation_loss(scores, scores[method], gains, cfg["topk"]).loss.mean()
            rows.append(dict(strategy=method, train_end=fold["train"][1], early_stop_end=fold["early_stop"][1],
                             valid_start=fold["validation"][0], loss=loss))
    table = pd.DataFrame(rows)
    table.to_csv(out / "fold_trials.csv", index=False)
    return table.groupby("strategy").loss.mean()


def evaluate_period(out, cfg, split, scores, dates, benchmark):
    """用共同股票日计算日 RankIC、先锁名单的前50原始收益和连续扣费账本。"""
    folder = out/split
    folder.mkdir(exist_ok=True)
    scores.to_parquet(folder/"predictions.parquet",index=False)
    idx = stock_day_index(scores)
    target = scores.label.where(scores.evaluable)
    diagnostic = []
    for method in METHODS:
        ic = daily_score_metrics(scores[method].set_axis(idx),target.set_axis(idx))
        picks = scores.assign(realized=target).sort_values(["trade_date",method,"ticker"],ascending=[True,False,True])
        top = picks.groupby("trade_date").head(cfg["topk"]).groupby("trade_date").realized
        summary = top.agg(top_return="mean",top_labeled="count",selected="size").reset_index()
        diagnostic.append(ic.merge(summary,on="trade_date",validate="one_to_one").assign(strategy=method))
    metrics = pd.concat(diagnostic,ignore_index=True)
    metrics.to_csv(folder/"signal_daily.csv",index=False)
    period_cfg = {**cfg,"segments":{**cfg["segments"],"test":cfg["segments"][split]}}
    panel, sessions = prepare_test_panel(period_cfg,dates)
    if not sessions.equals(pd.DatetimeIndex(sorted(scores.trade_date.unique()))):
        raise ValueError("打分交易日与回测日历不一致")
    run_backtests({**INPUTS,"benchmark":benchmark}, period_cfg,sessions,scores,panel,folder,strategies=METHODS)
    ledger = pd.read_parquet(folder/"ledger.parquet")
    joined = ledger.merge(metrics,on=["trade_date","strategy"],validate="one_to_one")
    if len(joined) != len(sessions)*len(METHODS):
        raise ValueError("策略账本与诊断交易日不一致")
    rows = []
    for method, data in joined.groupby("strategy"):
        years = list(data.groupby(data.trade_date.dt.year))
        if split == "test":
            years.append(("全期",data))
        for period, group in years:
            rows.append(dict(split=split,period=period,strategy=method,days=len(group),
                             signal_days=group.rank_ic.count(),rank_ic=group.rank_ic.mean(),
                             top_return=group.top_return.mean(),top_coverage=group.top_labeled.sum()/group.selected.sum(),
                             turnover=group.turnover_two_sided.mean()*QLIB_DAILY_ANNUALIZATION/2,
                             **qlib_excess_metrics(group.daily_return,group.benchmark_return)))
    return pd.DataFrame(rows)


def write_report(out, summary, validation, choice):
    """汇总同股票日的三组结果，比较堆叠和直接投票的增量。"""
    boot = pd.read_csv(out/"bootstrap"/"coverage_daily.csv",parse_dates=["trade_date"]).set_index("trade_date")
    by_year = boot.groupby(boot.index.year).sum()
    params = json.loads((out/"config.json").read_text())["catrank_regressor"]
    fits = pd.read_csv(out/"first_layer/fits.csv")
    first_members = pd.read_parquet(INPUTS["membership"], columns=["source_snapshot_date"],
                                   filters=[("trade_date","==",boot.index[0])])
    first_snapshot = pd.Timestamp(first_members.source_snapshot_date.max()).date()
    coverage = markdown_table(["年度","成员股票日","有行情","有 Alpha158 行","158 列均有效","截至 2015 年末标签已兑现"],
                              [(y,*row) for y,row in by_year.iterrows()])
    table = markdown_table(["区间","组别","日均 RankIC","前50平均标签收益（bp/日）","扣费年化超额","年化单边换手"],
            [(f"{r.period} {'验证' if r.split=='validation' else '测试'}", r.strategy,f"{r.rank_ic:.4f}",
              f"{r.top_return*10000:.2f}",f"{r.annualized_excess_return:.2%}",f"{r.turnover:.2f} 倍")
             for r in summary.itertuples()])
    full = summary.loc[summary.period.eq("全期")].set_index("strategy")
    differences = []
    for left,right in [("B","A"),("C","A"),("C","B")]:
        d = full.loc[left,["rank_ic","top_return","annualized_excess_return","turnover"]] - full.loc[right,["rank_ic","top_return","annualized_excess_return","turnover"]]
        differences.append(f"{left} 相比 {right}，日均 RankIC {d.rank_ic:+.4f}，前 50 平均收益 {10000*d.top_return:+.2f} bp/日，扣费年化超额 {100*d.annualized_excess_return:+.2f} 个百分点。")
    lines = ["# 中证500 CatRank 实验","",
        "A 用 Alpha158 训练四个 LightGBM 和四个 XGBoost 排序器。B 给 A 的八张百分位选票加上 CatBoost 的一票，每票占 1/9。C 把 CatRank 作为第 159 列重训同配置的八个排序器，仍各占 1/8。三组第二层均从 2014 年开始，用相同的模型输入股票日、2021 年早停和相同的交易规则。",
        "", "## 数据与时间边界","",
        f"日历、行情及历史成员从 {boot.index[0].date()} 开始，首日对应 {first_snapshot} 的权重快照。成员按当日或之前最近的月度快照确定，是近似历史成分。沿用 60 个交易日的预热要求，首个 Alpha158 日期为 {fits.fit_start.iloc[0]}。部分特征缺失时保留样本。第一层使用完整可观测成员截面的 MAD 截尾 Alpha158。第二层复用正式八树实验的特征存档与股票日。",
        "", "表 1　启动期覆盖","",coverage,"",
        f"第一层 CatBoostRegressor 用 RMSE 拟合日内已兑现收益百分位（平均秩除以有效数）。实验预设 {params['iterations']} 轮，深度 {params['depth']}，学习率 {params['learning_rate']}，L2 为 {params['l2_leaf_reg']}。首版用了 {fits.train_rows.iloc[0]:,} 个股票日，冻结版用了 {fits.train_rows.iloc[-1]:,} 个。CatRank 按预测分数的日内平均秩计算 (秩−1)/(有效数−1)。2014–2020 年逐年用截至上年末已经兑现的标签拟合。2020 年底冻结的 CatBoost 用于 2021 年早停、2022 年验证及全部测试年。三组第二层用 2014–2020 年 T+1 至 T+2 复权收盘收益产生的五档标签训练。",
        "", "## 验证与测试","",
        "预先按五个选参年的等权四目标日均 1−NDCG@50 选方案，越低越好。"+
        "，".join(f"{n} 为 {v:.6f}" for n,v in validation.items())+f"。据此锁定 **{choice}**，再统一评价 2023–2025 年。该历史测试期在项目中已用于多轮研究。",
        "", "表 2　三组同日期对照","",table,"",
        "前 50 在全部有分股票中选定，同分按代码升序，只对原名单中有有效原始收益的股票取均值，再按信号日等权。RankIC 使用 Qlib 日截面 Spearman。两个回测区间各用一亿元起步，测试区间跨年连续持仓。每日 Top50、计划换 5 只，次日收盘成交，买 5 bp、卖 15 bp。扣费日收益减中证500指数日收益，按 Qlib 的 238 日口径年化。年化单边换手为日均双边换手 × 238 / 2。",
        "", *differences, "",
        "[逐年原始数值](comparison.csv) · [验证选型记录](validation_lock.json) · [第一层各折兑现边界](first_layer/fits.csv) · [启动期逐日覆盖](bootstrap/coverage_daily.csv) · [配置](config.json)"]
    text = "\n".join(lines)+"\n"
    (out/"report.md").write_text(text)
    docs = ROOT/"docs/results"
    archive = Path(os.path.relpath(out, docs)).as_posix()
    lines[-1] = lines[-1].replace("](", f"]({archive}/")
    (docs/"catrank_daily.md").write_text("\n".join(lines)+"\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path)
    args = parser.parse_args(argv)
    out = (args.out or new_run(1,tag="catrank")).resolve()
    out.mkdir(parents=True,exist_ok=True)
    cfg = json.loads((BASE/"config.json").read_text())
    cfg.update(json.loads((BASE/"winners.json").read_text()))
    settings = json.loads(CONFIG_FILE.read_text())
    cfg.update(catrank_regressor=settings["catrank_regressor"], walk_forward=settings["walk_forward"])
    early_year = cfg["final_early_stop_year"]
    cfg["segments"].update(train=settings["segments"]["train"],
                           early_stop=[f"{early_year}-01-01",f"{early_year}-12-31"])
    cfg["segments"] = {name: cfg["segments"][name] for name in ("train","early_stop","validation","test")}
    cfg["catrank_start"] = START
    source = Path(cfg["feature_source"])
    names = alpha158_columns(source)
    config_file = out/"config.json"
    if config_file.exists() and json.loads(config_file.read_text()) != cfg:
        raise ValueError("当前配置与归档不符，请使用新目录")
    config_file.write_text(json.dumps(cfg,indent=2,ensure_ascii=False))
    dates = load_market_calendar(INPUTS["calendar"])
    benchmark = validation_benchmark(out,cfg["segments"]["validation"],dates)
    ranking, model = fit_first_layer(out,cfg,dates,names)
    losses = select_method(out, cfg, names, ranking)
    periods = {}
    for split in ("train","early_stop","validation"):
        frame = read_period(source,cfg["segments"][split])
        if split != "validation":
            frame = frame.loc[frame.evaluable]
        periods[split] = merge_scores(frame,ranking)
    del frame, ranking
    scored = layer_two(out,cfg,names,periods["train"],periods["early_stop"],periods["validation"],fit=True)
    del periods
    gc.collect()
    gains = {name:cfg["label_gains"][name] for name in BASES}
    daily_loss = pd.concat([validation_loss(scored,scored[m],gains,cfg["topk"]).assign(strategy=m) for m in METHODS])
    daily_loss.to_csv(out/"validation_loss_daily.csv",index=False)
    choice = losses.idxmin()
    validation = evaluate_period(out,cfg,"validation",scored,dates,benchmark)
    validation.to_csv(out/"validation_summary.csv",index=False)
    (out/"validation_lock.json").write_text(json.dumps(dict(selection="min_mean_four_target_1_minus_ndcg_at_50",
               loss=losses.to_dict(),winner=choice,test=cfg["segments"]["test"]),indent=2))
    print(f"验证完成，锁定 {choice}，各组损失 {losses.to_dict()}",flush=True)
    del scored
    test = read_period(source,cfg["segments"]["test"])
    panel, test_dates = prepare_test_panel(cfg,dates)
    members = prepare_membership(pd.read_parquet(INPUTS["membership"]))
    observed = panel.daily.loc[panel.daily.observation_status.eq("observed"),KEY]
    eligible = members.loc[members.trade_date.isin(test_dates),KEY].merge(observed,on=KEY,validate="one_to_one")
    pd.testing.assert_index_equal(stock_day_index(test).sort_values(),stock_day_index(eligible).sort_values())
    del panel, members, eligible
    ranks = cat_scores(model,test,names)
    ranks.assign(model_cutoff=cfg["segments"]["train"][1]).to_parquet(out/"first_layer/test_scores.parquet",index=False)
    test = merge_scores(test,ranks)
    scored = layer_two(out,cfg,names,None,None,test,fit=False)
    del test
    testing = evaluate_period(out,cfg,"test",scored,dates,INPUTS["benchmark"])
    summary = pd.concat([validation,testing],ignore_index=True).sort_values(["split","period","strategy"],
                            key=lambda s:s.map({"validation":"0","test":"1"}) if s.name=="split" else s.astype(str))
    summary.to_csv(out/"comparison.csv",index=False)
    write_report(out,summary,losses,choice)
    print(out/"report.md",flush=True)


if __name__ == "__main__":
    main()
