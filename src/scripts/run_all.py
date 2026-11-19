"""一次完成中证500特征生成、模型训练、回测与报告。"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.research_panel import CSI500_INPUTS as INPUTS
from src.data.research_panel import ROOT, load_market_calendar, prepare_membership, prepare_panel
from src.ml.alpha158 import build_features


RUNS = ROOT / "results/ml_study"


def new_run() -> Path:
    """为一次新实验选择带运行时间的目录。"""
    return RUNS / datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def latest_run(*required: str) -> Path:
    """找最近一个具备当前步骤所需文件的实验目录。"""
    for path in sorted(RUNS.glob("20*"), reverse=True):
        if path.is_dir() and all((path / name).is_file() for name in required):
            return path
    raise FileNotFoundError("没有可继续的实验，请先运行 run_all，或用 --out 指定已有实验目录")


def load_predictions(config: dict, out: Path) -> pd.DataFrame:
    """读取当前批次的预测并确认训练参数匹配。"""
    from src.ml.models import training_settings

    log = json.loads((out / "training_log.json").read_text())
    if any(log.get(key) != value for key, value in training_settings(config).items()):
        raise ValueError("Saved predictions use a different training configuration")
    return pd.read_parquet(out / "predictions.parquet")


def evaluate_predictions(config: dict, config_path: Path, out: Path, dates: pd.DatetimeIndex,
                         predictions: pd.DataFrame) -> None:
    """用一份打分表完成所有方法的诊断、组合回测与统一报告。"""
    from src.ml.models import ROLLING_SCORE, SCORE_COLUMNS, prediction_diagnostics
    from src.ml.report import write_results
    from src.ml.rolling import check_prediction_windows
    from src.ml.rolling_report import evaluate_windows
    from src.ml.strategy import compare_scores, run_backtests

    if ROLLING_SCORE not in predictions:
        raise ValueError("实验缺少滚动模型预测，请用 run_all 完整运行")
    specs = json.loads((out / "windows.json").read_text())
    if specs["config"] != config:
        raise ValueError("配置与当前存档的滚动训练或回测参数不同")
    test = predictions.loc[predictions.split.eq("test")]
    if not np.isfinite(test[ROLLING_SCORE]).all() or test.window.isna().any():
        raise ValueError("滚动模型缺少测试期分数或窗口编号")
    check_prediction_windows(test, specs["windows"], test)
    strategies = (*SCORE_COLUMNS, ROLLING_SCORE)
    scores, panel, test_dates = compare_scores(predictions, INPUTS, config, dates)
    daily, groups = prediction_diagnostics(scores, config, dates, out, strategies=strategies)
    run_backtests(INPUTS, config, test_dates, scores, panel, out, strategies=strategies)
    evaluate_windows(config, test_dates, scores, daily, groups, out)
    scores.to_parquet(out / "predictions.pending", index=False)
    (out / "predictions.pending").replace(out / "predictions.parquet")
    write_results(config, config_path, INPUTS, out, ROOT)


def prepare_features(config: dict, dates: pd.DatetimeIndex, out: Path, workers: int) -> None:
    """准备交易日行情和成分股，生成全实验共用的 Alpha158 特征。"""
    daily = pd.read_parquet(INPUTS["prices"])
    members = prepare_membership(pd.read_parquet(INPUTS["membership"]))
    if not pd.DatetimeIndex(members.trade_date.unique()).as_unit("ns").equals(dates.as_unit("ns")):
        raise ValueError("Membership does not cover the market calendar")
    panel = prepare_panel(daily, dates)
    build_features(panel, members, config, out, workers)
    audit_file = out / "data_audit.json"
    audit = json.loads(audit_file.read_text())
    audit.update(raw_price_rows=len(daily), raw_start=str(daily.trade_date.min()), raw_end=str(daily.trade_date.max()),
                 raw_tickers=int(daily.ticker.nunique()), market_sessions=len(dates),
                 raw_input_sorted_by_ticker_date=pd.MultiIndex.from_frame(daily[["ticker", "trade_date"]]).is_monotonic_increasing,
                 panel_observation_status=panel.daily.observation_status.value_counts().to_dict(),
                 price_keys_unique=True, member_keys_unique=True, snapshot_dates_at_or_before_signals=True)
    audit_file.write_text(json.dumps(audit, indent=2) + "\n")


def main() -> None:
    """执行一次完整实验，或用同一批预测刷新回测与报告。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--workers", type=int, default=6, help="计算 Qlib 表达式的并行进程数")
    parser.add_argument("--stage", choices=("all", "evaluate"), default="all", help="evaluate 使用已有预测刷新比较")
    args = parser.parse_args()
    evaluate = args.stage == "evaluate"
    out = (args.out or (latest_run("manifest.json", "windows.json") if evaluate else new_run())).resolve()
    protected = (ROOT / "data", ROOT / "docs", ROOT / "results/factor_study")
    if out == RUNS or any(out == path or out in path.parents or path in out.parents for path in protected):
        parser.error("请选择单次实验的输出目录")
    config_path = args.config or (out / "config.json" if evaluate else ROOT / "configs/ml_qlib.json")
    config = json.loads(config_path.read_text())
    dates = load_market_calendar(INPUTS["calendar"])

    if evaluate:
        predictions = load_predictions(config, out)
    else:
        from src.ml.models import train_and_predict
        from src.ml.rolling import merge_rolling_predictions, train_rolling

        out.mkdir(parents=True, exist_ok=True)
        if any(out.iterdir()):
            parser.error("新实验需要空目录，请选择其他 --out")
        print(f"本次实验目录 {out}", flush=True)
        prepare_features(config, dates, out, args.workers)
        predictions = train_and_predict(config, out)
        rolling = train_rolling(config, dates, out)
        specs = json.loads((out / "windows.json").read_text())
        predictions = merge_rolling_predictions(predictions, rolling, specs["windows"])
    evaluate_predictions(config, config_path, out, dates, predictions)
    if not evaluate:
        (out / "features.parquet").unlink()
    print(f"结果已生成 {out}")
    print(f"报告已生成 {ROOT / 'docs/ml_study.md'}")


if __name__ == "__main__":
    main()
