"""按统一因子研究流程生成分数、评价并回测中证500选股方法。"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import CONFIG_FILE, load_config, load_tuning, rebalance_frequency, validate_lightgbm_grid, walk_forward_policy
from src.data.download import save_parquet
from src.data.research_panel import (
    CSI500_INPUTS as INPUTS, ROOT, ResearchPanel, load_market_calendar, prepare_inputs,
    prepare_membership, prepare_test_panel,
)
from src.factors import factor_batches
from src.factors.neutralization import attach_neutral_scores
from src.factors.price_volume import SINGLE_FACTOR_SCORES

RUNS = ROOT / "results/ml_study"
FACTOR_RUNS = ROOT / "results/factors"


def new_run(rebalance_days: int | None = None, *, tag: str = "", root: Path = RUNS) -> Path:
    """按调仓频率、实验类别和运行时间命名实验目录。"""
    parts = [rebalance_frequency(rebalance_days)[0]] if rebalance_days is not None else []
    if tag:
        parts.append(tag)
    parts.append(f"{datetime.now():%Y%m%d_%H%M%S_%f}")
    return root / "_".join(parts)


def latest_run(*required: str, rebalance_days: int) -> Path:
    """按末尾时间戳查找同一调仓周期的最近完整实验，兼容旧目录名。"""
    for path in sorted(RUNS.glob("*"), key=lambda item: item.name[-22:], reverse=True):
        config_path = path / "config.json"
        if path.is_dir() and config_path.is_file() and all((path / name).is_file() for name in required):
            if json.loads(config_path.read_text())["rebalance_days"] == rebalance_days:
                return path
    raise FileNotFoundError("没有可继续的实验，请先运行 run_all，或用 --out 指定已有实验目录")


def prepare_test_scores(predictions: pd.DataFrame, panel: ResearchPanel, dates: pd.DatetimeIndex,
                        config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """将中性化后的传统因子按股票日左连接到模型分数，返回分数和回归控制变量。"""
    members = prepare_membership(pd.read_parquet(INPUTS["membership"]))
    factors = pd.concat(factor_batches("price_volume", panel, members, dates, config), ignore_index=True)
    factors = attach_neutral_scores(factors, pd.read_parquet(INPUTS["caps"]), pd.read_parquet(INPUTS["industries"]),
                                    SINGLE_FACTOR_SCORES)
    columns = {f"{name}_neutral": name for name in SINGLE_FACTOR_SCORES}
    keys = ["trade_date", "ticker"]
    scores = predictions.drop(columns=list(SINGLE_FACTOR_SCORES), errors="ignore").merge(
        factors[[*keys, *columns]].rename(columns=columns), on=keys, how="left", validate="one_to_one",
    )
    return scores, factors[[*keys, "industry", "total_mv", "circ_mv"]]


def evaluate_predictions(config: dict, config_path: Path, out: Path, dates: pd.DatetimeIndex,
                         predictions: pd.DataFrame, *, lightgbm_only: bool = False) -> Path:
    """接收股票日分数表，为选定方法执行同一套评价、Top-K 回测与报告生成。"""
    from src.backtest.engine import run_backtests
    from src.backtest.factor_metrics import common_scores, evaluate_regression, prediction_diagnostics
    from src.backtest.portfolio import evaluate_windows
    from src.ml.models import MODEL_SCORES, ROLLING_SCORE
    from src.report.research import write_results

    panel, test_dates = prepare_test_panel(config, dates)
    windows = []
    if lightgbm_only:
        strategies = validation = ("lightgbm",)
        scores = predictions
        used = ["calendar", "membership", "prices", "benchmark"]
        if config.get("limit_threshold") is None:
            used.append("limits")
        if config.get("alpha158_processing") == "neutral":
            used.extend(["caps", "industries"])
        inputs = {name: INPUTS[name] for name in used}
    else:
        from src.ml.rolling import study_windows

        windows = study_windows(config, out, predictions)
        strategies = (MODEL_SCORES[0], ROLLING_SCORE, MODEL_SCORES[1], *SINGLE_FACTOR_SCORES)
        if "lightgbm" not in predictions and (out / "manifest.json").is_file():
            archive = json.loads((out / "manifest.json").read_text())
            if "lightgbm" in archive.get("excluded_from_archive", []):
                strategies = tuple(name for name in strategies if name != "lightgbm")
        validation = tuple(name for name in MODEL_SCORES if name in strategies)
        scores, controls = prepare_test_scores(predictions, panel, test_dates, config)
        inputs = INPUTS
        common = common_scores(scores.loc[scores.split.eq("test")], strategies)
        evaluate_regression(common, controls, strategies, config.get("min_regression_pairs", 100), out)

    evaluation = scores.loc[scores.split.isin(("validation", "test"))]
    test = evaluation.loc[evaluation.split.eq("test")]
    
    # 评估分数与未来收益的关系，如计算RankIC等
    daily = prediction_diagnostics(evaluation, config, dates, out,
                                   strategies_by_split={"validation": validation, "test": strategies})
    run_backtests(INPUTS, config, test_dates, test, panel, out, strategies=strategies)
    if windows:
        evaluate_windows(windows, config["rebalance_days"], test_dates, len(test), daily, out)
    save_parquet(scores, out / "predictions.parquet")
    return write_results(config, config_path, inputs, out, ROOT, strategies)


def main(argv=None, *, lightgbm_only: bool = False) -> None:
    """执行完整实验、验证集调参或复用预测刷新报告。"""
    parser = argparse.ArgumentParser(description="固定 LightGBM 与中证500指数比较" if lightgbm_only else __doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", help="指定实验目录，默认自动选择")
    parser.add_argument("--workers", type=int, help="临时覆盖配置中的 feature_workers")
    if lightgbm_only:
        parser.set_defaults(stage="all")
    else:
        parser.add_argument("--tuning-config", type=Path, help="仅用于 --stage tune，默认 configs/tuning.json")
        parser.add_argument("--stage", choices=("all", "evaluate", "tune"), default="all",
                            help="tune 用验证集调参，evaluate 使用已有预测刷新比较")
    args = parser.parse_args(argv)
    evaluate = args.stage == "evaluate"
    tuning = args.stage == "tune"
    if not lightgbm_only and args.tuning_config and not tuning:
        parser.error("--tuning-config 需要 --stage tune")
    out = Path(args.out) if args.out else None
    if evaluate and out is None:
        days = load_config(args.config or CONFIG_FILE)["rebalance_days"]
        out = latest_run("manifest.json", "windows.json", rebalance_days=days)
    config_path = args.config or (out / "config.json" if evaluate and out is not None else CONFIG_FILE)
    # 加载研究配置，确定输出目录等
    config = load_config(config_path)
    if out is None:
        out = new_run(tag="tuning") if tuning else new_run(config["rebalance_days"], tag="lightgbm" if lightgbm_only else "")
    out = out.resolve()
    protected = (ROOT / "data", ROOT / "docs")
    if out == RUNS or any(out == path or out in path.parents or path in out.parents for path in protected):
        parser.error("请选择单次实验的输出目录")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers 必须是正整数")
    if evaluate:
        dates = load_market_calendar(INPUTS["calendar"])
        from src.ml.models import load_predictions

        predictions = load_predictions(config, out)
    else:
        if tuning:
            from src.ml.tuning import tune_lightgbm

            config["walk_forward"] = walk_forward_policy(config)
            config["lightgbm_tuning"] = load_tuning("lightgbm_tuning", config, args.tuning_config)
            validate_lightgbm_grid(config["lightgbm_tuning"], config["lightgbm"])
        out.mkdir(parents=True, exist_ok=True)
        if any(out.iterdir()):
            parser.error("新实验需要空目录，请选择其他 --out")
        print(f"本次实验目录 {out}", flush=True)
        neutral = config.get("alpha158_processing") == "neutral"
        # 检查交易日历、历史成分股、行情和行业数据，返回交易日历
        dates = prepare_inputs(config, need_industry=neutral or not (tuning or lightgbm_only))
        if neutral:
            from src.data.download import download_caps
            download_caps(config)
        workers = args.workers if args.workers is not None else config.get("feature_workers", 6)
        through = config["segments"]["validation"][1] if tuning else None
        from src.ml.dataset import prepare_features

        # 生成训练表，计算每一个交易日的alpha158和未来收益标签
        prepare_features(config, dates, out, workers, through=through)
        if tuning:
            (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            tune_lightgbm(config, out)
            (out / "features.parquet").unlink()
            print(f"调参完成。候选结果 {out / 'trials.csv'}，最佳配置 {out / 'best_config.json'}")
            return
        from src.ml.models import train_and_predict

        # 用生成的训练表训练固定的lightGBM(MSE)和Ridge
        predictions = train_and_predict(config, out, include_ridge=not lightgbm_only)
        if not lightgbm_only:
            from src.ml.rolling import merge_rolling_predictions, train_rolling

            # 训练 Rolling lightGBM，给各自的测试窗口打分
            rolling = train_rolling(config, dates, out)
            windows = json.loads((out / "windows.json").read_text())["windows"]
            predictions = merge_rolling_predictions(predictions, rolling, windows)
    # 接手预测表，此时内部有固定LightGBM(MSE)，Rolling LightGBM(MSE)，Ridge(MSE) 三类
    # 补充alpha3,13,44三种传统因子分数，经过了行业、市值中性化，再按照日期和股票代码合并 
    # 这里最后有六种方法计算出的测试集分数
    report = evaluate_predictions(config, config_path, out, dates, predictions, lightgbm_only=lightgbm_only)
    if not evaluate:
        (out / "features.parquet").unlink()
    print(f"结果已生成 {out}")
    print(f"报告已生成 {report}")


def lightgbm_main(argv=None):
    """单独训练固定 LightGBM，评价与报告共用综合研究流程。"""
    main(argv, lightgbm_only=True)


if __name__ == "__main__":
    main()
