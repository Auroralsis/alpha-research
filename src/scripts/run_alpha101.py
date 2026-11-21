"""计算中证500的 Alpha101 原值、中性化值和逐日 IC 指标。"""

import argparse
import json
from pathlib import Path

import pandas as pd

from src.backtest.factor_metrics import evaluate_factors
from src.config import CONFIG_FILE, load_config
from src.data.download import write_parquet_batches
from src.data.research_panel import (
    CSI500_INPUTS, execution_labels, load_market_calendar, prepare_membership, prepare_panel,
    segment_masks,
)
from src.factors import factor_batches
from src.factors.alpha101 import HISTORY_DAYS
from src.report.research import write_alpha101_report
from src.scripts.run_all import new_run


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
    names = [f"Alpha{i}" for i in range(1, 102)]
    frame = pd.read_parquet(out / "alpha101_raw.parquet")
    summary = evaluate_factors(frame, names, out, config["feature_workers"])
    return write_alpha101_report(summary, frame, out, config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_FILE)
    args = parser.parse_args()
    config = load_config(args.config)
    out = new_run(tag="daily_alpha101")
    out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    build_raw(out, config)
    print(evaluate(out, config))


if __name__ == "__main__":
    main()
