"""读取项目统一研究配置。"""

import json
import math
from pathlib import Path

CONFIG_FILE = Path(__file__).resolve().parents[1] / "configs/research.json"
RIDGE_LOSS = "mse"
ZSCORE_LABEL = "zscore"


def normalize_training_label(mode: str) -> str:
    """校验标签模式，并兼容历史实验中的旧名称。"""
    if mode == "daily_sample_zscore":
        return ZSCORE_LABEL
    if mode not in (ZSCORE_LABEL, "raw"):
        raise ValueError("training_label 必须为 zscore 或 raw")
    return mode


def rebalance_frequency(days: int) -> tuple[str, str]:
    """根据调仓交易日数返回文件名后缀和中文研究频率。"""
    if isinstance(days, bool) or not isinstance(days, int) or days <= 0:
        raise ValueError("rebalance_days 必须是正整数")
    return {1: ("daily", "日频"), 5: ("weekly", "周频")}.get(days, (f"{days}d", f"{days}日调仓"))


def load_config(path: Path = CONFIG_FILE) -> dict:
    """读取训练、标签、交易、运行和下载参数。"""
    config = json.loads(path.read_text(encoding="utf-8"))
    rebalance_frequency(config["rebalance_days"])
    if not 0 < config["label_start_days"] < config["label_end_days"]:
        raise ValueError("标签偏移必须满足 0 < label_start_days < label_end_days")
    config["training_label"] = normalize_training_label(config.get("training_label"))
    if config.setdefault("ridge_loss", RIDGE_LOSS) != RIDGE_LOSS:
        raise ValueError("ridge_loss 必须为 mse，Ridge 使用平方误差拟合")
    if not 0 <= config["n_drop"] <= config["topk"] or config["topk"] <= 0:
        raise ValueError("持股数必须为正，且 0 <= n_drop <= topk")
    pairs = config.get("min_regression_pairs", 100)
    if isinstance(pairs, bool) or not isinstance(pairs, int) or pairs < 2:
        raise ValueError("min_regression_pairs 必须为不小于 2 的整数")
    if config["initial_capital_cny"] <= 0:
        raise ValueError("initial_capital_cny 必须为正")
    if config.get("buy_price", "vwap") not in ("vwap", "close"):
        raise ValueError("buy_price 必须为 vwap 或 close")
    limit = config.get("limit_threshold")
    if limit is not None and (isinstance(limit, bool) or not 0 < limit < 1):
        raise ValueError("limit_threshold 必须在 0 与 1 之间")
    if not 0 <= config.get("min_trade_cost_cny", 0) < float("inf"):
        raise ValueError("min_trade_cost_cny 必须为有限非负数")
    unit = config.get("trade_unit")
    if unit is not None and (isinstance(unit, bool) or not isinstance(unit, int) or unit <= 0):
        raise ValueError("trade_unit 必须为正整数或 null")
    for key in ("feature_workers", "feature_batch_size", "ridge_num_threads", "rolling_test_windows"):
        if key in config and (not isinstance(config[key], int) or config[key] <= 0):
            raise ValueError(f"{key} 必须为正整数")
    if "lightgbm_tuning" in config:
        tuning = config["lightgbm_tuning"]
        if not isinstance(tuning, dict):
            raise ValueError("lightgbm_tuning 必须为 JSON 对象")
        space = tuning.get("search_space")
        supported = {"lambda_l1", "lambda_l2", "num_leaves", "learning_rate", "max_depth"}
        if not isinstance(space, dict) or not space or not space.keys() <= supported & config["lightgbm"].keys():
            raise ValueError("search_space 应从 lightgbm 段已有的 lambda_l1、lambda_l2、num_leaves、learning_rate 和 max_depth 中选参数")
        for key, values in space.items():
            if not isinstance(values, list) or not values or any(
                type(v) not in (int, float) or not math.isfinite(v)
                or (key != "max_depth" and v < 0)
                or (key == "max_depth" and type(v) is not int)
                or (key == "num_leaves" and (type(v) is not int or v < 2))
                or (key == "learning_rate" and v <= 0) for v in values
            ):
                raise ValueError(f"search_space.{key} 包含无效的候选值")
    return config
