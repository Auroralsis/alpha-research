"""读取项目统一研究配置。"""

import json
import math
from pathlib import Path

CONFIG_FILE = Path(__file__).resolve().parents[1] / "configs/research.json"
TUNING_FILE = CONFIG_FILE.with_name("tuning.json")
RIDGE_LOSS = "mse"
ZSCORE_LABEL = "zscore"
SAMPLING_RATES = {"subsample", "colsample_bytree"}


def load_tuning(section: str, settings: dict, path: Path | None = None) -> dict | list:
    """仅在调参时读取候选，兼容旧实验配置内嵌的网格。"""
    if path is None:
        if section in settings:
            return settings[section]
        if section == "catboost_tuning" and "tuning" in settings.get("catboost_four_equal", {}):
            return settings["catboost_four_equal"]["tuning"]
    source = path or TUNING_FILE
    document = json.loads(source.read_text(encoding="utf-8"))
    if section not in document:
        raise ValueError(f"{source} 缺少调参配置 {section}")
    return document[section]


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
    # 将面向读者的实验分组展开为训练端的参数；旧实验快照本身已是展开格式。
    if "single_lightgbm" in config:
        single = config.pop("single_lightgbm")
        config["lightgbm"] = single["params"]
        if "tuning" in single:
            config["lightgbm_tuning"] = single["tuning"]
    if "single_xgboost" in config:
        config["xgboost"] = config.pop("single_xgboost")["params"]
    if "xgboost_four_equal" in config:
        ensemble = config.pop("xgboost_four_equal")
        if "tuning" in ensemble:
            config["xgboost_tuning"] = ensemble["tuning"]
    if "four_model_targets" in config:
        targets = config["four_model_targets"]
        if set(targets) != {"top50", "top100", "avoid50", "avoid100"} or any(
            not isinstance(gains, list) or len(gains) != 5 or any(type(n) is not int or n < 0 for n in gains)
            or sorted(gains) != gains or gains[0] == gains[-1] for gains in targets.values()
        ):
            raise ValueError("four_model_targets 的四项目标各需五个非负递增或持平的整数增益，首尾须不同")
    if "lightgbm_xgboost_blend" in config:
        weight = config["lightgbm_xgboost_blend"]["xgboost_weight"]
        if type(weight) not in (int, float) or not 0 <= weight <= 1:
            raise ValueError("xgboost_weight 必须在 0 到 1 之间")
    if config.get("alpha158_processing", "raw") not in ("raw", "mad", "neutral"):
        raise ValueError("alpha158_processing 必须为 raw、mad 或 neutral")
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
    model = config["lightgbm"]
    for key in SAMPLING_RATES & model.keys():
        if type(model[key]) not in (int, float) or not 0 < model[key] <= 1:
            raise ValueError(f"single_lightgbm.params.{key} 必须在 (0, 1] 内")
    freq = model.get("bagging_freq", model.get("subsample_freq", 0))
    if type(freq) is not int or freq < 0:
        raise ValueError("bagging_freq 必须是非负整数")
    if "lightgbm_tuning" in config:
        validate_lightgbm_grid(config["lightgbm_tuning"], model)
    return config


def validate_lightgbm_grid(tuning: dict, model: dict) -> None:
    """在准备特征前校验 LightGBM 网格及各候选的取值范围。"""
    if not isinstance(tuning, dict):
        raise ValueError("lightgbm_tuning 必须为 JSON 对象")
    space = tuning.get("search_space")
    supported = {"lambda_l1", "lambda_l2", "num_leaves", "learning_rate", "max_depth"} | SAMPLING_RATES
    if not isinstance(space, dict) or not space or not space.keys() <= supported & model.keys():
        raise ValueError("search_space 应选用 single_lightgbm.params 中已配置的参数，可选 " + "、".join(sorted(supported)))
    for key, values in space.items():
        if not isinstance(values, list) or not values or any(
            type(v) not in (int, float) or not math.isfinite(v)
            or (key in SAMPLING_RATES and not 0 < v <= 1)
            or (key != "max_depth" and v < 0)
            or (key == "max_depth" and type(v) is not int)
            or (key == "num_leaves" and (type(v) is not int or v < 2))
            or (key == "learning_rate" and v <= 0) for v in values
        ):
            raise ValueError(f"search_space.{key} 包含无效的候选值")


def with_model_params(config: dict, family: str, params: dict) -> dict:
    """以输入配置原有的分段格式导出选定参数，随机种子继续由原配置控制。"""
    key = f"single_{family}"
    if key in config:
        section = config[key]
        selected = {k: params[k] for k in section["params"]}
        return {**config, key: {**section, "params": selected}}
    return {**config, family: {k: params[k] for k in config[family]}}


def write_best_config(out: Path, config: dict) -> None:
    """导出可直接用于固定训练的参数，搜索范围保存在当次实验快照中。"""
    fixed = {k: v for k, v in config.items()
             if not k.endswith("_tuning") and k not in ("ridge_strong_alphas", "lambdarank_bottom_gaps")}
    for key in ("single_lightgbm", "xgboost_four_equal", "catboost_four_equal"):
        if key in fixed:
            fixed[key] = {k: v for k, v in fixed[key].items() if k != "tuning"}
    (out / "best_config.json").write_text(json.dumps(fixed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
