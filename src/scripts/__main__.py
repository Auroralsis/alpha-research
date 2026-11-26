"""按任务运行中证500研究。通过“任务名 --help”查看该任务的参数。"""
import argparse
import sys
from importlib import import_module

# 延迟导入使查看帮助和运行单个任务只加载所需模块。
COMMANDS = {
    "study": ("run_all", "main", "综合研究、验证集调参和归档复评"),
    "lightgbm": ("train_lightgbm", "main", "LightGBM 四等权排序或 MSE 回归"),
    "lambdarank": ("compare_rank_labels", "main", "比较排序标签和增益"),
    "eight-rankers": ("train_eight_rankers", "main", "LightGBM + XGBoost 八排序器等权"),
    "xgboost": ("train_xgboost", "main", "独立 XGBoost 四排序器等权"),
    "regime-fusion": ("regime_fusion", "main", "复用四个排序器评价六种市场状态融合方案"),
    "lightgbm-equal": ("train_lightgbm", "equal_main", "LightGBM 四目标等权训练、选参与回测"),
    "rank-stack": ("compare_tree_models", "stack_main", "四目标排序、等权与堆叠"),
    "xgboost-compare": ("compare_tree_models", "xgboost_main", "XGBoost 与 LightGBM 排名组合"),
    "rolling-pair": ("ml.tree_experiment", "rolling_pair_main", "按年用扩展或滑动窗口重训 LightGBM 与 XGBoost 等权组合"),
    "pair-tuning": ("ml.tree_experiment", "main", "按验证期组合排名损失联合调优八个 LightGBM/XGBoost 排序器"),
    "tra": ("run_tra", "main", "Alpha158 时序 TRA 与冻结树模型的五组对照"),
    "mlp": ("train_mlp", "main", "MLP 固定或滚动训练"),
}


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="python -m src.scripts", description=__doc__)
    commands = parser.add_subparsers(dest="task", metavar="任务", required=True)
    for name, (_, _, description) in COMMANDS.items():
        commands.add_parser(name, help=description, add_help=False)
    if not args:
        parser.print_help()
        return
    task = parser.parse_args(args[:1]).task
    module, entry, _ = COMMANDS[task]
    package = "src" if module.startswith("ml.") else "src.scripts"
    getattr(import_module(f"{package}.{module}"), entry)(args[1:])


if __name__ == "__main__":
    main()
