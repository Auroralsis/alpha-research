"""按任务运行中证500研究。通过“任务名 --help”查看该任务的参数。"""
import argparse
import sys
from importlib import import_module

# 延迟导入使查看帮助和运行单个任务只加载所需模块。
COMMANDS = {
    "study": ("run_all", "main", "综合研究、验证集调参和归档复评"),
    "lightgbm": ("run_all", "lightgbm_main", "固定 LightGBM 与指数比较"),
    "alpha101": ("factors", "alpha101_main", "Alpha101 单因子检验"),
    "factor-ranking": ("factors", "ranking_main", "比较 Alpha101、Alpha158 及合并特征"),
    "equal-weight": ("factors", "equal_weight_main", "Alpha158 因子等权和 Ridge 对照"),
    "lambdarank": ("run_lambdarank", "main", "比较排序标签和增益"),
    "rank-stack": ("run_rank_stack", "main", "四目标排序、等权与堆叠"),
    "six-grades": ("run_rank_stack", "six_grades_main", "比较五档和六档排序"),
    "xgboost": ("boosting", "xgboost_main", "XGBoost 与 LightGBM 排名组合"),
    "catboost": ("boosting", "catboost_main", "CatBoost 候选、组合及回放"),
    "rolling-pair": ("run_pair_tuning", "rolling_pair_main", "固定训练起点，按年重训 LightGBM 与 XGBoost 等权组合"),
    "pair-tuning": ("run_pair_tuning", "main", "按组合验证损失联合调参"),
    "listfold": ("run_listfold", "main", "ListFold 与四目标 MLP；用 --model 选择"),
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
    getattr(import_module(f"src.scripts.{module}"), entry)(args[1:])


if __name__ == "__main__":
    main()
