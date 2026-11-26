"""训练 LightGBM 和 XGBoost 各四个排序器，按日排名后八等权。"""
import argparse

from src.ml import tree_experiment

DEFAULT_XGBOOST_LOSS = None  # 默认读取配置中的 rank:ndcg。


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    tree_experiment.model_arguments(parser)
    parser.add_argument("--xgboost-loss", choices=("rank:ndcg", "rank:pairwise"), default=DEFAULT_XGBOOST_LOSS)
    args = parser.parse_args(argv)
    tree_experiment.model_run(args, ("lightgbm", "xgboost"))


if __name__ == "__main__":
    main()
