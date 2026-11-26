"""独立训练四个 XGBoost 排序器，每日排名百分位等权融合。"""
import argparse

import pandas as pd
import xgboost as xgb

DEFAULT_LOSS = None  # 默认使用配置中的 XGBoost 训练目标。


def fit_target(params: dict, cfg: dict, train_set: xgb.DMatrix, early_stop: xgb.DMatrix):
    """在本组标签上训练，用早停期选出的轮数截取模型和记录曲线。"""
    history = {}
    model = xgb.train(params, train_set, cfg["num_boost_round"], evals=[(early_stop, "early_stop")],
                      early_stopping_rounds=cfg["early_stopping_rounds"], maximize=True,
                      evals_result=history, verbose_eval=False)
    rounds = model.best_iteration + 1
    curve = pd.Series(history["early_stop"][params["eval_metric"]], name="ndcg_at_50")
    curve.index = pd.RangeIndex(1, len(curve) + 1, name="iteration")
    return model[:rounds], curve


def main(argv=None):
    """独立训练四个 XGBoost，按每日排名百分位等权验证和回测。"""
    from src.ml.tree_experiment import model_arguments, model_run

    parser = argparse.ArgumentParser(description=main.__doc__)
    model_arguments(parser)
    parser.add_argument("--loss", dest="xgboost_loss", choices=("rank:ndcg", "rank:pairwise"), default=DEFAULT_LOSS,
                        help="训练目标，默认读取配置中的 XGBoost objective")
    model_run(parser.parse_args(argv), ("xgboost",))


if __name__ == "__main__":
    main()
