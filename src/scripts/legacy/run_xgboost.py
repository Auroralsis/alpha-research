"""兼容原有命令和导入，功能实现在 boosting。"""
from src.scripts.boosting import (
    fit_target,
    train_xgboost as train,
    predict_xgboost as predict,
    evaluate_xgboost as evaluate,
    paired_signals,
    compare_target_weights,
    xgboost_main as main,
)


if __name__ == "__main__":
    main()
