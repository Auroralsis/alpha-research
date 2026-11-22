"""兼容原有命令和导入，功能实现在 boosting。"""
from src.scripts.boosting import (
    fit_catboost as fit,
    score_candidate,
    evaluate_catboost as compare,
    replay_candidate,
    catboost_main as main,
)


if __name__ == "__main__":
    main()
