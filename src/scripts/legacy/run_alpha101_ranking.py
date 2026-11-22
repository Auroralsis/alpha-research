"""兼容原有命令和导入，功能实现在 factors。"""
from src.scripts.factors import (
    FACTORS,
    KEYS,
    ENSEMBLES,
    build_inputs,
    fit_arm,
    run_ranking as run,
    train_and_compare,
    ranking_main as main,
)


if __name__ == "__main__":
    main()
