"""兼容原有命令和导入，功能实现在 run_rank_stack。"""
from src.scripts.run_rank_stack import (
    SIX_GRADE_METHODS as METHODS,
    train_six_grades as train,
    evaluate_six_grades as evaluate,
    six_grades_main as main,
)


if __name__ == "__main__":
    main()
