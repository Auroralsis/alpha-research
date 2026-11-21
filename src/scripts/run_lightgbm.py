"""只训练固定 LightGBM，使用综合研究的评价、回测与报告流程。"""

from src.scripts.run_all import main as run_study


def main() -> None:
    """执行固定 LightGBM 与指数的比较。"""
    run_study(lightgbm_only=True)


if __name__ == "__main__":
    main()
