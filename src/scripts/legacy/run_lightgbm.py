"""兼容原有命令和导入，功能实现在 run_all。"""
from src.scripts.run_all import (
    main as run_study,
    lightgbm_main as main,
)


if __name__ == "__main__":
    main()
