"""兼容原有命令和导入，功能实现在 factors。"""
from src.scripts.factors import (
    build_raw,
    evaluate,
    alpha101_main as main,
)


if __name__ == "__main__":
    main()
