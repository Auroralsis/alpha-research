"""兼容原有命令和导入，功能实现在 run_listfold。"""
from src.scripts.run_listfold import (
    run,
    mlp_main as main,
)


if __name__ == "__main__":
    main()
