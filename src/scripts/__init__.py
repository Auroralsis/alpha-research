"""研究运行入口；历史命令仍可按原模块名导入或使用 python -m 执行。"""
from pathlib import Path

__path__.append(str(Path(__file__).with_name("legacy")))
