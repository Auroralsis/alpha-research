"""供研究报告共用的 Markdown 表格与文件校验。"""

import hashlib
import re
from collections.abc import Iterable
from itertools import count
from pathlib import Path


def markdown_table(columns: Iterable[object], rows: Iterable[Iterable[object]]) -> str:
    """输入列名和逐行数据，返回 Markdown 表格文本。"""
    headers = list(columns)
    return "\n".join("| " + " | ".join(map(str, row)) + " |" for row in [headers, ["---"] * len(headers), *rows])


def number_table_captions(report: str) -> str:
    """按整份报告的出现顺序为表题编号，已有编号也一并更新。"""
    numbers = count(1)
    return re.sub(r"^\*\*表(?: [0-9]+)?　", lambda _: f"**表 {next(numbers)}　", report, flags=re.MULTILINE)


def sha256(path: Path) -> str:
    """分块读取文件，返回 SHA-256 十六进制校验值。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
