"""用已知历史快照重建每个交易日的中证500股票池。"""

import pandas as pd

from src.data.download import SNAPSHOT_FILE, save_parquet
from src.data.research_panel import CSI500_INPUTS


def build_membership(snapshots: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """按交易日向前查找最近的成员快照，返回逐股逐日成员表。"""
    dates = pd.DataFrame({"trade_date": pd.to_datetime(calendar.cal_date.astype(str))})
    dates = dates.drop_duplicates().sort_values("trade_date")
    snapshots = snapshots.rename(columns={"con_code": "ticker", "trade_date": "source_snapshot_date"}).copy()
    snapshots["source_snapshot_date"] = pd.to_datetime(snapshots.source_snapshot_date.astype(str))
    if snapshots.duplicated(["ticker", "source_snapshot_date"]).any():
        raise ValueError("成员快照包含重复股票日期")
    known = snapshots[["source_snapshot_date"]].drop_duplicates().sort_values("source_snapshot_date")
    aligned = pd.merge_asof(dates, known, left_on="trade_date", right_on="source_snapshot_date", direction="backward")
    if aligned.source_snapshot_date.isna().any():
        raise ValueError("日历起点缺少此前已知的成员快照，请补齐预热期快照")
    return snapshots.drop(columns="index_code").merge(
        aligned, on="source_snapshot_date", validate="many_to_many",
    ).sort_values(["trade_date", "ticker"]).reset_index(drop=True)


def main() -> None:
    """读取下载产物，保存覆盖预热期及完整研究期的历史成员表。"""
    result = build_membership(pd.read_parquet(SNAPSHOT_FILE), pd.read_parquet(CSI500_INPUTS["calendar"]))
    save_parquet(result, CSI500_INPUTS["membership"])
    print(f"成员数据 {result.trade_date.min().date()} 至 {result.trade_date.max().date()}，{result.ticker.nunique()} 只股票")


if __name__ == "__main__":
    main()
