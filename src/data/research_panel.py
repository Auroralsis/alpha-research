"""从日历、成分和行情构建研究面板，并按交易日生成收益标签。"""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

DateValues = pd.Series | pd.Index | np.ndarray | Sequence[date | str | int]
ROOT = Path(__file__).resolve().parents[2]
CSI500_INPUTS = {
    "prices": ROOT / "data/processed/daily/csi500_processed_daily.parquet",
    "membership": ROOT / "data/processed/universe/csi500_membership.parquet",
    "calendar": ROOT / "data/raw/calender/trade_calender_2013_2025.parquet",
    "caps": ROOT / "data/reference/csi500/daily_basic.parquet",
    "industries": ROOT / "data/reference/csi500/industry_l1_history.parquet",
    "limits": ROOT / "data/reference/csi500/stk_limit.parquet",
    "benchmark": ROOT / "data/reference/csi500/benchmark.csv",
}


def as_date_index(values: DateValues) -> pd.DatetimeIndex:
    """把一维日期数据转为日期索引，保留原有日期顺序。"""
    if isinstance(values, pd.DatetimeIndex):
        return values
    return pd.DatetimeIndex(pd.Series(values).to_numpy(), name=getattr(values, "name", None))


def stock_day_index(rows: pd.DataFrame) -> pd.MultiIndex:
    """把项目的股票日键转换成 Qlib 使用的索引。"""
    return pd.MultiIndex.from_frame(rows[["trade_date", "ticker"]], names=["datetime", "instrument"])


def parse_dates(values: DateValues) -> pd.Series:
    """输入一维日期，如 `["20230103", "20230104"]`；返回重新编号的日期列。"""
    # 丢弃输入的旧标签并重新编号，不保存旧标签结果作为新列
    date_series = pd.Series(values).reset_index(drop=True)
    # 如果输入是日期类型直接转换，否则先转string再处理
    if pd.api.types.is_datetime64_any_dtype(date_series):
        result = pd.to_datetime(date_series, errors="raise")
    else:
        text = date_series.astype("string")
        # 检查values是否都是类似"20230103"的8位数字标准格式
        if not text.str.fullmatch(r"\d{8}").fillna(False).all():
            raise ValueError("Dates must be datetime values or exactly YYYYMMDD")
        # 注意这里%Y指4位年份，%m指月份，%d指日期
        result = pd.to_datetime(text, format="%Y%m%d", errors="raise")
    # 这里normalize把时间归到当天零点，保证数据没有非零的时分秒
    if result.isna().any() or (result != result.dt.normalize()).any():
        raise ValueError("Missing dates and intraday timestamps are not allowed")
    return result


def load_market_calendar(path: Path) -> pd.DatetimeIndex:
    """读取含 cal_date 的日历 Parquet 表，返回升序交易日索引。

    若表中包含 is_open，只保留开市日。
    """
    frame = pd.read_parquet(path)
    if "is_open" in frame:
        frame = frame.loc[frame.is_open.astype(int).eq(1)]
    return pd.DatetimeIndex(parse_dates(frame.cal_date)).sort_values()


@dataclass
class ResearchPanel:
    """`daily` 是股票日行情长表，`calendar` 和 `tickers` 定义宽表的行列。"""

    daily: pd.DataFrame
    calendar: pd.DatetimeIndex
    tickers: pd.Index

    def matrix(self, column: str) -> pd.DataFrame:
        """取 `daily` 的一个字段，例如 `close_adj`，返回 [交易日 × 股票] 宽表。

        行索引是交易日，列名是股票代码。非 observed 行情记为 NaN。
        """
        data = self.daily
        values = data[column].where(data.observation_status.eq("observed"))
        # assign是往原两列数据中添加一列，随后pivot将其转换成矩阵
        # 其中index是行索引，columns是列索引，values是对应数值
        # reindex这里，使用全交易日日期和全股票，没有数据的部分自动补充NaN
        return (data[["ticker", "trade_date"]].assign(value=values).pivot(index="trade_date", columns="ticker", values="value")
                .reindex(index=self.calendar, columns=self.tickers))


def prepare_panel(daily: pd.DataFrame, calendar: DateValues) -> ResearchPanel:
    """整理行情长表：每行是一只股票在一天的记录。

    daily 必需列：ticker、trade_date、open、high、low、close、vol、amount、adj_factor。
    calendar 是研究用的一维市场交易日历。先用完整行情定复权锚点，再按日历裁剪。
    """
    dates = pd.DatetimeIndex(parse_dates(calendar)).sort_values()
    # 确保交易日历不重复，非空
    if dates.has_duplicates or len(dates) == 0:
        raise ValueError("Calendar must have unique dates")
    data = daily.copy()
    # to_numpy把结果取成不带行标签的数组，按照当前位置进行赋值避免错位
    data["trade_date"] = parse_dates(data.trade_date).to_numpy()
    # 检查是否存在股票，交易日期存在重复。随后排序，先按股票代码排序，再对交易日期排序
    if data.ticker.isna().any() or data.duplicated(["ticker", "trade_date"]).any():
        raise ValueError("Missing ticker or duplicate ticker/date")
    data = data.sort_values(["ticker", "trade_date"]).reset_index(drop=True)
    # 如果行情日期处于本次交易日期工作区间内，却不在交易日历中就报错
    inside = data.trade_date.between(dates.min(), dates.max())
    if (inside & ~data.trade_date.isin(dates)).any():
        raise ValueError("Observed date inside the working interval is not a market day")
    
    # 此处开始处理行情数据
    columns = ["open", "high", "low", "close", "vol", "amount", "adj_factor"]
    data[columns] = data[columns].astype(float)
    # isfinite要求每一行中不能存在NaN或者正负无穷
    # 接着判断所有数值是否在合理范围内。le是小于，gt是大于。
    # 这里得到valid后不会因为某一行数据出问题直接删除全部
    valid = np.isfinite(data[columns]).all(axis=1) & data[columns].gt(0).all(axis=1)
    valid &= (data.low.le(data.open) & data.low.le(data.close)
              & data.high.ge(data.open) & data.high.ge(data.close))
    # 确定每一支股票第一次出现的时候的复权因子作为锚点，并检查数据是否有问题
    # 这里复权因子可以选在交易日历外
    anchor = data.ticker.map(data.drop_duplicates("ticker").set_index("ticker").adj_factor)
    if (~np.isfinite(anchor) | anchor.le(0)).any():
        raise ValueError("Invalid fixed adjustment anchor")
    data["adjustment_anchor"] = anchor
    # 计算复权价格
    for column in ["open", "high", "low", "close"]:
        data[column + "_adj"] = data[column] * data.adj_factor / anchor
    # Tushare 的成交额为千元，成交量为手，每手 100 股。
    vwap = 10 * data.amount / data.vol.where(valid)
    vwap = vwap.where(vwap.between(data.low - .01, data.high + .01))
    data["vwap_adj"] = vwap * data.adj_factor / anchor
    # 非observed的数据，要么数据本身有问题，要么交易量为0
    data["observation_status"] = np.where(valid, "observed", "invalid_input")
    data.loc[data.vol.eq(0), "observation_status"] = "no_trade"
    # 注意这里才根据交易日历截断数据
    data = data.loc[data["trade_date"].isin(dates.to_list()), :].copy()
    # 这里返回的ticker是所有daily数据中都出现过的ticker，不是只在交易日历内出现的ticker
    return ResearchPanel(data, dates, pd.Index(sorted(daily.ticker.unique()), name="ticker"))


def prepare_membership(membership: pd.DataFrame) -> pd.DataFrame:
    """整理成员长表，每行是某日的一个股票池成员。

    必需列：trade_date（信号日）、ticker（股票代码）、source_snapshot_date（快照日）。
    """
    data = membership.copy()
    for column in ["trade_date", "source_snapshot_date"]:
        data[column] = parse_dates(pd.Series(data[column])).to_numpy()
    if data.ticker.isna().any() or data.duplicated(["ticker", "trade_date"]).any():
        raise ValueError("Invalid membership key")
    # 这里确保取每个交易日时，应该采用历史快照，而不能使用未来的快照
    if (data.source_snapshot_date > data.trade_date).any():
        raise ValueError("Future snapshot used for membership")
    data["approximate_pit"] = True
    return data.sort_values(["trade_date", "ticker"]).reset_index(drop=True)


def execution_labels(panel: ResearchPanel, keys: pd.DataFrame, start: int, end: int) -> pd.DataFrame:
    """给逐股信号长表补上收益端点和复权收盘收益。

    start、end 表示相对信号日的交易日偏移，按市场日历定位价格。
    """
    date_pos = panel.calendar.get_indexer(keys.trade_date)
    ticker_pos = panel.tickers.get_indexer(keys.ticker)
    if (date_pos < 0).any() or (ticker_pos < 0).any() or not 0 < start < end:
        raise ValueError("Invalid label keys or offsets")
    close = panel.matrix("close_adj")
    calendar = pd.Series(panel.calendar, index=panel.calendar)
    start_close = close.shift(-start).to_numpy()[date_pos, ticker_pos]
    end_close = close.shift(-end).to_numpy()[date_pos, ticker_pos]
    return keys.assign(feature_date=keys.trade_date, label_start_date=keys.trade_date.map(calendar.shift(-start)),
                       label_end_date=keys.trade_date.map(calendar.shift(-end)), label=end_close / start_close - 1)


def segment_masks(rows: pd.DataFrame, segments: dict[str, list[str]]) -> dict[str, pd.Series]:
    """按阶段返回布尔掩码，要求信号日与收益终点同属该阶段且收益有效。"""
    boundaries = [(pd.Timestamp(start), pd.Timestamp(end)) for start, end in segments.values()]
    if any(lo > hi for lo, hi in boundaries) or any(a[1] >= b[0] for a, b in zip(boundaries, boundaries[1:])):
        raise ValueError("Research splits must be chronologically ordered and disjoint")
    return {name: rows.trade_date.between(lo, hi) & rows.label_end_date.le(hi) & np.isfinite(rows.label)
            for name, (lo, hi) in zip(segments, boundaries)}


def prepare_inputs(config: dict, *, need_industry: bool = True) -> pd.DatetimeIndex:
    """按依赖顺序准备日历、成员、行情，以及单因子需要的行业历史。"""
    from src.data.download import download_calendar, download_industry, download_prices
    from src.data.universe_process import main as build_members

    if not CSI500_INPUTS["calendar"].is_file():
        print("补充交易日历与历史成员快照", flush=True)
        download_calendar()
    dates = load_market_calendar(CSI500_INPUTS["calendar"])
    members_path = CSI500_INPUTS["membership"]
    members = pd.read_parquet(members_path, columns=["trade_date", "ticker"]) if members_path.is_file() else pd.DataFrame()
    if members.empty or not pd.DatetimeIndex(pd.to_datetime(members.trade_date).unique()).sort_values().as_unit("ns").equals(dates.as_unit("ns")):
        print("重建覆盖扩展训练期的历史成员表", flush=True)
        build_members()
        members = pd.read_parquet(members_path, columns=["trade_date", "ticker"])
    prices_path = CSI500_INPUTS["prices"]
    prices = pd.read_parquet(prices_path, columns=["trade_date", "ticker", "adj_factor"]) if prices_path.is_file() else pd.DataFrame()
    if (prices.empty or pd.to_datetime(prices.trade_date.astype(str)).min() > dates[0]
            or pd.to_datetime(prices.trade_date.astype(str)).max() < pd.Timestamp(config["segments"]["test"][1])
            or not set(members.ticker).issubset(prices.ticker)
            or prices.adj_factor.isna().any()):
        print("补充历史行情与复权因子，已完成的股票会复用下载检查点", flush=True)
        download_prices()
    if need_industry:
        path = CSI500_INPUTS["industries"]
        if not path.is_file() or path.stat().st_mtime_ns < members_path.stat().st_mtime_ns:
            download_industry()
    return dates


def prepare_test_panel(config: dict, market_dates: pd.DatetimeIndex) -> tuple[ResearchPanel, pd.DatetimeIndex]:
    """读取回测及传统因子预热所需行情，返回测试期交易日。"""
    start, end = config["segments"]["test"]
    dates = market_dates[market_dates.to_series().between(start, end).to_numpy()]
    if dates.empty:
        raise ValueError("测试期缺少交易日")
    first = market_dates.get_indexer(dates[:1])[0]
    working = market_dates[max(first - 21, 0):first + len(dates)]
    return prepare_panel(pd.read_parquet(CSI500_INPUTS["prices"]), working), dates
