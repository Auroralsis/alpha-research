from dataclasses import dataclass

import numpy as np
import pandas as pd

def parse_dates(values):
    # 丢弃输入的旧标签并重新编号，不保存旧标签结果作为新列
    values = pd.Series(values).reset_index(drop=True)
    # 如果输入是日期类型直接转换，否则先转string再处理
    if pd.api.types.is_datetime64_any_dtype(values):
        result = pd.to_datetime(values, errors="raise")
    else:
        text = values.astype("string")
        # 检查values是否都是类似"20230103"的8位数字标准格式
        if not text.str.fullmatch(r"\d{8}").fillna(False).all():
            raise ValueError("Dates must be datetime values or exactly YYYYMMDD")
        # 注意这里%Y指4位年份，%m指月份，%d指日期
        result = pd.to_datetime(text, format="%Y%m%d", errors="raise")
    # 这里normalize把时间归到当天零点，保证数据没有非零的时分秒
    if result.isna().any() or (result != result.dt.normalize()).any():
        raise ValueError("Missing dates and intraday timestamps are not allowed")
    return result


@dataclass
class ResearchPanel:
    daily: pd.DataFrame
    calendar: pd.DatetimeIndex
    tickers: pd.Index

    def matrix(self, column, clean=True):
        data = self.daily
        values = data[column]
        if clean:
            # 满足条件保留原值，否则换成NaN。即将非observed的数据换成NaN
            values = values.where(data.observation_status.eq("observed"))
        # assign是往原两列数据中添加一列，随后pivot将其转换成矩阵
        # 其中index是行索引，columns是列索引，values是对应数值
        # reindex这里，使用全交易日日期和全股票，没有数据的部分自动补充NaN
        return (data[["ticker", "trade_date"]].assign(value=values).pivot(index="trade_date", columns="ticker", values="value")
                .reindex(index=self.calendar, columns=self.tickers))

    def observation_status(self):
        # 这里返回observation_status，fillna时填充"no_record_unknown"
        return (self.daily.pivot(index="trade_date", columns="ticker", values="observation_status")
                .reindex(index=self.calendar, columns=self.tickers).fillna("no_record_unknown"))
    

def prepare_panel(daily : pd.DataFrame, calendar):
    """在按照日历时间范围裁剪数据之前，先确定锚定基准，并保留进入时间窗口之前和退出时间窗口之后的历史数据。"""
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
    # 非observed的数据，要么数据本身有问题，要么交易量为0
    data["observation_status"] = np.where(valid, "observed", "invalid_input")
    data.loc[data.vol.eq(0), "observation_status"] = "no_trade"
    # 注意这里才根据交易日历截断数据
    data = data[data.trade_date.isin(dates)].copy()
    # 这里返回的ticker是所有daily数据中都出现过的ticker，不是只在交易日历内出现的ticker
    return ResearchPanel(data, dates, pd.Index(sorted(daily.ticker.unique()), name="ticker"))


def prepare_membership(membership):
    """处理历史股票名单"""
    data = membership.copy()
    for column in ["trade_date", "source_snapshot_date"]:
        data[column] = parse_dates(data[column]).to_numpy()
    if data.ticker.isna().any() or data.duplicated(["ticker", "trade_date"]).any():
        raise ValueError("Invalid membership key")
    # 这里确保取每个交易日时，应该采用历史快照，而不能使用未来的快照
    if (data.source_snapshot_date > data.trade_date).any():
        raise ValueError("Future snapshot used for membership")
    data["approximate_pit"] = True
    return data.sort_values(["trade_date", "ticker"]).reset_index(drop=True)
