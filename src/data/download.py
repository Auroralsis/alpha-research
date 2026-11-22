"""下载中证500交易日历、历史成分、行情复权因子与行业数据。"""

import argparse
import os
import time
from collections.abc import Callable, Iterable
from contextlib import ExitStack, contextmanager
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import requests
from tqdm import tqdm

from src.config import load_config
from src.data.research_panel import CSI500_INPUTS, ROOT

DATA_SETTINGS = load_config()["data"]
DATA_START = DATA_SETTINGS["start_date"]
DATA_END = DATA_SETTINGS["end_date"]
SNAPSHOT_FILE = ROOT / "data/raw/universe/csi500_weight_snapshots.parquet"
MARKET_END = DATA_SETTINGS["market_end_date"]
RAW_INDUSTRY = ROOT / "data/raw/industry"
SINA_URL = "https://vip.stock.finance.sina.com.cn/corp/go.php/vCI_CorpXiangGuan/stockid/{code}.phtml"


@contextmanager
def pending_file(path: Path):
    """先写同目录临时文件，成功后再替换目标文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, suffix=".pending", delete=False) as handle:
        pending = Path(handle.name)
    try:
        yield pending
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def save_parquet(frame: pd.DataFrame, path: Path) -> None:
    """原子保存单个数据表。"""
    with pending_file(path) as pending:
        frame.to_parquet(pending, index=False)


def write_parquet_batches(frames: Iterable[pd.DataFrame], path: Path) -> None:
    """按批流式保存 Parquet，所有批次成功后再发布实验文件。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    with pending_file(path) as pending, ExitStack() as files:
        writer = None
        for frame in frames:
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = files.enter_context(pq.ParquetWriter(pending, table.schema, compression="zstd"))
            writer.write_table(table)
        if writer is None:
            raise ValueError("没有可写入的 Parquet 批次")


def tushare_client():
    """为日历、成分和行情请求创建 TuShare 客户端。"""
    import tushare as ts

    return ts.pro_api(os.environ["TUSHARE_TOKEN"])


def download_calendar() -> None:
    """按月补齐成分股快照，成功后更新日历和合并快照。"""
    pro = tushare_client()
    calendar = pro.trade_cal(exchange="SSE", start_date=DATA_START, end_date=DATA_END, is_open=1)
    if calendar.empty:
        raise ValueError("交易日历为空")
    calendar = calendar.drop_duplicates("cal_date").sort_values("cal_date")
    # 多取一个月，供日历首个交易日向前查找当时已知的成员快照。
    months = pd.period_range(pd.Timestamp(DATA_START) - pd.DateOffset(months=1), DATA_END, freq="M")
    previous = pd.read_parquet(SNAPSHOT_FILE) if SNAPSHOT_FILE.exists() else pd.DataFrame()
    if not previous.empty:
        previous["trade_date"] = pd.to_datetime(previous.trade_date.astype(str)).dt.strftime("%Y%m%d")
    frames = []
    for month in months:
        checkpoint = SNAPSHOT_FILE.parent / f"csi500_weight_{month.strftime('%Y%m')}.parquet"
        if checkpoint.exists():
            frame = pd.read_parquet(checkpoint)
        else:
            frame = previous.loc[previous.trade_date.str.startswith(month.strftime("%Y%m"))].copy() if not previous.empty else previous
            if frame.empty:
                frame = pro.index_weight(
                    index_code="000905.SH",
                    start_date=month.start_time.strftime("%Y%m%d"),
                    end_date=month.end_time.strftime("%Y%m%d"),
                    fields="con_code,trade_date,weight,index_code",
                )
                time.sleep(DATA_SETTINGS["request_interval"])
            if frame.empty:
                raise ValueError(f"{month} 缺少成员快照，下载中止以便核查")
            save_parquet(frame, checkpoint)
        frames.append(frame)
        print(f"历史成员 {month}，{len(frame)} 行", flush=True)
    snapshots = pd.concat(frames, ignore_index=True)
    snapshots = snapshots.drop_duplicates(["con_code", "trade_date"]).sort_values(["trade_date", "con_code"])
    save_parquet(snapshots, SNAPSHOT_FILE)
    save_parquet(calendar, CSI500_INPUTS["calendar"])
    print(f"交易日历已保存到 {CSI500_INPUTS['calendar']}", flush=True)


def merge_prices(daily: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """按股票和交易日合并行情与复权因子，拒绝缺失复权因子。"""
    frame = daily.merge(factors, on=["ts_code", "trade_date"], how="left", validate="one_to_one")
    if frame.adj_factor.isna().any() or frame.adj_factor.le(0).any():
        raise ValueError("部分行情缺少有效复权因子，请检查下载响应")
    return frame.rename(columns={"ts_code": "ticker"}).sort_values(["ticker", "trade_date"])


def download_prices() -> None:
    """按股票保存下载检查点，完成后生成一份合并行情。"""
    pro = tushare_client()
    membership = pd.read_parquet(CSI500_INPUTS["membership"], columns=["ticker"])
    codes = sorted(membership.ticker.unique())
    raw = ROOT / "data/raw/daily"
    frames = []
    for code in tqdm(codes, desc="下载含早期成分股的历史行情"):
        checkpoint = raw / f"{code}_{DATA_START}_{MARKET_END}.parquet"
        if checkpoint.exists():
            frame = pd.read_parquet(checkpoint)
        else:
            daily = pro.daily(
                ts_code=code, start_date=DATA_START, end_date=MARKET_END,
                fields="ts_code,trade_date,open,high,low,close,vol,amount",
            )
            time.sleep(DATA_SETTINGS["request_interval"])
            factors = pro.adj_factor(
                ts_code=code, start_date=DATA_START, end_date=MARKET_END,
                fields="ts_code,trade_date,adj_factor",
            )
            time.sleep(DATA_SETTINGS["request_interval"])
            if daily.empty:
                raise ValueError(f"{code} 属于历史成员，但未下载到行情")
            frame = merge_prices(daily, factors)
            save_parquet(frame, checkpoint)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True).sort_values(["ticker", "trade_date"])
    if result.duplicated(["ticker", "trade_date"]).any():
        raise ValueError("合并行情包含重复股票日期")
    save_parquet(result, CSI500_INPUTS["prices"])
    print(f"行情已保存到 {CSI500_INPUTS['prices']}")


def tushare_query(api: str, fields: str = "", **params) -> pd.DataFrame:
    """通过 TuShare 官方 HTTPS 接口读取一批数据。"""
    response = requests.post("https://api.tushare.pro", timeout=30,
                             json={"api_name": api, "token": os.environ["TUSHARE_TOKEN"], "params": params, "fields": fields})
    response.raise_for_status()
    result = response.json()
    if result["code"] != 0:
        raise ValueError(f"TuShare {api} 请求失败，错误码 {result['code']}")
    time.sleep(DATA_SETTINGS["request_interval"])
    return pd.DataFrame(result["data"]["items"], columns=result["data"]["fields"])


def download_caps(config: dict | None = None) -> None:
    """按历史成员的实际成交日补齐每日总市值和流通市值，按股票缓存响应。"""
    dest = CSI500_INPUTS["caps"]
    columns = ["ts_code", "trade_date", "total_mv", "circ_mv"]
    existing = pd.read_parquet(dest) if dest.exists() else pd.DataFrame(columns=columns)
    members = pd.read_parquet(CSI500_INPUTS["membership"], columns=["ticker", "trade_date"])
    prices = pd.read_parquet(CSI500_INPUTS["prices"], columns=["ticker", "trade_date", "vol"])
    members["trade_date"] = pd.to_datetime(members.trade_date.astype(str)).dt.strftime("%Y%m%d")
    prices["trade_date"] = pd.to_datetime(prices.trade_date.astype(str)).dt.strftime("%Y%m%d")
    keys = members.merge(prices.loc[prices.vol.gt(0)], on=["trade_date", "ticker"]).rename(columns={"ticker": "ts_code"})
    segments = (config or load_config())["segments"]
    start, end = segments["train"][0], segments["test"][1]
    keys = keys.loc[keys.trade_date.between(start.replace("-", ""), end.replace("-", "")), columns[:2]]
    existing["trade_date"] = existing.trade_date.astype(str)
    missing = keys.merge(existing.loc[existing.total_mv.gt(0), columns[:2]], how="left", indicator=True)
    groups = missing.loc[missing._merge.eq("left_only")].groupby("ts_code")
    parts = []
    for ticker, group in tqdm(groups, desc="补齐历史市值"):
        begin, end = group.trade_date.min(), group.trade_date.max()
        cache = ROOT / "data/raw/daily_basic" / f"{ticker}_{begin}_{end}.parquet"
        if cache.exists():
            frame = pd.read_parquet(cache)
        else:
            frame = tushare_query("daily_basic", fields=",".join(columns), ts_code=ticker, start_date=begin, end_date=end)
            if frame.empty or not frame.ts_code.eq(ticker).all() or frame.duplicated(columns[:2]).any():
                raise ValueError(f"{ticker} 历史市值响应为空或股票日期键无效")
            save_parquet(frame, cache)
        parts.append(group[columns[:2]].merge(frame, on=columns[:2], validate="one_to_one"))
    if parts:
        combined = pd.concat([existing, *parts], ignore_index=True).drop_duplicates(columns[:2], keep="last")
        save_parquet(combined, dest)


def fetch_tushare() -> pd.DataFrame:
    """按一级行业下载当前与历史记录，保留原始响应中的分类和生效区间。"""
    classes = tushare_query("index_classify", level="L1", src="SW2021")
    if classes.empty or classes.index_code.duplicated().any():
        raise ValueError("申万一级行业目录为空或重复")
    frames = []
    for code in classes.index_code:
        for flag in ("Y", "N"):
            frame = tushare_query("index_member_all", l1_code=code, is_new=flag)
            if len(frame) >= 2000 or (flag == "Y" and frame.empty):
                raise ValueError(f"{code}/{flag} 返回空的当前成分或达到接口上限，请核查")
            if not frame.l1_code.eq(code).all() or not frame.is_new.eq(flag).all():
                raise ValueError(f"{code}/{flag} 接口返回了其他行业或状态")
            frames.append(frame)
    history = pd.concat(frames, ignore_index=True).drop_duplicates()
    save_parquet(history, RAW_INDUSTRY / "tushare_sw_history.parquet")
    return history


def fetch_sina(ticker: str) -> pd.DataFrame:
    """读取指定股票的指数历史成分表，保留来源给出的进入和退出日期。"""
    code = ticker.split(".")[0]
    url = SINA_URL.format(code=code)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    html = response.content.decode("gb18030")
    if f"({code})相关资料" not in html:
        raise ValueError(f"{ticker} 历史成分网页与请求股票不符")
    tables = pd.read_html(StringIO(html), match="进入日期", flavor="lxml")
    for table in tables:
        if table.iloc[0].tolist() == ["指数名称", "指数代码", "进入日期", "退出日期"]:
            records = table.iloc[1:].copy()
            records.columns = ["index_name", "index_code", "in_date", "out_date"]
            records["index_code"] = records.index_code.astype(str)
            return records
    raise ValueError(f"{ticker} 缺少可识别的历史成分表")


def supplement_history(
    members: pd.DataFrame, history: pd.DataFrame, load_sina: Callable[[str], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐日检查历史成分股，只在主源没有生效记录且次源行业唯一时补一级代码。"""
    keys = members[["ticker", "trade_date"]].reset_index(drop=True)
    primary = history.loc[history.ts_code.isin(keys.ticker.unique())].copy()
    dates = primary[["ts_code", "l1_code", "in_date", "out_date"]].copy()
    for col in ("in_date", "out_date"):
        dates[col] = pd.to_datetime(dates[col], format="%Y%m%d", errors="raise")
    if dates.in_date.isna().any() or dates.l1_code.isna().any() or dates.out_date.le(dates.in_date).any():
        raise ValueError("TuShare 行业缺失或生效区间无效")
    joined = keys.reset_index().merge(dates, left_on="ticker", right_on="ts_code", how="inner")
    active = joined.trade_date.ge(joined.in_date) & (joined.out_date.isna() | joined.trade_date.lt(joined.out_date))
    counts = joined.loc[active].groupby("index").l1_code.nunique().reindex(keys.index, fill_value=0)
    audit = keys.loc[counts.ne(1)].copy()
    audit["status"] = counts.loc[audit.index].map(lambda n: "主源行业冲突" if n else "无历史一级行业记录")
    for col in ("l1_code", "evidence_in", "evidence_out"):
        audit[col] = pd.Series(pd.NA, index=audit.index, dtype="string")
    names = history.drop_duplicates("l1_code").set_index("l1_code").l1_name
    for ticker, group in audit.loc[counts.eq(0)].groupby("ticker"):
        records = load_sina(ticker).copy()
        records["l1_code"] = records.index_code + ".SI"
        records = records.loc[records.l1_code.isin(names.index) & records.in_date.notna()
                              & records.in_date.ne("1900-01-01")].copy()
        for col in ("in_date", "out_date"):
            records[col] = pd.to_datetime(records[col], format="%Y-%m-%d", errors="raise")
        matches = group[["trade_date"]].reset_index().merge(records, how="cross")
        matches = matches.loc[matches.trade_date.ge(matches.in_date)
                              & (matches.out_date.isna() | matches.trade_date.lt(matches.out_date))]
        distinct = matches.groupby("index").l1_code.nunique()
        audit.loc[distinct[distinct.gt(1)].index, "status"] = "次源行业冲突"
        unique = matches.loc[matches["index"].isin(distinct[distinct.eq(1)].index)]
        unique = unique.sort_values("in_date").drop_duplicates("index", keep="last").set_index("index")
        audit.loc[unique.index, "status"] = "已补一级行业"
        audit.loc[unique.index, "l1_code"] = unique.l1_code
        audit.loc[unique.index, "evidence_in"] = unique.in_date.dt.strftime("%Y%m%d")
        audit.loc[unique.index, "evidence_out"] = unique.out_date.dt.strftime("%Y%m%d")
    audit["source_url"] = audit.ticker.map(lambda s: SINA_URL.format(code=s.split(".")[0]))
    audit.loc[audit.status.eq("主源行业冲突"), "source_url"] = pd.NA
    filled = audit.loc[audit.status.eq("已补一级行业")].sort_values(["ticker", "trade_date"])
    primary["source"] = "TuShare index_member_all"
    if not filled.empty:
        groups = ["ticker", "l1_code", "source_url", "evidence_in", "evidence_out"]
        # 仅合并相邻自然日，保证补充区间不会跨过主源记录或未经核验的日期。
        runs = filled.groupby(groups, dropna=False).trade_date.diff().ne(pd.Timedelta(days=1)).cumsum()
        extra = filled.groupby([*groups, runs.rename("run")], dropna=False).trade_date.agg(["min", "max"]).reset_index()
        extra["in_date"] = extra["min"].dt.strftime("%Y%m%d")
        extra["out_date"] = (extra["max"] + pd.Timedelta(days=1)).dt.strftime("%Y%m%d")
        extra["l1_name"] = extra.l1_code.map(names)
        extra["source"] = "新浪历史指数成分"
        primary = pd.concat([primary, extra.rename(columns={"ticker": "ts_code"}).drop(columns=["min", "max", "run"])],
                            ignore_index=True)
    return primary, audit


def download_industry() -> None:
    """下载全股票池的行业历史，将补齐的正式输入和缺口核验明细写入 data。"""
    def fresh(path: Path) -> bool:
        return path.is_file() and datetime.fromtimestamp(path.stat().st_mtime).date() == date.today()

    raw_path = RAW_INDUSTRY / "tushare_sw_history.parquet"
    history = pd.read_parquet(raw_path) if fresh(raw_path) else fetch_tushare()
    members = pd.read_parquet(CSI500_INPUTS["membership"], columns=["ticker", "trade_date"])

    def load_sina(ticker: str) -> pd.DataFrame:
        path = RAW_INDUSTRY / "sina" / f"{ticker}.parquet"
        if fresh(path):
            return pd.read_parquet(path)
        records = fetch_sina(ticker)
        save_parquet(records, path)
        time.sleep(DATA_SETTINGS["request_interval"])
        return records

    result, audit = supplement_history(members, history, load_sina)
    dest = CSI500_INPUTS["industries"]
    save_parquet(audit, dest.with_name("industry_gap_audit.parquet"))
    save_parquet(result, dest)
    print(f"历史行业已保存到 {dest}。{audit.status.value_counts().to_dict()}", flush=True)


def main(argv: list[str] | None = None) -> None:
    """默认按依赖顺序准备全部下载数据，也可单独运行一个环节。"""
    from src.data.universe_process import main as rebuild_membership

    steps = {"calendar": download_calendar, "membership": rebuild_membership,
             "prices": download_prices, "industry": download_industry}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("part", nargs="?", default="all", choices=["all", *steps, "caps"],
                        help="默认全部。calendar 含成分快照，membership 重建逐日成员表，prices 含复权因子。")
    part = parser.parse_args(argv).part
    if part == "caps":
        download_caps()
        return
    for name, run in steps.items():
        if part in ("all", name):
            run()


if __name__ == "__main__":
    main()
