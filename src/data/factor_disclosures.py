"""按历史成分逐股缓存本次增量特征需要的 TuShare 原始版本。"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

from src.data.download import save_parquet, tushare_query
from src.data.research_panel import CSI500_INPUTS, ROOT

CACHE = ROOT / "data/raw/factor_extensions"
FIN = "ts_code,ann_date,f_ann_date,end_date,report_type,update_flag,"
SOURCES = {
    "eps": ("fina_indicator", "ts_code,ann_date,end_date,q_eps,update_flag", [None]),
    "income": ("income", FIN + "basic_eps,n_income_attr_p", ["2", "3"]),
    "equity": ("balancesheet", FIN + "total_hldr_eqy_exc_min_int", ["1", "4", "5"]),
    "limit": ("stk_limit", "ts_code,trade_date,up_limit,down_limit", [None]),
    "shares": ("daily_basic", "ts_code,trade_date,total_share", [None]),
}


def fetch(item):
    kind, ticker = item
    path = CACHE / kind / f"{ticker}.parquet"
    if path.exists():
        return
    api, fields, versions = SOURCES[kind]
    frames = []
    for version in versions:
        params = dict(ts_code=ticker, start_date="20100101" if version or kind == "eps" else "20130901", end_date="20251231")
        if version:
            params["report_type"] = version
        for attempt in range(6):
            try:
                rows = tushare_query(api, fields, **params)
                if kind == "eps":
                    pages = [rows]
                    while len(pages[-1]) == 100:
                        page = tushare_query(api, fields, offset=sum(map(len, pages)), **params)
                        if page.equals(pages[-1]):
                            raise ValueError(f"单季 EPS 接口未翻页 {ticker}")
                        pages.append(page)
                    rows = pd.concat([page for page in pages if len(page)], ignore_index=True)
                break
            except (requests.RequestException, ValueError):
                if attempt == 5:
                    raise
                time.sleep(min(60, 2 ** (attempt + 1)))
        if len(rows) >= (500 if version else 5000):
            raise ValueError(f"需要分页 {kind} {ticker}")
        if len(rows) and (not rows.ts_code.eq(ticker).all() or
                          (version and not rows.report_type.astype(str).eq(version).all())):
            raise ValueError(f"返回的股票或报表类型错误 {ticker}")
        frames.append(rows)
    save_parquet(pd.concat(frames, ignore_index=True), path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kinds", nargs="+", choices=SOURCES)
    args = parser.parse_args()
    members = pd.read_parquet(CSI500_INPUTS["membership"], columns=["ticker"])
    tickers = sorted(members.ticker.unique())
    for kind in args.kinds:
        with ThreadPoolExecutor(max_workers=4) as pool:
            for _ in pool.map(fetch, ((kind, ticker) for ticker in tickers)):
                pass
        print(kind, len(tickers), flush=True)


if __name__ == "__main__":
    main()
