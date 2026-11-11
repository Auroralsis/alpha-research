import os
import time
import tushare as ts
import pandas as pd
from pathlib import Path

project_root_dir = Path("/home/lxy/quant/alpha-research")
raw_daily_dir = project_root_dir / "data" / "raw" / "daily"
# raw_daily_dir.mkdir(parents=True, exist_ok=True)
# raw_output_path = raw_daily_dir / "daily_2024_csi.parquet"

universe_raw_dir = project_root_dir / "data" / "raw" / "universe"
universe_raw_dir.mkdir(parents=True, exist_ok=True)
universe_raw_data = universe_raw_dir / "csi500_weight_snapshots.parquet"

calender_raw_dir = project_root_dir / "data" / "raw" / "calender"
calender_raw_dir.mkdir(parents=True, exist_ok=True)
calender_raw_data = calender_raw_dir / "trade_calender_2016_2025.parquet"

pro = ts.pro_api(os.environ["TUSHARE_TOKEN"])
trade_cal = pro.trade_cal(
    exchange='SSE',
    start_date='20160101',
    end_date='20251231',
    is_open=1
)

# 这里请求数据时必须按照每个月请求，最后concat
months = pd.period_range(
    start='20160101',
    end='20251231',
    freq='M'
)

csi_frame = []

# 注意这里的strftime的写法 %Y%m%d，大写表示四位，小写表示两位
for month in months:
    csi_frame.append(pro.index_weight(
        index_code='000905.SH',
        start_date=month.start_time.strftime("%Y%m%d"),
        end_date=month.end_time.strftime("%Y%m%d"),
        fields='con_code, trade_date, weight, index_code'
    ))
csi_df = pd.concat(csi_frame, ignore_index=True)
csi_df.to_parquet(universe_raw_data)
trade_cal.to_parquet(calender_raw_data)