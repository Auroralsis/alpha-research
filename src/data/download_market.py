import os
import tushare as ts
import pandas as pd
import time
from tqdm import tqdm

from pathlib import Path

project_root_dir = Path("/home/lxy/quant/alpha-research")
membership_data = project_root_dir / "data" / "processed" / "universe" / "csi500_membership.parquet"
daily_data = project_root_dir / "data" / "processed" / "daily" / "csi500_processed_daily.parquet"

pro = ts.pro_api(os.environ["TUSHARE_TOKEN"])
membership_df = pd.read_parquet(membership_data)
market_code = membership_df['ticker'].unique()

start_date = '20150901'
end_date = '20260228'

request_interval = 0.4

frames = []
adj_factor_frames = []

for code in tqdm(market_code, desc='下载股票历史数据'):
    daily = pro.daily(
        ts_code=code,
        start_date=start_date,
        end_date=end_date,
        fields='ts_code,trade_date,open,high,low,close,vol,amount',
    )
    frames.append(daily)
    time.sleep(request_interval)

    daily_adj_factor = pro.adj_factor(
        ts_code=code,
        start_date=start_date,
        end_date=end_date,
        fields='ts_code,trade_date,adj_factor',
    )
    adj_factor_frames.append(daily_adj_factor)
    time.sleep(request_interval)
    
temp_df = pd.concat(frames, ignore_index=True)
temp_adj_factor_df = pd.concat(adj_factor_frames, ignore_index=True)
daily_market_df = pd.merge(
    temp_df, 
    temp_adj_factor_df, 
    on=['ts_code', 'trade_date'], 
    how='left', 
    validate='one_to_one'
)

if pd.isna(daily_market_df).any().any():
    print("adj factor and df does not match")
daily_market_df = daily_market_df.rename(columns={'ts_code': 'ticker'})
daily_market_df.to_parquet(daily_data)