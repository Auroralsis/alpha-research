import os
import tushare as ts
import pandas as pd
import time

from pathlib import Path

project_root_dir = Path("/home/lxy/quant/alpha-research")
membership_data = project_root_dir / "data" / "processed" / "universe" / "csi500_membership.parquet"
daily_data = project_root_dir / "data" / "processed" / "daily" / "csi500_processed_daily.parquet"

pro = ts.pro_api(os.environ["TUSHARE_TOKEN"])
membership_df = pd.read_parquet(membership_data)
market_code = membership_df['ticker'].unique()

months = pd.period_range(
    start='20150901',
    end='20260228',
    freq='M'
)

frames = []
adj_factor_frames = []
for month in months:
    for code in market_code:
        daily = pro.daily(
            ts_code=code, 
            start_date=month.start_time.strftime('%Y%m%d'),
            end_date=month.end_time.strftime('%Y%m%d'), 
            fields='ts_code, trade_date, open, high, low, close, vol, amount'
        )
        frames.append(daily)

        time.sleep(0.35)
        daily_adj_factor = pro.adj_factor(
            ts_code=code,
            start_date=month.start_time.strftime('%Y%m%d'),
            end_date=month.end_time.strftime('%Y%m%d'),
        )
        adj_factor_frames.append(daily_adj_factor)
temp_df = pd.concat(frames, ignore_index=True)
temp_adj_factor_df = pd.concat(adj_factor_frames, ignore_index=True)
daily_market_df = pd.merge(temp_df, temp_adj_factor_df, on=['ticker', 'trade_date'], how='left')

daily_market_df.to_parquet(daily_data)