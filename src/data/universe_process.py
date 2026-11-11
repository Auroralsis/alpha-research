import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_root_dir = Path("/home/lxy/quant/alpha-research")
universe_raw_data = project_root_dir / "data" / "raw" / "universe" / "csi500_weight_snapshots.parquet"
calender_raw_data = project_root_dir / "data" / "raw" / "calender" / "trade_calender_2016_2025.parquet"
membership_data = project_root_dir / "data" / "processed" / "universe" / "csi500_membership.parquet"

df = pd.read_parquet(universe_raw_data)
trade_cal_df = pd.read_parquet(calender_raw_data)
trade_date = pd.to_datetime(trade_cal_df['cal_date'])
trade_date = trade_date.dropna().drop_duplicates()
trade_date = trade_date.to_frame(name='trade_date').sort_values('trade_date')

df = df.rename(columns={'con_code' : 'ticker', 'trade_date' : 'snapshot_date'})
df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
df = df.dropna().drop_duplicates()
snapshot_date = df[['snapshot_date']].sort_values('snapshot_date')

merge_date_df = (
    pd.merge_asof(trade_date, snapshot_date, direction='backward', left_on='trade_date', right_on='snapshot_date')
    .sort_values(['trade_date'])
).rename(columns={'snapshot_date': 'source_snapshot_date'})

result_df = (pd.merge(df, merge_date_df, how='inner', left_on='snapshot_date', right_on='source_snapshot_date')
             .drop(columns=['snapshot_date', 'index_code'])
             .reset_index(drop=True)
             .sort_values('trade_date')
)
result_df.to_parquet(membership_data)