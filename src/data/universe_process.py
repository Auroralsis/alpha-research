import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_root_dir = Path("/home/lxy/quant/alpha-research")
# raw_daily_data = project_root_dir / "data" / "raw" / "daily" / "daily_2024_csi"

# interim_daily_dir = project_root_dir / "data" / "interim" / "daily"
# interim_daily_dir.mkdir(parents=True, exist_ok=True)

# processed_daily_dir = project_root_dir / "data" / "processed" / "daily"
# processed_daily_dir.mkdir(parents=True, exist_ok=True)s

# df = pd.read_parquet(raw_daily_data)

# df['trade_date'] = pd.to_datetime(df['trade_date'])

# # sort_values根据输入列名称，按照输入顺序依次进行升序排序
# # drop_duplicates丢弃第一次出现之后再次重复的行
# df = df.sort_values(['ts_code', 'trade_date'])
# df = df.drop_duplicates(['ts_code', 'trade_date'])

# interim_output_path = interim_daily_dir / "daily_2024_clean_csi"
# processed_output_path = processed_daily_dir / "processed_2024_clean_csi"

# df.to_parquet(interim_output_path,
#               index=False)

# df = df.rename(columns={'ts_code': 'ticker'})

# # 这里df.groupby得到根据ticker分组的DataFrameGroupBy，取其中adj_factor的一列
# # 得到SeriesGroupBy，这里之所以直接取last是因为之前是根据trade_date最后做了
# # sort_value，所以每组的最后一个值就是每支股票最后一天的复权因子，用于计算前复权
# # 因子。注意这里用了.transform，是为了得到和原DataFrame长度和位置匹配的复权因子。
# last_factor = df.groupby('ticker')['adj_factor'].transform('last')

# for col in ['open', 'high', 'low', 'close']:
#     df[f'{col}_hfq'] = df[f'{col}'] * df['adj_factor']
#     df[f'{col}_qfq'] = df[f'{col}_hfq'] / last_factor

# df.to_parquet(processed_output_path,
#               index=False)

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