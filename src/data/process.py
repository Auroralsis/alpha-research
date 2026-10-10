import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_root_dir = Path("/home/lxy/quant/alpha-research")
raw_daily_data = project_root_dir / "data" / "raw" / "daily" / "daily_2024_test"

interim_daily_dir = project_root_dir / "data" / "interim" / "daily"
interim_daily_dir.mkdir(parents=True, exist_ok=True)

processed_daily_dir = project_root_dir / "data" / "processed" / "daily"
processed_daily_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(raw_daily_data)

df['trade_date'] = pd.to_datetime(df['trade_date'])

# sort_values根据输入列名称，按照输入顺序依次进行升序排序
# drop_duplicates丢弃第一次出现之后再次重复的行
df = df.sort_values(['ts_code', 'trade_date'])
df = df.drop_duplicates(['ts_code', 'trade_date'])

interim_output_path = interim_daily_dir / "daily_2024_clean_test"
processed_output_path = processed_daily_dir / "processed_2024_clean_test"

df.to_parquet(interim_output_path,
              index=False)

df = df.rename(columns={'ts_code': 'ticker'})

# 这里df.groupby得到根据ticker分组的DataFrameGroupBy，取其中adj_factor的一列
# 得到SeriesGroupBy，这里之所以直接取last是因为之前是根据trade_date最后做了
# sort_value，所以每组的最后一个值就是每支股票最后一天的复权因子，用于计算前复权
# 因子。注意这里用了.transform，是为了得到和原DataFrame长度和位置匹配的复权因子。
last_factor = df.groupby('ticker')['adj_factor'].transform('last')

for col in ['open', 'high', 'low', 'close']:
    df[f'{col}_hfq'] = df[f'{col}'] * df['adj_factor']
    df[f'{col}_qfq'] = df[f'{col}_hfq'] / last_factor

df.to_parquet(processed_output_path,
              index=False)

# 选取000001.SZ的股票对应的行，以及对应close的列构成新的DataFrame，注意这里的
# 索引值不会重新调整，还是和原本的DataFrame中的索引一致。
# close_qfq = df.loc[df['ticker'] == '000001.SZ', 
#                    ['trade_date', 'close', 'close_qfq', 'adj_factor']].copy()

# 设置子图
# fig, axes = plt.subplots(
#     nrows=2,
#     ncols=1,
#     sharex=True,
#     figsize=(12,7)
# )

# close_qfq.plot(
#     x='trade_date',
#     y=['close', 'close_qfq'],
#     kind='line',
#     ax=axes[0]
# )
# axes[0].set_ylabel("Price")
# axes[0].legend(['Raw close', 'QFQ close'])
# axes[0].grid(alpha=0.3)

# close_qfq.plot(
#     x='trade_date',
#     y='adj_factor',
#     ax=axes[1],
#     kind='line',
#     color='purple'
# )
# axes[1].set_ylabel("Adjustment factor")
# axes[1].set_xlabel("Date")
# axes[1].grid(alpha=0.3)

# plt.tight_layout()
# plt.show()