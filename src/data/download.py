import os
import tushare as ts
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

project_root_dir = Path("/home/lxy/quant/alpha-research")
raw_daily_dir = project_root_dir / "data" / "raw" / "daily"
raw_daily_dir.mkdir(parents=True, exist_ok=True)

pro = ts.pro_api(os.environ["TUSHARE_TOKEN"])

codes = [
    "600519.SH",  # 贵州茅台
    "000001.SZ",  # 平安银行
    "600036.SH",  # 招商银行
    "000858.SZ",  # 五粮液
    "601318.SH",  # 中国平安
]

# 获取每日行情
df = pro.daily(ts_code=",".join(codes), start_date='20240101', end_date='20241231', 
               fields='ts_code, trade_date, open, high, low, close, vol, amount')

# 获取每日每支股票的复权因子，并合并
# 将df和adj_factor两个DataFrame合并，on表示依靠哪几列作为合并的依据，而how表示
# 如何合并，left表示依照左边的df合并，如果df有而adj_factor没有就填写nan，如果
# df没有而adj_factor有就直接丢弃。
adj_factor = pro.adj_factor(ts_code=",".join(codes), start_date='20240101', end_date='20241231')
df = pd.merge(df, adj_factor, on=['ts_code', 'trade_date'], how='left')

raw_output_path = raw_daily_dir / "daily_2024_test"

df.to_parquet(raw_output_path, 
              index=False)

# 查看基本信息
print(df.head(5))
print(df.info())

# isna检查是否有异常值，sum沿着列求和
print(df.isna().sum())

# duplicated检查特定的两列是否存在完全一致的情况，得到一个序列(Series)，随后求和得到重复的个数
print(df.duplicated(['ts_code', 'trade_date']).sum())

# groupby根据ts_code分组，得到一个DataFrameGroupBy的结构，使用size得到每个分组包含的个数
print(df.groupby("ts_code").size().sort_values())


