import pandas as pd
from pathlib import Path

from ..features.momentum import calc_momentum
from ..features.targets import calc_future_return

project_root_dir = Path("/home/lxy/quant/alpha-research")
processed_daily_data = project_root_dir / "data" / "processed" / "daily" / "processed_2024_clean_csi"

factor_panel_dir = project_root_dir / "data" / "processed" / "factors"
factor_panel_dir.mkdir(parents=True, exist_ok=True)
factor_panel_data = factor_panel_dir / "factor_data"

df = pd.read_parquet(processed_daily_data)

df['mom_20'] = (df.groupby('ticker')['close_qfq']
                .transform(calc_momentum, 20))
df['mom_5'] = (df.groupby('ticker')['close_qfq']
               .transform(calc_momentum, 5))
df['mom_10'] = (df.groupby('ticker')['close_qfq']
                .transform(calc_momentum, 10))
df['mom_60'] = (df.groupby('ticker')['close_qfq']
                .transform(calc_momentum, 60))

df['future_ret_1'] = (df.groupby('ticker')['close_qfq']
                        .transform(calc_future_return, 1))
df['future_ret_5'] = (df.groupby('ticker')['close_qfq']
                        .transform(calc_future_return, 5))
df['future_ret_10'] = (df.groupby('ticker')['close_qfq']
                        .transform(calc_future_return, 10))
df['future_ret_20'] = (df.groupby('ticker')['close_qfq']
                        .transform(calc_future_return, 20))

df = df.dropna()
factor_df = df[['ticker', 'trade_date', 
                'mom_5', 'mom_10', 'mom_20', 'mom_60', 
                'future_ret_1', 'future_ret_5', 'future_ret_10', 'future_ret_20']].copy()
factor_df.to_parquet(factor_panel_data, index=False)