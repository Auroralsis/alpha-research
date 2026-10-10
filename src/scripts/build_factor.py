import pandas as pd
from pathlib import Path

from ..features.momentum import calc_momentum
from ..features.targets import calc_future_return

project_root_dir = Path("/home/lxy/quant/alpha-research")
processed_daily_test_data = project_root_dir / "data" / "processed" / "daily" / "processed_2024_clean_test"

factor_panel_dir = project_root_dir / "data" / "processed" / "factors"
factor_panel_dir.mkdir(parents=True, exist_ok=True)
factor_panel_data = factor_panel_dir / "factor_data"

df = pd.read_parquet(processed_daily_test_data)

df['mom_20'] = (df.groupby('ticker')['close_qfq']
                .transform(lambda prices : calc_momentum(prices, 20)))
df['future_ret_5'] = (df.groupby('ticker')['close_qfq']
                        .transform(lambda prices : calc_future_return(prices, 5)))

factor_df = df[['ticker', 'trade_date', 'close_qfq', 'mom_20', 'future_ret_5']].copy()
factor_df.to_parquet(factor_panel_data)

