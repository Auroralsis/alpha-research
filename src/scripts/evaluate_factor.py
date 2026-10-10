import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ..evaluation.factor_metrics import calc_ic_rank
from ..evaluation.quantile_analysis import assign_quantiles, calc_quantile_returns

project_root_dir = Path("/home/lxy/quant/alpha-research")
factor_panel_dir = project_root_dir / "data" / "processed" / "factors"
factor_panel_dir.mkdir(parents=True, exist_ok=True)
factor_panel_data = factor_panel_dir / "factor_data"

# factor_df ['ticker', 'trade_date', 'close_qfq', 'mom_20', 'future_ret_5']
factor_df = pd.read_parquet(factor_panel_data)
rank_ic = (factor_df.groupby('trade_date')[['mom_20', 'future_ret_5']]
          .apply(lambda x : calc_ic_rank(x['mom_20'], x['future_ret_5'])))

print(rank_ic.describe())

mean_rank_ic = rank_ic.mean()
std_rank_ic = rank_ic.std()
rank_icir = mean_rank_ic / std_rank_ic
positive_rank_ic_ratio = sum(rank_ic > 0) / rank_ic.count()

# print(mean_rank_ic, std_rank_ic, rank_icir, positive_rank_ic_ratio)

factor_df['quantiles'] = factor_df.groupby('trade_date')['mom_20'].transform(assign_quantiles)
date_R_q = factor_df.groupby('trade_date').apply(calc_quantile_returns)

# here use unstack to turn double indexes series into dataframe
date_R_q = date_R_q.unstack(level='quantiles')
mean_R_q = date_R_q.mean()
spread_t = date_R_q[5.0] - date_R_q[1.0]
mean_spread_t = spread_t.mean()

print(date_R_q.describe())
print(rank_ic.describe())

# plt.plot(spread_t)
# plt.show()