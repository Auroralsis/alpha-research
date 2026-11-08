import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ..evaluation.factor_metrics import calc_ic_rank
from ..evaluation.quantile_analysis import assign_quantiles, calc_quantile_returns

project_root_dir = Path("/home/lxy/quant/alpha-research")
factor_panel_dir = project_root_dir / "data" / "processed" / "factors"
factor_panel_dir.mkdir(parents=True, exist_ok=True)
factor_panel_data = factor_panel_dir / "factor_data"

mom_ret_rank_ic = project_root_dir / "results" / "tables" / "mom_ret_rank_ic.csv"
mom20_ret5_quantile_returns = project_root_dir / "results" / "tables" / "mom20_ret5_quantile_returns.csv"
mom_ret_summary = project_root_dir / "results" / "tables" / "mom_ret_summary.csv"

# factor_df = df[['ticker', 'trade_date', 'mom_5', 'mom_10', 'mom_20', 'mom_60', 'future_ret_5']].copy()
factor_df = pd.read_parquet(factor_panel_data)

mom_type = ['mom_5', 'mom_10', 'mom_20', 'mom_60']
future_ret_type = ['future_ret_1', 'future_ret_5', 'future_ret_10', 'future_ret_20']

mean_rank_ic_df = pd.DataFrame(
    index=mom_type,
    columns=future_ret_type,
    dtype=float
)

std_rank_ic_df = pd.DataFrame(
    index=mom_type,
    columns=future_ret_type,
    dtype=float
)

for mom in mom_type:
    for future_ret in future_ret_type:
        mean_rank_ic_df.loc[mom, future_ret] = (
            (factor_df.groupby('trade_date')[[mom, future_ret]]
             .apply(lambda x: calc_ic_rank(x[mom], x[future_ret])))
             .mean()
        )
        std_rank_ic_df.loc[mom, future_ret] = (
            (factor_df.groupby('trade_date')[[mom, future_ret]]
             .apply(lambda x: calc_ic_rank(x[mom], x[future_ret])))
             .std()
        )

mean_rank_ic_df.to_csv(mom_ret_rank_ic)

date_R_q_df = pd.DataFrame(
    index=mom_type,
    columns=future_ret_type,
    dtype=object
)

for mom in mom_type:
    factor_df['quantile_' + mom] = (
        factor_df.groupby('trade_date')[mom]
        .transform(assign_quantiles)
    )
    for future_ret in future_ret_type:
        date_R_q_df.at[mom, future_ret] = (
            factor_df.groupby('trade_date')
            .apply(calc_quantile_returns, 'quantile_' + mom, future_ret)
        )

mean_Q5_Q1_df = pd.DataFrame(
    index=mom_type,
    columns=future_ret_type,
    dtype=float
)

for mom in mom_type:
    for future_ret in future_ret_type:
        mean_Q5_Q1_df.loc[mom, future_ret] = (
            ((date_R_q_df.loc[mom, future_ret].mean())[5.0] 
             - (date_R_q_df.loc[mom, future_ret].mean())[1.0])
        )

mean_Q5_Q1_df.to_csv(mom_ret_summary)

# plt.plot(spread_t)
# plt.show()