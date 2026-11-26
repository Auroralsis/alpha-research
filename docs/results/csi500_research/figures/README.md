# 中证500研究报告绘图数据

每张图对应一份 SVG 和一份按日期排列的 CSV，可直接用 Excel 打开。2023-01-01 是图上的起点。其后 727 行为 2023-01-03 至 2025-12-31 的真实交易日，含首个交易日的指数收益。

| 图 | 图片 | 绘图数据 | 纵轴 |
|---|---|---|---|
| 图 1：七条模型与指数净值曲线 | [SVG](figure_1_model_nav.svg) | [CSV](figure_1_model_nav.csv) | 从 1 起按日收益复利累计，策略已扣费。 |
| 图 2：固定组合、三个单因子和指数 | [SVG](figure_2_factor_returns.svg) | [CSV](figure_2_factor_returns.csv) | 前五列数值为累计收益，后五列为累计超额收益，均以百分数表示。 |

图 2 的累计收益等于净值减一。累计超额收益为逐日累加的策略扣费收益率与中证500收益率之差，指数相对自身的超额为零。CSV 中的 1 表示 1%。

图 1 的固定组合和扩展窗口取自[固定与扩展组合账本](../sources/ml_study/daily_expanding_pair_20261122_133152_853377/ledger.parquet)，其余来源为[滑动组合账本](../sources/ml_study/daily_sliding_pair_20261122_150725_707868/ledger.parquet)、[LightGBM（MSE）与指数净值](../sources/ml_study/daily_xgboost_tuned_20261121_172019_402981/nav_comparison/daily_nav.csv)、[Ridge 账本](../sources/ml_study/daily_mad_ridge_20261122_145614_730197/ledger.parquet)和[MLP 账本](../sources/ml_study/mlp_report_median_20261126/ledger.parquet)。图 2 使用同一固定组合账本中的策略及基准日收益，并从[单因子账本](../sources/ml_study/daily_20261121_092926_223980/ledger.parquet)提取 Alpha3、Alpha13、Alpha44 的扣费日收益。
