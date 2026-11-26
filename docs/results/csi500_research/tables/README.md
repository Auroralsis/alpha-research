# 中证500日频选股研究表格数据

每个 CSV 对应报告中的一张表，行列顺序与报告一致。标有（%）的列按百分数保存，例如 27.30 表示 27.30%。换手单位为倍，手续费单位为万元。空单元格对应报告中的“—”。文件采用 UTF-8 BOM 编码，可直接用 Excel 打开。

## 表 1　三个单因子处理前后的单日收益预测及扣费回测

[table_1_factors.csv](table_1_factors.csv)，共 6 行。

- [factors/daily_factor_processing_20261122_142859_680463/paired_ic.csv](../sources/factors/daily_factor_processing_20261122_142859_680463/paired_ic.csv)
- [factors/daily_factor_processing_20261122_142859_680463/portfolio_metrics.csv](../sources/factors/daily_factor_processing_20261122_142859_680463/portfolio_metrics.csv)

## 表 2　回归模型的验证误差、单日收益预测及扣费回测

[table_2_regression.csv](table_2_regression.csv)，共 2 行。

- [ml_study/daily_mad_ridge_20261122_145614_730197/ridge_trials.csv](../sources/ml_study/daily_mad_ridge_20261122_145614_730197/ridge_trials.csv)
- [ml_study/daily_mad_ridge_20261122_145614_730197/rank_ic_summary.csv](../sources/ml_study/daily_mad_ridge_20261122_145614_730197/rank_ic_summary.csv)
- [ml_study/daily_mad_ridge_20261122_145614_730197/portfolio_metrics.csv](../sources/ml_study/daily_mad_ridge_20261122_145614_730197/portfolio_metrics.csv)
- [ml_study/daily_alpha158_comparison_20261121_105840_864298/preprocessing_test/sampling_mse/summary.csv](../sources/ml_study/daily_alpha158_comparison_20261121_105840_864298/preprocessing_test/sampling_mse/summary.csv)

## 表 3　四个排序目标的收益分档与增益设置

[table_2_ranking_gains.csv](table_2_ranking_gains.csv)，共 4 行。

- [ml_study/daily_expanding_pair_20261122_133152_853377/config.json](../sources/ml_study/daily_expanding_pair_20261122_133152_853377/config.json)

## 表 4　预测前 50 名单中各真实收益组的占比

[table_3_top50_hits.csv](table_3_top50_hits.csv)，共 6 行。

树模型按每天可评价股票的真实收益排名分组，同收益按股票代码排序。从每个模型的预测前 50 只中计算四项占比，再对 725 个交易日取均值。其他方法读取对应归档中的测试期汇总。

- [ml_study/daily_pair_tuning_20261122_100328_550844/predictions.parquet](../sources/ml_study/daily_pair_tuning_20261122_100328_550844/predictions.parquet)
- [ml_study/daily_expanding_pair_20261122_133152_853377/predictions.parquet](../sources/ml_study/daily_expanding_pair_20261122_133152_853377/predictions.parquet)
- [ml_study/daily_sliding_pair_20261122_150725_707868/predictions.parquet](../sources/ml_study/daily_sliding_pair_20261122_150725_707868/predictions.parquet)
- [ml_study/daily_mlp_lambdarank_20261122_131215_786064/hits_summary.csv](../sources/ml_study/daily_mlp_lambdarank_20261122_131215_786064/hits_summary.csv)

## 表 5　两种滚动训练的样本区间、扣费年度收益和名单重合度

[table_4_windows.csv](table_4_windows.csv)，共 3 行。

- [ml_study/daily_expanding_pair_20261122_133152_853377/2023/periods.csv](../sources/ml_study/daily_expanding_pair_20261122_133152_853377/2023/periods.csv)
- [ml_study/daily_sliding_pair_20261122_150725_707868/2023/periods.csv](../sources/ml_study/daily_sliding_pair_20261122_150725_707868/2023/periods.csv)
- [ml_study/daily_sliding_pair_20261122_150725_707868/window_comparison.csv](../sources/ml_study/daily_sliding_pair_20261122_150725_707868/window_comparison.csv)
- [ml_study/daily_expanding_pair_20261122_133152_853377/2024/periods.csv](../sources/ml_study/daily_expanding_pair_20261122_133152_853377/2024/periods.csv)
- [ml_study/daily_sliding_pair_20261122_150725_707868/2024/periods.csv](../sources/ml_study/daily_sliding_pair_20261122_150725_707868/2024/periods.csv)
- [ml_study/daily_expanding_pair_20261122_133152_853377/2025/periods.csv](../sources/ml_study/daily_expanding_pair_20261122_133152_853377/2025/periods.csv)
- [ml_study/daily_sliding_pair_20261122_150725_707868/2025/periods.csv](../sources/ml_study/daily_sliding_pair_20261122_150725_707868/2025/periods.csv)

## 表 6　中证500日频选股结果（2023–2025 年）

[table_5_performance.csv](table_5_performance.csv)，共 12 行。

- [ml_study/daily_pair_tuning_20261122_100328_550844/comparison.csv](../sources/ml_study/daily_pair_tuning_20261122_100328_550844/comparison.csv)
- [ml_study/daily_expanding_pair_20261122_133152_853377/comparison.csv](../sources/ml_study/daily_expanding_pair_20261122_133152_853377/comparison.csv)
- [ml_study/daily_sliding_pair_20261122_150725_707868/comparison.csv](../sources/ml_study/daily_sliding_pair_20261122_150725_707868/comparison.csv)
- [ml_study/daily_xgboost_tuned_20261121_172019_402981/nav_comparison/metrics.csv](../sources/ml_study/daily_xgboost_tuned_20261121_172019_402981/nav_comparison/metrics.csv)
- [ml_study/daily_mlp_lambdarank_20261122_131215_786064/portfolio_metrics.csv](../sources/ml_study/daily_mlp_lambdarank_20261122_131215_786064/portfolio_metrics.csv)
- [ml_study/daily_mlp_lambdarank_20261122_131215_786064/rank_ic_summary.csv](../sources/ml_study/daily_mlp_lambdarank_20261122_131215_786064/rank_ic_summary.csv)
- [ml_study/daily_mad_ridge_20261122_145614_730197/portfolio_metrics.csv](../sources/ml_study/daily_mad_ridge_20261122_145614_730197/portfolio_metrics.csv)
- [ml_study/daily_mad_ridge_20261122_145614_730197/rank_ic_summary.csv](../sources/ml_study/daily_mad_ridge_20261122_145614_730197/rank_ic_summary.csv)
- [ml_study/daily_20261121_092926_223980/portfolio_metrics.csv](../sources/ml_study/daily_20261121_092926_223980/portfolio_metrics.csv)
- [ml_study/daily_20261121_092926_223980/rank_ic_summary.csv](../sources/ml_study/daily_20261121_092926_223980/rank_ic_summary.csv)
