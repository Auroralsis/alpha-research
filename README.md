# alpha-research

在 2023—2025 年动态中证500上，按原研报的方法检验 Alpha3、Alpha13、Alpha44：每日加权横截面回归、5/10/20日 Rank IC，以及次日收盘换仓的20层等权回测。

```bash
python -B -m src.scripts.run_factor_study
python -B -m unittest discover -s tests -v
```

[研究报告](docs/factor_study.md) · [固定参数](results/factor_study/design.json) · [因子评价汇总](results/factor_study/factor_summary.csv) · [分层回测汇总](results/factor_study/portfolio_summary.csv)

流程离线运行，报告写入 `docs/`，数值结果和图表写入 `results/factor_study/`。依赖见 [requirements-research.txt](requirements-research.txt)。运行需要以下固定输入：

- `data/processed/daily/csi500_processed_daily.parquet`：行情。
- `data/processed/universe/csi500_membership.parquet`：历史成员名单。
- `data/raw/calender/trade_calender_2016_2025.parquet`：独立交易日历。
- [辅助输入](data/reference/csi500/)：历史行业、市值、涨跌停价和中证500价格指数。[数据来源与原始请求记录](data/reference/csi500/provenance.json)随输入保存。

主要模块：[因子](src/features/price_volume.py)、[收益标签](src/features/targets.py)、[每日横截面评价](src/evaluation/cross_section.py)、[跨日期推断](src/evaluation/inference.py)、[分层回测](src/evaluation/layer_backtest.py)与[交易账本](src/evaluation/portfolio.py)。

主要限制：成员和行业记录不是严格按原始公告时间重建；统计端用次日正常报价近似筛选停牌。回测未重建集合竞价流动性、冲击、整手、公司行动现金流或退市结算。
