# alpha-research

在 2023—2025 年动态中证500上，以复权行情计算 Alpha3、Alpha13 和 Alpha44，做每日加权横截面回归、5/10/20 日 Rank IC 和 20 层等权回测。回测使用下一交易日的复权 VWAP 买入、复权收盘价卖出。

```bash
python -B -m src.scripts.run_factor_study
python -B -m unittest discover -s tests -v
```

[统一研究报告](docs/ml_study.md) · [固定参数](results/factor_study/design.json) · [因子评价汇总](results/factor_study/factor_summary.csv) · [分层回测汇总](results/factor_study/portfolio_summary.csv)

流程离线运行，更新统一报告中的传统单因子章节，数值结果和图表写入 `results/factor_study/`。项目依赖见 [requirements.txt](requirements.txt)。运行需要以下固定输入：

- `data/processed/daily/csi500_processed_daily.parquet`：行情。
- `data/processed/universe/csi500_membership.parquet`：历史成员名单。
- `data/raw/calender/trade_calender_2016_2025.parquet`：独立交易日历。
- [辅助输入](data/reference/csi500/)：历史行业、市值、涨跌停价和中证500价格指数。[数据来源与原始请求记录](data/reference/csi500/provenance.json)随输入保存。

主要模块：[因子](src/features/price_volume.py)、[收益标签](src/features/targets.py)、[每日横截面评价与汇总](src/evaluation/cross_section.py)、[分层回测](src/evaluation/layer_backtest.py)与[交易账本](src/evaluation/portfolio.py)。

## 共用回测接口

传统分层策略按因子分数生成每日分层名单，机器学习策略按模型分数和实际持仓生成 Top-50 换股名单。两者都调用 `backtest_long_only(quotes, signals, fee_bps=..., rebalance_every=..., offset=...)`。其中 `quotes` 包含交易日 × 股票的行情表，`signals` 是信号日到股票名单的字典。返回值包含每日账本、成交记录和期末持仓，`performance()` 从账本计算收益、风险、换手率和月胜率。[共用引擎](src/evaluation/portfolio.py) · [分层策略](src/evaluation/layer_backtest.py) · [Top-50 策略](src/ml/strategy.py)

成员和行业记录按可取得的历史快照重建，统计端用次日正常报价近似筛选停牌。回测使用日频行情，并省略集合竞价流动性、冲击、整手、公司行动现金流和退市结算。

## 中证500模型与单因子对比

ML 研究使用本地中证500历史行情和成分股。首次可用 `conda create -n quant python=3.10 pip -y` 创建环境。依赖安装后，从项目根目录运行 `run_all` 即可生成统一对比。

```bash
conda activate quant
python -m pip install -r requirements.txt
python -B -m src.scripts.run_all
```

固定 LightGBM 和 Ridge 使用 2016–2020 年训练、2021–2022 年验证、2023–2025 年测试。滚动 LightGBM 复用同一批 Alpha158 特征，逐窗训练后合并预测并做一次连续周频回测。报告统一比较两种 LightGBM、Ridge、Alpha13、Alpha3、Alpha44 和中证500价格指数。各策略每 5 个交易日执行一次 Top-50 调仓，每次最多计划换出 5 只，费用为买入 5 bp、卖出 15 bp。[训练配置](configs/ml_qlib.json) · [统一研究报告](docs/ml_study.md)

每次完整运行在 `results/ml_study/` 下建一个时间戳目录。该目录中的 `portfolio_metrics.csv`、`rank_ic_summary.csv`、`quintile_summary.csv` 和账本明细都包含各方法，用 `strategy` 区分。逐股分数统一存成 `predictions.parquet`，`window` 可筛选滚动模型的预测窗口。完整运行成功后清理 Alpha158 中间特征表，模型文件、训练日志和指标可在本次实验目录查看。

用 `python -B -m src.scripts.run_all --stage evaluate` 可用最近一次完整实验的预测刷新回测和报告。加 `--out 实验目录` 可指定已有实验，或在完整运行时选择一个新的目录。报告统一写到 `docs/ml_study.md`。

统一报告同时纳入已有的传统 20 层因子研究结果。重新计算这一部分时运行 `run_factor_study`。
