# alpha-research

本项目研究历史中证500成分股的日频多头选股。在同一股票池、成交规则和费率下，比较量价单因子、收益回归和截面排序三种生成选股信号的方法。研究既检验分数与下一期收益的关联，也跟踪实际持仓、换手和扣费净值。

**[阅读研究报告《中证500多头选股策略研究——量价单因子、收益回归与截面排序的实证比较》](research.md)** · [报告表格与数据来源](docs/results/csi500_research/tables/README.md) · [净值图与逐日数据](docs/results/csi500_research/figures/README.md)

## 研究内容

| 方法 | 比较内容 |
| --- | --- |
| 量价单因子 | 分别用 Alpha3、Alpha13 和 Alpha44 选股，比较原始分数与行业、市值中性化分数的 Rank IC、收益和回撤。 |
| 收益回归 | Ridge 与 LightGBM 使用相同的 Alpha158 特征和逐日标准化收益标签，按预测值选股。 |
| 截面排序 | LightGBM、XGBoost 和 MLP 各训练四个排序器，分别强调选入真实前 50、前 100 或避开后 50、后 100。八模型组合将四个 LightGBM 和四个 XGBoost 的当日排名百分位各赋予 12.5% 权重，并与单算法及逐年重训方案比较。 |

研报测试期为 2023–2025 年。以交易日 T 的收盘信息打分，按 T+1 收盘价模拟成交，预测标签为 T+1 至 T+2 的复权收盘收益。Qlib 的 `TopkDropoutStrategy` 目标持有 50 只股票，每日计划替换 5 只。初始资金 1 亿元，买入和卖出费率分别为 5、15 个基点，每笔最低 5 元，基准为中证500价格指数。[研究配置](configs/research.json) · [回测实现](src/backtest/engine.py)

## 实验结果

2023–2025 年，按 **2022 年组合排序损失**选参的固定八模型组合扣费累计收益为 **70.41%**，同期指数上涨 **27.30%**。该组合先用 2014–2020 年训练，以 2021 年确定各排序器的迭代轮数，再以 2022 年八模型融合后的平均 `1 − NDCG@50` 选择树参数。[选参记录](docs/results/csi500_research/sources/ml_study/daily_pair_tuning_20261122_100328_550844/stages.csv) · [回测指标](docs/results/csi500_research/sources/ml_study/daily_pair_tuning_20261122_100328_550844/comparison.csv)

在同一测试期，调参前的八模型等权组合扣费累计收益为 70.64%，八个排序器分别按自身验证损失选参后再等权为 63.95%。三种方案均按同一交易规则回测，研究数据截至 **2025 年 12 月 31 日**。

`results/` 保存本机的特征缓存、模型和实验输出，研报引用的表格、图形及归档数据位于 [docs/results/csi500_research/](docs/results/csi500_research/)。

## 运行与复核

在仓库根目录使用已有 Conda 环境 `quant`。项目使用 Python 3.10 和 Qlib 0.9.7，依赖版本见 [requirements.txt](requirements.txt)。下载 TuShare 数据时须设置环境变量 `TUSHARE_TOKEN`，下载日期由研究配置的 `data` 段控制。下载器可按任务更新交易日历、历史成分、个股行情与复权因子等输入。[下载入口](src/data/download.py)

```bash
conda activate quant
python -B -m src.scripts --help
python -B -m unittest discover -s tests -v
```

训练在已有 tmux 会话 `train` 的新窗口运行。以下示例使用本地已归档的八模型参数和 Alpha158 特征缓存。

```bash
tmux new-window -t train -n eight-rankers "conda run --no-capture-output -n quant python -B -m src.scripts eight-rankers --validation-mode split"
```

| 任务 | `python -B -m src.scripts` 后的命令 |
| --- | --- |
| 按研报组合参数训练固定八排序器 | `eight-rankers --validation-mode split` |
| 按验证期组合排序损失分阶段调参 | `pair-tuning` |
| 按年重训八模型组合 | `rolling-pair --window expanding` 或 `rolling-pair --window sliding` |
| 训练四个 MLP 排序器 | `mlp --loss lambdarank` |
| 训练固定 LightGBM 收益回归模型 | `lightgbm --loss mse` |
| 比较固定和滚动 LightGBM、Ridge 及三个单因子 | `study` |
| 复用上述六方法实验的预测重新评价 | `study --stage evaluate --out <实验目录>` |

八排序器入口默认读取本地七阶段调参归档的 `best_config.json` 及基线归档指定的特征缓存。使用其他数据时，传入 `--source <特征缓存目录>`（含 `features.parquet` 和 `alpha158.json`）及 `--config <配置文件>`。复现研报中的 MLP 时，给 `mlp` 传入下列归档配置，其中隐藏层宽度为 256、512、128、32，特征仅做训练集中位数填补，`dropout=0`。[MLP 归档配置](docs/results/csi500_research/sources/ml_study/mlp_report_median_20261126/config.json) · [当前默认配置](configs/research.json) · [调参网格](configs/tuning.json)

## 代码与产物

| 入口 | 用途 |
| --- | --- |
| [数据处理](src/data/) · [因子计算](src/factors/) | 按当时可用的快照重建逐日成分股，准备复权量价、Alpha158 特征和收益标签。 |
| [模型训练](src/ml/) · [任务命令表](src/scripts/__main__.py) | 构建样本、训练和选择模型，生成按交易日和股票代码对齐的预测。 |
| [评价与回测](src/backtest/) · [报告生成](src/report/) | 计算 IC、Rank IC 和组合绩效，输出交易账本及各任务的实验报告。 |
| [Alpha158 中性化检验](docs/factors/alpha158_neutral_ic_daily.md) · [开发指南](AGENTS.md) | 查阅专项分析、模块入口和影响复现的计算口径。 |

复核某次实验时，先看实验目录中的 `config.json` 确认数据区间和模型参数，再查 `predictions.parquet`、`rank_ic_summary.csv` 和 `portfolio_metrics.csv`。逐笔成交、逐日持仓和扣费收益分别记录在 `trades.parquet`、`holdings.parquet` 和 `ledger.parquet`。
