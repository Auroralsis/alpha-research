# alpha-research

本项目在动态中证500上比较固定 LightGBM、滚动 LightGBM、Ridge、Alpha13、Alpha3 和 Alpha44。机器学习模型用 Alpha158 生成分数，传统单因子用复权量价公式和行业市值中性化生成分数。各方法用自身的有效分数和收益计算 Qlib IC 和 Rank IC。五组远期收益和加权横截面回归使用各方法都有分数的股票日。各方法用自己的有效分数运行 Qlib 原生 TopkDropoutStrategy。中证500价格指数作为组合绩效基准。

`run_backtests()` 将分数、调仓日期和行情传给 Qlib，由其原生策略、交易所和模拟执行器完成选股、成交和记账。默认 T 日收盘生成分数，T+1 按复权收盘价买卖和估值。模拟器先卖后买，按可用现金的 95% 计算买单预算。指数日收益从当日收盘和前收盘点位计算，包含首日实际涨跌。[日频研究报告](docs/results/ml_study_daily.md) · [统一回测](src/backtest/engine.py)

## 研究报告

| 分类 | 报告 |
| --- | --- |
| 因子研究 | [Alpha101 排名](docs/factors/alpha101_ic_daily.md) · [Alpha158 中性化对照](docs/factors/alpha158_neutral_ic_daily.md) · [单因子 IC 排名](docs/factors/factor_rank_ic_daily.md) · [因子缺分分析](docs/factors/factor_missingness_daily.md) |
| 最终结果研究 | [六种选股方法对比](docs/results/ml_study_daily.md) · [固定 LightGBM 与指数对比](docs/results/lightgbm_study_daily.md) |

## 项目结构

项目按 Qlib 工作流组织数据集、模型、信号记录和组合评价，所有运行参数仍由 `configs/research.json` 提供。 [Qlib Alpha158 工作流](https://github.com/microsoft/qlib/blob/v0.9.7/examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158.yaml)

从入口看，一次完整研究经过六部分。`data` 下载和预处理行情，`factors` 计算因子，`ml` 训练模型并输出股票日预测。运行脚本 `scripts` 把模型分数和传统因子交给同一个 `backtest`，最后由 `report` 读取指标和账本生成报告。[六个目录的代码索引](AGENTS.md)

Alpha158、Alpha101 和传统量价因子共用 `factor_batches()`。下面假定已有含预热行情的 `panel`、逐日成员表 `members` 和信号日期 `dates`。每批都是带有 `trade_date`、`ticker` 的 DataFrame，将 `"alpha158"` 换成 `"alpha101"` 可计算研报的 101 项公式，换成 `"price_volume"` 可计算策略比较使用的传统量价因子。

```python
from src.config import load_config
from src.factors import factor_batches

for batch in factor_batches("alpha158", panel, members, dates, load_config()):
    print(batch.head())
```

## 运行

```bash
conda activate quant
python -m pip install -r requirements.txt
python -B -m src.scripts.run_all
python -B -m unittest discover -s tests -v
```

固定 LightGBM 和 Ridge 默认使用 2014–2020 年训练、2021–2022 年验证、2023–2025 年测试。滚动 LightGBM 复用 Alpha158，逐窗训练、合并分数后连续回测。完整研究使用唯一的配置文件。目前收益标签从 T+1 到 T+2，默认每日调仓，买入 5 bp、卖出 15 bp。[研究配置](configs/research.json)

### 只跑固定 LightGBM

```bash
python -B -m src.scripts.run_lightgbm
```

脚本使用固定训练和验证窗口，在验证集早停后对测试集打分。候选池由当日有有效 LightGBM 分数的中证500历史成分股组成。报告列出固定 LightGBM 与中证500价格指数的绩效，并对照同区间净值。报告写入 `docs/results/lightgbm_study_<频率>.md`，结果位于 `results/ml_study/<频率>_lightgbm_<时间戳>/`。可用 `--config 调参目录/best_config.json` 采用 Optuna 导出的参数。

### LightGBM 验证集调参

```bash
python -B -m src.scripts.run_all --stage tune
```

先用原配置训练基线，再用 Optuna 的 `GridSampler` 跑完 `lightgbm_tuning.search_space` 里的每组候选。候选总数等于各参数去重后的取值个数相乘，另加一次基线训练。每组用 Qlib 的 `DatasetH` 和 `LGBModel.fit()` 在训练期拟合，用验证期 MSE 早停并选参，同时记录验证期 Rank IC 和 Rank ICIR。固定窗口选出的参数可以给固定和滚动 LightGBM 共用。

搜索空间支持 `max_depth`、`num_leaves`、`lambda_l1`、`lambda_l2` 和 `learning_rate`，候选值以数组给出。

每次 LightGBM 拟合结束后输出一行，包含训练和验证的 Mean Rank IC、MSE 与最佳轮数。两组指标都取验证 MSE 最低的同一轮，Mean Rank IC 按交易日计算 Spearman 相关系数后取平均。调参摘要写入 `trials.csv`。固定和滚动训练的 Mean Rank IC 写入各自的 `training_log.json`，逐轮 MSE 见 `training_mse.csv`。

调参产物默认放在 `results/ml_study/tuning_<时间戳>/`。`trials.csv` 保存基线和每次候选的指标，`optuna.sqlite3` 保存 Optuna 试验，`best_config.json` 是完整的研究配置，`lightgbm.txt` 和 `best_training_mse.csv` 保存验证 MSE 最佳模型及损失曲线。用 `python -B -m src.scripts.run_lightgbm --config 调参目录/best_config.json` 只跑固定 LightGBM，或用 `run_all` 跑完整研究。

### 数据准备

完整运行 `run_all` 时，会自动检查新增训练期所需的日历、历史成员、行情和复权因子，缺少时依次补齐。需预先设置环境变量 `TUSHARE_TOKEN`。行情从 2013 年 9 月开始，为 2014 年的 Alpha158 提供预热。历史成员额外从 2013 年 8 月开始查询，供首个交易日向前查找快照。下载按月、按股票保存检查点，中断后重跑会复用已完成部分。手动下载使用同一个入口，默认依次下载日历与成分快照、重建逐日成员表、下载行情与复权因子、补齐行业。

```bash
python -B -m src.data.download
```

只更新一类数据时，在命令末尾加上 `calendar`、`membership`、`prices` 或 `industry`。例如 `python -B -m src.data.download industry` 只更新行业，使用已有的逐日成员表。

下载器按历史成员逐日检查 TuShare 申万一级行业，遇到空档时查询该股票的历史指数成分表。只有在当天能匹配到唯一一级行业时才补入。原始响应位于 `data/raw/industry/`，研究直接读取 [正式行业表](data/reference/csi500/industry_l1_history.parquet)。[缺口核验明细](data/reference/csi500/industry_gap_audit.parquet) 每行记录一个股票日、核验结果、行业代码、来源地址和证据区间。复查旧实验可用 [原 TuShare 快照](data/reference/csi500/industry_history.parquet)。历史记录提供生效日期，未提供逐条公告时间。

回测还使用 [辅助输入](data/reference/csi500/) 中的市值、涨跌停价和指数点位。

日频和周频共用以下配置项，每次实验目录里的 `config.json` 记录运行时使用的参数。

| 参数 | 含义 |
| --- | --- |
| `segments` | 训练、验证和测试日期区间 |
| `label_start_days`、`label_end_days` | 标签的起止交易日偏移，收益跨度为两者之差 |
| `training_label` | 三种模型共用的训练标签处理。`zscore` 为每日截面标准化，`raw` 为原始未来收益 |
| `lightgbm` | LightGBM 模型参数，包括损失、学习率、树结构、采样、正则化和线程数 |
| `seed` | LightGBM 随机种子 |
| `num_boost_round`、`early_stopping_rounds` | 最大训练轮数和早停轮数 |
| `ridge_loss`、`ridge_alpha`、`ridge_solver`、`ridge_num_threads` | Ridge 损失函数（固定为 `mse`）、正则化强度、求解器及计算线程数 |
| `feature_warmup_days`、`feature_workers`、`feature_batch_size` | 特征预热交易日数、计算进程数和每批股票数量 |
| `rolling_test_windows` | 将测试期划分为多少段，据此确定滑动步长 |
| `topk`、`n_drop` | 目标持股数量和每次计划替换数量 |
| `min_regression_pairs` | 回归所需的最低有效股票数 |
| `rebalance_days` | 调仓间隔。报告对全部起始偏移的轨道绩效取算术平均 |
| `buy_price` | Qlib 策略买入价格，默认 `close` 为复权收盘价 |
| `limit_threshold`、`trade_unit`、`min_trade_cost_cny` | Qlib 涨跌幅限制、每手股数和最低单笔手续费，默认 9.5%、100 股和 5 元 |
| `buy_fee_bps`、`sell_fee_bps` | 买入和卖出费率 |
| `initial_capital_cny` | 初始资金，单位为元 |
| `log_evaluation_period` | 每隔多少训练轮打印损失，设为 0 可关闭逐轮日志 |
| `data` | 历史数据下载起止日、行情终点和请求间隔 |
| `qlib_version` | Qlib 版本 |

每次完整运行在 `results/ml_study/` 下建立一个实验目录。Qlib 组合评价保存到 `portfolio_metrics.csv`，日账本保存到 `ledger.parquet`，成交与持仓可查看 `trades.parquet` 和 `holdings.parquet`。结果使用 `strategy` 区分打分方法，`offset` 区分调仓起始偏移。排序评价见 `rank_ic_summary.csv`，五组远期收益仅作为排序诊断，保存到 `quintile_summary.csv`。逐股分数存入 `predictions.parquet`，其中 `window` 是滚动模型的预测窗口。完整运行成功后清理 Alpha158 中间特征表。

Qlib 导出的 LightGBM 文本模型按位置命名特征，例如 `Column_0`。用同一实验的 `alpha158.json` 按顺序查到每列的因子名称和公式。

用 `python -B -m src.scripts.run_all --stage evaluate` 可用最近一次完整实验的预测刷新回测和报告。加 `--out 实验目录` 可指定已有实验，或在完整运行时选择一个新的目录。报告按该次实验的调仓周期写到日频或周频报告。

完整研究的报告和默认结果目录按 `rebalance_days` 命名。

| 调仓间隔 | 报告 | 结果目录示例 |
| --- | --- | --- |
| 1 | `docs/results/ml_study_daily.md` | `results/ml_study/daily_时间戳/` |
| 5 | `docs/results/ml_study_weekly.md` | `results/ml_study/weekly_时间戳/` |
| 10 | `docs/results/ml_study_10d.md` | `results/ml_study/10d_时间戳/` |
| 20 | `docs/results/ml_study_20d.md` | `results/ml_study/20d_时间戳/` |

`--stage evaluate` 默认选择当前配置中同一调仓周期的最近完整实验，也可用 `--out` 指定某次历史实验。显式传入 `--out` 时，结果目录使用指定的名称。

单独运行华泰研报的 101 项技术因子检验可用 `python -B -m src.scripts.run_alpha101`。每项均保存原始和中性化后的 IC、ICIR、Rank IC、Rank ICIR，完整比较见 [Alpha101 单因子报告](docs/factors/alpha101_ic_daily.md)。
