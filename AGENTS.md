# alpha-research 开发指南

项目用历史动态中证500股票池比较六种打分方法。固定 LightGBM、滚动 LightGBM 和 Ridge 从 Alpha158 学习分数，Alpha3、Alpha13、Alpha44 用复权量价计算分数并做行业、市值中性化。六种分数进入同一套评价和 Qlib Top-K 回测，中证500价格指数是组合基准。

代码注释和文档字符串统一使用中文，函数名、参数名、模型名和指标名沿用代码中的写法。

## 沿一条信号读代码

默认配置在 [configs/research.json](configs/research.json)，调参网格和联合搜索阶段在 [configs/tuning.json](configs/tuning.json)，读取入口在 [src/config.py](src/config.py)。阅读时先看 `run_all.main()` 怎样准备数据并生成预测，再看同文件的 `evaluate_predictions()` 怎样依次计算 IC、回测和写报告。

| 模块 | 输入、输出与主要入口 |
| --- | --- |
| [data](src/data) · 下载与预处理 | [download.py](src/data/download.py) 是日历、成分快照、行情和行业的统一下载入口；[universe_process.py](src/data/universe_process.py) 用快照重建逐日成员表。[research_panel.py](src/data/research_panel.py) 区分缺失观测与零成交量，并用 `execution_labels()` 按交易日生成收益标签，[qlib_provider.py](src/data/qlib_provider.py) 提供本地 Qlib 行情。 |
| [factors](src/factors) · 因子 | [factor_batches()](src/factors/__init__.py) 接收行情面板、成员和信号日，分批返回含 `trade_date`、`ticker` 的因子长表。`alpha158` 用 Qlib 官方表达式，`price_volume` 用项目约定的成分股截面排名计算 Alpha3、Alpha13、Alpha44 等。[neutralization.py](src/factors/neutralization.py) 做行业、市值中性化。 |
| [ml](src/ml) · 模型预测 | [dataset.py](src/ml/dataset.py) 将 Alpha158 与标签对齐，组装 Qlib 拟合样本。[models.py](src/ml/models.py)、[rolling.py](src/ml/rolling.py) 生成固定和滚动预测，[tuning.py](src/ml/tuning.py) 按验证 MSE 搜参。 |
| [backtest](src/backtest) · 评价与回测 | 股票日分数 → IC、五组收益、回归、交易账本。[engine.py](src/backtest/engine.py) 为各方法运行 Qlib Top-K，[factor_metrics.py](src/backtest/factor_metrics.py) 计算 IC、分组收益和回归，[portfolio.py](src/backtest/portfolio.py) 从账本计算全期及逐窗绩效。 |
| [report](src/report) · 报告 | [research.py](src/report/research.py) 的 `write_results()` 生成策略比较报告，[charts.py](src/report/charts.py) 绘图，[strategy_report.md](src/report/strategy_report.md) 存放报告正文模板；复现清单由 `research.write_manifest()` 生成。 |
| [scripts](src/scripts) · 运行 | [任务命令表](src/scripts/__main__.py) 连接各任务入口。[run_all.py](src/scripts/run_all.py) 负责综合研究和固定 LightGBM，[train_xgboost.py](src/scripts/train_xgboost.py) 负责独立 XGBoost，八模型见 [train_eight_rankers.py](src/scripts/train_eight_rankers.py)。 |

## 容易改变结果的口径

- 标签按市场交易日定位，用 T+`label_end_days` 的复权收盘价除以 T+`label_start_days` 的复权收盘价再减一。训练与验证样本的收益终点必须留在各自区间。固定和滚动训练、Ridge 共用 `training_label` 的 `raw` 或每日横截面 `zscore`。预测可保留尚无未来标签的股票日。
- IC 和 Rank IC 各用该方法自己的有效分数与原始收益配对。ICIR 是每日相关系数的均值除以样本标准差。五组收益和横截面回归使用各方法共同有分的股票日。每种方法按自己的有效分数选股。
- T 日收盘打分，T+1 计划成交。`rebalance_days` 控制调仓间隔与起始偏移数。`n_drop` 控制每次计划替换的股票数。逐窗绩效从连续账本截取，按每条轨道计算后等权平均。
- `daily_return = daily_return_before_cost - cost_rate`，`cost_cny` 是人民币手续费。扣费日收益减指数日收益进入 Qlib `risk_analysis(mode="sum")`，用于年化超额收益、信息比率与最大超额回撤。绝对复利收益使用 `product`。项目锁定 Qlib 0.9.7，日频年化基数为 238。
- 行业在下载阶段按历史成员日期核验。正式输入是 `data/reference/csi500/industry_l1_history.parquet`，逐日证据在同目录的 `industry_gap_audit.parquet`。来源冲突或缺乏历史证据时保留缺失。

## 运行与复查

使用已有 Conda 环境 `quant`，依赖见 [requirements.txt](requirements.txt)。在仓库根目录运行。

每次启动训练都使用已有的 tmux 会话 `train`，在其中新开窗口运行，方便通过 `tmux attach -t train` 查看进度。

| 任务 | 命令 |
| --- | --- |
| 完整比较六种方法 | `python -B -m src.scripts study` |
| 仅固定 LightGBM 与指数 | `python -B -m src.scripts lightgbm --loss mse` |
| 按验证集 MSE 网格调参 | `python -B -m src.scripts study --stage tune` |
| 复用已有预测，重新评价、回测并写报告 | `python -B -m src.scripts study --stage evaluate --out <实验目录>` |
| 回归测试 | `python -B -m unittest discover -s tests -v` |

四个训练入口为 [MLP](src/scripts/train_mlp.py)、[LightGBM](src/scripts/train_lightgbm.py)、[XGBoost](src/scripts/train_xgboost.py) 和 [八等权排序器](src/scripts/train_eight_rankers.py)。树排序模型共用 [tree_experiment.py](src/ml/tree_experiment.py) 的拟合与选参流程，神经网络见 [mlp.py](src/ml/mlp.py)。训练目标在模型入口选择，选参指标由 `--tpe-objective` 控制。

单因子排名、缺分规则对照和中性化分析写入 `results/factors/`，因子报告在 `docs/factors/`。综合回测、LightGBM 调参、Ridge 和等权组合共用 `results/ml_study/`。完整实验写入 `results/ml_study/<频率>_<时间戳>/`。复查时先看该次的 `config.json` 和 `manifest.json`，再按问题查 `predictions.parquet`、`daily_rank_ic.csv`、`ledger.parquet` 或 `portfolio_metrics.csv`。评价命令会更新指定实验的产物和对应频率的报告。综合报告在 `docs/results/ml_study_<频率>.md`，单模型报告在 `docs/results/lightgbm_study_<频率>.md`。调参单独写入 `tuning_<时间戳>/`。`results/` 仅保存在本地。

测试分为[数据准备与因子计算](tests/test_data_factors.py)和[模型训练、预测评价与组合回测](tests/test_ml_backtest.py)。改指标时同时核对逐日明细、汇总 CSV 和报告中全期、分年、逐窗的同名列。
