# alpha-research

本项目在动态中证500上比较固定 LightGBM、滚动 LightGBM、Ridge、Alpha13、Alpha3 和 Alpha44。机器学习模型用 Alpha158 生成分数，传统单因子用复权量价公式和行业市值中性化生成分数。各方法用自身的有效分数和收益计算 Qlib IC 和 Rank IC。五组远期收益和加权横截面回归使用各方法都有分数的股票日。各方法用自己的有效分数运行 Qlib 原生 TopkDropoutStrategy。中证500价格指数作为组合绩效基准。

`run_backtests()` 将分数、调仓日期和行情传给 Qlib，由其原生策略、交易所和模拟执行器完成选股、成交和记账。默认 T 日收盘生成分数，T+1 按复权收盘价买卖和估值。模拟器先卖后买，按可用现金的 95% 计算买单预算。指数日收益从当日收盘和前收盘点位计算，包含首日实际涨跌。[日频研究报告](docs/results/ml_study_daily.md) · [统一回测](src/backtest/engine.py)

## 研究报告

**[中证500日频选股研究：从单因子到前50名排序](docs/results/csi500_research.md)** 汇总因子检验、树模型与 MLP 对照、逐年扩展训练的结果。报告附有选股命中率表、扣费净值和累计收益曲线。

| 分类 | 结果目录 | 报告 |
| --- | --- | --- |
| 因子分析 | [results/factors](results/factors) | [Alpha101 排名](docs/factors/alpha101_ic_daily.md) · [Alpha158 中性化对照](docs/factors/alpha158_neutral_ic_daily.md) · [单因子 IC 排名](docs/factors/factor_rank_ic_daily.md) · [因子缺分分析](docs/factors/factor_missingness_daily.md) |
| 模型与组合研究 | [results/ml_study](results/ml_study) | [树模型与 MLP 研究总结](docs/results/strategy_summary_daily.md) · [LightGBM、XGBoost、回归基线和指数的五线净值对照](docs/results/lightgbm_xgboost_nav_daily.md) · [滚动 LightGBM、Ridge 与传统单因子的综合回测](docs/results/ml_study_daily.md) |

五线报告同时记录 CatBoost、强正则 Ridge 的研究结论。复现排序组合时，使用原实验的 MAD 特征、种子 158 的模型、预测和交易账本。[排序基线归档](results/ml_study/baselines/nav_comparison_daily/baseline.json)

## 运行入口

从模型名选择训练入口，在仓库根目录运行 `python -B -m src.scripts <命令>`。

| 命令 | 训练内容 | 选择目标 |
| --- | --- | --- |
| `mlp` | 四个排序网络或单个收益回归网络 | `--loss lambdarank` 或 `--loss mse` |
| `lightgbm` | 四个 LightGBM 排序器或单个 MSE 回归模型 | `--loss lambdarank` 或 `--loss mse` |
| `xgboost` | 四个独立 XGBoost 排序器，各占 25% | `--loss rank:ndcg` 或 `--loss rank:pairwise` |
| `eight-rankers` | 四个 LightGBM 加四个 XGBoost，八列每日百分位各占 12.5% | LightGBM 用 LambdaRank，XGBoost 可用 `--xgboost-loss` |

八等权的默认参数来自七阶段优胜归档（LightGBM 深度 7、XGBoost 深度 5）。独立 LightGBM 固定训练默认使用四等权基线（深度 5）。用 `--config` 可覆盖入口默认参数。三个树排序入口默认用固定参数，加 `--tpe` 后按 `--tpe-objective ndcg` 或 `composite` 选参。固定研究通过 `--validation-mode split` 或 `joint` 选择窗口。每次训练在 tmux 会话 `train` 的新窗口启动。

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
python -B -m src.scripts study
python -B -m unittest discover -s tests -v
```

测试按[数据与因子](tests/test_data_factors.py)和[模型与回测](tests/test_ml_backtest.py)分成两组。

固定 LightGBM 和 Ridge 默认使用 2014–2020 年训练、2021–2022 年验证、2023–2025 年测试。滚动 LightGBM 复用 Alpha158，逐窗训练、合并分数后连续回测。日常训练和回测读取 `configs/research.json`。默认收益标签从 T+1 到 T+2，可用 `--label-days 5` 或 `--label-days 10` 切换期限，默认每日调仓，买入 5 bp、卖出 15 bp。[研究配置](configs/research.json)

### 按实验修改参数

| 配置段 | 生效的位置 |
| --- | --- |
| `single_lightgbm` | 固定和滚动的收益回归。单跑固定模型用 `python -B -m src.scripts lightgbm --loss mse`，调参用 `python -B -m src.scripts study --stage tune`。 |
| `single_xgboost` | `python -B -m src.scripts xgboost` 训练 A、B、C、D 中每一个排序器时读取的参数，各组单独输出预测。 |
| `four_model_targets` | A 选前 50、B 选前 100、C 避后 50、D 避后 100，各组五档收益的增益从低到高排列。 |
| `lightgbm_four_equal` | `rank-stack --source … --historical …` 新训练使用的 LightGBM 排序参数，四组排名各占 25%。 |
| `xgboost_four_equal` | 四个 XGBoost 排名各占 25%，每组使用 `single_xgboost.params`。 |
| `four_model_weights` | 运行 `python -B -m src.scripts xgboost-compare --reweight`，读取已调参的 Alpha158 双模型预测，比较四目标按此配比与原等权的验证损失、持仓尾部比例及回测收益。 |
| `lightgbm_xgboost_blend.xgboost_weight` | 两类模型混合时 XGBoost 的权重。默认 0.5 表示两边各占一半。 |

各调参命令默认读取 `configs/tuning.json`，可以用 `--tuning-config 路径` 指定另一份候选配置。文件包含 LightGBM 回归、XGBoost、双模型联合搜索的阶段顺序与候选，以及 Ridge 强正则和 LambdaRank 末档增益间距。固定模型参数、种子和早停轮数在 `configs/research.json`。[调参配置](configs/tuning.json)

配对实验复用指定 LightGBM 归档的股票日、收益标签、回测设置和预测。修改四组增益后，先运行新的 LightGBM 排序实验，再通过 `python -B -m src.scripts xgboost-compare --reference 实验目录` 与同目标的 XGBoost 配对。单改 XGBoost 树参数时可以直接复用原 LightGBM 归档。

### 只跑固定 LightGBM

```bash
python -B -m src.scripts lightgbm --loss mse
```

脚本使用固定训练和验证窗口，在验证集早停后对测试集打分。候选池由当日有有效 LightGBM 分数的中证500历史成分股组成。报告列出固定 LightGBM 与中证500价格指数的绩效，并对照同区间净值。报告写入 `docs/results/lightgbm_study_<频率>.md`，结果位于 `results/ml_study/<频率>_lightgbm_<时间戳>/`。可用 `--config 调参目录/best_config.json` 采用 Optuna 导出的参数。

### 前 50 名排序实验

```bash
python -B -m src.scripts lambdarank --source results/ml_study/daily_alpha158_comparison_20261121_105840_864298/preprocessing_test/sampling_mse
```

该命令从存档读取 MAD 特征和 MSE 基线，按交易日分组训练 LambdaRank，以验证集 NDCG@50 早停，并生成新的 Top-50 回测结果。

当前日频研究基线为 A、B、C、D 四模型等权，每个模型占 25%，种子 158。[归档训练配置](results/ml_study/baselines/nav_comparison_daily/rank_config.json) · [LightGBM、XGBoost 及二者混合与 MSE、指数的五线净值对照](docs/results/lightgbm_xgboost_nav_daily.md)

固定排名五档与模型堆叠使用以下命令。它生成 2018–2020 年样本外预测，再比较四个目标、线性增益、等权和逻辑回归。[五线净值对照](docs/results/lightgbm_xgboost_nav_daily.md)

```bash
python -B -m src.scripts rank-stack \
  --source results/ml_study/daily_alpha158_comparison_20261121_105840_864298/preprocessing_test/sampling_mse \
  --historical results/ml_study/daily_lambdarank_quintile_gap3_20261121_155554_734020
```

比较连续排名堆叠、等权和原逻辑回归时，直接复用四个模型的样本外预测。

```bash
python -B -m src.scripts rank-stack --blend-from results/ml_study/daily_rank_stack_20261121_160245_719888
```

四模型等权种子对照只重训 A、B、C、D，每个模型固定占 25%。例如将原种子 158 与 42 对比：

```bash
python -B -m src.scripts rank-stack --equal-seed-from results/ml_study/daily_rank_stack_20261121_160245_719888 --seed 42
```

### XGBoost 与 LightGBM 四组配对

`python -B -m src.scripts xgboost-compare` 读取种子 158 的正式归档，用同一份 MAD 特征、五档增益及早停期训练四个 XGBoost 排序模型。报告分别比较 A、B、C、D 的每日预测相关性、前 50 名重合率，以及两类模型各占一半时的扣费收益。[五线净值对照](docs/results/lightgbm_xgboost_nav_daily.md)

默认训练直接读取 `research.json` 的 `single_xgboost.params` 和公共种子、最大轮数、早停轮数。当前固定参数取自 2022 年双模型等权调参的优胜组合。[当次选定参数](results/ml_study/daily_pair_tuning_20261122_100328_550844/winners.json)

运行 `python -B -m src.scripts xgboost-compare --tune` 时，脚本从独立配置读取 `xgboost_tuning.search_space`，四组共享同一网格。默认用 2014–2020 年训练、2021 年早停，按 2022 年两模型混合排名的日均损失选参，随后进行 2023–2025 年回测。可用 `--tuning-config 路径` 指定其他网格。导出的 `best_config.json` 可通过 `--config` 传回固定训练入口。[排序调参配置](configs/tuning.json)

Qlib 0.9.7 的 `XGBModel.fit()` 创建训练矩阵时只传特征、标签和样本权重。本项目通过 XGBoost 的 `group` 显式传入每日截面大小，让同一天的股票互相比较；IC 评价和组合回测共用 Qlib。[Qlib XGBModel 源码](https://github.com/microsoft/qlib/blob/v0.9.7/qlib/contrib/model/xgboost.py)

### LightGBM 与 XGBoost 等权组合调参

独立运行 LightGBM 四等权用 `python -B -m src.scripts lightgbm-equal`。默认读取四等权基线的固定参数，训练 A/B/C/D，按每日排名百分位各占 25% 生成 `equal`，输出验证损失与测试期 Top50/Drop5 回测。可用 `--nplhr` 指定附加因子归档。

在 `python -B -m src.scripts pair-tuning --loss <选项目标>` 中，`mse` 使用原 LightGBM 回归调参流程，`lambdarank` 用八排序器组合的各折平均损失选参，`lambdarank_stable` 供显式启用多窗口的实验使用，按各期平均损失加 0.5 倍总体标准差选参。每折损失是该验证期的日均 1 − NDCG@50。逐阶段保留目标值最低的组合，稳定性系数固定。多窗口排序调参的 `fold_trials.csv` 和 `trials.csv` 可按 `stage, trial` 关联，分别给出各期损失及其汇总。

`python -B -m src.scripts pair-tuning` 从已归档的五档双模型基线开始，按 `configs/tuning.json` 中的 `rank_pair_tuning` 逐阶段搜索。每一步在验证期交叉组合两类模型，比较 1 − NDCG@50，保留上一阶段的优胜者。每类模型内部四目标等权。默认使用原始五档基线的固定区间，即 2014–2020 年拟合、2021 年早停、2022 年选参。显式设为 `"all"` 可复用历史多窗口策略，明细见 `fold_trials.csv`。参数冻结后评价测试期。可用 `--reference` 指定基线、`--tuning-config` 指定网格，或用 `--resume 实验目录` 从当次配置接续运行。输出的 `best_config.json` 分别更新四模型排序 LightGBM 和 XGBoost 的参数。候选与逐阶段结果见 `trials.csv` 和 `stages.csv`。

八模型 Optuna 联合调参使用 `python -B -m src.scripts pair-tuning --tpe`。默认从七阶段优胜组合的 MAD Alpha158 验证预测起步。每个 Trial 为 LightGBM 和 XGBoost 分别提出一套参数，同族四目标共用；按 2022 年八模型等权排名的日均 1−NDCG@50 选参。预算和两类搜索空间见 `configs/tuning.json` 的 `rank_pair_tpe`，前两个 Trial 是原七阶段优胜组合和调整 LightGBM 正则与叶数的历史候选。实验目录保存 SQLite、逐轮验证损失、胜出参数和最终回测；续跑使用 `pair-tuning --resume 实验目录`。

固定训练可通过同一个 `--validation-mode` 选择日期。两种模式均用 2014–2020 年训练，2023–2025 年用于参数冻结后的历史诊断。

| 参数 | 早停窗口 | 超参数或候选方案评分窗口 |
| --- | --- | --- |
| `--validation-mode split` | 2021 | 2022 |
| `--validation-mode joint` | 2021–2022 | 同一段连续的 2021–2022 |

例如，`python -B -m src.scripts pair-tuning --tpe --tpe-models lightgbm --tpe-objective composite --validation-mode joint` 用两年 NDCG@50 选择各排序器轮数，再用两年连续扣费回测选择 Trial。将 `lightgbm` 换为 `pair` 即为八排序器。仅训练固定参数时，`lightgbm-equal --validation-mode split` 按 2021 年早停，生成 2022 年验证预测。显式月度或年度滚动重训继续由每个预测窗口决定历史区间。续跑读取原实验配置。

四等权两年验证回测选参可使用 `pair-tuning --tpe --tpe-models lightgbm --tpe-objective composite`。LightGBM 四目标各占 25%；`--tpe-models pair` 使用八等权。默认 `--tpe-objective ndcg`。新的四等权实验用 2021–2022 年全部有效交易日的 NDCG@50 早停，按同两年连续扣费账本选参。八模型 Optuna 缺省用 2021 年早停和 2022 年验证。综合目标从对应账本取得绝对复利年化收益 R、项目年化 Sharpe SR 和绝对最大回撤 D。按论文式 (2) 的单年度形式最小化损失，收益尺度在基础项和负收益惩罚中均为 0.20。SQLite 逐 Trial 保存 R、SR、D、四个惩罚、综合损失和 NDCG 损失，完成后导出 `trials.csv`。续跑从实验目录读取原选参目标。

`python -B -m src.scripts rolling-pair` 用最近一次已完成的联合调参归档，按自然年重训四个 LightGBM 和四个 XGBoost。默认使用固定起点的扩展窗口（expanding window）。加上 `--window sliding` 使用滑动窗口，训练年数与首窗一致，起点和终点每年同时后移。两者均用上一整年早停，跨年连续运行日频 Top-50 回测。可用 `--reference 实验目录` 指定固定对照，或用 `--resume 实验目录` 接续本次模型。参见[扩展窗口](docs/results/expanding_pair_daily.md)和[滑动窗口](docs/results/sliding_pair_daily.md)的回测结果。


### MLP

`python -B -m src.scripts mlp --loss lambdarank` 按 A/B/C/D 训练四个网络，每日排名后等权融合。`--loss mse` 用一个网络拟合原始收益。网络宽度、学习率、批次和设备在 `configs/research.json` 的 `mlp` 中，命令行可用 `--device cpu`。固定训练写入 `predictions.parquet` 和各目标的权重、训练曲线。加 `--walk-forward 1` 每月重训，`--train-months` 和 `--stop-months` 控制移动窗口。

### LightGBM 验证集调参

```bash
python -B -m src.scripts study --stage tune
```

先用原配置训练基线，再用 Optuna 的 `GridSampler` 跑完 `tuning.json` 的 `lightgbm_tuning.search_space` 里的每组候选。候选总数等于各参数去重后的取值个数相乘，另加基线候选。每折用独立的下一年按 MSE 早停，随后一年计算选参 MSE。默认以最近一个完整选参年的 MSE 选择树参数，最终模型按固定模型的训练和早停配置拟合。固定和滚动 LightGBM 都可使用选出的参数。

默认单独比较采样强度，固定树结构、学习率与 L1/L2。当前 MAD 基线学习率为 0.02554240953288225，树深为 5，每轮使用 50% 的特征和 80% 的训练行。网格交叉比较列采样率 `colsample_bytree` 的 0.5、0.7 与行采样率 `subsample` 的 1.0、0.8，共四个候选。行指股票日样本，`single_lightgbm.params.bagging_freq=1` 表示每轮重新抽样，`subsample=1.0` 使用全部行。只调列时可从 `search_space` 移除 `subsample`，固定值取自 `single_lightgbm.params`。树深、叶子数、L1/L2 和学习率仍可加入同一网格。

最终模型拟合结束后输出一行，包含训练和验证的 Mean Rank IC、MSE 与最佳轮数。两组指标都取验证 MSE 最低的同一轮，Mean Rank IC 按交易日计算 Spearman 相关系数后取平均。调参摘要写入 `trials.csv`，每折的区间、最佳轮数和选参 MSE 写入 `fold_trials.csv`。固定和滚动训练的 Mean Rank IC 写入各自的 `training_log.json`，逐轮 MSE 见 `training_mse.csv`。

调参产物默认放在 `results/ml_study/tuning_<时间戳>/`。`trials.csv` 保存基线和每次候选的指标，`optuna.sqlite3` 保存 Optuna 试验，`best_config.json` 是完整的研究配置，`lightgbm.txt` 和 `best_training_mse.csv` 保存验证 MSE 最佳模型及损失曲线。用 `python -B -m src.scripts lightgbm --config 调参目录/best_config.json` 只跑固定 LightGBM，或用 `run_all` 跑完整研究。

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
| `alpha158_processing` | `raw` 使用原始 Alpha158；`mad`（默认）仅按日、按因子做中位数 ± 5 MAD 截尾，保留缺失及零 MAD 截面；`neutral` 另做行业、市值中性化及残差标准化 |
| `--label-days {1,5,10}` 或配置中的 `label_days` | 训练、早停和排序评价从 T+1 收盘计算 1、5、10 个交易日的复权收益。命令行优先，例如 `python -B -m src.scripts eight-rankers --label-days 5`。旧配置沿用原起止日，项目默认 1 日。 |
| `label_start_days`、`label_end_days` | 标签的起止交易日偏移，收益跨度为两者之差 |
| `training_label` | 三种模型共用的训练标签处理。`zscore` 为每日截面标准化，`raw` 为原始未来收益 |
| `single_lightgbm.params` | 固定及滚动回归模型的参数，包括损失、学习率、树结构、采样、正则化和线程数 |
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

用 `python -B -m src.scripts study --stage evaluate` 可用最近一次完整实验的预测刷新回测和报告。加 `--out 实验目录` 可指定已有实验，或在完整运行时选择一个新的目录。报告按该次实验的调仓周期写到日频或周频报告。

完整研究的报告和默认结果目录按 `rebalance_days` 命名。

| 调仓间隔 | 报告 | 结果目录示例 |
| --- | --- | --- |
| 1 | `docs/results/ml_study_daily.md` | `results/ml_study/daily_时间戳/` |
| 5 | `docs/results/ml_study_weekly.md` | `results/ml_study/weekly_时间戳/` |
| 10 | `docs/results/ml_study_10d.md` | `results/ml_study/10d_时间戳/` |
| 20 | `docs/results/ml_study_20d.md` | `results/ml_study/20d_时间戳/` |

`--stage evaluate` 默认选择当前配置中同一调仓周期的最近完整实验，也可用 `--out` 指定某次历史实验。显式传入 `--out` 时，结果目录使用指定的名称。
