# alpha-research

复现华泰证券2019年的Alpha3、Alpha13、Alpha44，并用MOM20对照检验2023—2025年动态中证500。研究包含IC、收益分布诊断、MOM20增量信息、行业市值控制和明确标记的简化多头账本；不接入实盘。

在项目根目录，以已有Python环境运行：

```bash
python -B -m src.scripts.run_huatai_research
```

一个入口依次运行全部核心测试、正式实验、四张图、两份报告、输入哈希与重构前后比较。依赖版本见[requirements-research.txt](requirements-research.txt)（已验证Python 3.13.11）。仅运行测试：`python -B -m unittest discover -s tests -v`。

流程完全离线，不覆盖已有`data/`或旧2024研究结果；新结果只写入`results/huatai_alpha/`。它要求以下固定输入已经存在，不能从空仓库自动恢复受权限限制的数据：

- 原行情：`data/processed/daily/csi500_processed_daily.parquet`。
- 历史成员：`data/processed/universe/csi500_membership.parquet`。
- 独立开市日历：`data/raw/calender/trade_calender_2016_2025.parquet`。
- 辅助输入：`results/huatai_alpha/auxiliary/`内历史行业区间、每日市值、涨跌停价及中证500价格指数，共四个文件。保留数据版本，不在重跑时重新下载。
- `config.json`及`evidence/provenance.json`、`evidence/baseline.json`固定设计、输入校验与旧版数值基线。

输入来源、实际TuShare请求参数、原始获取日志和旧版证据合并保存在`evidence/provenance.json`；72个月度辅助分片已逐值验证等于所保留的两个合并表后删除。原已提交的事前研究方案已逐字归档到该JSON的`original_research_design`，保留原文件路径、内容及SHA-256；`docs/`仅保留主报告和英文摘要。现行模块和运行命令以此README为准。

| 核心文件 | 用途 |
| --- | --- |
| [研究报告](docs/huatai_alpha_research_report.md) | 主结果、正IC/负价差诊断、组合与局限 |
| [英文摘要](docs/huatai_alpha_english_summary.md) | 300—500词MAFM申请素材 |
| [config.json](results/huatai_alpha/config.json) | 原始冻结参数，未为诊断优化 |
| [ic_summary.csv](results/huatai_alpha/ic_summary.csv) | 主、分年、敏感性、中性化及增量均值推断，按analysis区分 |
| [quantile_summary.csv](results/huatai_alpha/quantile_summary.csv) | 五组、单调性和同样本控制比较；小数收益 |
| [diagnostics.csv](results/huatai_alpha/diagnostics.csv) | 均值/中位数/截尾、年度、行业与调样窗口；只作诊断 |
| [portfolio_summary.csv](results/huatai_alpha/portfolio_summary.csv) | 完整5/20日、费用及所有起始偏移；主策略仍为20日/偏移0/15bp |
| [manifest.json](results/huatai_alpha/manifest.json) | 输入/源码/产物哈希、测试、独立复核及重构比较 |
| [最终验收证据](results/huatai_alpha/evidence/final_acceptance.json) | 尾部及重叠标签复核、独立交易重放、波动日指数来源与固定输入复现记录；不是另一份分析报告 |

`details/`集中保留一份研究面板、每日IC/分组、增量、分布及实际模拟成交/现金账本；`figures/`只有四张SVG。`influential_stocks.csv`完整列出个股对价差的加性贡献，不用来事后删除股票。所有CSV收益、IC和波动均为小数；报告表格另注明百分数/bp。重叠5/20日标签均值不是净值。

核心源码共八个研究模块：`data/research_panel.py`；`features/huatai.py`（含数值算子）与`neutralization.py`；`evaluation/huatai_study.py`（含HAC/Holm）、`diagnostics.py`与`portfolio.py`；`scripts/run_huatai_research.py`与`huatai_report.py`。复用原Momentum、标签及评价助手，不维护第二套旧输出入口。

已删除一次性下载器和续跑分支；取得新版输入时应另立版本和研究配置，不能静默替换历史数据。原始数据下载脚本仍留在`src/data/`，不会被本研究入口执行。

主要限制：月度成员和行业有效区间均非公告时间PIT；2024曾用于实验；缺失标签可能非随机。组合采用分数复权单位、下一日开盘及官方开盘涨跌停拦截，缺完整竞价流动性、退市结算与精确公司行动现金流；不能等同精确可执行超额收益。
