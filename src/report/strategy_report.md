# 中证500{frequency}选股研究 · {title}

{test_start} 至 {test_end}。{result}{track_rule}。

**表　测试期预测能力与扣费组合绩效**

{overview}

超额收益以扣费日收益减指数日收益计算，Qlib 按 238 日算术年化。信息比率与最大超额回撤使用同一序列。[组合指标]({links}/portfolio_metrics.csv)

## 模型训练

**表　固定模型区间与有效股票日数**

| 阶段 | 日期 | 股票日数 |
| --- | --- | ---: |
| 训练 | {train_start} 至 {train_end} | {fit[fitted_rows]:,} |
| 验证 | {validation_start} 至 {validation_end} | {valid[fitted_rows]:,} |
| 测试 | {test_start} 至 {test_end} | {held_out[evaluable_rows]:,} |

模型用{feature_mode}预测 T+{label_end} 相对 T+{label_start} 的复权收盘收益。{training_details}{ridge_losses} [训练日志]({links}/training_log.json)

{training_chart}

## 预测排序

各方法用自身有效分数与原始收益计算每日 Pearson IC 和 Spearman Rank IC。ICIR 为日均值除以日样本标准差，正 IC 比例按 Rank IC 大于零计。[逐日 IC]({links}/daily_rank_ic.csv)

<details>
<summary>验证期与逐年结果</summary>

**表　验证期与分年度 IC**

{yearly_ic_table}

</details>

![20 日滚动平均 Rank IC]({links}/rank_ic_timeline.svg)

**图 2　测试期 Rank IC。** 每个点是截至当日最近 20 个交易日的平均 Rank IC。{switch_caption}

**表　预测五组平均收益**

{group_table}

![五组远期收益]({links}/quintile_returns.svg)

**图 3　测试期五组原始 {horizon} 日收益。** 对{group_rule}按分数从低到高分组，先取组内等权收益，再取日均值。

## 组合回测

Qlib Top-K 目标持有 {config[topk]} 只，每 {config[rebalance_days]} 个交易日计划替换 {config[n_drop]} 只。{candidate_rule}，T 日收盘打分，T+1 按{buy_rule}。每条轨道初始资金 {config[initial_capital_cny]:,} 元，{fees}。[执行设置]({links}/execution_audit.json) · [每日账本]({links}/ledger.parquet)

![组合净值与指数]({links}/portfolio_nav.svg)

**图 4　测试期扣费净值。** 彩线为各方法全部轨道的每日平均净值，灰色虚线为中证500价格指数，初始净值为 1。{switch_caption}

**表　各调仓偏移的扣费年化超额收益**

{offset_table}

![调仓偏移的超额收益]({links}/offset_returns.svg)

**图 5　各轨道相对指数的年化超额收益。** Offset 表示首次取信号相对测试首日延后几个交易日，零线表示与指数持平。

{rolling_details}
