"""基于完整 Alpha158 的 Qlib TRA，按交易日构造窗口，标签兑现后更新个股误差记忆。"""
from collections import deque
from itertools import chain

import numpy as np
import pandas as pd
import torch
from qlib.constant import EPS
from qlib.contrib.model.pytorch_tra import TRAModel, device, minmax_norm

MODEL = dict(input_size=158, hidden_size=256, num_layers=2, rnn_arch="LSTM", use_attn=True, dropout=0.2)
ROUTER = dict(num_states=3, rnn_arch="LSTM", hidden_size=32, num_layers=1, dropout=0.0,
              tau=1.0, src_info="LR_TPE")
TRAINING = dict(model_type="RNN", lr=1e-3, n_epochs=100, early_stop=20, seed=0, lamb=1.0,
                rho=0.99, alpha=0.5, transport_method="router", memory_mode="sample",
                eval_train=False, eval_test=False, pretrain=True)
LENGTH, HORIZON, BATCH = 60, 2, 1024
KEYS = ["trade_date", "ticker"]


class Panel:
    """每个有行情的股票日存一组特征，按需取出最近 60 个交易日的序列。"""

    def __init__(self, history, samples, names, calendar, train_bounds, scaler=None, horizon=HORIZON):
        self.horizon = horizon
        self.calendar = calendar
        self.samples = samples.sort_values(KEYS).reset_index(drop=True)
        s = self.samples
        if history.duplicated(KEYS).any() or s.duplicated(KEYS).any():
            raise ValueError("Duplicate stock-day in TRA inputs")
        tickers = pd.Index(history.ticker.unique())
        d = calendar.get_indexer(history.trade_date)
        k = tickers.get_indexer(history.ticker)
        self.days = calendar.get_indexer(s.trade_date)
        self.stocks = tickers.get_indexer(s.ticker)
        if min(d.min(), self.days.min(), self.stocks.min()) < 0:
            raise ValueError("Unknown market session or stock")
        # 两张索引表都用 -1 表示缺失，对应数据数组末尾预留的全零行。
        self.features_at = np.full((len(calendar) + LENGTH, len(tickers)), -1, dtype=np.int32)
        self.features_at[d + LENGTH, k] = np.arange(len(history))
        self.samples_at = np.full_like(self.features_at, -1)
        self.samples_at[self.days + LENGTH, self.stocks] = np.arange(len(s))
        current = self.features_at[self.days + LENGTH, self.stocks]
        if (current < 0).any():
            raise ValueError("Missing signal-day features")
        for offset, column in ((1, "label_start_date"), (self.horizon, "label_end_date")):
            actual = calendar.get_indexer(s[column])
            known = s[column].notna()
            if (actual[known] != self.days[known] + offset).any():
                raise ValueError(f"Unexpected {column} relative to the exchange calendar")
        if (s.source_snapshot_date.isna() | (s.source_snapshot_date > s.trade_date)).any():
            raise ValueError("Missing or future membership snapshot")
        self.raw = s.label.where(np.isfinite(s.label) & s.label_end_date.notna())
        # 按 Qlib 的 CSRankNorm 对标签做日内排名，同一信号日的股票共用收益终点。
        self.labels = ((self.raw.groupby(s.trade_date).rank(pct=True) - 0.5) * 3.46).to_numpy(np.float32)
        self.memory = np.zeros((len(s) + 1, ROUTER["num_states"]), dtype=np.float32)
        self.features = np.zeros((len(history) + 1, len(names)), dtype=np.float32)
        fit_rows = current[s.trade_date.between(*train_bounds)]
        stats = []
        for j, name in enumerate(names):
            values = history[name].to_numpy(np.float32, copy=True)
            values[~np.isfinite(values)] = np.nan
            if scaler is None:
                fit = values[fit_rows]
                center = np.nanmedian(fit)
                scale = (np.nanmedian(np.abs(fit - center)) + EPS) * 1.4826
            else:
                center, scale = scaler.loc[name, ["median", "scale"]]
            stats.append((center, scale))
            # 按 Qlib 的稳健标准化规则截到正负 3，再将缺失值填零。
            values -= center
            values /= scale
            np.clip(values, -3, 3, out=values)
            self.features[:-1, j] = np.nan_to_num(values, copy=False)
        self.scaler = pd.DataFrame(stats, index=pd.Index(names, name="feature"), columns=["median", "scale"])

    def window(self, rows):
        positions = self.days[rows, None] + np.arange(1, LENGTH + 1)
        ticker = self.stocks[rows, None]
        features = self.features_at[positions, ticker]
        # 只取截至当前交易日已经兑现的预测误差。
        memory = self.samples_at[positions[:, :-self.horizon], ticker]
        return (torch.as_tensor(self.features[features], device=device),
                torch.as_tensor(self.memory[memory], device=device))

    def segment(self, bounds, *, supervised):
        s = self.samples
        mask = s.trade_date.between(*bounds)
        if supervised:
            mask &= s.label_end_date.le(bounds[1]) & self.raw.notna()
        return Segment(self, np.flatnonzero(mask), self.calendar.searchsorted(pd.Timestamp(bounds[1]), side="right") - 1)


class Segment:
    """为 Qlib 的训练循环按批提供样本，每轮打乱符合训练条件的股票日。"""

    def __init__(self, panel, rows, end_day):
        self.panel, self.rows, self.end_day = panel, rows, end_day
        if not len(rows):
            raise ValueError("Empty TRA segment")
        self.daily = np.split(rows, np.flatnonzero(np.diff(panel.days[rows])) + 1)

    def train(self):
        pass  # 评价时直接按交易日顺序重放。

    def __len__(self):
        return len(self.rows) // BATCH

    def __iter__(self):
        indices = np.random.permutation(self.rows)
        for rows in indices[:len(self) * BATCH].reshape(-1, BATCH):
            x, state = self.panel.window(rows)
            yield dict(data=x, state=state, label=torch.as_tensor(self.panel.labels[rows], device=device),
                       index=rows, daily_count=[])

    def assign_data(self, index, losses):
        self.panel.memory[index] = losses.detach().cpu().numpy()

    def clear_memory(self):
        self.panel.memory.fill(0)


@torch.no_grad()
def replay(model, segment, *, output_start=None):
    """逐日预测，各日各专家的误差在对应收益兑现的收盘时刻写入记忆。"""
    panel = segment.panel
    model.model.eval()
    model.tra.eval()
    panel.memory[segment.rows] = 0
    pending = deque()
    records, ics = [], []

    def settle(day):
        while pending and pending[0][0] <= day:
            _, rows, experts, score = pending.popleft()
            valid = np.isfinite(panel.labels[rows])
            if not valid.any():
                continue
            known = rows[valid]
            labels = panel.labels[known]
            loss = torch.from_numpy((experts[valid] - labels[:, None]) ** 2)
            panel.memory[known] = minmax_norm(loss).numpy()
            if output_start is None and len(known) > 1:
                ics.append(pd.Series(score[valid]).corr(pd.Series(labels), method="spearman"))

    for rows in segment.daily:
        day = panel.days[rows[0]]
        settle(day)
        x, state = panel.window(rows)
        experts, _, prob = model.tra(model.model(x), state)
        route = prob.argmax(dim=1)
        score = experts.gather(1, route[:, None]).flatten().cpu().numpy()
        pending.append((day + panel.horizon, rows, experts.cpu().numpy(), score))
        if output_start is not None and panel.calendar[day] >= output_start:
            records.append(panel.samples.loc[rows, KEYS].assign(tra_raw=score, expert=route.cpu().numpy()))
    settle(segment.end_day)
    if output_start is not None:
        return pd.concat(records, ignore_index=True)
    return float(pd.Series(ics, dtype=float).mean())


class CausalTRA(TRAModel):
    def __init__(self):
        super().__init__(model_config=MODEL.copy(), tra_config=ROUTER.copy(), **TRAINING)

    def test_epoch(self, epoch, data_set, return_pred=False, prefix="test", is_pretrain=False):
        # 两阶段均以实际路由预测的 RankIC 选取最佳轮次，包括预训练阶段的验证。
        return {"IC": replay(self, data_set)}, None, None, None

    def fit_windows(self, train, early):
        history = []
        for pretrain in (True, False):
            params = self.tra.predictors.parameters() if pretrain else self.tra.parameters()
            self.optimizer = torch.optim.Adam(chain(self.model.parameters(), params), lr=self.lr)
            logs = dict(train=[], valid=[])
            # Qlib 在每个阶段结束时恢复该阶段 2021 年日均 RankIC 最高的编码器和路由器。
            self._fit(train, early, None, logs, is_pretrain=pretrain)
            if not any(np.isfinite(entry["IC"]) for entry in logs["valid"]):
                raise ValueError("No finite daily RankIC for TRA checkpoint selection")
            history.extend(dict(phase="pretrain" if pretrain else "router", epoch=i+1, rank_ic=v["IC"])
                           for i, v in enumerate(logs["valid"]))
        self.fitted = True
        return pd.DataFrame(history)
