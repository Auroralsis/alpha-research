"""ListFold-exp（论文式 4、6）。每天按真实名次从两端向中间折叠。"""
from functools import partial

import numpy as np
import pandas as pd


def fold_day(scores, derivatives=True):
    """输入已按真实收益降序排列。返回当日负对数似然与各股票的梯度、对角 Hessian。"""
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    pairs = n // 2
    if not pairs:
        return 0., np.zeros(n), np.zeros(n)
    pos = np.exp(scores - scores.mean())
    neg = 1. / pos
    hi, lo = slice(0, pairs), slice(n - 1, n - pairs - 1, -1)

    def remaining(values):
        ends = values[hi] + values[lo]
        middle = values[pairs] if n % 2 else 0.
        return ends[::-1].cumsum()[::-1] + middle

    a, b = remaining(pos), remaining(neg)
    size = n - 2 * np.arange(pairs)
    z = a * b - size
    loss = (np.log(z) - scores[hi] + scores[lo]).sum()
    if not derivatives:
        return loss, None, None

    # 第 j 只股票仅参加包含它的阶段。前缀和把所有阶段的导数累加至它退出时。
    j = np.arange(n)
    last = np.minimum(np.minimum(j, n - 1 - j), pairs - 1)
    def total(values):
        return values.cumsum()[last]

    u, v = b / z, a / z
    plus, minus = pos * total(u), neg * total(v)
    grad = plus - minus
    grad[hi] -= 1.
    grad[lo] += 1.
    # 每阶段的二阶导是“被选为首端/尾端”的符号变量 (+1/-1/0) 的方差。
    hess = plus + minus - 2 * total(1/z) - pos**2 * total(u**2) - neg**2 * total(v**2) + 2 * total(u*v)
    return loss, grad, hess


class ListFold:
    """预先缓存各日的真实名次，向 LightGBM 返回原行顺序的导数。"""

    def __init__(self, labels, sizes):
        bounds = np.r_[0, np.cumsum(sizes)]
        if bounds[-1] != len(labels) or not np.isfinite(labels).all():
            raise ValueError("ListFold 需要分组完整的有效收益")
        self.order = np.concatenate([a + np.argsort(-labels[a:b], kind="stable")
                                     for a, b in zip(bounds[:-1], bounds[1:])])
        self.bounds = bounds
        self.pairs = sum(sizes // 2)

    def objective(self, pred, dataset):
        ranked = pred[self.order]
        grad, hess = np.empty_like(ranked), np.empty_like(ranked)
        for a, b in zip(self.bounds[:-1], self.bounds[1:]):
            _, grad[a:b], hess[a:b] = fold_day(ranked[a:b])
        if not np.isfinite(grad).all() or not np.isfinite(hess).all() or hess.min() < -1e-8:
            raise ValueError("ListFold 导数无效")
        g, h = np.empty_like(grad), np.empty_like(hess)
        g[self.order], h[self.order] = grad, hess
        return g, h

    def metric(self, pred, dataset):
        ranked = pred[self.order]
        loss = sum(fold_day(ranked[a:b], False)[0] for a, b in zip(self.bounds[:-1], self.bounds[1:]))
        return "listfold_nll_per_pair", loss / self.pairs, False


def torch_fold_day(ranked_scores):
    """对单日真实收益从高到低排列的网络分数，计算可反向传播的 ListFold-exp 总损失。"""
    import torch

    s = ranked_scores.double()
    pairs = s.numel() // 2
    if not pairs:
        return s.sum() * 0
    s = s - s.mean()

    def remaining(values):
        ends = values[:pairs] + values.flip((0,))[:pairs]
        return ends.flip((0,)).cumsum(0).flip((0,)) + values[pairs:len(s)-pairs].sum()

    size = len(s) - 2 * torch.arange(pairs, device=s.device)
    z = remaining(s.exp()) * remaining((-s).exp()) - size
    return (z.log() - s[:pairs] + s.flip((0,))[:pairs]).sum()


def fit_tree(sections, names, cfg, out):
    """逐日分组拟合树模型，返回按最佳轮数打分的函数。"""
    import lightgbm as lgb

    params = dict(cfg["lightgbm"])
    params.pop("objective")
    params.pop("metric")
    datasets, losses = [], []
    for period, rows, sizes in sections:
        target = rows.label.to_numpy()
        losses.append(ListFold(target, sizes))
        data = lgb.Dataset(rows[names], label=target, group=sizes, params=params,
                           reference=datasets[0] if datasets else None).construct()
        datasets.append(data)
        print(period, len(rows), flush=True)
        del rows
    history = {}
    model = lgb.train(dict(params, objective=losses[0].objective, metric="None"), datasets[0],
                      num_boost_round=cfg["num_boost_round"], valid_sets=[datasets[1]], valid_names=["early_stop"],
                      feval=losses[1].metric, callbacks=[lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False),
                      lgb.record_evaluation(history), lgb.log_evaluation(50)])
    curve = pd.Series(history["early_stop"]["listfold_nll_per_pair"], name="early_stop_nll_per_pair")
    curve.index = pd.RangeIndex(1, len(curve)+1, name="iteration")
    assert model.best_iteration == curve.idxmin()
    curve.to_csv(out / "early_stop.csv")
    model.save_model(out / "listfold.txt", num_iteration=model.best_iteration)
    print("最佳轮数", model.best_iteration, "早停损失", curve.min(), flush=True)
    return partial(model.predict, num_iteration=model.best_iteration, num_threads=params["num_threads"])


def fit_scale(x, names):
    """只在训练样本上拟合每列的填充值与缩放范围。无训练观测的列保持零输入。"""
    stats = []
    for col in x.T:
        if np.isinf(col).any():
            raise ValueError("输入存在无穷值")
        seen = col[~np.isnan(col)]
        stats.append((len(seen), np.min(seen), np.median(seen), np.max(seen)) if len(seen) else (0, 0., 0., 0.))
    return pd.DataFrame(stats, columns=["count", "min", "median", "max"], index=pd.Index(names, name="feature"))


def transform(values, stats):
    """使用训练期统计量填补、截尾并缩放特征。"""
    import torch

    x = np.ascontiguousarray(values, dtype=np.float32).copy()
    if np.isinf(x).any():
        raise ValueError("输入存在无穷值")
    lo, fill, hi = stats[["min", "median", "max"]].to_numpy(dtype=np.float32).T
    np.copyto(x, fill, where=np.isnan(x))
    np.clip(x, lo, hi, out=x)
    x -= lo
    x /= np.where(hi > lo, hi - lo, 1)
    return torch.from_numpy(x)


def network(widths):
    """逐股共享全连接打分器，按论文在各层使用 ReLU。"""
    from torch import nn

    layers = []
    for n_in, n_out in zip(widths, widths[1:]):
        layers.extend([nn.Linear(n_in, n_out), nn.ReLU()])
    # 正偏置避免初始化时输出 ReLU 对所有股票的梯度都为零。
    nn.init.constant_(layers[-2].bias, 1.)
    return nn.Sequential(*layers)


def batch_loss(model, days):
    """一次前向计算多个完整截面，各日期独立折叠，再按总配对数平均。"""
    import torch

    sizes = [len(x) for x in days]
    device = next(model.parameters()).device
    scores = model(torch.cat(days).to(device)).flatten().split(sizes)
    return sum(torch_fold_day(s) for s in scores) / sum(n // 2 for n in sizes)


def ndcg_day(scores, gains, cutoff, truncation, *, evaluate=False):
    """按当前分数稳定排序。评价返回 1−NDCG，训练返回带 LambdaRank 权重的成对损失。"""
    import torch
    from torch.nn.functional import softplus

    order = scores.detach().argsort(descending=True, stable=True)
    ranked, gains = scores[order], gains[order]
    positions = torch.arange(len(scores), device=scores.device)
    discount = (positions + 2).to(scores.dtype).log2().reciprocal()
    k = min(cutoff if evaluate else truncation, len(scores))
    ideal = (gains.sort(descending=True).values[:k] * discount[:k]).sum()
    denominator = ideal.clamp_min(torch.finfo(scores.dtype).tiny)
    if evaluate:
        return torch.where(ideal > 0, 1 - (gains[:k] * discount[:k]).sum() / denominator, 0.)
    gain_diff = gains[:k, None] - gains
    margin = (ranked[:k, None] - ranked) * gain_diff.sign()
    with torch.no_grad():
        weight = gain_diff.abs() * (discount[:k, None] - discount).abs() / denominator
        weight *= positions[:k, None] < positions
        weight /= torch.where(ranked[0] != ranked[-1], .01 + margin.abs(), 1.)
        magnitude = 2 * (weight * torch.sigmoid(-margin)).sum()
        # LightGBM 按日归一化 lambda；对冻结的权重反传，避免对排序和权重本身求导。
        weight *= torch.log1p(magnitude) / (np.log(2.) * magnitude.clamp_min(torch.finfo(scores.dtype).tiny))
    return (weight * softplus(-margin)).sum()


def mlp_periods(sections, names, out, *, ranking):
    """共享训练期填补和缩放统计；LambdaRank 同分按代码排序，ListFold 按真实收益折叠。"""
    import torch

    periods = []
    for period, rows, sizes in sections:
        if ranking:
            ordered = rows.sort_values(["trade_date", "ticker"])
        else:
            ordered = rows.sort_values(["trade_date", "label", "ticker"], ascending=[True, False, True])
        values = ordered[names].to_numpy(dtype=np.float32)
        if period == "train":
            stats = fit_scale(values, names)
            stats.to_csv(out / "preprocessing.csv")
        features = transform(values, stats).split(sizes.tolist())
        grades = torch.from_numpy(ordered.grade.to_numpy(dtype=np.int64)).split(sizes.tolist()) if ranking else ()
        periods.append((features, grades))
        print(period, len(sizes), len(rows), flush=True)
        del rows, ordered, values
    return periods


def fit_mlp(sections, names, cfg, out):
    """共用按日训练和早停流程；四个排序目标独立保存权重。"""
    ranking = cfg["objective"] == "lambdarank"
    periods = mlp_periods(sections, names, out, ranking=ranking)
    targets = cfg["label_gains"] if ranking else {"mlp": None}
    models = {}
    for target, gains in targets.items():
        folder = out / target if ranking else out
        folder.mkdir(exist_ok=True)
        models[target] = train_network(periods, gains, cfg, folder)
    return mlp_predictor(models, names, cfg, out)


def mlp_predictor(models, names, cfg, out):
    """用冻结网络和训练期缩放统计对所有股票打分。"""
    import torch

    stats = pd.read_csv(out / "preprocessing.csv", index_col="feature", float_precision="round_trip").loc[names]

    @torch.no_grad()
    def predict(features):
        chunks = transform(features.to_numpy(), stats).split(4096)
        result = {key: torch.cat([model(x.to(cfg["device"])).flatten().cpu() for x in chunks]).numpy()
                  for key, model in models.items()}
        return pd.DataFrame(result, index=features.index) if cfg["objective"] == "lambdarank" else result["mlp"]

    return predict


def train_network(periods, gains, cfg, out):
    """用一个目标拟合一个网络，按早停日期间的最佳指标恢复权重。"""
    import torch

    p = cfg["listfold_mlp"]
    widths = cfg["architecture"]
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(p["cpu_threads"])
    rng = np.random.default_rng(cfg["seed"])
    (train, train_grades), (stop, stop_grades) = periods
    device = torch.device(cfg["device"])
    model = network(widths).to(device)
    metric = "early_stop_nll_per_pair" if gains is None else "early_stop_one_minus_ndcg"
    gain_table = None if gains is None else torch.tensor(gains, device=device, dtype=torch.float32)

    def objective(days, grade_days, indices, *, evaluate=False):
        xs = [days[i] for i in indices]
        if gain_table is None:
            return batch_loss(model, xs), sum(len(x)//2 for x in xs)
        sizes = [len(x) for x in xs]
        scores = model(torch.cat(xs).to(device)).flatten().split(sizes)
        gain_days = gain_table[torch.cat([grade_days[i] for i in indices]).to(device)].split(sizes)
        losses = [ndcg_day(s, g, cfg["ndcg_at"], cfg["truncation"], evaluate=evaluate)
                  for s, g in zip(scores, gain_days)]
        return torch.stack(losses).mean(), len(xs)

    opt = torch.optim.Adam(model.parameters(), lr=p["learning_rate"])
    order = np.empty(0, dtype=int)
    best, history = float("inf"), []
    train_losses = []
    for step in range(1, p["updates"] + 1):
        if not len(order):
            order = rng.permutation(len(train))
        batch, order = order[:p["batch_days"]], order[p["batch_days"]:]
        opt.zero_grad()
        loss, _ = objective(train, train_grades, batch)
        if not torch.isfinite(loss):
            raise ValueError("训练损失无效")
        loss.backward()
        opt.step()
        train_losses.append(loss.item())
        if step % p["evaluate_every"] == 0 or step == p["updates"]:
            with torch.no_grad():
                total, count = 0., 0
                for i in range(0, len(stop), p["batch_days"]):
                    indices = range(i, min(i+p["batch_days"], len(stop)))
                    loss_value, n = objective(stop, stop_grades, indices, evaluate=True)
                    total += loss_value.item() * n
                    count += n
                val = total / count
            if not np.isfinite(val):
                raise ValueError("早停期损失无效")
            history.append({"step": step, "train_batch_loss": np.mean(train_losses), metric: val})
            train_losses.clear()
            pd.DataFrame(history).to_csv(out / "training_loss.csv", index=False)
            if val < best:
                best = val
                torch.save({"step":step, "widths":widths, "state_dict":model.state_dict()}, out / "mlp.pt")
            print(step, val, flush=True)
    checkpoint = torch.load(out / "mlp.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    print(out.name, "冻结步数", checkpoint["step"], flush=True)
    return model
