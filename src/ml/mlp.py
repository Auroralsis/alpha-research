"""按交易日训练 MLP，支持 LambdaRank 与收益回归。"""
import numpy as np
import pandas as pd

from src.ml.dataset import daily_zscore


def regression_output(values, dates, mode):
    """按交易日标准化回归目标或输出，同日无差异时全部取零。"""
    if mode == "raw":
        return values
    if mode != "zscore":
        raise ValueError(f"未知的回归输出标准化方式 {mode}")
    values = pd.Series(values, index=dates.index, dtype="float64")
    if not np.isfinite(values).all():
        raise ValueError("回归目标或预测含有无效值")
    return daily_zscore(values, dates).fillna(0.0)


def fit_scale(x, names):
    """只在训练样本上拟合每列的填充值与缩放范围。无训练观测的列以零填补缺失值。"""
    stats = []
    for col in x.T:
        if np.isinf(col).any():
            raise ValueError("输入存在无穷值")
        seen = col[~np.isnan(col)]
        stats.append((len(seen), np.min(seen), np.median(seen), np.max(seen)) if len(seen) else (0, 0., 0., 0.))
    return pd.DataFrame(stats, columns=["count", "min", "median", "max"], index=pd.Index(names, name="feature"))


def transform(values, stats, scaling="none"):
    """默认用训练期中位数填补缺失值，显式选择 minmax 时再截尾和缩放。"""
    import torch

    x = np.ascontiguousarray(values, dtype=np.float32).copy()
    if np.isinf(x).any():
        raise ValueError("输入存在无穷值")
    lo, fill, hi = stats[["min", "median", "max"]].to_numpy(dtype=np.float32).T
    np.copyto(x, fill, where=np.isnan(x))
    if scaling == "minmax":
        np.clip(x, lo, hi, out=x)
        x -= lo
        x /= np.where(hi > lo, hi - lo, 1)
    elif scaling != "none":
        raise ValueError(f"未知的输入缩放方式 {scaling}")
    return torch.from_numpy(x)


def network(widths, *, regression=False, dropout=0.0):
    """逐股共享全连接打分器，按论文在各层使用 ReLU。"""
    from torch import nn

    layers = []
    for i, (n_in, n_out) in enumerate(zip(widths, widths[1:])):
        layers.extend([nn.Linear(n_in, n_out), nn.ReLU()])
        if dropout and i < len(widths) - 2:
            layers.append(nn.Dropout(dropout))
    # 正偏置避免初始化时输出 ReLU 对所有股票的梯度都为零。
    nn.init.constant_(layers[-2].bias, 1.)
    return nn.Sequential(*(layers[:-1] if regression else layers))


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


def mlp_periods(sections, names, out, *, ranking, scaling="none", regression_mode="raw"):
    """只用训练期拟合缩放统计，按交易日和股票代码对齐标签。"""
    import torch

    periods = []
    for period, rows, sizes in sections:
        ordered = rows.sort_values(["trade_date", "ticker"])
        values = ordered[names].to_numpy(dtype=np.float32)
        if scaling == "daily_zscore":
            features = torch.from_numpy(values)
        else:
            if period == "train":
                stats = fit_scale(values, names)
                stats.to_csv(out / "preprocessing.csv")
            features = transform(values, stats, scaling)
        features = features.split(sizes.tolist())
        target = ordered.grade.to_numpy(dtype=np.int64) if ranking else np.asarray(
            regression_output(ordered.label, ordered.trade_date, regression_mode), dtype=np.float32)
        grades = torch.from_numpy(target).split(sizes.tolist())
        periods.append((features, grades))
        print(period, len(sizes), len(rows), flush=True)
        del rows, ordered, values
    return periods


def fit_mlp(sections, names, cfg, out):
    """共用按日训练和早停流程；四个排序目标独立保存权重。"""
    ranking = cfg["objective"] == "lambdarank"
    periods = mlp_periods(sections, names, out, ranking=ranking, scaling=cfg.get("mlp_input_scaling", cfg["mlp"].get("input_scaling", "none")),
                          regression_mode=cfg["mlp"].get("regression_output", "raw"))
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

    scaling = cfg.get("mlp_input_scaling", cfg["mlp"].get("input_scaling", "none"))
    daily_zscore = scaling == "daily_zscore"
    stats = None if daily_zscore else pd.read_csv(out / "preprocessing.csv", index_col="feature", float_precision="round_trip").loc[names]

    @torch.no_grad()
    def predict(features):
        values = features.to_numpy(dtype=np.float32)
        inputs = torch.from_numpy(values) if daily_zscore else transform(values, stats, scaling)
        chunks = inputs.split(4096)
        result = {key: torch.cat([model(x.to(cfg["device"])).flatten().cpu() for x in chunks]).numpy()
                  for key, model in models.items()}
        return pd.DataFrame(result, index=features.index) if cfg["objective"] == "lambdarank" else result["mlp"]

    return predict


def train_network(periods, gains, cfg, out):
    """用一个目标拟合一个网络，按早停日期间的最佳指标恢复权重。"""
    import torch

    p = cfg["mlp"]
    widths = cfg["architecture"]
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(p["cpu_threads"])
    rng = np.random.default_rng(cfg["seed"])
    (train, train_grades), (stop, stop_grades) = periods
    device = torch.device(cfg["device"])
    model = network(widths, regression=gains is None, dropout=p.get("dropout", 0.0)).to(device)
    metric = "early_stop_mse" if gains is None else "early_stop_one_minus_ndcg"
    gain_table = None if gains is None else torch.tensor(gains, device=device, dtype=torch.float32)

    def objective(days, grade_days, indices, *, evaluate=False):
        xs = [days[i] for i in indices]
        if gain_table is None:
            prediction = model(torch.cat(xs).to(device)).flatten()
            target = torch.cat([grade_days[i] for i in indices]).to(device)
            return (prediction - target).square().mean(), len(target)
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
            model.eval()
            with torch.no_grad():
                total, count = 0., 0
                for i in range(0, len(stop), p["batch_days"]):
                    indices = range(i, min(i+p["batch_days"], len(stop)))
                    loss_value, n = objective(stop, stop_grades, indices, evaluate=True)
                    total += loss_value.item() * n
                    count += n
                val = total / count
            model.train()
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
