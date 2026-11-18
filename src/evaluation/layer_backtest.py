"""Equal-weight layer and fixed-count portfolios using the shared cash ledger."""

import numpy as np
import pandas as pd

from src.evaluation.portfolio import backtest_long_only, performance


def build_targets(frame, score, layer_count, top_counts):
    layers = [{} for _ in range(layer_count)]
    top = {count: {} for count in top_counts}
    for day, group in frame.groupby("trade_date", sort=True):
        usable = group.loc[np.isfinite(group[score]), ["ticker", score]]
        tickers = usable.sort_values([score, "ticker"], ascending=[False, True]).ticker.to_numpy()
        for targets, partition in zip(layers, np.array_split(tickers, layer_count)):
            targets[day] = partition.tolist()
        for count, targets in top.items():
            targets[day] = tickers[:count].tolist()
    return layers, top


def simulate_track(quotes, signals, holding, offset, fee_bps):
    return backtest_long_only(*quotes, signals, fee=fee_bps / 10000,
                              rebalance_every=holding, offset=offset, record_trades=False)[0]


def spread_track(top, bottom):
    """Compound the difference of the two funded legs' daily return series."""
    spread = top.daily_return - bottom.daily_return
    if (spread <= -1).any():
        raise ValueError("Spread NAV cannot be compounded after losing 100%")
    return top[["trade_date"]].assign(daily_return=spread, nav=(1 + spread).cumprod(),
                                    turnover_two_sided=np.nan, cost=np.nan, stale_value=np.nan)


def track_metrics(ledger, benchmark=None):
    stats = performance(ledger, benchmark)
    monthly = ledger.groupby(ledger.trade_date.dt.to_period("M")).daily_return.apply(
        lambda returns: (1 + returns).prod() - 1)
    stats["month_win_rate"] = monthly.gt(0).mean()
    if benchmark is None:
        stats["annual_turnover_one_way"] = stats["cost_initial_nav"] = np.nan
    else:
        stats["blocked_buys"] = ledger.n_blocked_buys.sum()
        stats["blocked_sells"] = ledger.n_blocked_sells.sum()
    return stats


def run_layer_tracks(frame, close_quotes, benchmark, design):
    rows, curves = [], []
    layers_count = design["layers"]
    base_fee = design["main_fee_bps"]

    def store(ledger, kind, number, fee, relative=True, save_curve=False):
        tags = dict(factor=factor, holding_days=holding, offset=offset,
                    fee_bps=fee, kind=kind, number=number)
        rows.append(dict(**tags, **track_metrics(ledger, benchmark if relative else None)))
        if save_curve:
            curves.append(ledger[["trade_date", "nav", "daily_return"]].assign(**tags))

    for factor in design["factors"]:
        print(f"Backtesting {factor}", flush=True)
        layers, top_counts = build_targets(frame, f"{factor}_neutral", layers_count, design["top_counts"])
        for holding in design["horizons"]:
            for offset in range(holding):
                for number, signals in enumerate(layers, start=1):
                    ledger = simulate_track(close_quotes, signals, holding, offset, base_fee)
                    extreme = number in (1, layers_count)
                    store(ledger, "layer", number, base_fee, save_curve=extreme)
                    if number == 1:
                        top_layer = ledger
                store(spread_track(top_layer, ledger), "spread", 0, base_fee, relative=False, save_curve=True)
                for fee in design["top_layer_fee_bps"]:
                    ledger = simulate_track(close_quotes, layers[0], holding, offset, fee)
                    store(ledger, "layer", 1, fee)
                if holding == design["main_holding_days"]:
                    for count, signals in top_counts.items():
                        ledger = simulate_track(close_quotes, signals, holding, offset, base_fee)
                        store(ledger, "top_n", count, base_fee)
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)


def summarize_tracks(tracks):
    grouped = tracks.groupby(["factor", "holding_days", "fee_bps", "kind", "number"], sort=True)
    return grouped.agg(n_tracks=("offset", "size"), mean_annual=("annual_return", "mean"),
                       min_annual=("annual_return", "min"), max_annual=("annual_return", "max"),
                       mean_total=("total_return", "mean"), mean_sharpe=("sharpe", "mean"),
                       mean_drawdown=("max_drawdown", "mean"), mean_month_win=("month_win_rate", "mean"),
                       mean_annual_relative=("annual_relative_return", "mean"),
                       mean_total_relative=("relative_total_return", "mean"),
                       outperform_tracks=("relative_total_return", lambda returns: returns.gt(0).sum(min_count=1)
                                          if returns.notna().any() else np.nan)).reset_index()
