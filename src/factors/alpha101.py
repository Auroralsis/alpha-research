"""按信号日的历史成分名单计算 Alpha101，连续相同名单共用一次窗口计算。"""

from functools import lru_cache
from itertools import groupby

import numpy as np
import pandas as pd
from numpy import abs, log, maximum, minimum, sign

from src.data.research_panel import CSI500_INPUTS, parse_dates
from src.factors.neutralization import align_industry_history
from src.factors.price_volume import (
    at_most, choose, correlation, covariance, cross_sectional_rank as rank, decay_linear, delay, delta,
    divide, indneutralize, less, product, scale, signedpower, stddev, ts_argmax, ts_argmin, ts_max,
    ts_min, ts_rank, ts_sum,
)

# Alpha48 的 250 日相关包含两次单日差分。其他公式的嵌套窗口更短。
HISTORY_DAYS = 251


def alpha101_batches(panel, keys):
    """返回成分变动区间的因子长表；历史窗口中的行业使用各历史日的有效行业。"""
    bars = [panel.matrix(name) for name in ('open_adj','high_adj','low_adj','close_adj','vol','vwap_adj')]
    caps = pd.read_parquet(CSI500_INPUTS['caps'], columns=['trade_date','ts_code','total_mv'])
    caps['trade_date'] = parse_dates(caps.trade_date).to_numpy()
    caps = caps.pivot(index='trade_date', columns='ts_code', values='total_mv').reindex_like(bars[0])
    history = pd.read_parquet(CSI500_INPUTS['industries'])
    stock_days = pd.MultiIndex.from_product([panel.calendar, panel.tickers], names=['trade_date','ticker']).to_frame(index=False)
    codes = pd.factorize(align_industry_history(stock_days, history))[0].reshape(bars[0].shape)
    membership = keys.sort_values(['trade_date','ticker']).groupby('trade_date').ticker.apply(tuple)
    for symbols, group in groupby(membership.items(), key=lambda item: item[1]):
        dates = pd.DatetimeIndex([day for day, _ in group])
        rows = panel.calendar.get_indexer(dates)
        if rows.min() < 0:
            raise ValueError("信号日不在交易日历中")
        begin = max(0, rows[0] - HISTORY_DAYS)
        span = slice(begin, rows[-1]+1)
        cols = panel.tickers.get_indexer(symbols)
        if (cols < 0).any():
            raise ValueError("成员不在行情面板中")
        inputs = [np.asfortranarray(field.to_numpy()[span, cols]) for field in [*bars, caps]]
        universe = pd.MultiIndex.from_product([dates, symbols], names=['trade_date', 'ticker'])
        output = {}
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            for name, scores in formula_values(*inputs, codes[span, cols]):
                values = scores[rows-begin].ravel()
                output[name] = np.where(np.isfinite(values), values, np.nan)
        yield pd.DataFrame(output, index=universe).reset_index()


def formula_values(open_price, high, low, close, volume, vwap, cap, industry):
    """输入同形状的 [交易日 × 股票] 矩阵，按研报编号逐个返回分数矩阵。"""
    @lru_cache(maxsize=None)
    def adv(days):
        return ts_sum(volume, days) / days

    change = delta(close, 1)
    returns = divide(change, delay(close, 1))
    mean2 = ts_sum(close, 2) / 2
    mean8 = ts_sum(close, 8) / 8
    slope_change = (delay(close, 20) - 2 * delay(close, 10) + close) / 10

    def by_industry(values):
        return indneutralize(values, industry)

    def blend(first, second, weight):
        return first * weight + second * (1 - weight)

    def rank_decay(values, days):
        return rank(decay_linear(values, days))

    def ts_rank_decay(values, decay_days, rank_days):
        return ts_rank(decay_linear(values, decay_days), rank_days)

    def ts_rank_corr(first, first_days, second, second_days, corr_days):
        return correlation(ts_rank(first, first_days), ts_rank(second, second_days), corr_days)

    def directional_change(days):
        """窗口内涨跌方向一致时取当日价差，其余取反。"""
        return choose(less(0, ts_min(change, days)), change,
                      choose(less(ts_max(change, days), 0), change, -change))

    volume_rank = rank(volume)
    close_rank = rank(close)
    high_rank = rank(high)
    open_rank = rank(open_price)
    vwap_rank = rank(vwap)
    mid_price = (high + low) / 2
    intraday_change = close - open_price
    relative_volume = divide(volume, adv(20))
    open_volume_corr = correlation(open_price, volume, 10)
    change7 = delta(close, 7)
    change3 = delta(close, 3)

    yield "Alpha1", rank(ts_argmax(signedpower(choose(less(returns, 0), stddev(returns, 20), close), 2), 5)) - 0.5
    yield "Alpha2", -correlation(rank(delta(log(volume), 2)), rank(divide(intraday_change, open_price)), 6)
    yield "Alpha3", -correlation(open_rank, volume_rank, 10)
    yield "Alpha4", -ts_rank(rank(low), 9)
    yield "Alpha5", rank(open_price - ts_sum(vwap, 10) / 10) * -abs(rank(close - vwap))
    yield "Alpha6", -open_volume_corr
    yield "Alpha7", choose(less(adv(20), volume), -ts_rank(abs(change7), 60) * sign(change7), -1)
    yield "Alpha8", -rank(delta(ts_sum(open_price, 5) * ts_sum(returns, 5), 10))
    yield "Alpha9", directional_change(5)
    yield "Alpha10", rank(directional_change(4))
    yield "Alpha11", (rank(ts_max(vwap - close, 3)) + rank(ts_min(vwap - close, 3))) * rank(delta(volume, 3))
    yield "Alpha12", sign(delta(volume, 1)) * -change
    yield "Alpha13", -rank(covariance(close_rank, volume_rank, 5))
    yield "Alpha14", -rank(delta(returns, 3)) * open_volume_corr
    yield "Alpha15", -ts_sum(rank(correlation(high_rank, volume_rank, 3)), 3)
    yield "Alpha16", -rank(covariance(high_rank, volume_rank, 5))
    yield "Alpha17", -rank(ts_rank(close, 10)) * rank(delta(change, 1)) * rank(ts_rank(relative_volume, 5))
    yield "Alpha18", -rank(stddev(abs(intraday_change), 5) + intraday_change + correlation(close, open_price, 10))
    yield "Alpha19", -sign(close - delay(close, 7) + change7) * (1 + rank(1 + ts_sum(returns, 250)))
    yield "Alpha20", (-rank(open_price - delay(high, 1)) * rank(open_price - delay(close, 1))
        * rank(open_price - delay(low, 1)))
    yield "Alpha21", (choose(less(mean8 + stddev(close, 8), mean2), -1,
        choose(less(mean2, mean8 - stddev(close, 8)), 1, choose(at_most(1, relative_volume), 1, -1))))
    yield "Alpha22", -(delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20)))
    yield "Alpha23", choose(less(ts_sum(high, 20) / 20, high), -delta(high, 2), 0)
    yield "Alpha24", (choose(at_most(divide(delta(ts_sum(close, 100) / 100, 100), delay(close, 100)), 0.05),
        ts_min(close, 100) - close, -change3))
    yield "Alpha25", rank(-returns * adv(20) * vwap * (high - close))
    yield "Alpha26", -ts_max(ts_rank_corr(volume, 5, high, 5, 5), 3)
    yield "Alpha27", choose(less(0.5, rank(ts_sum(correlation(volume_rank, vwap_rank, 6), 2) / 2)), -1, 1)
    yield "Alpha28", scale(correlation(adv(20), low, 5) + mid_price - close)
    yield "Alpha29", (ts_min(product(rank(rank(scale(log(ts_sum(ts_min(rank(rank(-rank(delta(close - 1, 5)))), 2), 1))))),
        1), 5) + ts_rank(delay(-returns, 6), 5))
    yield "Alpha30", (divide((1.0 - rank(sign(close - delay(close, 1)) + sign(delay(close, 1) - delay(close, 2))
        + sign(delay(close, 2) - delay(close, 3)))) * ts_sum(volume, 5), ts_sum(volume, 20)))
    yield "Alpha31", rank_decay(-rank(delta(close, 10)), 10) + rank(-change3) + sign(scale(correlation(adv(20), low, 12)))
    yield "Alpha32", scale(ts_sum(close, 7) / 7 - close) + 20 * scale(correlation(vwap, delay(close, 5), 230))
    yield "Alpha33", rank(-(1 - divide(open_price, close)) ** 1)
    yield "Alpha34", rank(1 - rank(divide(stddev(returns, 2), stddev(returns, 5))) + (1 - rank(change)))
    yield "Alpha35", ts_rank(volume, 32) * (1 - ts_rank(close + high - low, 16)) * (1 - ts_rank(returns, 32))
    yield "Alpha36", (2.21 * rank(correlation(intraday_change, delay(volume, 1), 15)) + 0.7 * rank(open_price - close)
        + 0.73 * rank(ts_rank(delay(-returns, 6), 5)) + rank(abs(correlation(vwap, adv(20), 6))) + 0.6
        * rank((ts_sum(close, 200) / 200 - open_price) * intraday_change))
    yield "Alpha37", rank(correlation(delay(open_price - close, 1), close, 200)) + rank(open_price - close)
    yield "Alpha38", -rank(ts_rank(close, 10)) * rank(divide(close, open_price))
    yield "Alpha39", -rank(change7 * (1 - rank_decay(relative_volume, 9))) * (1 + rank(ts_sum(returns, 250)))
    yield "Alpha40", -rank(stddev(high, 10)) * correlation(high, volume, 10)
    yield "Alpha41", (high * low) ** 0.5 - vwap
    yield "Alpha42", divide(rank(vwap - close), rank(vwap + close))
    yield "Alpha43", ts_rank(relative_volume, 20) * ts_rank(-change7, 8)
    yield "Alpha44", -correlation(high, volume_rank, 5)
    yield "Alpha45", (-rank(ts_sum(delay(close, 5), 20) / 20) * correlation(close, volume, 2)
        * rank(correlation(ts_sum(close, 5), ts_sum(close, 20), 2)))
    yield "Alpha46", choose(less(0.25, slope_change), -1, choose(less(slope_change, 0), 1, -change))
    yield "Alpha47", (divide(rank(divide(1, close)) * volume, adv(20))
        * divide(high * rank(high - close), ts_sum(high, 5) / 5) - rank(vwap - delay(vwap, 5)))
    yield "Alpha48", (divide(by_industry(divide(correlation(change, delta(delay(close, 1), 1), 250) * change, close)),
        ts_sum(divide(change, delay(close, 1)) ** 2, 250)))
    yield "Alpha49", choose(less(slope_change, -0.1), 1, -change)
    yield "Alpha50", -ts_max(rank(correlation(volume_rank, vwap_rank, 5)), 5)
    yield "Alpha51", choose(less(slope_change, -0.05), 1, -change)
    yield "Alpha52", ((-ts_min(low, 5) + delay(ts_min(low, 5), 5))
        * rank((ts_sum(returns, 240) - ts_sum(returns, 20)) / 220) * ts_rank(volume, 5))
    yield "Alpha53", -delta(divide(close - low - (high - close), close - low), 9)
    yield "Alpha54", divide(-((low - close) * open_price ** 5), (low - high) * close ** 5)
    yield "Alpha55", -correlation(rank(divide(close - ts_min(low, 12), ts_max(high, 12) - ts_min(low, 12))), volume_rank, 6)
    yield "Alpha56", 0 - 1 * (rank(divide(ts_sum(returns, 10), ts_sum(ts_sum(returns, 2), 3))) * rank(returns * cap))
    yield "Alpha57", 0 - 1 * divide(close - vwap, decay_linear(rank(ts_argmax(close, 30)), 2))
    yield "Alpha58", -ts_rank_decay(correlation(by_industry(vwap), volume, 3.92795), 7.89291, 5.50322)
    yield "Alpha59", -ts_rank_decay(correlation(by_industry(blend(vwap, vwap, 0.728317)), volume, 4.25197), 16.2289, 8.19648)
    yield "Alpha60", (0 - 1 * (2 * scale(rank(divide(close - low - (high - close), high - low) * volume))
        - scale(rank(ts_argmax(close, 10)))))
    yield "Alpha61", less(rank(vwap - ts_min(vwap, 16.1219)), rank(correlation(vwap, adv(180), 17.9282)))
    yield "Alpha62", (-less(rank(correlation(vwap, ts_sum(adv(20), 22.4101), 9.91009)),
        rank(less(open_rank * 2, rank(mid_price) + high_rank))))
    yield "Alpha63", (-(rank_decay(delta(by_industry(close), 2.25164), 8.22237)
        - rank_decay(correlation(blend(vwap, open_price, 0.318108), ts_sum(adv(180), 37.2467), 13.557), 12.2883)))
    yield "Alpha64", (-less(rank(correlation(ts_sum(blend(open_price, low, 0.178404), 12.7054), ts_sum(adv(120), 12.7054),
        16.6208)), rank(delta(blend(mid_price, vwap, 0.178404), 3.69741))))
    yield "Alpha65", (-less(rank(correlation(blend(open_price, vwap, 0.00817205), ts_sum(adv(60), 8.6911), 6.40374)),
        rank(open_price - ts_min(open_price, 13.635))))
    yield "Alpha66", (-(rank_decay(delta(vwap, 3.51013), 7.23052)
        + ts_rank_decay(divide(blend(low, low, 0.96633) - vwap, open_price - mid_price), 11.4157, 6.72611)))
    yield "Alpha67", (-rank(high - ts_min(high, 2.14593))
        ** rank(correlation(by_industry(vwap), by_industry(adv(20)), 6.02936)))
    yield "Alpha68", (-less(ts_rank(correlation(high_rank, rank(adv(15)), 8.91644), 13.9333),
        rank(delta(blend(close, low, 0.518371), 1.06157))))
    yield "Alpha69", (-rank(ts_max(delta(by_industry(vwap), 2.72412), 4.79344))
        ** ts_rank(correlation(blend(close, vwap, 0.490655), adv(20), 4.92416), 9.0615))
    yield "Alpha70", -rank(delta(vwap, 1.29456)) ** ts_rank(correlation(by_industry(close), adv(50), 17.8256), 17.9171)
    yield "Alpha71", (maximum(ts_rank_decay(ts_rank_corr(close, 3.43976, adv(180), 12.0647, 18.0175), 4.20501, 15.6948),
        ts_rank_decay(rank(low + open_price - (vwap + vwap)) ** 2, 16.4662, 4.4388)))
    yield "Alpha72", (divide(rank_decay(correlation(mid_price, adv(40), 8.93345), 10.1519),
        rank_decay(ts_rank_corr(vwap, 3.72469, volume, 18.5188, 6.86671), 2.95011)))
    yield "Alpha73", (-maximum(rank_decay(delta(vwap, 4.72775), 2.91864),
        ts_rank_decay(-divide(delta(blend(open_price, low, 0.147155), 2.03608),
        blend(open_price, low, 0.147155)), 3.33829, 16.7411)))
    yield "Alpha74", (-less(rank(correlation(close, ts_sum(adv(30), 37.4843), 15.1365)),
        rank(correlation(rank(blend(high, vwap, 0.0261661)), volume_rank, 11.4791))))
    yield "Alpha75", less(rank(correlation(vwap, volume, 4.24304)), rank(correlation(rank(low), rank(adv(50)), 12.4413)))
    yield "Alpha76", (-maximum(rank_decay(delta(vwap, 1.24383), 11.8259),
        ts_rank_decay(ts_rank(correlation(by_industry(low), adv(81), 8.14941), 19.569), 17.1543, 19.383)))
    yield "Alpha77", (minimum(rank_decay(mid_price + high - (vwap + high), 20.0451),
        rank_decay(correlation(mid_price, adv(40), 3.1614), 5.64125)))
    yield "Alpha78", (rank(correlation(ts_sum(blend(low, vwap, 0.352233), 19.7428), ts_sum(adv(40), 19.7428), 6.83313))
        ** rank(correlation(vwap_rank, volume_rank, 5.77492)))
    yield "Alpha79", (less(rank(delta(by_industry(blend(close, open_price, 0.60733)), 1.23438)),
        rank(ts_rank_corr(vwap, 3.60973, adv(150), 9.18637, 14.6644))))
    yield "Alpha80", (-rank(sign(delta(by_industry(blend(open_price, high, 0.868128)), 4.04545)))
        ** ts_rank(correlation(high, adv(10), 5.11456), 5.53756))
    yield "Alpha81", (-less(rank(log(product(rank(rank(correlation(vwap, ts_sum(adv(10), 49.6054), 8.47743)) ** 4),
        14.9655))), rank(correlation(vwap_rank, volume_rank, 5.07914))))
    yield "Alpha82", (-minimum(rank_decay(delta(open_price, 1.46063), 14.8717),
        ts_rank_decay(correlation(by_industry(volume), blend(open_price, open_price, 0.634196), 17.4842),
        6.92131, 13.4283)))
    yield "Alpha83", (divide(rank(delay(divide(high - low, ts_sum(close, 5) / 5), 2)) * volume_rank,
        divide(divide(high - low, ts_sum(close, 5) / 5), vwap - close)))
    yield "Alpha84", signedpower(ts_rank(vwap - ts_max(vwap, 15.3217), 20.7127), delta(close, 4.96796))
    yield "Alpha85", (rank(correlation(blend(high, close, 0.876703), adv(30), 9.61331))
        ** rank(ts_rank_corr(mid_price, 3.70596, volume, 10.1595, 7.11408)))
    yield "Alpha86", (-less(ts_rank(correlation(close, ts_sum(adv(20), 14.7444), 6.00049), 20.4195),
        rank(open_price + close - (vwap + open_price))))
    yield "Alpha87", (-maximum(rank_decay(delta(blend(close, vwap, 0.369701), 1.91233), 2.65461),
        ts_rank_decay(abs(correlation(by_industry(adv(81)), close, 13.4132)), 4.89768, 14.4535)))
    yield "Alpha88", (minimum(rank_decay(open_rank + rank(low) - (high_rank + close_rank), 8.06882),
        ts_rank_decay(ts_rank_corr(close, 8.44728, adv(60), 20.6966, 8.01266), 6.65053, 2.61957)))
    yield "Alpha89", (ts_rank_decay(correlation(blend(low, low, 0.967285), adv(10), 6.94279), 5.51607, 3.79744)
        - ts_rank_decay(delta(by_industry(vwap), 3.48158), 10.1466, 15.3012))
    yield "Alpha90", (-rank(close - ts_max(close, 4.66719))
        ** ts_rank(correlation(by_industry(adv(40)), low, 5.38375), 3.21856))
    yield "Alpha91", (-(ts_rank_decay(decay_linear(correlation(by_industry(close), volume, 9.74928), 16.398), 3.83219,
        4.8667) - rank_decay(correlation(vwap, adv(30), 4.01303), 2.6809)))
    yield "Alpha92", (minimum(ts_rank_decay(less(mid_price + close, low + open_price), 14.7221, 18.8683),
        ts_rank_decay(correlation(rank(low), rank(adv(30)), 7.58555), 6.94024, 6.80584)))
    yield "Alpha93", (divide(ts_rank_decay(correlation(by_industry(vwap), adv(81), 17.4193), 19.848, 7.54455),
        rank_decay(delta(blend(close, vwap, 0.524434), 2.77377), 16.2664)))
    yield "Alpha94", (-rank(vwap - ts_min(vwap, 11.5783))
        ** ts_rank(ts_rank_corr(vwap, 19.6462, adv(60), 4.02992, 18.0926), 2.70756))
    yield "Alpha95", (less(rank(open_price - ts_min(open_price, 12.4105)),
        ts_rank(rank(correlation(ts_sum(mid_price, 19.1351), ts_sum(adv(40), 19.1351), 12.8742)) ** 5, 11.7584)))
    price_volume = ts_rank_decay(correlation(vwap_rank, volume_rank, 3.83878), 4.16783, 8.38151)
    close_volume = ts_rank_corr(close, 7.45404, adv(60), 4.13242, 3.65459)
    peak_position = ts_argmax(close_volume, 12.6556)
    yield "Alpha96", -maximum(price_volume, ts_rank_decay(peak_position, 14.0365, 13.4143))
    yield "Alpha97", (-(rank_decay(delta(by_industry(blend(low, vwap, 0.721001)), 3.3705), 20.4523)
        - ts_rank_decay(ts_rank(ts_rank_corr(low, 7.87871, adv(60), 17.255, 4.97547), 18.5925), 15.7152,
        6.71659)))
    yield "Alpha98", (rank_decay(correlation(vwap, ts_sum(adv(5), 26.4719), 4.58418), 7.18088)
        - rank_decay(ts_rank(ts_argmin(correlation(open_rank, rank(adv(15)), 20.8187), 8.62571), 6.95668),
        8.07206))
    yield "Alpha99", (-less(rank(correlation(ts_sum(mid_price, 19.8975), ts_sum(adv(60), 19.8975), 8.8136)),
        rank(correlation(low, volume, 6.28259))))
    close_location = divide(close * 2 - low - high, high - low) * volume
    location_signal = scale(by_industry(by_industry(rank(close_location))))
    volume_signal = scale(by_industry(correlation(close, rank(adv(20)), 5) - rank(ts_argmin(close, 30))))
    yield "Alpha100", -((1.5 * location_signal - volume_signal) * relative_volume)
    yield "Alpha101", divide(intraday_change, high - low + 0.001)
