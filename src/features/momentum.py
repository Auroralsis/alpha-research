import pandas as pd

# calculate momentum, the adjusted prices must be sorted according to trade date
# window > 0
def calc_momentum(prices : pd.Series, window : int) -> pd.Series :
    return prices.pct_change(window)