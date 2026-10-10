import pandas as pd

def calc_future_return(prices : pd.Series , horizon):
    return prices.shift(-horizon) / prices - 1