'''
Docstring for HypoTest

How tf is this supposed to work lmao
Get ready for a mix of wtf is this

arguments:
- data (will figure out how to interweave this, lets just get this
  working first :P)
- hazard function (likely just 1/whatever number I feel like)
- Will probs just use t-distribution cause why not

So how I think this works is I just have to keep track of two probabilities,
one of the 
'''

from scipy.stats import t
import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t
import statsmodels.api as sm

tickers = ["BTC-USD", "ETH-USD"]

S1_ticker = yf.Ticker(tickers[0])
S2_ticker = yf.Ticker(tickers[1])

S1_data = S1_ticker.history(period = '1y')[["Close"]]
S2_data = S2_ticker.history(period = '1y')[["Close"]]

def hypothesis_test(S1, S2, train_i, h = 0.01, K = 2):
    S1 = S1.reset_index().merge(S2.reset_index(), on = "Date", how = "inner", suffixes = ["_S1", "_S2"]).set_index("Date")
    param_set = S1.iloc[:train_i]
    test_set = S1.iloc[train_i:]
    x = sm.add_constant(param_set["Close_S1"])
    results = sm.OLS(param_set["Close_S2"], x).fit()
    b = results.params["Close_S1"]

    initial_spread = param_set["Close_S2"] - b * param_set["Close_S1"] - results.params["const"]
    track = initial_spread
    mu0 = initial_spread.mean()
    sd0 = initial_spread.std()
    bs = S1["Close_S2"] - b * S1["Close_S1"] - results.params["const"]
    bs.plot()
    plt.axhline(mu0, color = "black")

    nu0 = 5
    nu1 = 3
    mu1 = mu0
    sd1 = sd0 * 3
    L = 0.0
    count = 0
    last_10 = param_set.tail(5)
    for i, row in test_set.iterrows():
        last_10 = last_10.iloc[1:]   
        last_10.loc[i] = row   
        
        spread = row["Close_S2"] - b * row["Close_S1"] - results.params["const"]
        track = pd.concat([track, pd.Series(spread, index = [i])])
        L0 = t.logpdf(spread, df = nu0, loc = mu0, scale = sd0)
        L1 = t.logpdf(spread, df = nu1, loc = mu1, scale = sd1)
        prior = np.log((1-h)/h)
        L += (L0 - L1 + prior)
        if L < -K:
            count += 1
            x = sm.add_constant(last_10["Close_S1"])
            results = sm.OLS(last_10["Close_S2"], x).fit()
            b = results.params["Close_S1"]
            last_10_spread = last_10["Close_S2"] - b * last_10["Close_S1"] - results.params["const"]
            mu0 = last_10_spread.mean()
            mu1 = mu0
            sd0 = last_10_spread.std()
            print(last_10_spread)
            sd1 = sd0 * 3
            L = 0.0
            plt.axvline(i)

    plt.show()
    print(count)
    track.plot()
    plt.axhline(0, color = 'black')
    plt.show()

hypothesis_test(S1_data, S2_data, 10, 0.5, 3)

# Tail length changes a lot of stuff, make sure to play with it
