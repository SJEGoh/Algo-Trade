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

import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t
import statsmodels.api as sm

tickers = ["KO", "PEP"]

S1_ticker = yf.Ticker(tickers[0])
S2_ticker = yf.Ticker(tickers[1])

S1_data = S1_ticker.history(period = '5y')[["Close"]]
S2_data = S2_ticker.history(period = '5y')[["Close"]]

def loglikelihood(spread, alpha, beta, kappa, mu):
    nu = 2.0 * alpha
    scale = np.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
    return t.logpdf(spread, df=nu, loc=mu, scale=scale)

def hypothesis_test(S1, S2, train_i, h = 0.01, K = 2):
    S1 = S1.reset_index().merge(S2.reset_index(), on = "Date", how = "inner", suffixes = ["_S1", "_S2"]).set_index("Date")
    param_set = S1.iloc[:train_i]
    test_set = S1.iloc[train_i:]
    x = sm.add_constant(param_set["Close_S1"])
    results = sm.OLS(param_set["Close_S2"], x).fit()
    b = results.params["Close_S1"]
    const = results.params["const"]

    initial_spread = param_set["Close_S2"] - b * param_set["Close_S1"] - const
    track = initial_spread
    mu0 = initial_spread.mean()
    sd0 = initial_spread.std()
    bs = S1["Close_S2"] - b * S1["Close_S1"] - const
    bs.plot()
    plt.axhline(mu0, color = "black")

    kappa0, alpha0, beta0 = 1.0, 2.0, (sd0**2)*1.0
    mu, kappa, alpha, beta = mu0, kappa0, alpha0, beta0

    prior = np.log((1-h)/h)
    L = 0.0
    count = 0
    last_10 = param_set.tail(5)
    for i, row in test_set.iterrows():
        last_10 = last_10.iloc[1:]   
        last_10.loc[i] = row   
        
        spread = row["Close_S2"] - b * row["Close_S1"] - const
        track = pd.concat([track, pd.Series(spread, index = [i])])
        L0 = loglikelihood(spread, alpha, beta, kappa, mu)
        L1 = loglikelihood(spread, alpha0, beta0, kappa0, mu0)
        L += (L1 - L0 + prior)
        if L < -K:
            count += 1
            x = sm.add_constant(last_10["Close_S1"])
            results = sm.OLS(last_10["Close_S2"], x).fit()
            b = results.params["Close_S1"]
            const = results.params["const"]
            last_10_spread = last_10["Close_S2"] - b * last_10["Close_S1"] - const
            mu0 = last_10_spread.mean()
            sd0 = last_10_spread.std()
            beta0 = (sd0**2)*1.0
            mu, kappa, alpha, beta = mu0, kappa0, alpha0, beta0
            L = 0.0
            plt.axvline(i)
            continue
        
        kappa += 1.0
        beta += 0.5 * ((kappa-1)*(spread-mu)**2)/kappa
        mu = ((kappa-1.0) * mu + spread)/kappa
        alpha += 0.5
        print(mu)

    plt.show()
    print(count)
    track.plot()
    plt.axhline(0, color = 'black')
    plt.show()

hypothesis_test(S1_data, S2_data, 10, 0.5, 10)

# Tail length changes a lot of stuff, make sure to play with it
