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
'''
tickers = ["BTC-USD", "ETH-USD"]

S1_ticker = yf.Ticker(tickers[0])
S2_ticker = yf.Ticker(tickers[1])

S1_data = S1_ticker.history(period = '1y')[["Close"]]
S2_data = S2_ticker.history(period = '1y')[["Close"]]

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
            count += 1
            continue
        new_kappa = kappa + 1.0
        new_beta = beta + 0.5 * (kappa*(spread-mu)**2)/new_kappa
        new_mu = ((kappa) * mu + spread)/new_kappa
        new_alpha = alpha + 0.5
        kappa, beta, mu, alpha = new_kappa, new_beta, new_mu, new_alpha

    plt.show()
    print(count)
    track.plot()
    plt.axhline(0, color = 'black')
    plt.show()

hypothesis_test(S1_data, S2_data, 10, 0.2, 10)
'''
# Tail length changes a lot of stuff, make sure to play with it
# Ok cool it works, now how to make it into an object

class RegimeDetector:
    def __init__(self, h: float = 0.1, K: int = 5):
        self.prior = np.log((1-h)/h)
        self.K = K

        self.mu0 = None
        self.sd0 = None
        self.kappa0 = 1.0
        self.alpha0 = 2.0
        self.beta0 = None

        self.mu = None
        self.kappa = None
        self.alpha = None
        self.beta = None

        self.L = 0.0
        self.initialized = False

        self.lam = 0.99
    
    def initialize(self, initial_spread):
        self.mu0 = initial_spread.mean()
        self.sd0 = initial_spread.std()
        self.L = 0.0

        self.beta0 = self.sd0 ** 2
        self.mu = self.mu0
        self.kappa = self.kappa0
        self.alpha = self.alpha0
        self.beta = self.beta0
        self.initialized = True
    
    def _loglikelihood(self, spread, alpha, beta, kappa, mu):
        nu = 2.0 * alpha
        scale = np.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
        return t.logpdf(spread, df=nu, loc=mu, scale=scale)
    
    def reset_baseline(self, last_k: pd.Series):

        self.mu0 = last_k.mean()
        self.sd0 = last_k.std()
        self.beta0 = (self.sd0 ** 2) * 1.0
        self.mu = self.mu0
        self.kappa = self.kappa0
        self.alpha = self.alpha0
        self.beta = self.beta0
        self.L = 0.0

    def update(self, spread: float):
        if not self.initialized:
            raise ValueError("Not initialized")

        L0 = self._loglikelihood(spread, self.alpha, self.beta, self.kappa, self.mu)
        L1 = self._loglikelihood(spread, self.alpha0, self.beta0, self.kappa0, self.mu0)
        self.L += L1 - L0 + self.prior

        if self.L < -self.K:
            return True

        self.kappa = 1.0 + self.lam * (self.kappa - 1.0)
        self.alpha = self.alpha0 + self.lam * (self.alpha - self.alpha0)
        self.beta  = self.beta0  + self.lam * (self.beta  - self.beta0)
        new_kappa = self.kappa + 1.0
        new_beta = self.beta + 0.5 * (self.kappa * (spread - self.mu) ** 2) / new_kappa
        new_mu = (self.kappa * self.mu + spread) / new_kappa
        new_alpha = self.alpha + 0.5

        self.kappa, self.beta, self.mu, self.alpha = new_kappa, new_beta, new_mu, new_alpha
        return False
    
    def get_params(self):
        return (self.mu, self.sd0)
