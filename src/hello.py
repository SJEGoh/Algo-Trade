import numpy as np
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import t
import statsmodels.api as sm

tickers = ["KO", "PEP"]

S1_ticker = yf.Ticker(tickers[0])
S2_ticker = yf.Ticker(tickers[1])

S1_data = S1_ticker.history(period = '15y')[["Close"]]
S2_data = S2_ticker.history(period = '15y')[["Close"]]

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

hypothesis_test(S1_data, S2_data, 10, 0.001, 2)
