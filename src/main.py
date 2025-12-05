from pair import PairModel, PairTrader, Portfolio
from HypoTest import RegimeDetector
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np


def backtest(df1, df2, h, K):
    prices = pd.concat([df1,df2], axis = 1).dropna()
    prices.columns = ["S1", "S2"]
    portfolio = Portfolio(100)

    train_i = 10
    train_set = prices.iloc[:train_i]
    test_set = prices.iloc[train_i:]
    train_rows = []   # collect rows here

    pair_model = PairModel()
    pair_model.fit_hedge(train_set)

    for i, row in train_set.iterrows():
        spread = pair_model.compute_spread(row["S1"], row["S2"])
        train_rows.append((i, spread))

    train_spread = pd.DataFrame(train_rows, columns=["Date", "Spread"]).set_index("Date")
    
    detector = RegimeDetector(h = h, K = K)
    detector.initialize(train_spread["Spread"])
    strategy = PairTrader(2, 0.75, train_spread["Spread"].mean(), train_spread["Spread"].std())
    last_k = train_set.tail(5)
    r = 0
    new_params = (0, 0)
    for i, row in test_set.iterrows():
        spread = pair_model.compute_spread(row["S1"], row["S2"])
        last_k = last_k.iloc[1:]
        last_k.loc[i] = row
        
        if rg := detector.update(spread):
            r += 1
            pair_model.fit_hedge(last_k)
            last_k_spread = []
            for j, row_k in last_k.iterrows():
                spread = pair_model.compute_spread(row_k["S1"], row_k["S2"])
                last_k_spread.append((j, spread))
            last_k_spread = pd.DataFrame(last_k_spread, columns=["Date", "Spread"]).set_index("Date")
            detector.reset_baseline(last_k_spread["Spread"])
            new_params = detector.get_params()
            strategy.set_params(new_params)
        position = strategy.generate(spread, rg, detector.L, K)
        prices = {"S1": row["S1"], "S2": row["S2"]}

        portfolio.update_position(i, prices, position, pair_model.b)

    d = pd.DataFrame(portfolio.history)
    d.columns = ["Date", "Value"]
    d = d.set_index("Date")
    daily_returns = d.pct_change().dropna()
    mean_daily = daily_returns.mean()
    std_daily = daily_returns.std()
    cumu_return = (d["Value"][-1]/d["Value"][0]) - 1
    std = std_daily * np.sqrt(252 * 10)

    rfr = 0.02

    sharpe = (cumu_return - rfr/100)/ std
    print(sharpe)
    print(r)
    d.plot()
    plt.show()




tickers = ["GLD", "SLV"]

S1_ticker = yf.Ticker(tickers[0])
S2_ticker = yf.Ticker(tickers[1])

S1_data = S1_ticker.history(period = '10y')[["Close"]]
S2_data = S2_ticker.history(period = '10y')[["Close"]]


print(backtest(S1_data, S2_data, h = 0.005, K = 1))

# MSFT + ADBE, 0.04, 5, 15 years, 2.29, z-entry = 2, z-exit = 0.75, tail: 5
# GLD + SLV, 0.005, 1, 10 year, 3.93
# BTC, ETH, 0.05, 5, 5 year, 2.08, z-entry = 2, z-exit = 0.75, tail: 5
