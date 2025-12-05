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
        last_k_spread = []
        if rg := detector.update(spread):
            r += 1
            pair_model.fit_hedge(last_k)
            for j, row_k in last_k.iterrows():
                spread = pair_model.compute_spread(row_k["S1"], row_k["S2"])
                last_k_spread.append((j, spread))
            last_k_spread = pd.DataFrame(last_k_spread, columns=["Date", "Spread"]).set_index("Date")
            detector.reset_baseline(last_k_spread["Spread"])
            new_params = detector.get_params()
            strategy.set_params(new_params)
        position = strategy.generate(spread, rg)
        prices = {"S1": row["S1"], "S2": row["S2"]}
        if detector.L < 0:
            print("L =", detector.L)
        portfolio.update_position(i, prices, position, pair_model.b)

    d = pd.DataFrame(portfolio.history)
    d.columns = ["Date", "Value"]
    d = d.set_index("Date")
    daily_return = d.copy().pct_change()
    mean = daily_return.mean()
    sd = daily_return.std()
    cumu_return = d.iloc[-1]/d.iloc[0] - 1
    sd = sd * np.sqrt(252)
    rfr = 0.02
    sharpe = (cumu_return-rfr)/sd
    print(sharpe)
    d.plot()
    plt.show()




tickers = ["BTC-USD", "ETH-USD"]

S1_ticker = yf.Ticker(tickers[0])
S2_ticker = yf.Ticker(tickers[1])

S1_data = S1_ticker.history(period = '1y')[["Close"]]
S2_data = S2_ticker.history(period = '1y')[["Close"]]


print(backtest(S1_data, S2_data, h = 0.05, K = 10))

# MSFT + ADBE, 0.04, 5, 10 years, 2.94, z-entry = 2, z-exit = 0.75, tail: 5
# KO + PEP, 0.004, 2, 15 years, 2.41, z-entry = 2, z-exit = 0.75, tail: 5
# BTC, ETH, 0.05, 10, 1 year, 2.05, z-entry = 2, z-exit = 0.75, tail: 10
