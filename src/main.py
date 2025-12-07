from pair import PairModel, PairTrader, Portfolio
from HypoTest import RegimeDetector
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def backtest(tickers: tuple[str, str], years: int, h: float, K: float, z_entry: float, z_exit: float, lam: float) -> pd.DataFrame:
    S1_ticker = yf.Ticker(tickers[0])
    S2_ticker = yf.Ticker(tickers[1])
    df1 = S1_ticker.history(period = str(years) + "y")[["Close"]]
    df2 = S2_ticker.history(period = str(years) + "y")[["Close"]]
    prices = pd.concat([df1,df2], axis = 1).dropna()
    prices.columns = ["S1", "S2"]
    portfolio = Portfolio(100)

    train_i = 10
    train_set = prices.iloc[:train_i]
    test_set = prices.iloc[train_i:]
    train_rows = []  

    pair_model = PairModel()
    pair_model.fit_hedge(train_set)

    for i, row in train_set.iterrows():
        spread = pair_model.compute_spread(row["S1"], row["S2"])
        train_rows.append((i, spread))

    train_spread = pd.DataFrame(train_rows, columns=["Date", "Spread"]).set_index("Date")
    
    detector = RegimeDetector(h = h, K = K, lam = lam)
    detector.initialize(train_spread["Spread"])
    strategy = PairTrader(z_entry, z_exit, train_spread["Spread"].mean(), train_spread["Spread"].std())
    last_k = train_set.tail(5)
    r = 0
    new_params = (0, 0)
    for i, row in test_set.iterrows():
        spread = pair_model.compute_spread(row["S1"], row["S2"])
        last_k = last_k.iloc[1:]
        last_k.loc[i] = row
        
        if rg := detector.update(spread):
            
            pair_model.fit_hedge(last_k)
            last_k_spread = []
            for j, row_k in last_k.iterrows():
                spread = pair_model.compute_spread(row_k["S1"], row_k["S2"])
                last_k_spread.append((j, spread))
            last_k_spread = pd.DataFrame(last_k_spread, columns=["Date", "Spread"]).set_index("Date")
            detector.reset_baseline(last_k_spread["Spread"])
            new_params = detector.get_params()
            strategy.set_params(new_params)
            r = 0
        r += 1
        position = strategy.generate(spread, rg, detector.L, K)
        prices = {"S1": row["S1"], "S2": row["S2"]}

        portfolio.update_position(i, prices, position, pair_model.b, r)

    d = pd.DataFrame(portfolio.history)
    d.columns = ["Date", "Value"]
    d = d.set_index("Date")
    daily_returns = d.pct_change().dropna()
    mean_daily = daily_returns.mean()
    std_daily = daily_returns.std()
    cumu_return = (d["Value"].iloc[-1]/d["Value"].iloc[0]) - 1
    std = std_daily * np.sqrt(252 * years)

    rfr = 0.02

    sharpe = (cumu_return - rfr)/ std
    sharpe = mean_daily/std_daily * np.sqrt(252 * years)
    print(f"Sharpe Ratio: {sharpe["Value"]}")
    return d

def main():
    data = backtest(["BTC-USD", "ETH-USD"],years = 5, h = 0.06, K = 5, z_entry = 3, z_exit = 0.75, lam = 0.99)
    data.plot()
    plt.show()

if __name__ == "__main__":
    main()

'''
Hyperparameters to be set:
- h
- K
- lam
- z_entry / z_exit
- tail length
- cool down
'''
