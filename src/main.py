from pair import PairModel, PairTrader, Portfolio
from HypoTest import RegimeDetector
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random


def backtest(prices, h: float, K: float, z_entry: float, z_exit: float, lam: float) -> pd.DataFrame:
    portfolio = Portfolio(100)

    train_i = 20
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
    rg_count = 0
    new_params = (0, 0)
    for i, row in test_set.iterrows():
        rg = False
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
        px = {"S1": row["S1"], "S2": row["S2"]}

        portfolio.update_position(i, px, position, pair_model.b, r)

    d = pd.DataFrame(portfolio.history)
    d.columns = ["Date", "Value"]
    d = d.set_index("Date")
    daily_returns = d.pct_change().dropna()
    std_daily = daily_returns.std()
    cumu_return = (d["Value"].iloc[-1]/d["Value"].iloc[0]) - 1
    std = std_daily * np.sqrt(252)

    rfr = 0.02

    daily_ret = d["Value"].pct_change().dropna()
    excess = daily_ret
    sharpe = float(excess.mean()/(1e-9 + excess.std())*np.sqrt(252))
    return d, sharpe

def test(prices):
    z_exit = [0.1, 0.5, 1, 1.5]
    best_train_sharpe = float('-inf')
    results = []
    for _ in range(200):  # 200 trials
        h = random.uniform(0.1, 0.5)
        K = random.randint(1, 5)
        lam = random.choice([0.9, 0.93, 0.96, 0.99])
        z_exit = random.choice([0.1, 0.5, 1.0, 1.5])
        z_entry = z_exit * random.choice([2,3,4,5])

        _, sharpe = backtest(prices = prices, h = h, K = K, lam = lam, z_exit = z_exit, z_entry = z_entry)
        if sharpe > best_train_sharpe:
            best_params = [h, K, lam, z_exit, z_entry]
            best_train_sharpe = sharpe
        results.append({
              "h": h, "K": K, "lam": lam,
              "z_exit": z_exit, "z_entry": z_entry,
              "sharpe": sharpe})
    return results

def run_one_window(window, split):
    split = int(split * len(window))
    train = window.iloc[:split]
    test_set = window.iloc[split:]

    results = test(train)
    df = pd.DataFrame(results).sort_values("sharpe", ascending=False)
    top_10 = df.head(10)

    best = df.iloc[0]
    best_params = [float(best["h"]), int(best["K"]), float(best["lam"]),
                   float(best["z_exit"]), float(best["z_entry"])]

    # evaluate top_10 on test
    test_rows = []
    for _, row in top_10.iterrows():
        _, test_sharpe = backtest(
            test_set,
            h=float(row["h"]),
            K=int(row["K"]),
            z_entry=float(row["z_entry"]),
            z_exit=float(row["z_exit"]),
            lam=float(row["lam"]),
        )
        test_rows.append({
            "h": float(row["h"]),
            "K": int(row["K"]),
            "lam": float(row["lam"]),
            "z_exit": float(row["z_exit"]),
            "z_entry": float(row["z_entry"]),
            "train_sharpe": float(row["sharpe"]),
            "test_sharpe": float(test_sharpe),
        })

    out = pd.DataFrame(test_rows).sort_values("test_sharpe", ascending=False)
    return best_params, out

def main():
    tickers = ["NVDA", "AMD"]
    S1_ticker = yf.download(tickers=tickers[0], period="60d", interval="15m", auto_adjust=True)
    S2_ticker = yf.download(tickers=tickers[1], period="60d", interval="15m", auto_adjust=True)

    prices = pd.concat([S1_ticker[["Close"]], S2_ticker[["Close"]]], axis = 1).dropna()
    prices.columns = ["S1", "S2"]
    l = len(prices)
    all_windows = []
    for i in range(0, l//2, l//12):
        window = prices.iloc[i:i + l//2]
        best_params, result_df = run_one_window(window, 0.8)
        print("best_params (train):", best_params)
        print(result_df.head(10))
        all_windows.append(result_df)
    wf = pd.concat(all_windows, ignore_index=True)
    print(wf.groupby(["h","K","lam","z_exit","z_entry"])["test_sharpe"].median().sort_values(ascending=False).head(10))
    return wf

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
