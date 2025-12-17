from pair import PairModel, PairTrader, Portfolio
from HypoTest import RegimeDetector
import pandas as pd
import random
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np


def backtest(prices, h: float, K: float, z_entry: float, z_exit: float, lam: float, train_i = 20) -> pd.DataFrame:
    portfolio = Portfolio(100)

    train_i = 30
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
    last_k = train_set.tail(30)
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

        portfolio.update_position(i, px, position, pair_model.b, r, detector.L, K)

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
    return d, sharpe, portfolio.max_drawdown, portfolio.trade_count

def sample_h(iteration, explore_every=20):
    if iteration % explore_every <= 5:
        return np.random.uniform(0.03, 0.07)
    if iteration % explore_every <= 10:
        return np.random.uniform(0.003, 0.007)
    else:
        return np.random.uniform(0.15, 0.30)
    
def test(prices):
    best_train_sharpe = float('-inf')
    results = []
    for _ in range(200): 
        h = sample_h(_)
        K = 5
        lam = 0.98 # fix for training, see how to actually get these some other time
        z_exit = 0.5 
        z_entry = z_exit * random.choice([4,8,12,16])

        _, sharpe, max_dd, trade_count = backtest(prices = prices, h = h, K = K, lam = lam, z_exit = z_exit, z_entry = z_entry)
        if sharpe > best_train_sharpe:
            best_train_sharpe = sharpe
        results.append({
              "h": h, "K": K, "lam": lam,
              "z_exit": z_exit, "z_entry": z_entry,
              "sharpe": sharpe,
              "max_dd": max_dd,
              "trade_count": trade_count})
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
        _, test_sharpe, test_max_dd, test_trade_count = backtest(
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
            "test_max_dd": float(test_max_dd),
            "test_trade_count": int(test_trade_count)
        })

    out = pd.DataFrame(test_rows).sort_values("test_sharpe", ascending=False)
    return best_params, out

def kmeans_cluster(bf):
    feats = ["test_sharpe", "test_max_dd", "test_trade_count"]

    df = bf.copy()

    df = df.replace([np.inf, -np.inf], np.nan)
    valid = df[feats].notna().all(axis=1)

    df_valid = df.loc[valid].copy()

    scaler = StandardScaler()
    X = scaler.fit_transform(df_valid[feats].values)

    best_k, best_model, best_score = None, None, -1.0
    for k in range(2, 9):
        model = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = model.fit_predict(X)
        score = silhouette_score(X, labels)
        if score > best_score:
            best_k, best_model, best_score = k, model, score

    df_valid["cluster"] = best_model.fit_predict(X)

    summary = (
        df_valid.groupby("cluster")[feats]
        .agg(["median", "mean", "count"])
        .sort_values(("test_sharpe", "median"), ascending=False)
    )

    return df_valid, summary, best_k, best_score


