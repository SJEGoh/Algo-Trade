from helper import run_one_window, kmeans_cluster, backtest, get_safe_range
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ChatGPT idea, might keep cause I find it funny lol. Will remove if I actually deploy
def sample_q(n, a=3, b=3, qmin=0.00001, qmax=0.8, explore_p=0.05, rng=None):
    rng = np.random.default_rng(rng)
    qs = []
    for _ in range(n):
        q = rng.beta(a, b)
        qs.append(float(q))
    return qs

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
        best_params, result_df = run_one_window(window, 0.7)
        print("best_params (train):", best_params)
        print(result_df.head(10))
        all_windows.append(result_df)
    wf = pd.concat(all_windows, ignore_index=True)
    print(wf)
    print(wf.groupby(["h","K","lam","z_exit","z_entry"])[["test_sharpe", "test_max_dd"]].median().sort_values(by = "test_sharpe", ascending=False).head(10))
    clustered_df, cluster_summary, best_k, sil = kmeans_cluster(wf)
    print("best_k =", best_k, "silhouette =", sil)
    print(cluster_summary)

    cs = clustered_df.groupby("cluster").agg(
    sharpe_med=("test_sharpe","median"),
    dd_med=("test_max_dd","median"),
    n=("test_sharpe","size"),
    )

    cs["score"] = cs["sharpe_med"] - 5*cs["dd_med"]
    best_cluster = cs.sort_values("score", ascending=False).index[0]
    print(cs.sort_values("score", ascending=False).head(5))

    best = clustered_df[clustered_df["cluster"] == best_cluster]
    med = best[["h","K","lam","z_exit","z_entry"]].median(numeric_only=True)
    param_cols = ["h","K","lam","z_exit","z_entry"]
    print("======= BEST ========")
    print(best[param_cols].describe(percentiles=[0.1,0.25,0.5,0.75,0.9]))
    
    safe_clusters = cs[(cs["sharpe_med"] > 0.5) & (cs["dd_med"] < 0.005)].index

    safe = clustered_df[clustered_df["cluster"].isin(safe_clusters)].copy()
    q = safe[param_cols].quantile([0.10, 0.25, 0.50, 0.75, 0.90]).T
    print(safe)

    combined_safe = get_safe_range(best, safe)
    print(combined_safe)
    q.columns = ["q10","q25","q50","q75","q90"]
    # Take overlap of best and safe
    n_draws = 20  
    # qs = sample_q(n_draws, a=4, b=4, qmin=0.2, qmax=0.8, explore_p=0.05, rng=42)

    compiled = pd.DataFrame(columns=["Date", "Value"])

    for n in range(n_draws):
        # med = combined_safe[["h","K","lam","z_exit","z_entry"]].quantile(q=q, numeric_only=True)

        d, sharpe, max_drawdown, trade_count = backtest(
            prices,
            h=float(np.random.uniform(combined_safe["h"][0], combined_safe["h"][1])),
            K=int(round(np.random.uniform(combined_safe["K"][0],combined_safe["K"][1]))),         
            z_entry=float(np.random.uniform(combined_safe["z_entry"][0],combined_safe["z_entry"][1])),
            z_exit=float(np.random.uniform(combined_safe["z_exit"][0],combined_safe["z_exit"][1])),
            lam=float(np.random.uniform(combined_safe["lam"][0],combined_safe["lam"][1])),
            train_i=20
        )

        print(f"n={n:.3f} sharpe={sharpe:.3f} dd={max_drawdown:.4f} trades={trade_count}")
        compiled = pd.concat([compiled, d.reset_index()], axis=0)

    compiled_mean = compiled.groupby("Date", as_index=False)["Value"].mean().set_index("Date")
    compiled_mean.plot()
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
