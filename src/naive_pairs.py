
'''
Assumptions for naive pairs trading model:
- Both stocks are cointegrated 
- Variance and Mean constant
'''

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

tickers = ["BTC-USD", "ETH-USD"]

S1_ticker = yf.Ticker(tickers[0])
S2_ticker = yf.Ticker(tickers[1])

S1_data = S1_ticker.history(period = '1y')[["Close"]]
S2_data = S2_ticker.history(period = '1y')[["Close"]]
'''
S1_data = sm.add_constant(S1_data)
results = sm.OLS(S2_data, S1_data).fit()
S1_data = S1_data.drop("const", axis = 1)
b = results.params[0]


norm = zscore(spread)
norm.plot()
plt.axhline(1, color='black')
plt.axhline(0, color='black')
plt.axhline(-1, color='black')
plt.show()


Idea is to buy when larger than 1 and sell when lower than 1.
This example has a lot of problems such as look ahead bias and assuming that
this relationship will hold for any different time period. In fact, changing the
time period to anything else will show a funny looking graph. 
'''

def zscore(series, mu, sd):
    return (series-mu)/sd

def backtest(S1, S2, test_prop):
    if not test_prop:
        return 0
    S1 = S1.reset_index().merge(S2.reset_index(), on = "Date", how = "inner", suffixes = ["_S1", "_S2"]).set_index("Date")
    split = int(len(S1) * test_prop)
    train = S1.iloc[:split]
    test  = S1.iloc[split:]
    x = sm.add_constant(train["Close_S1"])
    results = sm.OLS(train["Close_S2"], x).fit()
    b = results.params["Close_S1"]

    train_spread = train["Close_S2"] - b * train["Close_S1"] - results.params["const"]
    mu = train_spread.mean()
    sd = train_spread.std()
    test_spread = test["Close_S2"] - b * test["Close_S1"] - results.params["const"]
    
    test_zscores = zscore(test_spread, mu, sd)

    money = 100
    S1_holding = 0
    S2_holding = 0

    records = []
    for i, z in enumerate(test_zscores):
        if z < -1 and (not S2_holding and not S1_holding):
            tot = test["Close_S1"].iloc[i] * b + test["Close_S2"].iloc[i]
            S1_holding -= b * 100/tot
            S2_holding += 100/tot
            money += b * 100/tot * test["Close_S1"].iloc[i] - 100/tot * test["Close_S2"].iloc[i]
            continue
        if z > 1 and (not S2_holding and not S1_holding):
            tot = test["Close_S1"].iloc[i] * b + test["Close_S2"].iloc[i]
            S1_holding += b * 100/tot
            S2_holding -= 100/tot
            money -= b * 100/tot * test["Close_S1"].iloc[i] - 100/tot * test["Close_S2"].iloc[i]
            continue
        if abs(z) < 0.5:
            money += S1_holding * test["Close_S1"].iloc[i] + S2_holding * test["Close_S2"].iloc[i]
            S1_holding = 0
            S2_holding = 0
        records.append({
            "Date": test.index[i],
            "S1": S1_holding,
            "S2": S2_holding,
            "Money": money
        })
    if S1_holding != 0 or S2_holding != 0:
        p1 = test["Close_S1"].iloc[-1]
        p2 = test["Close_S2"].iloc[-1]
        money += S1_holding * p1 + S2_holding * p2
    records.append({
        "Date": test.index[-1],
        "S1": S1_holding,
        "S2": S2_holding,
        "Money": money
    })
    
    return pd.DataFrame(records).set_index("Date")


print(data := backtest(S1_data, S2_data, 0.2))


data[["Money"]].plot()
plt.show()



    


