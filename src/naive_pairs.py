
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

tickers = ["ADBE", "MSFT"]

S1_ticker = yf.Ticker(tickers[0])
S2_ticker = yf.Ticker(tickers[1])

S1_data = S1_ticker.history(period = '5y')[["Close"]]
S2_data = S2_ticker.history(period = '5y')[["Close"]]

S1_data = sm.add_constant(S1_data)
results = sm.OLS(S2_data, S1_data).fit()
S1_data = S1_data.drop("const", axis = 1)
b = results.params[0]

def zscore(series):
    return (series-series.mean())/np.std(series)
spread = S2_data - b * S1_data

norm = zscore(spread)
norm.plot()
plt.axhline(1, color='black')
plt.axhline(0, color='black')
plt.axhline(-1, color='black')
plt.show()

'''
Idea is to buy when larger than 1 and sell when lower than 1.
This example has a lot of problems such as look ahead bias and assuming that
this relationship will hold for any different time period. In fact, changing the
time period to anything else will show a funny looking graph. 
'''




