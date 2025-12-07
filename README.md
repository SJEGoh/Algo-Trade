# Regime-Aware Pairs Trading System using Online Changepoint Detection

The wonderful amalgamation that I have made is a pairs trader. This works by taking two stocks that are cointegrated with each other, and buying one while selling the other depending on how the spread deviates from the mean. 

Pairs trading depends on multiple assumptions:
- The two stocks are cointegrated, with the spread being stationary and mean-reverting
- The hedge ratio determining the spread stays constant
- The standard deviation of the spread is equal across time
- Spread residuals behave like IID Gaussian noise

Unfortunately, in actual market conditions these rarely hold up. My model handles these assumptions by readjusting parameters to fit a new market regime, rather than assuming it remains constant. I employ a Bayesian CUSUM algorithm to achieve this, with a recalculation of hedge ratio, standard deviation, and mean everytime a regime change is detected. This model will be a proof of concept until I do actually get around to making the one that can choose cointegrated stocks. 

Pairs that are cointegrated but experience major and violent shifts in regimes perform best with this bot. Stable pairs such as KO and PEP, while cointegrated, have regimes that evolve slowly, which may not trigger a regime change by the regime detector. In contrast, violent pairs such a NVDA and AMD, which are volatile and experience many regime changes perform better. While I could make this model handle evolving regimes, that would greatly increase complexity and perhaps affect the ability it currently has. As such, I will not do this for my own sanity, and because on the off chance that I want to convert this to C++ to compete with actual hedge funds, this change would slow down my bot significantly. 

Other problems with the current model:
- Very, very, very sensitive to changes in parameters. A weakness observed on most similar, BOCD style models. This would likely not make this viable for a actual markets, but a similar, more robust model could be adapted. But the great returns observed on backtests does show the potential of this type of regime shift pair trader.

Overall, very fun build. I foresee this being part of a bigger build, with another model handling slower moving regimes, and a meta-sorter that sorts pairs into one or the other. 


## 'main.py'

#### Description
This code runs the backtesting strategy using the parameters passed in, returning a dataframe of the account value over time as well as printing the sharpe ratio (using a risk-free rate of 2%).

### 'backtest(tickers: tuple[str,str], years: int, h: float, K: float, z_entry: float, z_exit: float, lam: float) -> DataFrame'

#### Params
- tickers: tuple container tickers of pair
- years: number of years for backtest
- h: hazard function, roughly 1/(expected length of regime)
- K: threshold for regime switch (higher leads to higher likelihood required)
- z_entry: deviation from mean before order is made
- z_exit: deviation from mean before order is closed
- lam: forgetting factor (prevents overweighting of past data)

## 'HypoTest.py'

#### Description
The coup-de-grace of this project. 

### 'class RegimeDetector'

#### '__init__(self, h: float = 0.1, K: int = 5, lam: float = 0.95):'
Sets all the greeks, hazard function, lambda, and threshold. 

#### 'initialize(self, initial_spread: pd.DataFrame)'
Takes in dataframe of spreads to initialize values on. Sets mu0 and sd0, and uses those values to set the rest of the greeks. 

#### '_loglikelihood(self, spread: float, alpha: float, beta: float, kappa: float, mu: float)'
Helper function to calculate loglikelihood based on t distribution

#### 'reset_baseline(self, last_k: pd.Series)'
Rests baseline when a regime change is detected. Uses last_k spreads. 

#### 'update(self, spread: float)'
Takes in current spread, and updates the cumulative loglikelihood. If loglikelihood goes below threshold, function returns 'True'. If not, parameters are updated and function returns False. 

#### 'get_params(self)'
Returns current mean and standard deviation in a tuple

## 'pair.py'
Contains strategy, trader, and portfolio objects. 

### 'class PairTrader'
Used to calculate the z score, and confidence for trades. 

#### '__init__(self, entry_z: float, exit_z: float, mu: float, sd: float)'
Sets entry and exit z scores, current mean and standard deviation. 

#### 'def calc_zscore(self, spread: float/series)'
Helper to calculate zscore

#### 'generate(self, spread: float, rg: bool, L: float = 0.0, K: float = 5)'
This function takes in the current spread, whether a regime change has happened, the current cumulative likelihood, and the threshold. If a regime change has happened, the function re-evaluates whether the current position still fits, and re-balances based on new confidence.

If a regime change has not happened, the z value is calculated. This z value is used to evaluate the type of position the model should take, or if the trader should close out the position. After this, the confidence of the position is evaluated through the difference between z-score and z_exit (how over/under sold the pair is) and likehlihood (confidence in current regime). These are used to adjust the confidence of the position. 

#### 'set_params(self, tup: tuple)'
Takes in tuple of new mean and standard deviation, and updates the object. 

### 'class PairModel'
Computes the hedge ratio using OLS and computes spread.

#### '__init__(self)'
Stores hedge ratio as 'b' and constant as 'const'.

#### 'fit_hedge(self, df: pd.DataFrame)'
Takes in dataframe with most current prices of both stocks, and updates hedgea ratio and constant through OLS. 

#### 'compute_spread(self, s1_price: float, s2_price: float)'
Computes instantaneous spread.

### 'class Portfolio'
Contains starting cash, current position, the position state, and history. 

#### 'update_position(self, date: np.datetime64, prices: dict, pos: float, b: float, r: int)'
Takes in signal (pos) and depending on that, rebalances, closes out, or buys into a position. Date and current value is then appended into history. 

#### 'get_value(self, prices: dict)'
Returns value of current portfolio

### 'hello.py'

#### 'hypothesis_test(tickers, years, train_i, h = 0.01, K = 2, lam = 0.95)'
Takes in parameters and returns normalised spread in a dataframe, and plots time of changepoint detection. 

References:




https://www.reddit.com/r/algotrading/comments/v7pchq/bayesian_hierarchical_models_for_algorithmic

