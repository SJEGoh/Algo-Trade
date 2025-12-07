# Regime-Aware Pairs Trading System using Online Changepoint Detection

The wonderful amalgamation that I have made is a pairs trader. This works by taking two stocks that are cointegrated with each other, and buying one while selling the other depending on how the spread deviates from the mean. 

Pairs trading depends on multiple assumptions:
- The two stocks are cointegrated, with the spread being stationary and mean-reverting
- The hedge ratio determining the spread stays constant
- The standard deviation of the spread is equal across time
- Spread residuals behave like IID Gaussian noise

Unfortunately, in actual market conditions these rarely hold up. My model handles these assumptions by readjusting parameters to fit a new market regime, rather than assuming it remains constant. My model performs online changepoint detection using a Bayesian CUSUM like algorithm. A regime shift causes a recalculation of hedge ratio, standard deviation, and mean everytime a regime change is detected. 

Pairs with abrupt structural changes perform best with this model. Slowly evolving pairs such as KO and PEP experience more of a regime drift and don't trigger clear changepoints, while more volatile pairs such as NVDA and AMD produce sharper structural breaks that are picked up by the detector.

Limitations of current model:
- Highly sensitive to hyperparameters:
This is a common issue in BOCD style models due to the use of likelihood ratios and hazard functions. While currently this model may be too sensitive for real markets, backtests show strong potential for more robust models of this type. 
- Performs best with distinct regimes
Current model struggles with picking up drifting regimes. However, rather than altering this model to handle those as well, it may be more efficient to develop a different model to handle slow moving regimes, and make a meta-layer that sorts pairs into one or the other. 

## `main.py`

#### Description
This code runs the backtesting strategy using the parameters passed in, returning a dataframe of the account value over time as well as printing the sharpe ratio (using a risk-free rate of 2%).

### `backtest(tickers: tuple[str,str], years: int, h: float, K: float, z_entry: float, z_exit: float, lam: float) -> DataFrame`

#### Params
- tickers: tuple container tickers of pair
- years: number of years for backtest
- h: hazard function, roughly 1/(expected length of regime)
- K: threshold for regime switch (higher leads to higher likelihood required)
- z_entry: deviation from mean before order is made
- z_exit: deviation from mean before order is closed
- lam: forgetting factor (prevents overweighting of past data)

## `HypoTest.py`

#### Description
The coup-de-grace of this project. 

### `class RegimeDetector`

#### `__init__(self, h: float = 0.1, K: int = 5, lam: float = 0.95):`
Sets all the greeks, hazard function, lambda, and threshold. 

#### `initialize(self, initial_spread: pd.DataFrame)`
Takes in dataframe of spreads to initialize values on. Sets mu0 and sd0, and uses those values to set the rest of the greeks. 

#### `_loglikelihood(self, spread: float, alpha: float, beta: float, kappa: float, mu: float)`
Helper function to calculate loglikelihood based on t distribution

#### `reset_baseline(self, last_k: pd.Series)`
Rests baseline when a regime change is detected. Uses last_k spreads. 

#### `update(self, spread: float)`
Takes in current spread, and updates the cumulative loglikelihood. If loglikelihood goes below threshold, function returns 'True'. If not, parameters are updated and function returns False. 

#### `get_params(self)`
Returns current mean and standard deviation in a tuple

## `pair.py`
Contains strategy, trader, and portfolio objects. 

### `class PairTrader`
Used to calculate the z score, and confidence for trades. 

#### `__init__(self, entry_z: float, exit_z: float, mu: float, sd: float)`
Sets entry and exit z scores, current mean and standard deviation. 

#### `def calc_zscore(self, spread: float/series)`
Helper to calculate zscore

#### `generate(self, spread: float, rg: bool, L: float = 0.0, K: float = 5)`
This function takes in the current spread, whether a regime change has happened, the current cumulative likelihood, and the threshold. If a regime change has happened, the function re-evaluates whether the current position still fits, and re-balances based on new confidence.

If a regime change has not happened, the z value is calculated. This z value is used to evaluate the type of position the model should take, or if the trader should close out the position. After this, the confidence of the position is evaluated through the difference between z-score and z_exit (how over/under sold the pair is) and likehlihood (confidence in current regime). These are used to adjust the confidence of the position. 

#### `set_params(self, tup: tuple)`
Takes in tuple of new mean and standard deviation, and updates the object. 

### `class PairModel`
Computes the hedge ratio using OLS and computes spread.

#### `__init__(self)`
Stores hedge ratio as 'b' and constant as 'const'.

#### `fit_hedge(self, df: pd.DataFrame)`
Takes in dataframe with most current prices of both stocks, and updates hedgea ratio and constant through OLS. 

#### `compute_spread(self, s1_price: float, s2_price: float)`
Computes instantaneous spread.

### `class Portfolio`
Contains starting cash, current position, the position state, and history. 

#### `update_position(self, date: np.datetime64, prices: dict, pos: float, b: float, r: int)`
Takes in signal (pos) and depending on that, rebalances, closes out, or buys into a position. Date and current value is then appended into history. 

#### `get_value(self, prices: dict)`
Returns value of current portfolio

### `hello.py`

#### `hypothesis_test(tickers, years, train_i, h = 0.01, K = 2, lam = 0.95)`
Takes in parameters and returns normalised spread in a dataframe, and plots time of changepoint detection. 

References:
Adams, R. P., & MacKay, D. J. C. (2007). Bayesian online changepoint detection. arXiv. https://arxiv.org/abs/0710.3742

Murphy, K. P. (2007). Conjugate Bayesian analysis of the Gaussian distribution. University of British Columbia. https://www.cs.ubc.ca/~murphyk/Papers/bayesGaussians.pdf

