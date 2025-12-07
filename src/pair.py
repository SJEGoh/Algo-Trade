import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

class PairTrader:
    def __init__(self, entry_z: float, exit_z: float, mu: float, sd: float):
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.position = 0
        self.mu = mu
        self.sd = sd
    
    def calc_zscore(self, spread: float):
        return (spread-self.mu)/self.sd
    
    def generate(self, spread: float, rg: bool, L: float = 0.0, K: float = 5):
        if rg:
            z = self.calc_zscore(spread)
            z_conf = min(1.0, 1 - np.exp(-abs(z)/self.entry_z))
            L_conf = max(0.0, min(1.0, 1 + L / K))
            k_conf = 1
            if self.position >= 0 and z < -self.exit_z:
                return self.position * (0.2 * z_conf + 0.8 * L_conf) * k_conf
            if self.position <= 0 and z > self.exit_z:
                return self.position * (0.2 * z_conf + 0.8 * L_conf) * k_conf
            return 0
        z = self.calc_zscore(spread)
        if z >= self.entry_z:
            self.position = -1
        if z <= -self.entry_z:
            self.position = 1
        if abs(z) < self.exit_z:
            self.position = 0
        z_conf = min(1.0, 1 - np.exp(-abs(z)/(2*self.entry_z)))
        L_conf = max(0.0, min(1.0, 1 + (L + K) / (2*K)))
        k_conf = 1
        return self.position * (0.2 * z_conf + 0.8 * L_conf) * k_conf
    
    def set_params(self, tup: tuple):
        self.mu = tup[0]
        self.sd = tup[1]
    

class PairModel:
    def __init__(self):
        self.b = None
        self.const = None

    def fit_hedge(self, df: pd.DataFrame):
        x = sm.add_constant(df["S1"])
        y = df["S2"]
        results = sm.OLS(y, x).fit()
        self.b = float(results.params["S1"])
        self.const = float(results.params["const"])
    
    def compute_spread(self, s1_price: float, s2_price: float):
        # Make sure it exists some other time
        return s2_price - self.b * s1_price - self.const
    

class Portfolio:
    def __init__(self, cash):
        self.cash = cash
        self.position = {"S1":0, "S2": 0}
        self.pos_state = 0
        self.history = []
    
    def update_position(self, date: np.datetime64, prices: dict, pos: float, b: float, r: bool):
        p_s1 = prices["S1"] 
        p_s2 = prices["S2"]
        cap = self.cash * 0.3
        tot = b * p_s1 + p_s2
        if not pos:
            self.cash += self.position["S1"] * p_s1 + self.position["S2"] * p_s2
            self.position["S1"] = 0
            self.position["S2"] = 0
            self.pos_state = 0
        elif r <= 5:
            pass
        elif self.pos_state:
            change = pos - self.pos_state
            self.position["S1"] += b * change * cap/tot
            self.position["S2"] -= change * cap/tot
            self.cash += (change * p_s2 - b * change * p_s1) * cap/tot
            self.pos_state = pos
            pass
        else:
            self.pos_state = pos
            self.position["S1"] += b * pos * cap/tot
            self.position["S2"] -= pos * cap/tot
            self.cash += (pos * p_s2 - b * pos * p_s1) * cap/tot
        self.history.append((date, self.get_value(prices)))
    
    def get_value(self, prices):
        return self.cash + self.position["S1"] * prices["S1"] + self.position["S2"] * prices["S2"]


