
import pandas as pd
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
                self.position *= (0.2 * z_conf + 0.8 * L_conf) * k_conf
                return self.position 
            if self.position <= 0 and z > self.exit_z:
                self.position *= (0.2 * z_conf + 0.8 * L_conf) * k_conf
                return self.position
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
        return self.position * (0.3 * z_conf + 0.7 * L_conf) * k_conf
    
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
    def __init__(self, cash, stop_loss = 0.05, cooldown_bars = 5, fee_bps = 1, slip_bps = 2, fixed_fee = 0.0):
        self.cash = cash
        self.position = {"S1":0, "S2": 0}
        self.pos_state = 0
        self.history = []
        self.stop_loss = float(stop_loss)         
        self.cooldown_bars = int(cooldown_bars)
        self.cooldown_left = 0
        self.entry_value = None
        self.peak_value = 0.0
        self.entry_peak_dd = 0.0
        self.max_drawdown = 0.0
        self.trade_count = 0

        self.fee_bps = float(fee_bps)    
        self.slip_bps = float(slip_bps)  
        self.fixed_fee = float(fixed_fee) 


    def update_drawdown(self, value):
        self.peak_value = max(self.peak_value, value)
        dd = (self.peak_value - value) / max(self.peak_value, 1e-9)
        self.max_drawdown = max(self.max_drawdown, dd)

    def _charge_costs(self, traded_notional: float):
        cost = traded_notional * ((self.fee_bps + self.slip_bps) / 1e4) + self.fixed_fee
        self.cash -= cost
        return cost
    
    def close_all(self, prices):
        p_s1, p_s2 = prices["S1"], prices["S2"]
        self.cash += self.position["S1"] * p_s1 + self.position["S2"] * p_s2
        self.position["S1"] = 0.0
        self.position["S2"] = 0.0
        self.pos_state = 0.0
        self.entry_value = None
        self.peak_value = 0.0
        self.entry_peak = None
        self.entry_peak_dd = 0.0

    def update_position(self, date: np.datetime64, prices: dict, pos: float, b: float, r: int, L, K):
        p_s1, p_s2 = prices["S1"], prices["S2"]
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
            val = self.get_value(prices)
            self.history.append((date, val))
            self.update_drawdown(val)
            return
        curr_val = self.get_value(prices)
        self.entry_peak = curr_val
        self.entry_peak_dd = 0.0
        cap = self.cash * 0.2
        tot = b * p_s1 + p_s2
        if self.pos_state != 0:
            if self.entry_value is None:
                self.entry_value = curr_val
                self.peak_value = curr_val

            self.peak_value = max(self.peak_value, curr_val)
            dd = (self.peak_value - curr_val) / max(1e-9, self.peak_value)
            self.entry_peak_dd = max(self.entry_peak_dd, dd)

            if dd >= self.stop_loss or L < (-K + K/4):
                traded_notional = abs(self.position["S1"]) * p_s1 + abs(self.position["S2"]) * p_s2
                if traded_notional:    
                    self._charge_costs(traded_notional)
                    self.trade_count += 1
                self.close_all(prices)
                self.cooldown_left = self.cooldown_bars
                self.history.append((date, self.get_value(prices)))
                return
        if not pos:
            traded_notional = abs(self.position["S1"]) * p_s1 + abs(self.position["S2"]) * p_s2
            if traded_notional:
                self._charge_costs(traded_notional)
                self.trade_count += 1
            self.close_all(prices)
        elif r <= 10 or L < -K + K/4:
            pass
        elif self.pos_state:
            change = pos - self.pos_state
            dS1 = b * change * cap / tot
            dS2 = change * cap / tot
            traded_notional = abs(dS1) * p_s1 + abs(dS2) * p_s2
            if traded_notional:
                self._charge_costs(traded_notional)
                self.trade_count += 1
            self.position["S1"] += dS1
            self.position["S2"] -= dS2
            self.cash += (dS2 * p_s2 - dS1 * p_s1)
            self.pos_state = pos
            pass
        else:
            self.pos_state = pos
            dS1 = b * pos * cap/tot
            dS2 = pos * cap/tot
            traded_notional = abs(dS1) * p_s1 + abs(dS2) * p_s2
            if traded_notional:
                self._charge_costs(traded_notional)
                self.trade_count += 1
            self.position["S1"] += dS1
            self.position["S2"] -= dS2
            self.cash += (dS2 * p_s2 - dS1* p_s1) 
        val = self.get_value(prices)
        if val > self.peak_value:
            self.peak_value = val
        self.update_drawdown(val)
        self.history.append((date, self.get_value(prices)))
    
    def get_value(self, prices):
        return self.cash + self.position["S1"] * prices["S1"] + self.position["S2"] * prices["S2"]


