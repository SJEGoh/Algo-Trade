import numpy as np
import pandas as pd
from scipy.stats import t
# Tail length changes a lot of stuff, make sure to play with it
# Ok cool it works, now how to make it into an object

class RegimeDetector:
    def __init__(self, h: float = 0.1, K: int = 5, lam: float = 0.95):
        self.prior = np.log((1-h)/h)
        self.K = K

        self.mu0 = None
        self.sd0 = None
        self.kappa0 = 1.0
        self.alpha0 = 2.0
        self.beta0 = None

        self.mu = None
        self.kappa = None
        self.alpha = None
        self.beta = None

        self.L = 0.0
        self.initialized = False

        self.lam = lam
    
    def initialize(self, initial_spread: pd.DataFrame):
        self.mu0 = initial_spread.mean()
        self.sd0 = initial_spread.std()
        self.L = 0.0

        self.beta0 = self.sd0 ** 2
        self.mu = self.mu0
        self.kappa = self.kappa0
        self.alpha = self.alpha0
        self.beta = self.beta0
        self.initialized = True
    
    def _loglikelihood(self, spread: float, alpha: float, beta: float, kappa: float, mu: float):
        nu = 2.0 * alpha
        scale = np.sqrt(beta * (kappa + 1.0) / (alpha * kappa))
        return t.logpdf(spread, df=nu, loc=mu, scale=scale)
    
    def reset_baseline(self, last_k: pd.Series):

        self.mu0 = last_k.mean()
        self.sd0 = last_k.std()
        self.beta0 = (self.sd0 ** 2) * 1.0
        self.mu = self.mu0
        self.kappa = self.kappa0
        self.alpha = self.alpha0
        self.beta = self.beta0
        self.L = 0.0

    def update(self, spread: float):
        if not self.initialized:
            raise ValueError("Not initialized")

        L1 = self._loglikelihood(spread, self.alpha, self.beta, self.kappa, self.mu)
        L0 = self._loglikelihood(spread, self.alpha0, self.beta0, self.kappa0, self.mu0)
        self.L += L0 - L1 + self.prior

        if self.L < -self.K:
            return True

        self.kappa = 1.0 + self.lam * (self.kappa - 1.0)
        self.alpha = self.alpha0 + self.lam * (self.alpha - self.alpha0)
        self.beta  = self.beta0  + self.lam * (self.beta  - self.beta0)
        new_kappa = self.kappa + 1.0
        new_beta = self.beta + 0.5 * (self.kappa * (spread - self.mu) ** 2) / new_kappa
        new_mu = (self.kappa * self.mu + spread) / new_kappa
        new_alpha = self.alpha + 0.5

        self.kappa, self.beta, self.mu, self.alpha = new_kappa, new_beta, new_mu, new_alpha
        return False
    
    def get_params(self):
        return (self.mu, self.sd0)
