import pandas as pd
import numpy as np

class Model:
    def __init__(self, variance, mu):
        self._variance = variance
        self._mu = mu
    
    @property
    def variance(self):
        return self._variance

    @property
    def mu(self):
        return self._mu

    @variance.setter
    def variance(self, new_var):
        self._variance = new_var

    @mu.setter
    def mu(self, new_mu):
        self._mu = new_mu

class Portfolio:
    def __init__(self, transactions, tickers):
