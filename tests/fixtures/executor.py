"""A real CentralExecutor + NettingCoordinator with the IB socket stubbed out.

Orders are captured as the ibapi Order objects that would have gone to IB (`ex.sent`),
cancels are recorded (`ex._cancelled`), and `fill` feeds an execution back through the real
execDetails path — so pooled attribution, pending release and chain reactions all run.
"""
import os

from ibapi.contract import Contract
from ibapi.execution import Execution

from execution.central_execution import CentralExecutor
from execution.netting import NettingCoordinator

CFG = {"s1": {"capital_allocation": 1e9, "max_drawdown": 0.2},
       "s2": {"capital_allocation": 1e9, "max_drawdown": 0.2}}


def inst(sym):
    return {"symbol": sym, "asset_class": "equity", "exchange": "SMART", "sec_type": "STK"}


class _NullLog:
    def __getattr__(self, name):
        return lambda *a, **k: None


def make_executor(cfg=CFG, state_path=None):
    os.environ.setdefault("EXECUTOR_API_KEY", "x")
    ex = CentralExecutor.__new__(CentralExecutor)
    CentralExecutor.__init__(ex)
    ex.risk_manager._config = cfg
    ex.risk_manager._active_strategies = set(cfg)
    ex._oid = 0

    def _next():
        ex._oid += 1
        return ex._oid

    ex.get_next_order_id = _next
    ex._cancelled = set()
    ex.sent = {}                                    # order id -> ibapi Order as placed
    ex.placeOrder = lambda oid, c, o: ex.sent.__setitem__(oid, o)
    ex.cancelOrder = lambda oid, *a, **k: ex._cancelled.add(oid)
    ex.logger_db = _NullLog()
    co = NettingCoordinator(ex, cfg, state_path=state_path)
    ex.coordinator = co
    return ex, co


def fill(ex, oid, shares=None, price=100.0):
    """Fill order `oid` — all of what is left, or `shares` of it."""
    st = ex.order_status[oid]
    left = st["pending_qty"] - st["exec_filled"]
    q = left if shares is None else shares * (1 if left > 0 else -1)
    c = Contract(); c.symbol = st["symbol"]
    e = Execution()
    e.orderId = oid; e.execId = f"e{oid}-{st['exec_filled']}"
    e.shares = abs(q); e.side = "BOT" if q > 0 else "SLD"; e.price = price
    ex.execDetails(1, c, e)


def held(ex, sym, sid="s1"):
    return ex.ledger.strategy_positions.get(sid, {}).get(sym, 0.0)
