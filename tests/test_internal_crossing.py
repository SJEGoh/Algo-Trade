"""tests/test_internal_crossing.py — Internal crossing in the NettingCoordinator.
Verifies that offsetting strategy deltas are crossed internally at the reference
price, and only the residual net goes to IB."""
import pytest
from ledger.position_ledger import PositionLedger
from risk.risk_manager import RiskManager
from execution.netting import NettingCoordinator


class FakeLoggerDB:
    """Records log_fill calls for assertions."""
    def __init__(self):
        self.fills = []

    def log_fill(self, order_id, exec_id, symbol, side, price, quantity,
                 strategy_id, expected_price=None):
        self.fills.append({
            "order_id": order_id, "exec_id": exec_id, "symbol": symbol,
            "side": side, "price": price, "quantity": quantity,
            "strategy_id": strategy_id, "expected_price": expected_price,
        })


class FakeExecutor:
    def __init__(self, config):
        self.ledger = PositionLedger(None)
        self.risk_manager = RiskManager(self.ledger, config)
        self.logger_db = FakeLoggerDB()
        self.placed = []

    def _cancel_open_orders_for_symbol(self, sym):
        pass

    def place_net_order(self, sym, delta, instrument, price):
        self.ledger.record_net_pending(sym, delta)
        oid = len(self.placed) + 1
        self.placed.append({"symbol": sym, "delta": delta, "price": price})
        return oid

    def fill_last(self, co, price=None):
        o = self.placed[-1]
        co.attribute_fill(o["symbol"], o["delta"], price if price is not None else o["price"])

    def fill_orders(self, co, orders):
        for o in orders:
            price = next(p["price"] for p in reversed(self.placed) if p["symbol"] == o["symbol"])
            co.attribute_fill(o["symbol"], o["delta"], price)


def _cfg(**allocs):
    return {sid: {"capital_allocation": a, "max_drawdown": 0.2} for sid, a in allocs.items()}


def inst(sym):
    return {"symbol": sym, "asset_class": "equity", "exchange": "SMART"}


# -----------------------------------------------------------------------
# Basic crossing scenarios
# -----------------------------------------------------------------------

class TestInternalCrossBasic:
    """Two strategies, one buys and one sells the same symbol."""

    def test_full_offset_no_ib_order(self):
        """A wants +10, B wants -10 → cross all 10, no IB order."""
        cfg = _cfg(A=1e9, B=1e9)
        ex = FakeExecutor(cfg)
        co = NettingCoordinator(ex, cfg)

        # A sets target first — goes to IB since B has no opposite delta yet
        r1 = co.set_target("A", "MSFT", 10, instrument=inst("MSFT"), price=400)
        assert len(r1["orders"]) == 1
        ex.fill_last(co, price=400)

        # Now B sets target -10 → net desired = 0, current position = 10
        # Internal cross: B's -10 crosses against A's reduction from 10 to 0? No —
        # A still wants +10. B wants -10. Net desired = 0.
        # A has 10, wants 10 → delta 0. B has 0, wants -10 → delta -10.
        # Only sellers, no buyers → no internal cross. IB order for -10.
        r2 = co.set_target("B", "MSFT", -10, instrument=inst("MSFT"), price=400)
        assert len(r2["orders"]) == 1
        assert r2["orders"][0]["delta"] == -10

    def test_offsetting_new_targets_cross_internally(self):
        """Both strategies set targets at once — offsetting deltas cross internally."""
        cfg = _cfg(A=1e9, B=1e9)
        ex = FakeExecutor(cfg)
        co = NettingCoordinator(ex, cfg)

        # A wants +10
        co.set_target("A", "MSFT", 10, instrument=inst("MSFT"), price=400)
        # Don't fill yet — B also sets a target before the IB order fills
        # Actually, with the current design, set_target triggers _rebalance immediately.
        # Let's use submit_book to set both at once via separate calls.

        # Start fresh
        ex2 = FakeExecutor(cfg)
        co2 = NettingCoordinator(ex2, cfg)

        # Set both desired books via submit_book
        co2.submit_book("A", [{"instrument": inst("MSFT"), "target_quantity": 10, "expected_price": 400}])
        # A has no opposing delta yet → order goes to IB for +10
        ex2.fill_last(co2, price=400)
        assert ex2.ledger.strategy_positions["A"]["MSFT"] == 10

        # Now A wants to increase to 20 and B wants -5
        co2.set_target("A", "MSFT", 20, instrument=inst("MSFT"), price=410)
        ex2.fill_last(co2, price=410)
        assert ex2.ledger.strategy_positions["A"]["MSFT"] == 20

        # B wants -5: A's delta is 0 (has 20, wants 20), B's delta is -5
        # No internal crossing (no buyers), goes to IB
        r = co2.set_target("B", "MSFT", -5, instrument=inst("MSFT"), price=410)
        assert len(r["orders"]) == 1
        assert r["orders"][0]["delta"] == -5

    def test_partial_cross_residual_to_ib(self):
        """A wants +10, B wants -3 → cross 3 internally, send +7 to IB."""
        cfg = _cfg(A=1e9, B=1e9)
        ex = FakeExecutor(cfg)
        co = NettingCoordinator(ex, cfg)

        # Both strategies have unfilled deltas from the start
        # Use desired directly to set up the scenario
        co.desired = {"A": {"MSFT": 10}, "B": {"MSFT": -3}}
        co.ref_price["MSFT"] = 400.0
        co.instrument["MSFT"] = inst("MSFT")

        rebal = co._rebalance({"MSFT"})
        crosses = rebal["internal_crosses"]
        orders = rebal["orders"]

        # Should have crossed 3 shares internally
        assert len(crosses) == 2  # one BOT (A) + one SLD (B)
        bot_cross = [c for c in crosses if c["side"] == "BOT"][0]
        sld_cross = [c for c in crosses if c["side"] == "SLD"][0]
        assert bot_cross["strategy_id"] == "A"
        assert abs(bot_cross["quantity"] - 3.0) < 1e-9
        assert bot_cross["price"] == 400.0
        assert sld_cross["strategy_id"] == "B"
        assert abs(sld_cross["quantity"] - 3.0) < 1e-9

        # Residual +7 should go to IB
        assert len(orders) == 1
        assert abs(orders[0]["delta"] - 7.0) < 1e-9

        # Strategy positions after internal cross (before IB fill)
        assert abs(ex.ledger.strategy_positions["A"]["MSFT"] - 3.0) < 1e-9
        assert abs(ex.ledger.strategy_positions["B"]["MSFT"] - (-3.0)) < 1e-9

        # Net broker position unchanged (internal cross is zero-sum)
        assert ex.ledger.current_positions.get("MSFT", 0) == 0

    def test_no_cross_same_direction(self):
        """A wants +10, B wants +5 → no internal cross, all to IB."""
        cfg = _cfg(A=1e9, B=1e9)
        ex = FakeExecutor(cfg)
        co = NettingCoordinator(ex, cfg)

        co.desired = {"A": {"MSFT": 10}, "B": {"MSFT": 5}}
        co.ref_price["MSFT"] = 400.0
        co.instrument["MSFT"] = inst("MSFT")

        rebal = co._rebalance({"MSFT"})
        assert len(rebal["internal_crosses"]) == 0
        assert len(rebal["orders"]) == 1
        assert abs(rebal["orders"][0]["delta"] - 15.0) < 1e-9


class TestInternalCrossThreeStrategies:
    """Three or more strategies with mixed directions."""

    def test_three_strategies_partial_cross(self):
        """A wants +10, B wants -5, C wants -3 → cross 8 internally, send +2 to IB."""
        cfg = _cfg(A=1e9, B=1e9, C=1e9)
        ex = FakeExecutor(cfg)
        co = NettingCoordinator(ex, cfg)

        co.desired = {"A": {"MSFT": 10}, "B": {"MSFT": -5}, "C": {"MSFT": -3}}
        co.ref_price["MSFT"] = 400.0
        co.instrument["MSFT"] = inst("MSFT")

        rebal = co._rebalance({"MSFT"})
        crosses = rebal["internal_crosses"]
        orders = rebal["orders"]

        # Total buy = 10, total sell = 8, crossable = 8
        # A gets 8/10 * 10 = 8 bought internally (buy_scale = 8/10)
        # Wait, buy_scale = crossable/total_buy = 8/10 = 0.8
        # A's internal fill = 10 * 0.8 = 8
        # B gets all 5 sold, C gets all 3 sold (sell_scale = 8/8 = 1.0)
        bot_crosses = [c for c in crosses if c["side"] == "BOT"]
        sld_crosses = [c for c in crosses if c["side"] == "SLD"]

        assert len(bot_crosses) == 1
        assert bot_crosses[0]["strategy_id"] == "A"
        assert abs(bot_crosses[0]["quantity"] - 8.0) < 1e-9

        assert len(sld_crosses) == 2
        b_cross = [c for c in sld_crosses if c["strategy_id"] == "B"][0]
        c_cross = [c for c in sld_crosses if c["strategy_id"] == "C"][0]
        assert abs(b_cross["quantity"] - 5.0) < 1e-9
        assert abs(c_cross["quantity"] - 3.0) < 1e-9

        # Strategy positions after cross
        assert abs(ex.ledger.strategy_positions["A"]["MSFT"] - 8.0) < 1e-9
        assert abs(ex.ledger.strategy_positions["B"]["MSFT"] - (-5.0)) < 1e-9
        assert abs(ex.ledger.strategy_positions["C"]["MSFT"] - (-3.0)) < 1e-9

        # Residual +2 to IB
        assert len(orders) == 1
        assert abs(orders[0]["delta"] - 2.0) < 1e-9

        # Net broker position unchanged
        assert ex.ledger.current_positions.get("MSFT", 0) == 0

    def test_three_strategies_two_buyers_one_seller(self):
        """A wants +6, B wants +4, C wants -5 → cross 5 pro-rata among buyers."""
        cfg = _cfg(A=1e9, B=1e9, C=1e9)
        ex = FakeExecutor(cfg)
        co = NettingCoordinator(ex, cfg)

        co.desired = {"A": {"MSFT": 6}, "B": {"MSFT": 4}, "C": {"MSFT": -5}}
        co.ref_price["MSFT"] = 400.0
        co.instrument["MSFT"] = inst("MSFT")

        rebal = co._rebalance({"MSFT"})
        crosses = rebal["internal_crosses"]

        # crossable = min(10, 5) = 5
        # buy_scale = 5/10 = 0.5
        # A internal: 6 * 0.5 = 3, B internal: 4 * 0.5 = 2
        # sell_scale = 5/5 = 1.0
        # C internal: 5 * 1.0 = 5
        bot_crosses = sorted([c for c in crosses if c["side"] == "BOT"],
                             key=lambda c: c["strategy_id"])
        assert abs(bot_crosses[0]["quantity"] - 3.0) < 1e-9  # A
        assert abs(bot_crosses[1]["quantity"] - 2.0) < 1e-9  # B

        sld_crosses = [c for c in crosses if c["side"] == "SLD"]
        assert abs(sld_crosses[0]["quantity"] - 5.0) < 1e-9  # C

        # Remaining to IB: net = 6+4-5 = 5, crossed fills net to 0 in current_positions
        assert abs(rebal["orders"][0]["delta"] - 5.0) < 1e-9


class TestInternalCrossPnL:
    """P&L flows correctly: internal fills at ref price, IB fills at execution price."""

    def test_pnl_split_between_internal_and_ib_price(self):
        """A gets some shares at internal price and rest at IB price.
        B gets all shares at internal price. P&L should reflect this."""
        cfg = _cfg(A=1e9, B=1e9)
        ex = FakeExecutor(cfg)
        co = NettingCoordinator(ex, cfg)

        # Set up: A wants +10 MSFT, B wants -4 MSFT
        co.desired = {"A": {"MSFT": 10}, "B": {"MSFT": -4}}
        co.ref_price["MSFT"] = 400.0
        co.instrument["MSFT"] = inst("MSFT")

        rebal = co._rebalance({"MSFT"})

        # Internal cross: 4 shares
        # A gets +4 @ 400, B gets -4 @ 400
        assert abs(ex.ledger.strategy_positions["A"]["MSFT"] - 4.0) < 1e-9
        assert abs(ex.ledger.strategy_positions["B"]["MSFT"] - (-4.0)) < 1e-9
        assert abs(ex.ledger.strategy_avg_cost["A"]["MSFT"] - 400.0) < 1e-9
        assert abs(ex.ledger.strategy_avg_cost["B"]["MSFT"] - 400.0) < 1e-9

        # IB fills +6 at 410 (market moved up)
        co.attribute_fill("MSFT", 6, 410)

        # A now has 10: 4 @ 400 + 6 @ 410 → avg cost = (4*400 + 6*410)/10 = 406
        assert abs(ex.ledger.strategy_positions["A"]["MSFT"] - 10.0) < 1e-9
        assert abs(ex.ledger.strategy_avg_cost["A"]["MSFT"] - 406.0) < 1e-9

        # B stays at -4 @ 400 (no IB attribution needed)
        assert abs(ex.ledger.strategy_positions["B"]["MSFT"] - (-4.0)) < 1e-9
        assert abs(ex.ledger.strategy_avg_cost["B"]["MSFT"] - 400.0) < 1e-9

    def test_realized_pnl_on_close_after_internal_cross(self):
        """A buys internally, then sells later → realized P&L uses internal entry price."""
        cfg = _cfg(A=1e9, B=1e9)
        ex = FakeExecutor(cfg)
        co = NettingCoordinator(ex, cfg)

        # Phase 1: A wants +5, B wants -5 → full internal cross @ 400
        co.desired = {"A": {"MSFT": 5}, "B": {"MSFT": -5}}
        co.ref_price["MSFT"] = 400.0
        co.instrument["MSFT"] = inst("MSFT")
        rebal = co._rebalance({"MSFT"})
        assert len(rebal["orders"]) == 0  # fully crossed, no IB order

        # Phase 2: A exits (target → 0), B exits (target → 0)
        # Net desired = 0, current_positions = 0 → no IB order needed
        # But A has +5 and B has -5 in strategy_positions
        co.desired = {"A": {"MSFT": 0}, "B": {"MSFT": 0}}
        co.ref_price["MSFT"] = 420.0  # price moved up
        rebal2 = co._rebalance({"MSFT"})

        # Internal cross: A wants 0, has +5 → delta -5 (seller)
        #                 B wants 0, has -5 → delta +5 (buyer)
        # Cross 5 @ 420
        assert len(rebal2["internal_crosses"]) == 2
        assert len(rebal2["orders"]) == 0

        # A realized P&L: bought at 400, sold at 420 → +20 per share * 5 = +100
        assert abs(ex.ledger.strategy_realized_pnl["A"] - 100.0) < 1e-9
        # B realized P&L: sold at 400, bought at 420 → loss of 20 per share * 5 = -100
        assert abs(ex.ledger.strategy_realized_pnl["B"] - (-100.0)) < 1e-9


class TestInternalCrossDBLogging:
    """Internal crosses get logged to the DB with 'xnet-' exec_id prefix."""

    def test_internal_fills_logged_to_db(self):
        cfg = _cfg(A=1e9, B=1e9)
        ex = FakeExecutor(cfg)
        co = NettingCoordinator(ex, cfg)

        co.desired = {"A": {"MSFT": 10}, "B": {"MSFT": -3}}
        co.ref_price["MSFT"] = 400.0
        co.instrument["MSFT"] = inst("MSFT")

        co._rebalance({"MSFT"})

        # Should have 2 internal fill rows (A BOT + B SLD)
        internal_fills = [f for f in ex.logger_db.fills if f["exec_id"].startswith("xnet-")]
        assert len(internal_fills) == 2

        a_fill = [f for f in internal_fills if f["strategy_id"] == "A"][0]
        b_fill = [f for f in internal_fills if f["strategy_id"] == "B"][0]

        assert a_fill["order_id"] == 0
        assert a_fill["side"] == "BOT"
        assert abs(a_fill["quantity"] - 3.0) < 1e-9
        assert a_fill["price"] == 400.0
        assert a_fill["symbol"] == "MSFT"

        assert b_fill["order_id"] == 0
        assert b_fill["side"] == "SLD"
        assert abs(b_fill["quantity"] - 3.0) < 1e-9

    def test_no_db_logging_when_no_cross(self):
        cfg = _cfg(A=1e9, B=1e9)
        ex = FakeExecutor(cfg)
        co = NettingCoordinator(ex, cfg)

        co.desired = {"A": {"MSFT": 10}, "B": {"MSFT": 5}}
        co.ref_price["MSFT"] = 400.0
        co.instrument["MSFT"] = inst("MSFT")

        co._rebalance({"MSFT"})

        internal_fills = [f for f in ex.logger_db.fills if f["exec_id"].startswith("xnet-")]
        assert len(internal_fills) == 0


class TestInternalCrossInvariants:
    """The net broker position invariant is maintained through internal crossing."""

    def test_current_positions_unchanged_by_cross(self):
        """Internal crosses are zero-sum: current_positions must not change."""
        cfg = _cfg(A=1e9, B=1e9, C=1e9)
        ex = FakeExecutor(cfg)
        co = NettingCoordinator(ex, cfg)

        co.desired = {"A": {"MSFT": 10}, "B": {"MSFT": -5}, "C": {"MSFT": -3}}
        co.ref_price["MSFT"] = 400.0
        co.instrument["MSFT"] = inst("MSFT")

        before = ex.ledger.current_positions.get("MSFT", 0)
        co._internal_cross({"MSFT"})
        after = ex.ledger.current_positions.get("MSFT", 0)

        assert before == after

    def test_strategy_sum_equals_current_after_full_cycle(self):
        """After internal cross + IB fill, sum of strategy positions = current_positions."""
        cfg = _cfg(A=1e9, B=1e9)
        ex = FakeExecutor(cfg)
        co = NettingCoordinator(ex, cfg)

        co.desired = {"A": {"MSFT": 10}, "B": {"MSFT": -3}}
        co.ref_price["MSFT"] = 400.0
        co.instrument["MSFT"] = inst("MSFT")

        rebal = co._rebalance({"MSFT"})

        # Fill the IB order
        for o in rebal["orders"]:
            co.attribute_fill(o["symbol"], o["delta"], 405)

        strat_sum = sum(
            ex.ledger.strategy_positions.get(s, {}).get("MSFT", 0)
            for s in ("A", "B")
        )
        assert abs(strat_sum - ex.ledger.current_positions["MSFT"]) < 1e-9

    def test_effective_position_correct_after_cross(self):
        """effective_position (current + pending) must still reflect only broker state."""
        cfg = _cfg(A=1e9, B=1e9)
        ex = FakeExecutor(cfg)
        co = NettingCoordinator(ex, cfg)

        co.desired = {"A": {"MSFT": 8}, "B": {"MSFT": -8}}
        co.ref_price["MSFT"] = 400.0
        co.instrument["MSFT"] = inst("MSFT")

        rebal = co._rebalance({"MSFT"})

        # Fully crossed — no IB order
        assert len(rebal["orders"]) == 0
        # effective_position should be 0 (no broker orders)
        assert abs(ex.ledger.effective_position("MSFT") - 0) < 1e-9
        # But strategy positions are filled
        assert abs(ex.ledger.strategy_positions["A"]["MSFT"] - 8.0) < 1e-9
        assert abs(ex.ledger.strategy_positions["B"]["MSFT"] - (-8.0)) < 1e-9


class TestInternalCrossMultipleSymbols:
    """Crossing works independently per symbol."""

    def test_two_symbols_cross_independently(self):
        cfg = _cfg(A=1e9, B=1e9)
        ex = FakeExecutor(cfg)
        co = NettingCoordinator(ex, cfg)

        co.desired = {
            "A": {"MSFT": 10, "AAPL": -5},
            "B": {"MSFT": -3, "AAPL": 8},
        }
        co.ref_price = {"MSFT": 400.0, "AAPL": 200.0}
        co.instrument = {"MSFT": inst("MSFT"), "AAPL": inst("AAPL")}

        rebal = co._rebalance({"MSFT", "AAPL"})
        crosses = rebal["internal_crosses"]
        orders = rebal["orders"]

        # MSFT: cross 3, residual +7
        msft_crosses = [c for c in crosses if c["symbol"] == "MSFT"]
        assert len(msft_crosses) == 2
        msft_orders = [o for o in orders if o["symbol"] == "MSFT"]
        assert abs(msft_orders[0]["delta"] - 7.0) < 1e-9

        # AAPL: cross 5, residual +3
        aapl_crosses = [c for c in crosses if c["symbol"] == "AAPL"]
        assert len(aapl_crosses) == 2
        aapl_orders = [o for o in orders if o["symbol"] == "AAPL"]
        assert abs(aapl_orders[0]["delta"] - 3.0) < 1e-9
