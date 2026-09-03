"""
tests/test_persistence.py

Tests for:
  1. Strategy state persistence (positions, avg cost, realized P&L, multipliers)
  2. Halted strategies persistence
  3. Decision journal (log + query)
  4. Rotation strategy state save/load

Uses tmp_path so each test gets a fresh database.
"""
import json
import pytest
import sys
from pathlib import Path

# ensure src/ and project root are on the path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))

from logger.event_logger import EventLogger


@pytest.fixture
def db(tmp_path):
    return EventLogger(db_path=tmp_path / "test.db")


# ── strategy state persistence ──────────────────────────────────────

class TestStrategyStatePersistence:
    def test_save_and_load_positions(self, db):
        positions = {"s1": {"AAPL": 100.0, "TSLA": -50.0}, "s2": {"MSFT": 200.0}}
        avg_cost = {"s1": {"AAPL": 150.0, "TSLA": 300.0}, "s2": {"MSFT": 400.0}}
        db.save_strategy_positions(positions, avg_cost)

        loaded_pos, loaded_cost = db.load_strategy_positions()
        assert loaded_pos == positions
        assert loaded_cost == avg_cost

    def test_save_positions_overwrites_previous(self, db):
        db.save_strategy_positions({"s1": {"AAPL": 100.0}}, {"s1": {"AAPL": 150.0}})
        db.save_strategy_positions({"s1": {"GOOGL": 50.0}}, {"s1": {"GOOGL": 2800.0}})

        loaded_pos, loaded_cost = db.load_strategy_positions()
        assert "AAPL" not in loaded_pos.get("s1", {})
        assert loaded_pos["s1"]["GOOGL"] == 50.0

    def test_zero_qty_positions_excluded(self, db):
        db.save_strategy_positions({"s1": {"AAPL": 0.0, "TSLA": 10.0}}, {"s1": {"AAPL": 0, "TSLA": 100}})
        loaded_pos, _ = db.load_strategy_positions()
        assert "AAPL" not in loaded_pos.get("s1", {})
        assert loaded_pos["s1"]["TSLA"] == 10.0

    def test_save_and_load_realized_pnl(self, db):
        pnl = {"s1": 1234.56, "s2": -789.0}
        db.save_realized_pnl(pnl)
        assert db.load_realized_pnl() == pnl

    def test_realized_pnl_overwrites(self, db):
        db.save_realized_pnl({"s1": 100.0})
        db.save_realized_pnl({"s1": 200.0, "s2": 50.0})
        loaded = db.load_realized_pnl()
        assert loaded["s1"] == 200.0
        assert loaded["s2"] == 50.0

    def test_save_and_load_multipliers(self, db):
        db.save_multipliers({"CL": 1000.0, "ES": 50.0, "AAPL": 1.0})
        loaded = db.load_multipliers()
        assert loaded["CL"] == 1000.0
        assert loaded["ES"] == 50.0
        assert "AAPL" not in loaded  # multiplier 1.0 is excluded

    def test_empty_state_loads_cleanly(self, db):
        pos, cost = db.load_strategy_positions()
        assert pos == {}
        assert cost == {}
        assert db.load_realized_pnl() == {}
        assert db.load_multipliers() == {}


# ── halted strategies persistence ────────────────────────────────────

class TestHaltedStrategies:
    def test_save_and_load_halted(self, db):
        config_keys = {"s1", "s2", "s3"}
        active = {"s1", "s3"}  # s2 is halted
        db.save_halted_strategies(set(), active, config_keys, "drawdown breach")
        assert db.load_halted_strategies() == {"s2"}

    def test_no_halts_returns_empty(self, db):
        config_keys = {"s1", "s2"}
        active = {"s1", "s2"}
        db.save_halted_strategies(set(), active, config_keys)
        assert db.load_halted_strategies() == set()

    def test_halted_overwrites_previous(self, db):
        db.save_halted_strategies(set(), {"s1"}, {"s1", "s2"})  # s2 halted
        db.save_halted_strategies(set(), {"s1", "s2"}, {"s1", "s2"})  # all active
        assert db.load_halted_strategies() == set()


# ── decision journal ─────────────────────────────────────────────────

class TestDecisionJournal:
    def test_log_and_query_decision(self, db):
        db.log_decision("s1", "signal", "5 active positions",
                        detail='{"weights": {}}', symbols=["AAPL", "TSLA"])
        entries = db.get_journal()
        assert len(entries) == 1
        assert entries[0]["strategy_id"] == "s1"
        assert entries[0]["event_type"] == "signal"
        assert entries[0]["summary"] == "5 active positions"
        assert entries[0]["symbols"] == "AAPL,TSLA"

    def test_journal_filter_by_strategy(self, db):
        db.log_decision("s1", "signal", "s1 signal")
        db.log_decision("s2", "signal", "s2 signal")
        entries = db.get_journal(strategy_id="s1")
        assert len(entries) == 1
        assert entries[0]["strategy_id"] == "s1"

    def test_journal_filter_by_event_type(self, db):
        db.log_decision("s1", "signal", "signal event")
        db.log_decision("s1", "halt", "halt event")
        entries = db.get_journal(event_type="halt")
        assert len(entries) == 1
        assert entries[0]["event_type"] == "halt"

    def test_journal_limit(self, db):
        for i in range(10):
            db.log_decision("s1", "signal", f"event {i}")
        entries = db.get_journal(limit=3)
        assert len(entries) == 3

    def test_journal_most_recent_first(self, db):
        db.log_decision("s1", "signal", "first")
        db.log_decision("s1", "signal", "second")
        entries = db.get_journal()
        assert entries[0]["summary"] == "second"
        assert entries[1]["summary"] == "first"

    def test_journal_dict_detail_serialized(self, db):
        db.log_decision("s1", "rebalance", "test",
                        detail={"orders": [1, 2, 3]})
        entries = db.get_journal()
        parsed = json.loads(entries[0]["detail"])
        assert parsed["orders"] == [1, 2, 3]

    def test_journal_long_detail_truncated(self, db):
        long_detail = "x" * 20_000
        db.log_decision("s1", "signal", "test", detail=long_detail)
        entries = db.get_journal()
        assert len(entries[0]["detail"]) < 11_000
        assert entries[0]["detail"].endswith("...(truncated)")

    def test_journal_no_symbols(self, db):
        db.log_decision("s1", "halt", "halted for drawdown")
        entries = db.get_journal()
        assert entries[0]["symbols"] is None


# ── rotation strategy state save/load ────────────────────────────────

class TestRotationStatePersistence:
    def test_save_and_load_state(self, tmp_path):
        from models.rotation import CombinedRotationStrategy

        # Minimal mock data provider
        class MockDP:
            def get_daily_bars(self, sym, lookback):
                import pandas as pd
                return pd.Series([100.0], index=pd.to_datetime(["2024-01-01"]))

        state_file = tmp_path / "rrg_state.json"
        strat = CombinedRotationStrategy(
            data_provider=MockDP(),
            universe=["AAPL", "TSLA"],
            state_path=str(state_file),
        )
        # Manually call _save_state
        signal = {"AAPL": {"x": 0.5, "y": 0.3, "score": 0.4, "quadrant": "Leading"}}
        weights = {"AAPL": 0.35}
        strat._save_state(signal, weights, {"AAPL": 1.0}, {"AAPL": 0.2})

        assert state_file.exists()
        loaded = strat.load_state()
        assert loaded is not None
        assert loaded["signal"]["AAPL"]["quadrant"] == "Leading"
        assert loaded["combined_weights"]["AAPL"] == 0.35
        assert "timestamp" in loaded

    def test_load_state_missing_file(self, tmp_path):
        from models.rotation import CombinedRotationStrategy

        class MockDP:
            def get_daily_bars(self, sym, lookback):
                import pandas as pd
                return pd.Series([100.0], index=pd.to_datetime(["2024-01-01"]))

        strat = CombinedRotationStrategy(
            data_provider=MockDP(),
            universe=["AAPL"],
            state_path=str(tmp_path / "nonexistent.json"),
        )
        assert strat.load_state() is None

    def test_load_state_no_path(self):
        from models.rotation import CombinedRotationStrategy

        class MockDP:
            def get_daily_bars(self, sym, lookback):
                import pandas as pd
                return pd.Series([100.0], index=pd.to_datetime(["2024-01-01"]))

        strat = CombinedRotationStrategy(
            data_provider=MockDP(),
            universe=["AAPL"],
        )
        assert strat.load_state() is None


# ── ledger save/restore roundtrip ────────────────────────────────────

class TestLedgerPersistenceRoundtrip:
    def test_save_and_restore(self, db):
        from ledger.position_ledger import PositionLedger

        ledger = PositionLedger(executor=None)
        ledger.strategy_positions = {"s1": {"AAPL": 100.0}, "s2": {"TSLA": -50.0}}
        ledger.strategy_avg_cost = {"s1": {"AAPL": 150.0}, "s2": {"TSLA": 300.0}}
        ledger.strategy_realized_pnl = {"s1": 500.0, "s2": -200.0}
        ledger.multipliers = {"CL": 1000.0}

        ledger.save_state(db)

        # Create a fresh ledger and restore
        ledger2 = PositionLedger(executor=None)
        ledger2.restore_state(db)

        assert ledger2.strategy_positions == {"s1": {"AAPL": 100.0}, "s2": {"TSLA": -50.0}}
        assert ledger2.strategy_avg_cost == {"s1": {"AAPL": 150.0}, "s2": {"TSLA": 300.0}}
        assert ledger2.strategy_realized_pnl == {"s1": 500.0, "s2": -200.0}
        assert ledger2.multipliers == {"CL": 1000.0}
