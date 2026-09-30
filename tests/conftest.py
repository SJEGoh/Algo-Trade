"""Shared test setup.

Keeps the real db/executor.db out of the suite. CentralExecutor() opens EventLogger() at its
default path — the live database, which is tracked in git — so every test that built an
executor wrote to it and left executor.db-shm / -wal behind. Several files patched this by
hand with a FakeDB; this makes the safe path the default. A test that patches EventLogger
itself still wins (its monkeypatch runs after this one).
"""
import pytest


@pytest.fixture(autouse=True)
def _executor_db_in_tmp(tmp_path, monkeypatch):
    try:
        import execution.central_execution as ce
    except ImportError:                     # no ibapi here: nothing can build an executor
        yield
        return
    real = ce.EventLogger
    opened = []

    def _tmp_logger(db_path=None):
        db = real(db_path=db_path or tmp_path / "executor.db")
        opened.append(db)
        return db

    monkeypatch.setattr(ce, "EventLogger", _tmp_logger)
    yield
    for db in opened:
        db.close()
