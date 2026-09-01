"""pytest path setup.

Puts the repo ROOT (for the `models` package) and `src/` (for vecm, indicators,
equity_signals, ledger, risk, execution, api, config, ...) on sys.path so the whole
suite imports cleanly with a plain `pytest` — no PYTHONPATH needed.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
