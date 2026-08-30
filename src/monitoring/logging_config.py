"""
logging_config.py

Central logging setup for the executor. Call setup_logging() once at startup,
before creating the CentralExecutor, so all subsequent log calls use this config.

Two things this handles beyond a plain basicConfig:
  1. Quiets ibapi's own very chatty loggers (ibapi.client / ibapi.wrapper), which
     otherwise flood the log with REQUEST/ANSWER/SENDING/byte-dump lines at INFO.
  2. Uses a rotating file handler so the log can't grow unbounded during a long
     unattended run.
"""

import logging
from logging.handlers import RotatingFileHandler
from monitoring.alerter import Alerter, AlertingHandler
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
LOG_DIR = MODULE_DIR / ".." / "logs"

BENIGN_IB_CODES = {2104, 2106, 2158, 2107, 2108, 2100, 2150, 2119, 2103, 2105, 202}

class IBNoiseFilter(logging.Filter):
    """Drops ibapi's benign status-code messages (2104 'connection is OK', etc.)
    that it logs at ERROR level even though nothing is wrong."""
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        # ibapi's error lines look like: "ERROR -1 2104 Market data farm connection is OK..."
        # keep the record (return True) unless it contains a benign code
        if record.name.startswith("ibapi"):
            for code in BENIGN_IB_CODES:
                if f" {code} " in msg:
                    return False  # drop it
        return True  # keep everything else
    
def setup_logging(level: int = logging.INFO, ibapi_level: int = logging.WARNING) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = (LOG_DIR / "executor.log").resolve()

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    noise_filter = IBNoiseFilter()

    file_handler = RotatingFileHandler(log_path, maxBytes=10_000_000, backupCount=5)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(noise_filter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(noise_filter)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(console_handler)

    alerter = Alerter()
    root.addHandler(AlertingHandler(alerter, level=logging.CRITICAL))

    logging.getLogger("ibapi").setLevel(ibapi_level)
