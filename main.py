import signal
import time

from execution.central_execution import CentralExecutor
from monitoring.logging_config import setup_logging

from tests.fixtures.orders import test_orders, get_test_order, list_test_case_ids, get_burst_test_orders
# main.py (or a dedicated test_recovery.py)
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).resolve().parent / ".env")

if __name__ == "__main__":
    setup_logging()
    app = CentralExecutor()

    def handle_sigint(sig, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, handle_sigint)

    try:
        recon = app.start(client_id=6)
        print("\n=== STARTUP RECONCILIATION ===")
        print(f"matched: {recon['matched']}")
        print(f"broker positions: {recon.get('broker_positions', {})}")
        print(f"internal positions: {dict(app.ledger.current_positions)}")
        print(f"open orders known to executor: "
              f"{[(oid, s['status']) for oid, s in app.order_status.items() if s['status'] in ('PreSubmitted', 'Submitted')]}")

        # Place an order that will STAY OPEN (limit far below market, won't fill)
        # so there's live state at IB to recover after a kill.
        open_order = get_test_order("test-014-far-limit")
        open_order["expected_price"] = 200.0
        # unique client_order_id each run so the DB / dedup doesn't collide across runs
        open_order["client_order_id"] = f"recovery-test-{int(time.time())}"
        r = app.process_intent(open_order)
        print(f"\nplaced open order: {r}")

        print("\n=== NOW KILL THIS PROCESS ===")
        print("In another terminal, run:  pkill -9 -f 'python main.py'")
        print("Then restart this script and check whether startup reconciliation")
        print("recovers the open order and any position from IB.\n")

        # keep the process alive so the order stays live at IB while you go kill it
        while True:
            time.sleep(5)

    finally:
        app.shutdown()
