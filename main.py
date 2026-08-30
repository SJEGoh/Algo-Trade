import signal
import time

from execution.central_execution import CentralExecutor
from monitoring.logging_config import setup_logging

from tests.fixtures.orders import test_orders, get_test_order, list_test_case_ids, get_burst_test_orders

if __name__ == "__main__":
    setup_logging()
    app = CentralExecutor()

    def handle_sigint(sig, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        app.start(client_id=6)
        time.sleep(3)
        o = get_test_order("test-014-far-limit")
        o["expected_price"] = 200.0
        r = app.process_intent(o)
        print(f"submitted: {r}")
        time.sleep(2)

        if r["accepted"]:
            app.cancelOrder(r["order_id"])   # match YOUR ibapi signature here
            print("cancel requested")
            time.sleep(4)   # longer — give the Cancelled status callback time to arrive

        # query AFTER the cancel callback has had time to land
        rows = app.logger_db._conn.execute(
            "SELECT order_id, symbol, side, order_type, expected_price, status FROM orders"
        ).fetchall()
        print("orders table:")
        for row in rows:
            print(" ", row)
        time.sleep(3)
    finally:
        app.shutdown()
