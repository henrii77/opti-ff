import csv
import time
from optibook.synchronous_client import Exchange

# --- CONFIGURATION ---
STOCK_IDS = ["NVDA", "NVDA_DUAL", "AMZN", "XOM", "NVO", "JPM"]
POLL_INTERVAL = 1.0  # seconds between polls

# --- CONNECT ---
exchange = Exchange()
exchange.connect()
print(f"Connected. Logging {len(STOCK_IDS)} instrument(s) every {POLL_INTERVAL}s")

# --- PREPARE HEADER ---
HEADER = [
    "timestamp",
    "bid_price",
    "bid_volume",
    "ask_price",
    "ask_volume",
    "last_trade_price",
]


def ensure_header(filename):
    """Write header if file does not exist or is empty."""
    try:
        with open(filename, "r") as f:
            pass
    except FileNotFoundError:
        with open(filename, "w", newline="") as f:
            csv.writer(f).writerow(HEADER)


# Ensure all CSV files have headers
for inst in STOCK_IDS:
    ensure_header(f"{inst}_data.csv")

print("Polling started. Press Ctrl+C to stop.")
try:
    while True:
        loop_start = time.time()
        for inst_id in STOCK_IDS:
            # Get order book snapshot
            book = exchange.get_last_price_book(inst_id)
            # Get last traded price (if any)
            ticks = exchange.get_trade_tick_history(inst_id)
            last_trade = ticks[-1].price if ticks else None

            bid_price = book.bids[0].price if book.bids else None
            bid_volume = book.bids[0].volume if book.bids else None
            ask_price = book.asks[0].price if book.asks else None
            ask_volume = book.asks[0].volume if book.asks else None

            row = [
                time.time(),
                bid_price,
                bid_volume,
                ask_price,
                ask_volume,
                last_trade,
            ]

            # Append row to instrument-specific CSV
            csv_filename = f"{inst_id}_data.csv"
            with open(csv_filename, "a", newline="") as f:
                csv.writer(f).writerow(row)

        # Maintain the polling interval
        elapsed = time.time() - loop_start
        if elapsed < POLL_INTERVAL:
            time.sleep(POLL_INTERVAL - elapsed)

except KeyboardInterrupt:
    print("\nPolling stopped by user.")
