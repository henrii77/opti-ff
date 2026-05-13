#!/usr/bin/env python3
"""
Plot NVDA and NVDA_DUAL prices over time on one chart from data written by
``collect_strategy_data.py`` (default filenames: ``NVDA_strategy_market_data.csv``,
``NVDA_DUAL_strategy_market_data.csv`` in ``OUTPUT_DIR``).

Uses the ``mid`` column (volume-weighted touch) when present; otherwise ``last_trade_price``.

Run from repo root (or pass ``--data-dir``):

    python common/plot_nvda_dual_prices.py
    python common/plot_nvda_dual_prices.py --data-dir /path/to/csvs --save nvda_dual.png
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

# Keep in sync with collect_strategy_data.OUTPUT_FILENAME_TEMPLATE default.
DEFAULT_FILENAME_TEMPLATE = "{instrument_id}_strategy_market_data.csv"


def _csv_path(data_dir: Path, instrument_id: str, template: str) -> Path:
    return data_dir / template.format(instrument_id=instrument_id)


def _parse_float(cell: str) -> Optional[float]:
    if cell is None or cell == "":
        return None
    try:
        return float(cell)
    except ValueError:
        return None


def load_price_series(
    path: Path,
) -> Tuple[List[datetime], List[float]]:
    """Return (timestamps as UTC datetime, prices) for rows with a usable price."""
    times: List[datetime] = []
    prices: List[float] = []
    if not path.is_file():
        return times, prices

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_raw = row.get("timestamp", "").strip()
            if not ts_raw:
                continue
            try:
                ts = float(ts_raw)
            except ValueError:
                continue
            mid = _parse_float((row.get("mid") or "").strip())
            last = _parse_float((row.get("last_trade_price") or "").strip())
            price = mid if mid is not None else last
            if price is None:
                continue
            # Interpret epoch seconds as UTC for axis labels (consistent regardless of local TZ).
            times.append(datetime.fromtimestamp(ts, tz=timezone.utc))
            prices.append(price)
    return times, prices


def plot_nvda_pair() -> None:
    parser = argparse.ArgumentParser(description="Plot NVDA vs NVDA_DUAL from strategy CSVs.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("."),
        help="Directory containing per-instrument CSV files (default: current directory).",
    )
    parser.add_argument(
        "--template",
        default=DEFAULT_FILENAME_TEMPLATE,
        help='Filename template with {instrument_id} (default matches collect_strategy_data).',
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="If set, save figure to this path instead of showing interactively.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    path_nvda = _csv_path(data_dir, "NVDA", args.template)
    path_dual = _csv_path(data_dir, "NVDA_DUAL", args.template)

    t1, p1 = load_price_series(path_nvda)
    t2, p2 = load_price_series(path_dual)

    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required. Install with: pip install matplotlib"
        ) from e

    fig, ax = plt.subplots(figsize=(10, 5))
    if t1:
        ax.plot(t1, p1, label="NVDA", linewidth=1.2)
    else:
        print(f"Warning: no plottable rows in {path_nvda}")
    if t2:
        ax.plot(t2, p2, label="NVDA_DUAL", linewidth=1.2)
    else:
        print(f"Warning: no plottable rows in {path_dual}")

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Price (mid or last trade)")
    ax.set_title("NVDA vs NVDA_DUAL")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S\n%m-%d"))
    fig.autofmt_xdate()

    plt.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"Saved {args.save.resolve()}")
    else:
        plt.show()