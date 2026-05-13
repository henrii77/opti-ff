#!/usr/bin/env python3
"""
Plot NVDA vs NVDA_DUAL from CSVs in this ``data/`` package directory by default.

Supports:

- **Strategy collector** (``collect_strategy_data.py``): ``{instrument_id}_strategy_market_data.csv``
- **Legacy collector** (``collect.py``): ``{instrument_id}_data.csv``

Reads UTF-8-with-BOM. Price per row: ``mid``, else microprice from book, else mid of bid-ask, else last trade.

**Run from repo root:**

    PYTHONPATH=. python data/plot_dual_prices.py
    PYTHONPATH=. python data/plot_dual_prices.py --data-dir data --save nvda_dual.png

**Notebook:**

    from data.plot_dual_prices import plot_nvda_pair
    plot_nvda_pair()
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

_DEFAULT_DATA_DIR = Path(__file__).resolve().parent

STRATEGY_TEMPLATE = "{instrument_id}_strategy_market_data.csv"
LEGACY_TEMPLATE = "{instrument_id}_data.csv"


def _strip_row(row: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in row.items():
        key = (k or "").strip()
        if not key:
            continue
        if v is None:
            out[key] = ""
        else:
            out[key] = str(v).strip()
    return out


def _parse_float(cell: str) -> Optional[float]:
    if cell is None or cell == "":
        return None
    try:
        return float(cell)
    except ValueError:
        return None


def _parse_int(cell: str) -> Optional[int]:
    if cell is None or cell == "":
        return None
    try:
        return int(float(cell))
    except ValueError:
        return None


def _price_from_row(row: Dict[str, str]) -> Optional[float]:
    mid = _parse_float(row.get("mid", ""))
    if mid is not None:
        return mid

    bid = _parse_float(row.get("bid_price", ""))
    ask = _parse_float(row.get("ask_price", ""))
    bv = _parse_int(row.get("bid_volume", ""))
    av = _parse_int(row.get("ask_volume", ""))

    if (
        bid is not None
        and ask is not None
        and bv is not None
        and av is not None
        and (bv + av) > 0
    ):
        return (bid * av + ask * bv) / float(bv + av)

    if bid is not None and ask is not None:
        return (bid + ask) / 2.0

    last = _parse_float(row.get("last_trade_price", ""))
    if last is not None:
        return last

    return _parse_float(row.get("last_trade", ""))


def resolve_csv_path(
    data_dir: Path,
    instrument_id: str,
    template: Optional[str],
) -> Path:
    if template:
        p = data_dir / template.format(instrument_id=instrument_id)
        if p.is_file():
            return p
    p_strat = data_dir / STRATEGY_TEMPLATE.format(instrument_id=instrument_id)
    if p_strat.is_file():
        return p_strat
    p_legacy = data_dir / LEGACY_TEMPLATE.format(instrument_id=instrument_id)
    return p_legacy


def load_price_series(path: Path) -> Tuple[List[datetime], List[float]]:
    times: List[datetime] = []
    prices: List[float] = []
    if not path.is_file():
        return times, prices

    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return times, prices
        for raw in reader:
            row = _strip_row(raw)
            ts_raw = row.get("timestamp", "")
            if not ts_raw:
                continue
            try:
                ts = float(ts_raw)
            except ValueError:
                continue
            price = _price_from_row(row)
            if price is None:
                continue
            times.append(datetime.fromtimestamp(ts, tz=timezone.utc))
            prices.append(price)
    return times, prices


def plot_nvda_pair(
    data_dir: Optional[Union[str, Path]] = None,
    template: Optional[str] = None,
    save: Optional[Union[str, Path]] = None,
    left_id: str = "NVDA",
    right_id: str = "NVDA_DUAL",
) -> None:
    """
    Plot two instruments. ``data_dir`` defaults to this package's directory (``data/``).
    """
    data_dir_p = Path(data_dir).resolve() if data_dir is not None else _DEFAULT_DATA_DIR
    path_left = resolve_csv_path(data_dir_p, left_id, template)
    path_right = resolve_csv_path(data_dir_p, right_id, template)

    t1, p1 = load_price_series(path_left)
    t2, p2 = load_price_series(path_right)

    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required. Install with: pip install matplotlib"
        ) from e

    fig, ax = plt.subplots(figsize=(10, 5))
    if t1:
        ax.plot(t1, p1, label=left_id, linewidth=1.2)
    else:
        print(f"Warning: no plottable rows in {path_left}")
    if t2:
        ax.plot(t2, p2, label=right_id, linewidth=1.2)
    else:
        print(f"Warning: no plottable rows in {path_right}")

    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Price (mid / microprice / mid of bid-ask / last trade)")
    ax.set_title(f"{left_id} vs {right_id}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S\n%m-%d"))
    fig.autofmt_xdate()

    plt.tight_layout()
    if save:
        out = Path(save)
        fig.savefig(out, dpi=150)
        print(f"Saved {out.resolve()}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot NVDA vs NVDA_DUAL from CSV data.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=_DEFAULT_DATA_DIR,
        help="Directory containing CSV files (default: this data/ package).",
    )
    parser.add_argument(
        "--template",
        default=None,
        help=(
            "Optional filename template with {instrument_id}. "
            "Default: try strategy then legacy _data.csv names."
        ),
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="If set, save figure to this path instead of showing interactively.",
    )
    args, _unknown = parser.parse_known_args()
    plot_nvda_pair(
        data_dir=args.data_dir,
        template=args.template,
        save=args.save,
    )


if __name__ == "__main__":
    main()
