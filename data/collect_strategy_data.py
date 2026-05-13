"""
Poll Optibook for strategy-oriented market snapshots across stocks, index ETFs, and index futures.

CSV output defaults to this package directory (``data/``). Override with ``run_collector(..., output_dir=...)``.

**Run from repo root:**

    PYTHONPATH=. python data/collect_strategy_data.py
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from optibook.common_types import InstrumentType
from optibook.synchronous_client import Exchange

_DATA_DIR = Path(__file__).resolve().parent

# --- CONFIGURATION ---
POLL_INTERVAL = 1.0  # seconds between poll cycles (defines bar spacing for offline MAs)
OUTPUT_DIR = _DATA_DIR
# Per-instrument files: one CSV per symbol (ETFs/futures use the same naming pattern).
OUTPUT_FILENAME_TEMPLATE = "{instrument_id}_strategy_market_data.csv"

# If non-empty, only these instrument IDs are polled (must exist on the exchange).
INSTRUMENT_IDS: List[str] = []

# Subset of InstrumentType to include when discovery is used (full universe mode).
INCLUDE_TYPES: Set[InstrumentType] = {
    InstrumentType.STOCK,
    InstrumentType.INDEX_TRACKING_ETF,
    InstrumentType.INDEX_FUTURE,
}

HEADER = [
    "timestamp",
    "instrument_id",
    "instrument_type",
    "instrument_group",
    "index_id",
    "bid_price",
    "bid_volume",
    "ask_price",
    "ask_volume",
    "mid",
    "spread",
    "last_trade_price",
]


def ensure_header(path: Union[str, Path]) -> None:
    try:
        with open(path, "r"):
            pass
    except FileNotFoundError:
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(HEADER)


def instrument_csv_path(
    instrument_id: str,
    output_dir: Path,
    filename_template: str,
) -> Path:
    """Stable filename for one instrument (slashes stripped from id)."""
    safe = str(instrument_id).replace("/", "_").replace("\\", "_")
    return output_dir / filename_template.format(instrument_id=safe)


def _str_id(inst_id: Any) -> str:
    return str(inst_id)


def _meta_for_instrument(inst: Any) -> Tuple[str, str, str]:
    """instrument_type name, instrument_group, index_id for CSV."""
    it = getattr(inst, "instrument_type", None)
    type_name = it.name if it is not None and hasattr(it, "name") else (str(it) if it else "")
    group = getattr(inst, "instrument_group", None)
    group_s = _str_id(group) if group is not None else ""
    idx = getattr(inst, "index_id", None)
    idx_s = _str_id(idx) if idx is not None else ""
    return type_name, group_s, idx_s


def discover_universe(
    instruments: Dict[Any, Any],
    include_types: Iterable[InstrumentType],
    explicit_ids: Optional[Sequence[str]],
) -> List[Tuple[str, Any]]:
    """
    Returns sorted list of (instrument_id_str, Instrument or None).
    If explicit_ids is set and non-empty, only those IDs are kept (Instrument may be None if missing).
    """
    include = set(include_types)
    out: List[Tuple[str, Any]] = []

    if explicit_ids:
        wanted = {_str_id(x) for x in explicit_ids}
        for key, inst in instruments.items():
            sid = _str_id(key)
            if sid not in wanted:
                continue
            if inst is not None and (
                getattr(inst, "paused", False) or getattr(inst, "expired", False)
            ):
                continue
            out.append((sid, inst))
        found = {t[0] for t in out}
        for wid in sorted(wanted):
            if wid not in found:
                out.append((wid, None))
        return sorted(out, key=lambda x: x[0])

    for key, inst in instruments.items():
        if inst is None:
            continue
        if getattr(inst, "paused", False) or getattr(inst, "expired", False):
            continue
        it = getattr(inst, "instrument_type", None)
        if it not in include:
            continue
        out.append((_str_id(key), inst))
    return sorted(out, key=lambda x: x[0])


def snapshot_row(
    ts: float,
    instrument_id: str,
    inst: Optional[Any],
    exchange: Exchange,
) -> List[Any]:
    """One CSV row for this instrument at timestamp ts."""
    if inst is not None:
        type_name, group_s, idx_s = _meta_for_instrument(inst)
    else:
        type_name, group_s, idx_s = "", "", ""

    book = exchange.get_last_price_book(instrument_id)
    ticks = exchange.get_trade_tick_history(instrument_id)
    last_trade = ticks[-1].price if ticks else None

    bid_price = book.bids[0].price if book.bids else None
    bid_volume = book.bids[0].volume if book.bids else None
    ask_price = book.asks[0].price if book.asks else None
    ask_volume = book.asks[0].volume if book.asks else None

    mid: Optional[float] = None
    spread: Optional[float] = None
    if bid_price is not None and ask_price is not None:
        spread = ask_price - bid_price
    # Microprice: weight bid by ask size and ask by bid size (imbalance-aware touch average).
    if (
        bid_price is not None
        and ask_price is not None
        and bid_volume is not None
        and ask_volume is not None
    ):
        denom = float(bid_volume) + float(ask_volume)
        if denom > 0:
            mid = (
                float(bid_price) * float(ask_volume)
                + float(ask_price) * float(bid_volume)
            ) / denom

    def fmt(x: Any) -> Union[str, float, int]:
        if x is None:
            return ""
        return x

    return [
        ts,
        instrument_id,
        type_name,
        group_s,
        idx_s,
        fmt(bid_price),
        fmt(bid_volume),
        fmt(ask_price),
        fmt(ask_volume),
        fmt(mid),
        fmt(spread),
        fmt(last_trade),
    ]


def run_collector(
    exchange: Exchange,
    poll_interval: float = POLL_INTERVAL,
    output_dir: Union[str, Path] = OUTPUT_DIR,
    filename_template: str = OUTPUT_FILENAME_TEMPLATE,
    instrument_ids: Optional[Sequence[str]] = None,
    include_types: Optional[Set[InstrumentType]] = None,
) -> None:
    inc = include_types if include_types is not None else INCLUDE_TYPES
    explicit = list(instrument_ids) if instrument_ids is not None else INSTRUMENT_IDS
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    instruments = exchange.get_instruments()
    universe = discover_universe(instruments, inc, explicit if explicit else None)

    paths = [instrument_csv_path(iid, out_dir, filename_template) for iid, _ in universe]
    for p in paths:
        ensure_header(p)

    mode = "explicit list" if explicit else "discovered"
    print(
        f"Universe ({mode}): {len(universe)} instrument(s). "
        f"Writing under {str(out_dir.resolve())!r} "
        f"({filename_template!r}) every {poll_interval}s. Ctrl+C to stop."
    )

    try:
        while True:
            loop_start = time.time()
            ts = time.time()
            for (inst_id, inst), path in zip(universe, paths):
                with open(path, "a", newline="") as f:
                    csv.writer(f).writerow(snapshot_row(ts, inst_id, inst, exchange))

            elapsed = time.time() - loop_start
            if elapsed < poll_interval:
                time.sleep(poll_interval - elapsed)
    except KeyboardInterrupt:
        print("\nPolling stopped by user.")


def main() -> None:
    exchange = Exchange()
    exchange.connect()
    run_collector(exchange)


if __name__ == "__main__":
    main()
