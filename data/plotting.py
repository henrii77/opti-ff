"""
Optibook strategy CSV plotting: dual listings, OB5X theory, and arbitrary stock pairs.

CSV files live under ``data/csv/`` (``*_strategy_market_data.csv``).
"""

from __future__ import annotations

import csv
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Shared CSV helpers
# ---------------------------------------------------------------------------

_DATA_ROOT = Path(__file__).resolve().parent
# All ``*_strategy_market_data.csv`` inputs are read from ``data/csv/`` only.
_CSV_DIR = _DATA_ROOT / "csv"
DEFAULT_CSV_DIR = _CSV_DIR

def _resolve_strategy_data_dir(data_dir: Union[str, Path]) -> Path:
    """
    Strategy CSVs live under ``data/csv/``.

    If ``data_dir`` is a directory named something other than ``csv`` and
    ``<data_dir>/csv`` exists, use that subdirectory (so passing the ``data/``
    package root still resolves to ``data/csv/``).
    """
    d = Path(data_dir).resolve()
    if d.name != "csv":
        nested = d / "csv"
        if nested.is_dir():
            return nested
    return d


def _strip_row(row: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for k, v in row.items():
        key = (k or "").strip()
        if not key:
            continue
        out[key] = "" if v is None else str(v).strip()
    return out


def _parse_float(cell: str) -> Optional[float]:
    if cell is None or cell == "":
        return None
    try:
        return float(cell)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Dual-listed pairs (NVDA / NVO) — **mid** prices
# ---------------------------------------------------------------------------

NVDA_STRATEGY_CSV = _CSV_DIR / "NVDA_strategy_market_data.csv"
NVDA_DUAL_STRATEGY_CSV = _CSV_DIR / "NVDA_DUAL_strategy_market_data.csv"
NVO_STRATEGY_CSV = _CSV_DIR / "NVO_strategy_market_data.csv"
NVO_DUAL_STRATEGY_CSV = _CSV_DIR / "NVO_DUAL_strategy_market_data.csv"

_DEFAULT_TAIL_N = 2000


def load_mid_series(path: Path) -> Tuple[List[datetime], List[float]]:
    """Load ``mid`` column only; rows without ``mid`` are skipped."""
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
            mid = _parse_float(row.get("mid", ""))
            if mid is None:
                continue
            times.append(datetime.fromtimestamp(ts, tz=timezone.utc))
            prices.append(mid)
    return times, prices


def _plot_mid_pair_on_ax(
    ax: Any,
    path_base: Path,
    path_dual: Path,
    base_id: str,
    dual_id: str,
    n: Optional[int] = None,
) -> None:
    _t1, p1 = load_mid_series(path_base)
    _t2, p2 = load_mid_series(path_dual)
    lim = _DEFAULT_TAIL_N if n is None else int(n)
    if lim > 0:
        if len(p1) > lim:
            _t1, p1 = _t1[-lim:], p1[-lim:]
        if len(p2) > lim:
            _t2, p2 = _t2[-lim:], p2[-lim:]
    if p1:
        x1 = range(len(p1))
        ax.plot(x1, p1, label=base_id, linewidth=1.2)
    else:
        print(f"Warning: no rows with mid in {path_base}")
    if p2:
        x2 = range(len(p2))
        ax.plot(x2, p2, label=dual_id, linewidth=1.2)
    else:
        print(f"Warning: no rows with mid in {path_dual}")
    ax.set_ylabel("Mid price")
    ax.set_title(f"{base_id} vs {dual_id} (mid, evenly spaced by row)")
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_dual_pairs(save: Optional[Union[str, Path]] = None, n: Optional[int] = None) -> None:
    """
    Plot **mid** prices for NVDA/NVDA_DUAL (top) and NVO/NVO_DUAL (bottom).

    ``n``: last ``n`` mid samples per leg (default ``2000``); ``n <= 0`` plots full series.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required. Install with: pip install matplotlib"
        ) from e

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    _plot_mid_pair_on_ax(ax_top, NVDA_STRATEGY_CSV, NVDA_DUAL_STRATEGY_CSV, "NVDA", "NVDA_DUAL", n=n)
    _plot_mid_pair_on_ax(ax_bot, NVO_STRATEGY_CSV, NVO_DUAL_STRATEGY_CSV, "NVO", "NVO_DUAL", n=n)
    ax_bot.set_xlabel("Event index (row in CSV; evenly spaced)")
    plt.tight_layout()
    if save:
        out = Path(save)
        fig.savefig(out, dpi=150)
        print(f"Saved {out.resolve()}")
    else:
        plt.show()


plot_dual_pairs_mid = plot_dual_pairs


# ---------------------------------------------------------------------------
# Arbitrary pair — **last trade** price, last ``n`` CSV rows (file order)
# ---------------------------------------------------------------------------

def load_trade_series_from_rows(path: Path, max_rows: int) -> Tuple[List[int], List[float]]:
    """
    Read up to the last ``max_rows`` **data lines** of the CSV (file order); return
    (event_index 0..k-1, last_trade_price) for rows with a numeric trade.

    If ``max_rows <= 0``, uses all rows in the file.
    """
    if not path.is_file():
        return [], []

    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return [], []
        rows = list(reader)

    if max_rows <= 0:
        tail = rows
    else:
        tail = rows[-max_rows:]

    ys: List[float] = []
    for raw in tail:
        row = _strip_row(raw)
        trade = _parse_float(row.get("last_trade_price", ""))
        if trade is None:
            trade = _parse_float(row.get("last_trade", ""))
        if trade is None:
            continue
        ys.append(trade)
    xs = list(range(len(ys)))
    return xs, ys


def plot_stock_pair_trade(
    stock_a: str,
    stock_b: str,
    n: int,
    save: Optional[Union[str, Path]] = None,
    data_dir: Optional[Union[str, Path]] = None,
) -> None:
    """
    Plot **last trade** prices for ``stock_a`` and ``stock_b`` on one chart.

    Uses the last ``n`` **rows** of each ``{id}_strategy_market_data.csv`` (after the header).
    If ``n <= 0``, all rows are read. X-axis: **event number** 0..k-1 for trades kept (rows
    without a trade are skipped).

    ``data_dir``: folder containing the CSVs (default: ``data/csv/``). If you pass the
    ``data/`` package directory, ``data/csv/`` is used when present.
    """
    base = _resolve_strategy_data_dir(data_dir) if data_dir is not None else _CSV_DIR
    path_a = base / f"{stock_a}_strategy_market_data.csv"
    path_b = base / f"{stock_b}_strategy_market_data.csv"

    xa, ya = load_trade_series_from_rows(path_a, n)
    xb, yb = load_trade_series_from_rows(path_b, n)

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("matplotlib is required. Install with: pip install matplotlib") from e

    fig, ax = plt.subplots(figsize=(10, 5))
    if ya:
        ax.plot(xa, ya, label=f"{stock_a} (last trade)", linewidth=1.2)
    else:
        print(f"Warning: no trade prices in last {n} rows of {path_a}")
    if yb:
        ax.plot(xb, yb, label=f"{stock_b} (last trade)", linewidth=1.2)
    else:
        print(f"Warning: no trade prices in last {n} rows of {path_b}")

    ax.set_xlabel("Event number (consecutive trades within last n CSV rows)")
    ax.set_ylabel("Trade price (last trade)")
    ax.set_title(f"{stock_a} vs {stock_b}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        out = Path(save)
        fig.savefig(out, dpi=150)
        print(f"Saved {out.resolve()}")
    else:
        plt.show()


# ---------------------------------------------------------------------------
# OB5X index / ETF / futures theory
# ---------------------------------------------------------------------------

INDEX_WEIGHTS: Dict[str, float] = {
    "AMZN": 953.21,
    "JPM": 129.25,
    "NVDA": 908.06,
    "XOM": 2245.39,
    "NVO": 124.78,
}
INDEX_DIVISOR = 1000.0

ETF_CASH_C = 2.50
ETF_MULTIPLIER_M = 0.25
ETF_ID = "OB5X_ETF"

R_ANNUAL = 0.03
FUTURE_ID_PATTERN = re.compile(r"^OB5X_(\d{4})(\d{2})_F$")

SECONDS_PER_YEAR = 365.25 * 24 * 3600

_STOCK_IDS: Tuple[str, ...] = tuple(INDEX_WEIGHTS.keys())


def _tail_plot_series(n: Optional[int], xs: List[float]) -> List[float]:
    lim = _DEFAULT_TAIL_N if n is None else int(n)
    if lim <= 0 or len(xs) <= lim:
        return xs
    return xs[-lim:]


def _percentile_linear(sorted_vals: List[float], p: float) -> float:
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = (n - 1) * (p / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo >= hi:
        return sorted_vals[lo]
    w = pos - lo
    return sorted_vals[lo] + w * (sorted_vals[hi] - sorted_vals[lo])


def _robust_y_limits(
    *series: Sequence[float],
    low_pct: float = 1.0,
    high_pct: float = 99.0,
    pad_ratio: float = 0.04,
) -> Optional[Tuple[float, float]]:
    vals: List[float] = []
    for s in series:
        for v in s:
            if v is not None and math.isfinite(v):
                vals.append(float(v))
    if not vals:
        return None
    if len(vals) < 5:
        mn, mx = min(vals), max(vals)
        span = mx - mn
        pad = span * pad_ratio if span > 0 else max(abs(mx) * 0.02, 1e-6)
        return mn - pad, mx + pad
    vals.sort()
    lo = _percentile_linear(vals, low_pct)
    hi = _percentile_linear(vals, high_pct)
    if hi <= lo:
        pad = max(abs(lo) * 0.02, 1e-6)
        return lo - pad, hi + pad
    span = hi - lo
    pad = span * pad_ratio
    return lo - pad, hi + pad


def load_strategy_rows_trade_ts(path: Path) -> List[Tuple[float, Optional[float]]]:
    rows: List[Tuple[float, Optional[float]]] = []
    if not path.is_file():
        return rows
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return rows
        for raw in reader:
            row = _strip_row(raw)
            ts_raw = row.get("timestamp", "")
            if not ts_raw:
                continue
            try:
                ts = float(ts_raw)
            except ValueError:
                continue
            trade = _parse_float(row.get("last_trade_price", ""))
            if trade is None:
                trade = _parse_float(row.get("last_trade", ""))
            rows.append((ts, trade))
    return rows


def load_strategy_rows_mid_ts(path: Path) -> List[Tuple[float, Optional[float]]]:
    rows: List[Tuple[float, Optional[float]]] = []
    if not path.is_file():
        return rows
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return rows
        for raw in reader:
            row = _strip_row(raw)
            ts_raw = row.get("timestamp", "")
            if not ts_raw:
                continue
            try:
                ts = float(ts_raw)
            except ValueError:
                continue
            mid = _parse_float(row.get("mid", ""))
            rows.append((ts, mid))
    return rows


def expiry_third_friday_noon_utc(year: int, month: int) -> datetime:
    d0 = datetime(year, month, 1, 12, 0, 0, tzinfo=timezone.utc)
    offset_days = (4 - d0.weekday()) % 7
    first_friday = d0 + timedelta(days=offset_days)
    third_friday = first_friday + timedelta(days=14)
    return third_friday


def parse_future_expiry(instrument_id: str) -> Optional[datetime]:
    m = FUTURE_ID_PATTERN.match(instrument_id.strip())
    if not m:
        return None
    y, mo = int(m.group(1)), int(m.group(2))
    try:
        return expiry_third_friday_noon_utc(y, mo)
    except ValueError:
        return None


def _tau_years(t_event: datetime, expiry: datetime) -> float:
    return max(0.0, (expiry - t_event).total_seconds() / SECONDS_PER_YEAR)


def discover_ob5x_future_ids(data_dir: Path) -> List[str]:
    data_dir = _resolve_strategy_data_dir(data_dir)
    found: List[str] = []
    for p in sorted(data_dir.glob("OB5X_*_F_strategy_market_data.csv")):
        stem = p.name.replace("_strategy_market_data.csv", "")
        if FUTURE_ID_PATTERN.match(stem):
            found.append(stem)
    return found


def build_aligned_theory_series(
    data_dir: Path,
    future_ids: Optional[Sequence[str]] = None,
) -> Tuple[
    List[int],
    List[float],
    List[float],
    Dict[str, Tuple[List[float], List[float]]],
]:
    data_dir = _resolve_strategy_data_dir(data_dir)
    paths = {sid: data_dir / f"{sid}_strategy_market_data.csv" for sid in _STOCK_IDS}
    etf_path = data_dir / f"{ETF_ID}_strategy_market_data.csv"

    stock_rows = {sid: load_strategy_rows_trade_ts(paths[sid]) for sid in _STOCK_IDS}
    etf_rows = load_strategy_rows_trade_ts(etf_path)

    fut_ids = list(future_ids) if future_ids else discover_ob5x_future_ids(data_dir)
    fut_rows = {
        fid: load_strategy_rows_trade_ts(data_dir / f"{fid}_strategy_market_data.csv")
        for fid in fut_ids
    }

    lengths = [len(stock_rows[sid]) for sid in _STOCK_IDS] + [len(etf_rows)]
    lengths += [len(fut_rows[fid]) for fid in fut_ids]
    if not lengths or min(lengths) == 0:
        return [], [], [], {}

    k_max = min(lengths)
    etf_trade_out: List[float] = []
    etf_theory_out: List[float] = []
    fut_data: Dict[str, Tuple[List[float], List[float]]] = {fid: ([], []) for fid in fut_ids}

    for k in range(k_max):
        price_stock: Dict[str, float] = {}
        ok = True
        for sid in _STOCK_IDS:
            _ts, trade = stock_rows[sid][k]
            if trade is None:
                ok = False
                break
            price_stock[sid] = trade
        if not ok:
            continue

        x_idx = sum(INDEX_WEIGHTS[sid] * price_stock[sid] for sid in _STOCK_IDS) / INDEX_DIVISOR
        nav_theory = ETF_CASH_C + ETF_MULTIPLIER_M * x_idx

        ts_k, etf_trade = etf_rows[k]
        if etf_trade is None:
            continue

        t_event = datetime.fromtimestamp(ts_k, tz=timezone.utc)

        etf_trade_out.append(etf_trade)
        etf_theory_out.append(nav_theory)

        for fid in fut_ids:
            exp = parse_future_expiry(fid)
            if exp is None:
                continue
            tau = _tau_years(t_event, exp)
            f_theory = x_idx * math.exp(R_ANNUAL * tau)
            fts, ftrade = fut_rows[fid][k]
            if ftrade is not None:
                fut_data[fid][0].append(ftrade)
                fut_data[fid][1].append(f_theory)

    fut_out: Dict[str, Tuple[List[float], List[float]]] = {}
    for fid in fut_ids:
        trades_f, theo_f = fut_data[fid]
        if trades_f:
            fut_out[fid] = (trades_f, theo_f)

    return (
        list(range(len(etf_trade_out))),
        etf_trade_out,
        etf_theory_out,
        fut_out,
    )


def plot_ob5x_index_theory(
    save: Optional[Union[str, Path]] = None,
    data_dir: Optional[Union[str, Path]] = None,
    future_ids: Optional[Sequence[str]] = None,
    y_percentile_low: float = 1.0,
    y_percentile_high: float = 99.0,
    y_pad_ratio: float = 0.04,
    n: Optional[int] = None,
) -> None:
    base = _resolve_strategy_data_dir(data_dir) if data_dir is not None else _CSV_DIR

    x_etf, etf_trade, etf_theory, futures_map = build_aligned_theory_series(base, future_ids)

    etf_trade = _tail_plot_series(n, etf_trade)
    etf_theory = _tail_plot_series(n, etf_theory)
    x_etf = list(range(len(etf_trade)))
    futures_map = {
        fid: (_tail_plot_series(n, ft), _tail_plot_series(n, th))
        for fid, (ft, th) in futures_map.items()
    }

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("matplotlib is required. Install with: pip install matplotlib") from e

    n_fut = len(futures_map)
    n_rows = 1 + n_fut
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, max(3.5, 3.5 * n_rows)), sharex=False)

    if n_rows == 1:
        axes = [axes]

    ax0 = axes[0]
    if x_etf:
        ax0.plot(x_etf, etf_trade, label=f"{ETF_ID} last trade (CSV)", linewidth=1.2)
        ax0.plot(x_etf, etf_theory, label="Theoretical NAV (C + M·X)", linewidth=1.2, linestyle="--")
        ylim = _robust_y_limits(
            etf_trade,
            etf_theory,
            low_pct=y_percentile_low,
            high_pct=y_percentile_high,
            pad_ratio=y_pad_ratio,
        )
        if ylim is not None:
            ax0.set_ylim(ylim)
    else:
        ax0.text(
            0.5,
            0.5,
            "No aligned rows with index last trades + ETF last trade",
            ha="center",
            va="center",
            transform=ax0.transAxes,
        )
    ax0.set_ylabel("Last trade price")
    ax0.set_title("OB5X ETF: last trade vs theoretical NAV")
    ax0.set_xlabel("Event index (aligned CSV rows)")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    for i, (fid, (ftrades, ftheo)) in enumerate(sorted(futures_map.items()), start=1):
        ax = axes[i]
        x_f = list(range(len(ftrades)))
        ax.plot(x_f, ftrades, label=f"{fid} last trade (CSV)", linewidth=1.2)
        ax.plot(x_f, ftheo, label="Theoretical F = X·exp(rτ)", linewidth=1.2, linestyle="--")
        ylim = _robust_y_limits(
            ftrades,
            ftheo,
            low_pct=y_percentile_low,
            high_pct=y_percentile_high,
            pad_ratio=y_pad_ratio,
        )
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_ylabel("Last trade price")
        ax.set_title(f"{fid}: last trade vs theoretical future")
        ax.set_xlabel("Event index (rows with index + future last trades)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        out = Path(save)
        fig.savefig(out, dpi=150)
        print(f"Saved {out.resolve()}")
    else:
        plt.show()


__all__ = [
    "INDEX_WEIGHTS",
    "INDEX_DIVISOR",
    "ETF_CASH_C",
    "ETF_MULTIPLIER_M",
    "ETF_ID",
    "R_ANNUAL",
    "NVDA_DUAL_STRATEGY_CSV",
    "NVDA_STRATEGY_CSV",
    "NVO_DUAL_STRATEGY_CSV",
    "NVO_STRATEGY_CSV",
    "DEFAULT_CSV_DIR",
    "discover_ob5x_future_ids",
    "expiry_third_friday_noon_utc",
    "load_mid_series",
    "load_strategy_rows_mid_ts",
    "load_strategy_rows_trade_ts",
    "load_trade_series_from_rows",
    "parse_future_expiry",
    "plot_dual_pairs",
    "plot_dual_pairs_mid",
    "plot_ob5x_index_theory",
    "plot_stock_pair_trade",
]
