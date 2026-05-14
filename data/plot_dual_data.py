"""
Plot NVDA vs NVDA_DUAL from fixed strategy CSV paths under ``data/``.

Notebooks (cwd may be ``optibook_guides/``; add repo root to ``sys.path`` first)::

    from data.plot_dual_data import plot_nvda_dual_data
    plot_nvda_dual_data()
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from data.plot_dual_prices import load_price_series

_DATA_DIR = Path(__file__).resolve().parent
NVDA_STRATEGY_CSV = _DATA_DIR / "NVDA_strategy_market_data.csv"
NVDA_DUAL_STRATEGY_CSV = _DATA_DIR / "NVDA_DUAL_strategy_market_data.csv"


def plot_nvda_dual_data(save: Optional[Union[str, Path]] = None) -> None:
    """
    Plot mid/microprice series from the NVDA and NVDA_DUAL strategy market CSVs.
    """
    path_left = NVDA_STRATEGY_CSV
    path_right = NVDA_DUAL_STRATEGY_CSV

    t1, p1 = load_price_series(path_left)
    t2, p2 = load_price_series(path_right)

    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required. Install with: pip install matplotlib"
        ) from e

    left_id, right_id = "NVDA", "NVDA_DUAL"
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


__all__ = [
    "NVDA_DUAL_STRATEGY_CSV",
    "NVDA_STRATEGY_CSV",
    "plot_nvda_dual_data",
]
