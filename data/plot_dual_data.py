"""
Plot NVDA vs NVDA_DUAL from the fixed strategy CSVs in ``data/``.

    from data.plot_dual_data import plot_nvda_dual_data
    plot_nvda_dual_data()
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

_DATA_DIR = Path(__file__).resolve().parent
NVDA_CSV = _DATA_DIR / "NVDA_strategy_market_data.csv"
NVDA_DUAL_CSV = _DATA_DIR / "NVDA_DUAL_strategy_market_data.csv"


def plot_nvda_dual_data(save: Optional[Union[str, Path]] = None) -> None:
    """Plot NVDA and NVDA_DUAL mid/price series from the bundled strategy market CSVs."""
    try:
        from data.plot_dual_prices import load_price_series
    except ModuleNotFoundError:
        from plot_dual_prices import load_price_series

    t1, p1 = load_price_series(NVDA_CSV)
    t2, p2 = load_price_series(NVDA_DUAL_CSV)

    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required. Install with: pip install matplotlib"
        ) from e

    left_id, right_id = "NVDA", "NVDA_DUAL"
    fig, ax = plt.subplots(figsize=(10, 5))
    if t1:
        ax.plot(t1, p1, label=left_id, linewidth=1.2)
    else:
        print(f"Warning: no plottable rows in {NVDA_CSV}")
    if t2:
        ax.plot(t2, p2, label=right_id, linewidth=1.2)
    else:
        print(f"Warning: no plottable rows in {NVDA_DUAL_CSV}")

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
