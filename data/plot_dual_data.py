"""
Plot both dual-listed pairs (NVDA/NVDA_DUAL and NVO/NVO_DUAL) from **mid** prices
in fixed strategy CSVs under ``data/``.

The x-axis is **event index** (0, 1, 2, …): each CSV row is one evenly spaced step,
not wall-clock time.

Notebooks (cwd may be ``optibook_guides/``; add repo root to ``sys.path`` first)::

    from data.plot_dual_data import plot_dual_pairs
    plot_dual_pairs()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from data.plot_dual_prices import load_mid_series

_DATA_DIR = Path(__file__).resolve().parent
NVDA_STRATEGY_CSV = _DATA_DIR / "NVDA_strategy_market_data.csv"
NVDA_DUAL_STRATEGY_CSV = _DATA_DIR / "NVDA_DUAL_strategy_market_data.csv"
NVO_STRATEGY_CSV = _DATA_DIR / "NVO_strategy_market_data.csv"
NVO_DUAL_STRATEGY_CSV = _DATA_DIR / "NVO_DUAL_strategy_market_data.csv"


def _plot_mid_pair_on_ax(
    ax: Any,
    path_base: Path,
    path_dual: Path,
    base_id: str,
    dual_id: str,
) -> None:
    _t1, p1 = load_mid_series(path_base)
    _t2, p2 = load_mid_series(path_dual)
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


def plot_dual_pairs(save: Optional[Union[str, Path]] = None) -> None:
    """
    Plot **mid** prices from CSV for NVDA/NVDA_DUAL (top) and NVO/NVO_DUAL (bottom).

    X-axis is the row / event index in each CSV so consecutive samples are evenly
    spaced (not scaled by timestamp).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit(
            "matplotlib is required. Install with: pip install matplotlib"
        ) from e

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 8), sharex=False)
    _plot_mid_pair_on_ax(ax_top, NVDA_STRATEGY_CSV, NVDA_DUAL_STRATEGY_CSV, "NVDA", "NVDA_DUAL")
    _plot_mid_pair_on_ax(ax_bot, NVO_STRATEGY_CSV, NVO_DUAL_STRATEGY_CSV, "NVO", "NVO_DUAL")
    ax_bot.set_xlabel("Event index (row in CSV; evenly spaced)")
    plt.tight_layout()
    if save:
        out = Path(save)
        fig.savefig(out, dpi=150)
        print(f"Saved {out.resolve()}")
    else:
        plt.show()


# Older notebooks and clones may still import this name.
plot_dual_pairs_mid = plot_dual_pairs

__all__ = [
    "NVDA_DUAL_STRATEGY_CSV",
    "NVDA_STRATEGY_CSV",
    "NVO_DUAL_STRATEGY_CSV",
    "NVO_STRATEGY_CSV",
    "plot_dual_pairs",
    "plot_dual_pairs_mid",
]
