"""
Alias entry point for ``plot_dual_prices``.

Notebooks (current working directory = repo root, ``opti-ff`` on ``sys.path``)::

    from data.plot_nvda_dual_prices import plot_nvda_pair
    plot_nvda_pair()
"""

from __future__ import annotations

try:
    from data.plot_dual_prices import load_price_series, main, plot_nvda_pair
except ModuleNotFoundError:
    from plot_dual_prices import load_price_series, main, plot_nvda_pair

__all__ = ["load_price_series", "main", "plot_nvda_pair"]

if __name__ == "__main__":
    main()
