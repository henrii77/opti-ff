"""
Dual-listing mean reversion: rolling z-score on (MAIN_mid − DUAL_mid).

Live: only NVDA / NVDA_DUAL and NVO / NVO_DUAL. Symmetric entry when
``z >= z_threshold_*`` (short spread) or ``z <= -z_threshold_*`` (long spread),
with optional flat-z partial exits.

Offline: :meth:`Trader.replay_dual_listing` for aligned mid arrays (rolling z, no sklearn).

Tick history: same CSV-shaped :class:`pandas.DataFrame` rows as ``collect_strategy_data``.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np
import pandas as pd

from data.collect_strategy_data import HEADER, snapshot_row_with_book

try:
    from optibook.synchronous_client import Exchange
except ImportError:  # pragma: no cover
    Exchange = Any  # type: ignore[misc, assignment]


class Trader:
    """Dual-listing IOC trader using rolling spread z-scores."""

    DUAL_PAIRS = [
        ("NVDA", "NVDA_DUAL"),
        ("NVO", "NVO_DUAL"),
    ]

    TICK_SIZE = 0.10
    MAX_POSITION = 99
    SAFE_POSITION = 80
    LOOP_SLEEP = 0.04
    RATE_LIMIT = 21
    BASE_VOLUME = 10

    Z_SCORE_WINDOW = 50
    Z_STD_EPS = 1e-9
    Z_EXIT = 0.3

    TICK_HISTORY_MAX_ROWS = 10_000

    def __init__(
        self,
        z_threshold_nvda: float = 2.0,
        z_threshold_nvo: float = 2.0,
        *,
        z_window: int = Z_SCORE_WINDOW,
    ) -> None:
        self._z_threshold_nvda = float(z_threshold_nvda)
        self._z_threshold_nvo = float(z_threshold_nvo)
        self._z_window = int(z_window)

        self._action_ts: Deque[float] = deque()
        self._start_time = 0.0
        self._last_status = 0.0
        self._all_assets: List[str] = []
        self._instrument_meta: Dict[str, Any] = {}
        self._tick_frames: Dict[str, pd.DataFrame] = {}
        self._spread_hist: Dict[str, Deque[float]] = {}
        self._loop_count = 0

    def _pair_key(self, main: str, dual: str) -> str:
        return f"{main}_{dual}_spread"

    def _z_threshold_for_main(self, main: str) -> float:
        if main == "NVDA":
            return self._z_threshold_nvda
        if main == "NVO":
            return self._z_threshold_nvo
        return max(self._z_threshold_nvda, self._z_threshold_nvo)

    @staticmethod
    def round_tick(price: float) -> float:
        return round(round(price / Trader.TICK_SIZE) * Trader.TICK_SIZE, 10)

    def can_trade(self, n: int = 1) -> bool:
        now = time.time()
        while self._action_ts and now - self._action_ts[0] > 1.0:
            self._action_ts.popleft()
        return (len(self._action_ts) + n) <= self.RATE_LIMIT

    def log_actions(self, n: int = 1) -> None:
        now = time.time()
        for _ in range(n):
            self._action_ts.append(now)

    @staticmethod
    def safe_vol(
        pos: int,
        requested: int,
        side: str,
        hard: int = MAX_POSITION,
        soft: int = SAFE_POSITION,
    ) -> int:
        cap = min(hard, soft)
        if side == "bid":
            return max(0, min(requested, cap - pos))
        return max(0, min(requested, cap + pos))

    @staticmethod
    def mid(book: Any) -> Optional[float]:
        if book and book.bids and book.asks:
            return (book.bids[0].price + book.asks[0].price) / 2.0
        return None

    @staticmethod
    def _resolve_instrument(instruments: Dict[Any, Any], aid: str) -> Any:
        for k, v in instruments.items():
            if str(k) == str(aid):
                return v
        return None

    def _bootstrap(self, exchange: Any) -> None:
        print("=" * 60)
        print("  Dual-listing z-score trader  —  Starting up")
        print("=" * 60)
        self._start_time = time.time()
        self._last_status = 0.0
        self._loop_count = 0

        self._all_assets = sorted({s for pair in self.DUAL_PAIRS for s in pair})
        self._spread_hist = {
            self._pair_key(m, d): deque(maxlen=self._z_window) for m, d in self.DUAL_PAIRS
        }

        instruments = exchange.get_instruments()
        self._instrument_meta = {
            aid: self._resolve_instrument(instruments, aid) for aid in self._all_assets
        }
        cols = list(HEADER)
        self._tick_frames = {aid: pd.DataFrame(columns=cols) for aid in self._all_assets}

        print(
            f"  Instruments: {self._all_assets}  "
            f"z_nvda=±{self._z_threshold_nvda}  z_nvo=±{self._z_threshold_nvo}  "
            f"window={self._z_window}"
        )

    def run(self, exchange: Any) -> None:
        """Main loop (blocks forever)."""
        self._bootstrap(exchange)
        while True:
            try:
                self._iteration(exchange)
            except Exception as e:  # pragma: no cover
                print(f"  [LOOP ERR] {e}")
            time.sleep(self.LOOP_SLEEP)

    def _iteration(self, exchange: Any) -> None:
        now = time.time()
        elapsed = now - self._start_time
        self._loop_count += 1

        books: Dict[str, Any] = {}
        mids_snap: Dict[str, float] = {}
        for asset in self._all_assets:
            try:
                bk = exchange.get_last_price_book(asset)
                books[asset] = bk
                m = self.mid(bk)
                if m is not None:
                    mids_snap[asset] = m
            except Exception:
                pass

        positions = exchange.get_positions()
        virt_pos: Dict[str, int] = dict(positions)

        if now - self._last_status > 15:
            try:
                pnl = exchange.get_pnl()
                rps = len(self._action_ts)
                print(f"\n[{elapsed:6.0f}s] PnL={pnl:,.2f}  RPS={rps}  Pos={dict(positions)}")
            except Exception:
                pass
            self._last_status = now

        self._strategy_dual_zscore(books, mids_snap, virt_pos, exchange)
        self.poll_tick_history(exchange, now, books)

    def poll_tick_history(self, exchange: Any, ts: float, books: Dict[str, Any]) -> None:
        """
        Append one CSV-shaped row per instrument (same columns as ``collect_strategy_data``).

        Uses the books already fetched this tick (no duplicate ``get_last_price_book``).
        """
        if not exchange.is_connected():
            return
        cols = list(HEADER)
        for inst_id in self._all_assets:
            book = books.get(inst_id)
            if not book:
                continue
            inst = self._instrument_meta.get(inst_id)
            try:
                row = snapshot_row_with_book(ts, inst_id, inst, exchange, book)
            except AssertionError:
                if not exchange.is_connected():
                    return
                raise
            df = self._tick_frames.setdefault(inst_id, pd.DataFrame(columns=cols))
            df.loc[len(df)] = row
            if len(df) > self.TICK_HISTORY_MAX_ROWS:
                self._tick_frames[inst_id] = df.iloc[1:].reset_index(drop=True)

    @property
    def tick_frames(self) -> Dict[str, pd.DataFrame]:
        return self._tick_frames

    def _rolling_spread_z(self, spread_key: str, spread: float) -> Optional[float]:
        dq = self._spread_hist[spread_key]
        dq.append(spread)
        if len(dq) < self._z_window:
            return None
        arr = np.array(dq, dtype=float)
        mu = float(np.mean(arr))
        sig = float(np.std(arr, ddof=1))
        if sig < self.Z_STD_EPS:
            return None
        return (spread - mu) / sig

    def _strategy_dual_zscore(
        self,
        books: Dict[str, Any],
        mids_snap: Dict[str, float],
        virt_pos: Dict[str, int],
        exchange: Any,
    ) -> None:
        vol_size = max(1, int(self.BASE_VOLUME))

        for main, dual in self.DUAL_PAIRS:
            bk_m = books.get(main)
            bk_d = books.get(dual)
            if not (bk_m and bk_d):
                continue
            if not (bk_m.bids and bk_m.asks and bk_d.bids and bk_d.asks):
                continue

            main_mid = mids_snap.get(main)
            dual_mid = mids_snap.get(dual)
            if main_mid is None or dual_mid is None:
                continue

            spread = main_mid - dual_mid
            sk = self._pair_key(main, dual)
            z = self._rolling_spread_z(sk, spread)
            z_thr = self._z_threshold_for_main(main)

            pos_m = virt_pos.get(main, 0)
            pos_d = virt_pos.get(dual, 0)

            if z is None:
                continue

            if z >= z_thr:
                v_m = self.safe_vol(pos_m, vol_size, "ask")
                v_d = self.safe_vol(pos_d, vol_size, "bid")
                v = min(v_m, v_d)
                if v > 0 and self.can_trade(2):
                    print(
                        f"  [DUAL Z SHORT SPREAD] {main}/{dual}  z={z:.3f}  "
                        f"spread={spread:.4f}  thr=±{z_thr}"
                    )
                    exchange.insert_order(
                        main, price=bk_m.bids[0].price, volume=v, side="ask", order_type="ioc"
                    )
                    exchange.insert_order(
                        dual, price=bk_d.asks[0].price, volume=v, side="bid", order_type="ioc"
                    )
                    self.log_actions(2)
                    virt_pos[main] = pos_m - v
                    virt_pos[dual] = pos_d + v

            elif z <= -z_thr:
                v_m = self.safe_vol(pos_m, vol_size, "bid")
                v_d = self.safe_vol(pos_d, vol_size, "ask")
                v = min(v_m, v_d)
                if v > 0 and self.can_trade(2):
                    print(
                        f"  [DUAL Z LONG SPREAD ] {main}/{dual}  z={z:.3f}  "
                        f"spread={spread:.4f}  thr=±{z_thr}"
                    )
                    exchange.insert_order(
                        main, price=bk_m.asks[0].price, volume=v, side="bid", order_type="ioc"
                    )
                    exchange.insert_order(
                        dual, price=bk_d.bids[0].price, volume=v, side="ask", order_type="ioc"
                    )
                    self.log_actions(2)
                    virt_pos[main] = pos_m + v
                    virt_pos[dual] = pos_d - v

            elif abs(z) < self.Z_EXIT and (
                virt_pos.get(main, 0) != 0 or virt_pos.get(dual, 0) != 0
            ):
                pos_m = virt_pos.get(main, 0)
                pos_d = virt_pos.get(dual, 0)
                if pos_m < 0 and bk_m.asks and self.can_trade(1):
                    v = self.safe_vol(pos_m, min(10, abs(pos_m)), "bid")
                    if v > 0:
                        exchange.insert_order(
                            main, price=bk_m.asks[0].price, volume=v, side="bid", order_type="ioc"
                        )
                        self.log_actions(1)
                        virt_pos[main] = pos_m + v
                        print(f"  [DUAL Z EXIT MAIN] {main} z={z:.3f}")
                elif pos_m > 0 and bk_m.bids and self.can_trade(1):
                    v = self.safe_vol(pos_m, min(10, pos_m), "ask")
                    if v > 0:
                        exchange.insert_order(
                            main, price=bk_m.bids[0].price, volume=v, side="ask", order_type="ioc"
                        )
                        self.log_actions(1)
                        virt_pos[main] = pos_m - v
                        print(f"  [DUAL Z EXIT MAIN] {main} z={z:.3f}")
                pos_m = virt_pos.get(main, 0)
                pos_d = virt_pos.get(dual, 0)
                if pos_d > 0 and bk_d.bids and self.can_trade(1):
                    v = self.safe_vol(pos_d, min(10, pos_d), "ask")
                    if v > 0:
                        exchange.insert_order(
                            dual, price=bk_d.bids[0].price, volume=v, side="ask", order_type="ioc"
                        )
                        self.log_actions(1)
                        virt_pos[dual] = pos_d - v
                        print(f"  [DUAL Z EXIT DUAL] {dual} z={z:.3f}")
                elif pos_d < 0 and bk_d.asks and self.can_trade(1):
                    v = self.safe_vol(pos_d, min(10, abs(pos_d)), "bid")
                    if v > 0:
                        exchange.insert_order(
                            dual, price=bk_d.asks[0].price, volume=v, side="bid", order_type="ioc"
                        )
                        self.log_actions(1)
                        virt_pos[dual] = pos_d + v
                        print(f"  [DUAL Z EXIT DUAL] {dual} z={z:.3f}")

    @staticmethod
    def replay_dual_listing(
        main_mids: np.ndarray,
        dual_mids: np.ndarray,
        *,
        z_threshold: float = 2.0,
        z_window: int = Z_SCORE_WINDOW,
        z_std_eps: float = Z_STD_EPS,
    ) -> List[Dict[str, Any]]:
        """
        Replay rolling z on spread (aligned mid arrays; no positions / exits).

        Returns events with keys ``i``, ``kind`` in ``{"short_spread", "long_spread"}``,
        plus ``spread`` and ``z``.
        """
        m = min(len(main_mids), len(dual_mids))
        main_mids = np.asarray(main_mids[:m], dtype=float)
        dual_mids = np.asarray(dual_mids[:m], dtype=float)
        spread_series = main_mids - dual_mids

        hist: Deque[float] = deque(maxlen=z_window)
        events: List[Dict[str, Any]] = []

        for i in range(m):
            s_now = float(spread_series[i])
            hist.append(s_now)
            if len(hist) < z_window:
                continue
            arr = np.array(hist, dtype=float)
            mu = float(np.mean(arr))
            sig = float(np.std(arr, ddof=1))
            if sig < z_std_eps:
                continue
            z = (s_now - mu) / sig
            if z >= z_threshold:
                events.append({"i": i, "kind": "short_spread", "spread": s_now, "z": z})
            elif z <= -z_threshold:
                events.append({"i": i, "kind": "long_spread", "spread": s_now, "z": z})

        return events


def main() -> None:
    from optibook.synchronous_client import Exchange as _Exchange

    exchange = _Exchange()
    exchange.connect()
    Trader().run(exchange)


if __name__ == "__main__":
    main()
