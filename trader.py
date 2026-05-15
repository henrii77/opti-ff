"""
Dual-listing mean reversion: rolling z-score on (MAIN_mid − DUAL_mid).

Live: only NVDA / NVDA_DUAL and NVO / NVO_DUAL. Symmetric entry when
``z >= z_threshold_*`` (short spread) or ``z <= -z_threshold_*`` (long spread),
with optional flat-z partial exits.

Offline: :meth:`Trader.replay_dual_listing` for aligned mid arrays (rolling z, no sklearn).

Tick history: same CSV-shaped :class:`pandas.DataFrame` rows as ``collect_strategy_data``.
On :meth:`start`, optional **CSV warm-start** loads recent rows from
``{instrument_id}_strategy_market_data.csv`` under ``csv_dir`` (same layout as the collector)
into ``tick_frames`` and seeds rolling spread deques from aligned mids.

Live loop: :meth:`Trader.run` (blocking). To interleave with another coroutine (e.g.
``collector_poll_once`` in a notebook), call :meth:`Trader.start` once then :meth:`Trader.step` each cycle.
"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from data.collect_strategy_data import (
    HEADER,
    OUTPUT_FILENAME_TEMPLATE,
    instrument_csv_path,
    snapshot_row_with_book,
)

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

    @staticmethod
    def default_csv_dir() -> Path:
        """Default directory for ``*_strategy_market_data.csv`` (repo ``data/csv``)."""
        return Path(__file__).resolve().parent / "data" / "csv"

    def __init__(
        self,
        z_threshold_nvda: float = 2.0,
        z_threshold_nvo: float = 2.0,
        *,
        z_window: int = Z_SCORE_WINDOW,
        csv_dir: Optional[Union[str, Path]] = None,
        csv_warm_start: bool = True,
    ) -> None:
        self._z_threshold_nvda = float(z_threshold_nvda)
        self._z_threshold_nvo = float(z_threshold_nvo)
        self._z_window = int(z_window)
        self._csv_dir = Path(csv_dir) if csv_dir is not None else self.default_csv_dir()
        self._csv_warm_start = bool(csv_warm_start)

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
    def _order_ack_ok(resp: Any) -> bool:
        """
        Best-effort: True if ``insert_order`` looks successful (fill or acceptance).

        Optibook returns a small response object; attribute names vary by version.
        Prefer explicit fill volume when present; otherwise ``success`` or ``order_id``.
        """
        if resp is None:
            return False
        if getattr(resp, "success", None) is False:
            return False
        for attr in (
            "filled_volume",
            "traded_volume",
            "executed_volume",
            "volume_executed",
            "filled",
        ):
            if hasattr(resp, attr):
                raw = getattr(resp, attr)
                if raw is None:
                    continue
                try:
                    fv = float(raw)
                except (TypeError, ValueError):
                    continue
                if fv > 0:
                    return True
                return False
        if getattr(resp, "success", None) is True:
            return True
        oid = getattr(resp, "order_id", None)
        if oid is not None and oid != "" and oid != 0:
            return True
        return False

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

        if self._csv_warm_start:
            self._load_csv_warm_start()

    def _normalize_csv_to_header(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure columns match ``HEADER`` (missing columns filled like collector blanks)."""
        for c in HEADER:
            if c not in df.columns:
                df[c] = np.nan
        out = df[list(HEADER)].copy()
        # Match string empties for optional text fields the collector writes as ""
        for c in ("instrument_id", "instrument_type", "instrument_group", "index_id"):
            if c in out.columns:
                out[c] = out[c].fillna("").astype(str).replace("nan", "")
        return out

    def _read_instrument_csv(self, instrument_id: str) -> pd.DataFrame:
        path = instrument_csv_path(
            instrument_id, self._csv_dir, OUTPUT_FILENAME_TEMPLATE
        )
        if not path.is_file():
            return pd.DataFrame(columns=list(HEADER))
        try:
            raw = pd.read_csv(path, encoding="utf-8-sig")
        except Exception as e:  # pragma: no cover
            print(f"  [CSV WARN] {instrument_id}: could not read {path}: {e}")
            return pd.DataFrame(columns=list(HEADER))
        if raw.empty:
            return pd.DataFrame(columns=list(HEADER))
        try:
            norm = self._normalize_csv_to_header(raw)
        except Exception as e:  # pragma: no cover
            print(f"  [CSV WARN] {instrument_id}: column normalize failed: {e}")
            return pd.DataFrame(columns=list(HEADER))
        norm["timestamp"] = pd.to_numeric(norm["timestamp"], errors="coerce")
        for c in ("bid_price", "bid_volume", "ask_price", "ask_volume", "mid", "spread", "last_trade_price"):
            if c in norm.columns:
                norm[c] = pd.to_numeric(norm[c], errors="coerce")
        norm = norm.dropna(subset=["timestamp", "mid"], how="any")
        if norm.empty:
            return pd.DataFrame(columns=list(HEADER))
        norm = norm.sort_values("timestamp").tail(self.TICK_HISTORY_MAX_ROWS).reset_index(drop=True)
        return norm

    def _load_csv_warm_start(self) -> None:
        """Load per-instrument CSVs into ``_tick_frames`` and seed ``_spread_hist`` from aligned mids."""
        for aid in self._all_assets:
            df = self._read_instrument_csv(aid)
            if not df.empty:
                self._tick_frames[aid] = df
                print(f"  [CSV] Loaded {len(df)} row(s) for {aid} from {self._csv_dir!s}")
            else:
                p = instrument_csv_path(aid, self._csv_dir, OUTPUT_FILENAME_TEMPLATE)
                print(f"  [CSV] No data for {aid} (missing or empty: {p})")

        for main, dual in self.DUAL_PAIRS:
            sk = self._pair_key(main, dual)
            d_m = self._tick_frames.get(main, pd.DataFrame())
            d_d = self._tick_frames.get(dual, pd.DataFrame())
            if d_m.empty or d_d.empty:
                continue
            m = d_m[["timestamp", "mid"]].rename(columns={"mid": "mid_main"})
            d = d_d[["timestamp", "mid"]].rename(columns={"mid": "mid_dual"})
            merged = m.merge(d, on="timestamp", how="inner").sort_values("timestamp")
            if merged.empty:
                print(f"  [CSV] No overlapping timestamps for {main}/{dual}; spread deque empty")
                continue
            spreads = (merged["mid_main"] - merged["mid_dual"]).astype(float).tolist()
            tail = spreads[-self._z_window :]
            self._spread_hist[sk].clear()
            self._spread_hist[sk].extend(tail)
            print(f"  [CSV] Seeded {sk} with {len(self._spread_hist[sk])} spread sample(s)")

    def start(self, exchange: Any) -> None:
        """Initialize books/meta/spread history once after :meth:`Exchange.connect`."""
        self._bootstrap(exchange)

    def step(self, exchange: Any) -> None:
        """Run a single trading tick (fetch books, strategy, tick frames). Safe to call from a shared driver loop."""
        if not self._all_assets:
            self._bootstrap(exchange)
        try:
            self._iteration(exchange)
        except Exception as e:  # pragma: no cover
            print(f"  [LOOP ERR] {e}")

    def run(self, exchange: Any) -> None:
        """Blocking 25 Hz loop (``start`` + repeated ``step`` + sleep)."""
        self.start(exchange)
        while True:
            self.step(exchange)
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
                    r_m = exchange.insert_order(
                        main, price=bk_m.bids[0].price, volume=v, side="ask", order_type="ioc"
                    )
                    r_d = exchange.insert_order(
                        dual, price=bk_d.asks[0].price, volume=v, side="bid", order_type="ioc"
                    )
                    self.log_actions(2)
                    virt_pos[main] = pos_m - v
                    virt_pos[dual] = pos_d + v
                    if self._order_ack_ok(r_m) and self._order_ack_ok(r_d):
                        print(
                            f"  [DUAL Z SHORT SPREAD] {main}/{dual}  z={z:.3f}  "
                            f"spread={spread:.4f}  thr=±{z_thr}  v={v}"
                        )

            elif z <= -z_thr:
                v_m = self.safe_vol(pos_m, vol_size, "bid")
                v_d = self.safe_vol(pos_d, vol_size, "ask")
                v = min(v_m, v_d)
                if v > 0 and self.can_trade(2):
                    r_m = exchange.insert_order(
                        main, price=bk_m.asks[0].price, volume=v, side="bid", order_type="ioc"
                    )
                    r_d = exchange.insert_order(
                        dual, price=bk_d.bids[0].price, volume=v, side="ask", order_type="ioc"
                    )
                    self.log_actions(2)
                    virt_pos[main] = pos_m + v
                    virt_pos[dual] = pos_d - v
                    if self._order_ack_ok(r_m) and self._order_ack_ok(r_d):
                        print(
                            f"  [DUAL Z LONG SPREAD ] {main}/{dual}  z={z:.3f}  "
                            f"spread={spread:.4f}  thr=±{z_thr}  v={v}"
                        )

            elif abs(z) < self.Z_EXIT and (
                virt_pos.get(main, 0) != 0 or virt_pos.get(dual, 0) != 0
            ):
                pos_m = virt_pos.get(main, 0)
                pos_d = virt_pos.get(dual, 0)
                if pos_m < 0 and bk_m.asks and self.can_trade(1):
                    v = self.safe_vol(pos_m, min(10, abs(pos_m)), "bid")
                    if v > 0:
                        r = exchange.insert_order(
                            main, price=bk_m.asks[0].price, volume=v, side="bid", order_type="ioc"
                        )
                        self.log_actions(1)
                        virt_pos[main] = pos_m + v
                        if self._order_ack_ok(r):
                            print(f"  [DUAL Z EXIT MAIN] {main} z={z:.3f}  v={v}")
                elif pos_m > 0 and bk_m.bids and self.can_trade(1):
                    v = self.safe_vol(pos_m, min(10, pos_m), "ask")
                    if v > 0:
                        r = exchange.insert_order(
                            main, price=bk_m.bids[0].price, volume=v, side="ask", order_type="ioc"
                        )
                        self.log_actions(1)
                        virt_pos[main] = pos_m - v
                        if self._order_ack_ok(r):
                            print(f"  [DUAL Z EXIT MAIN] {main} z={z:.3f}  v={v}")
                pos_m = virt_pos.get(main, 0)
                pos_d = virt_pos.get(dual, 0)
                if pos_d > 0 and bk_d.bids and self.can_trade(1):
                    v = self.safe_vol(pos_d, min(10, pos_d), "ask")
                    if v > 0:
                        r = exchange.insert_order(
                            dual, price=bk_d.bids[0].price, volume=v, side="ask", order_type="ioc"
                        )
                        self.log_actions(1)
                        virt_pos[dual] = pos_d - v
                        if self._order_ack_ok(r):
                            print(f"  [DUAL Z EXIT DUAL] {dual} z={z:.3f}  v={v}")
                elif pos_d < 0 and bk_d.asks and self.can_trade(1):
                    v = self.safe_vol(pos_d, min(10, abs(pos_d)), "bid")
                    if v > 0:
                        r = exchange.insert_order(
                            dual, price=bk_d.asks[0].price, volume=v, side="bid", order_type="ioc"
                        )
                        self.log_actions(1)
                        virt_pos[dual] = pos_d + v
                        if self._order_ack_ok(r):
                            print(f"  [DUAL Z EXIT DUAL] {dual} z={z:.3f}  v={v}")

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
