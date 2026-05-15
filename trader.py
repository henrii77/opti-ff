"""
Dual-listing market making on the **DUAL** line only: resting **limit** orders
anchored to the **MAIN** mid (e.g. NVDA_DUAL bid/ask around NVDA).

- Initial quotes: ``bid_dual = main_mid - quote_diff``, ``ask_dual = main_mid + quote_diff``
  (prices rounded to :data:`Trader.TICK_SIZE`).
- After our **ask** on the dual is lifted (filled / size drops), both targets move up by
  ``increment``, clamped so ``bid_dual <= main_mid`` and ``ask_dual >= main_mid``.
- Resting dual orders that see ``cancel_after_market_trades`` **public** trade ticks on
  the dual with **no** fill are cancelled.

Live loop: :meth:`Trader.run` (blocking). For a single tick: :meth:`Trader.step`
(:meth:`Trader.step` enforces at most :data:`Trader.MAX_UPDATES_PER_SEC` calls per wall-clock second).

Offline replay helper: :meth:`Trader.replay_dual_listing` (rolling z on spread; unchanged API).
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

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


# Used only by :meth:`Trader.replay_dual_listing` (notebook / offline).
Z_SCORE_WINDOW = 50
Z_STD_EPS = 1e-9
Z_ROLLING_STD_FLOOR = 0.22


@dataclass
class _RestingSide:
    order_id: Optional[Any] = None
    placed_trade_seq: int = -1
    initial_volume: int = 0
    # If insert succeeded but book/API lagged, avoid duplicate inserts for a few loops.
    pending_target_px: Optional[float] = None
    pending_until_loop: int = -1


@dataclass
class _DualQuoteState:
    """Per-(main,dual) pair: track lift skew and resting dual orders."""

    lift_steps: int = 0
    market_trade_seq: int = 0
    bid: _RestingSide = field(default_factory=_RestingSide)
    ask: _RestingSide = field(default_factory=_RestingSide)
    prev_ask_outstanding_vol: Optional[int] = None


class Trader:
    """Dual-listing **limit** quoter on DUAL anchored to MAIN mid."""

    DUAL_PAIRS = [
        ("NVDA", "NVDA_DUAL"),
        ("NVO", "NVO_DUAL"),
    ]

    TICK_SIZE = 0.10
    MAX_POSITION = 99
    SAFE_POSITION = 80
    # Cap strategy loop / step() rate and trailing-1s exchange action budget.
    MAX_UPDATES_PER_SEC = 23
    RATE_LIMIT = MAX_UPDATES_PER_SEC
    DEFAULT_QUOTE_VOLUME = 10

    TICK_HISTORY_MAX_ROWS = 10_000

    @staticmethod
    def default_csv_dir() -> Path:
        return Path(__file__).resolve().parent / "data" / "csv"

    def __init__(
        self,
        *,
        quote_diff: float = 0.50,
        quote_increment: float = 0.10,
        cancel_after_market_trades: int = 50,
        quote_volume: int = 10,
        csv_dir: Optional[Union[str, Path]] = None,
        csv_warm_start: bool = False,
    ) -> None:
        self._quote_diff = float(quote_diff)
        self._quote_increment = float(quote_increment)
        self._cancel_after_market_trades = int(cancel_after_market_trades)
        self._quote_volume = int(quote_volume)
        self._csv_dir = Path(csv_dir) if csv_dir is not None else self.default_csv_dir()
        self._csv_warm_start = bool(csv_warm_start)

        self._action_ts: Deque[float] = deque()
        self._start_time = 0.0
        self._last_status = 0.0
        self._all_assets: List[str] = []
        self._instrument_meta: Dict[str, Any] = {}
        self._tick_frames: Dict[str, pd.DataFrame] = {}
        self._loop_count = 0
        self._pair_states: Dict[str, _DualQuoteState] = {}
        self._next_step_perf: float = -1.0

    def _throttle_step_rate(self) -> None:
        period = 1.0 / float(type(self).MAX_UPDATES_PER_SEC)
        now = time.perf_counter()
        if self._next_step_perf >= 0.0:
            wait = self._next_step_perf - now
            if wait > 0:
                time.sleep(wait)
        self._next_step_perf = time.perf_counter() + period

    def _state_for(self, main: str, dual: str) -> _DualQuoteState:
        key = f"{main}|{dual}"
        if key not in self._pair_states:
            self._pair_states[key] = _DualQuoteState()
        return self._pair_states[key]

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

    @staticmethod
    def _normalize_oid(oid: Any) -> Optional[str]:
        if oid is None or oid == "" or oid == 0:
            return None
        return str(oid)

    @staticmethod
    def _order_id_from_insert(resp: Any) -> Optional[Any]:
        if resp is None:
            return None
        if getattr(resp, "success", None) is False:
            return None
        for attr in ("order_id", "orderId", "id", "orderID"):
            oid = getattr(resp, attr, None)
            if oid is not None and oid != "" and oid != 0:
                return oid
        return None

    @staticmethod
    def _order_limit_price(o: Any) -> Optional[float]:
        for attr in ("price", "limit_price", "limitPrice", "px"):
            px = getattr(o, attr, None)
            if px is None and isinstance(o, dict):
                px = o.get(attr)
            if px is not None:
                try:
                    return float(px)
                except (TypeError, ValueError):
                    continue
        return None

    def _side_token(self, o: Any) -> str:
        s = getattr(o, "side", None)
        if s is None and isinstance(o, dict):
            s = o.get("side")
        if hasattr(s, "name"):
            return str(getattr(s, "name")).lower()
        if s is not None:
            return str(s).lower()
        return ""

    def _order_matches_side(self, o: Any, side: str) -> bool:
        t = self._side_token(o)
        if side == "bid":
            return any(k in t for k in ("bid", "buy"))
        return any(k in t for k in ("ask", "sell"))

    def _find_resting_at(
        self,
        exchange: Any,
        dual: str,
        side: str,
        target_px: float,
        eps: float,
    ) -> Optional[Tuple[Any, int, float]]:
        """Our resting limit at ``target_px`` (best-effort: price + side match)."""
        tgt = self.round_tick(target_px)
        for o in self._outstanding_orders(exchange, dual):
            if not self._order_matches_side(o, side):
                continue
            fp = self._order_limit_price(o)
            if fp is None:
                continue
            fp = self.round_tick(fp)
            if abs(fp - tgt) > eps:
                continue
            oid = getattr(o, "order_id", getattr(o, "id", None))
            v = getattr(o, "volume", None)
            if v is None:
                continue
            try:
                rem = int(v)
            except (TypeError, ValueError):
                continue
            if rem <= 0:
                continue
            return (oid, rem, fp)
        return None

    @staticmethod
    def _poll_trade_tick_len(exchange: Any, dual: str) -> int:
        try:
            ticks = exchange.poll_new_trade_ticks(dual)
        except Exception:
            return 0
        if ticks is None:
            return 0
        try:
            return len(ticks)
        except TypeError:
            return 0

    def _outstanding_orders(self, exchange: Any, dual: str) -> List[Any]:
        try:
            oo = exchange.get_outstanding_orders(dual)
        except Exception:
            return []
        if oo is None:
            return []
        return list(oo)

    def _price_for_order_id(self, exchange: Any, dual: str, order_id: Any) -> Optional[float]:
        want = self._normalize_oid(order_id)
        if want is None:
            return None
        for o in self._outstanding_orders(exchange, dual):
            oid = getattr(o, "order_id", None)
            if oid is None and o is not None:
                oid = getattr(o, "id", None)
            if self._normalize_oid(oid) != want:
                continue
            return self._order_limit_price(o)
        return None

    def _volume_for_order_id(self, exchange: Any, dual: str, order_id: Any) -> Optional[int]:
        want = self._normalize_oid(order_id)
        if want is None:
            return None
        for o in self._outstanding_orders(exchange, dual):
            oid = getattr(o, "order_id", None)
            if oid is None and o is not None:
                oid = getattr(o, "id", None)
            if self._normalize_oid(oid) != want:
                continue
            v = getattr(o, "volume", None)
            if v is not None:
                try:
                    return int(v)
                except (TypeError, ValueError):
                    pass
        return None

    def _cancel_order_safe(self, exchange: Any, dual: str, order_id: Any) -> None:
        if order_id is None:
            return
        try:
            exchange.delete_order(dual, order_id=order_id)
            self.log_actions(1)
        except Exception:
            pass

    def _intended_dual_prices(self, main_mid: float, st: _DualQuoteState) -> Tuple[float, float]:
        ref = self.round_tick(main_mid)
        qd = self._quote_diff
        inc = self._quote_increment
        k = st.lift_steps
        raw_bid = ref - qd + k * inc
        raw_ask = ref + qd + k * inc
        bid_px = min(self.round_tick(raw_bid), ref)
        ask_px = max(self.round_tick(raw_ask), ref)
        if bid_px >= ask_px:
            bid_px = self.round_tick(ref - self.TICK_SIZE)
            ask_px = self.round_tick(ref + self.TICK_SIZE)
        return bid_px, ask_px

    def _maybe_cancel_stale_resting(
        self,
        exchange: Any,
        dual: str,
        st: _DualQuoteState,
        side: _RestingSide,
        side_name: str,
    ) -> None:
        if side.order_id is None:
            return
        if side.placed_trade_seq < 0:
            return
        if st.market_trade_seq - side.placed_trade_seq < self._cancel_after_market_trades:
            return
        rem = self._volume_for_order_id(exchange, dual, side.order_id)
        if rem is None:
            # Order gone (filled/cancelled elsewhere) — clear handle
            side.order_id = None
            side.placed_trade_seq = -1
            side.pending_target_px = None
            side.pending_until_loop = -1
            return
        if rem < side.initial_volume:
            return
        if not self.can_trade(1):
            return
        self._cancel_order_safe(exchange, dual, side.order_id)
        side.order_id = None
        side.initial_volume = 0
        side.placed_trade_seq = -1
        side.pending_target_px = None
        side.pending_until_loop = -1
        print(
            f"  [DUAL MM STALE {side_name.upper()}] {dual}  "
            f"cancel after {self._cancel_after_market_trades} mkt ticks w/o fill"
        )

    def _ensure_limit(
        self,
        exchange: Any,
        dual: str,
        pos_d: int,
        side_name: str,
        side: str,
        target_px: float,
        st_side: _RestingSide,
        st: _DualQuoteState,
    ) -> None:
        vol_req = self.safe_vol(pos_d, self._quote_volume, "bid" if side == "bid" else "ask")
        if vol_req <= 0:
            if st_side.order_id is not None and self.can_trade(1):
                self._cancel_order_safe(exchange, dual, st_side.order_id)
                st_side.order_id = None
                st_side.placed_trade_seq = -1
            st_side.pending_target_px = None
            st_side.pending_until_loop = -1
            return

        target_px = self.round_tick(target_px)
        eps = self.TICK_SIZE * 0.5 + 1e-9

        # Same-target insert may not appear in outstanding immediately; wait a few loops.
        if (
            st_side.pending_target_px is not None
            and st_side.pending_until_loop >= 0
            and self._loop_count <= st_side.pending_until_loop
        ):
            if abs(self.round_tick(float(st_side.pending_target_px)) - target_px) <= self.TICK_SIZE / 4:
                hitp = self._find_resting_at(exchange, dual, side, target_px, eps)
                if hitp is not None:
                    oidp, remp, _ = hitp
                    st_side.order_id = oidp
                    st_side.initial_volume = remp
                    if st_side.placed_trade_seq < 0:
                        st_side.placed_trade_seq = st.market_trade_seq
                    st_side.pending_target_px = None
                    st_side.pending_until_loop = -1
                    return
                return
        st_side.pending_target_px = None
        st_side.pending_until_loop = -1

        hit = self._find_resting_at(exchange, dual, side, target_px, eps)
        if hit is not None:
            oid, rem, _ = hit
            st_side.order_id = oid
            st_side.initial_volume = rem
            if st_side.placed_trade_seq < 0:
                st_side.placed_trade_seq = st.market_trade_seq
            st_side.pending_target_px = None
            st_side.pending_until_loop = -1
            return

        # Tracked order still at target but _find_resting_at missed (side repr quirks) — keep.
        if st_side.order_id is not None:
            rem = self._volume_for_order_id(exchange, dual, st_side.order_id)
            px_known = self._price_for_order_id(exchange, dual, st_side.order_id)
            if rem is not None and rem > 0 and px_known is not None:
                rp = self.round_tick(px_known)
                if abs(rp - target_px) <= eps:
                    st_side.initial_volume = rem
                    if st_side.placed_trade_seq < 0:
                        st_side.placed_trade_seq = st.market_trade_seq
                    st_side.pending_target_px = None
                    st_side.pending_until_loop = -1
                    return
            if rem is not None and rem > 0 and self.can_trade(1):
                if px_known is None or abs(self.round_tick(px_known) - target_px) > eps:
                    self._cancel_order_safe(exchange, dual, st_side.order_id)
            st_side.order_id = None
            st_side.placed_trade_seq = -1

        if not self.can_trade(1):
            return
        try:
            r = exchange.insert_order(
                dual,
                price=target_px,
                volume=vol_req,
                side=side,
                order_type="limit",
            )
        except Exception as e:  # pragma: no cover
            print(f"  [DUAL MM ERR] {dual} {side} insert: {e}")
            return
        if getattr(r, "success", None) is False:
            return
        self.log_actions(1)
        oid = self._order_id_from_insert(r)
        if oid is None:
            time.sleep(0.02)
            hit2 = self._find_resting_at(exchange, dual, side, target_px, eps)
            if hit2 is not None:
                oid, rem2, _ = hit2
                st_side.initial_volume = rem2
                st_side.pending_target_px = None
                st_side.pending_until_loop = -1
            else:
                st_side.initial_volume = vol_req
                st_side.pending_target_px = float(target_px)
                st_side.pending_until_loop = self._loop_count + 5
        else:
            st_side.initial_volume = vol_req
            st_side.pending_target_px = None
            st_side.pending_until_loop = -1
        st_side.order_id = oid
        st_side.placed_trade_seq = st.market_trade_seq
        print(
            f"  [DUAL MM {side_name.upper()}] {dual}  px={target_px:.2f}  v={vol_req}  "
            f"oid={oid}  ref_seq={st.market_trade_seq}"
        )

    def _process_pair(
        self,
        exchange: Any,
        main: str,
        dual: str,
        books: Dict[str, Any],
        virt_pos: Dict[str, int],
    ) -> None:
        bk_m = books.get(main)
        bk_d = books.get(dual)
        if not (bk_m and bk_d and bk_m.bids and bk_m.asks and bk_d.bids and bk_d.asks):
            return

        mm = self.mid(bk_m)
        if mm is None:
            return

        st = self._state_for(main, dual)
        n_ticks = self._poll_trade_tick_len(exchange, dual)
        st.market_trade_seq += n_ticks

        ask_vol_now: Optional[int] = None
        if st.ask.order_id is not None:
            ask_vol_now = self._volume_for_order_id(exchange, dual, st.ask.order_id)
        if (
            st.prev_ask_outstanding_vol is not None
            and ask_vol_now is not None
            and ask_vol_now < st.prev_ask_outstanding_vol
        ):
            st.lift_steps += 1
            print(f"  [DUAL MM LIFT] {dual}  lift_steps={st.lift_steps}")

        bid_px, ask_px = self._intended_dual_prices(mm, st)
        pos_d = virt_pos.get(dual, 0)

        self._maybe_cancel_stale_resting(exchange, dual, st, st.bid, "bid")
        self._maybe_cancel_stale_resting(exchange, dual, st, st.ask, "ask")

        self._ensure_limit(exchange, dual, pos_d, "bid", "bid", bid_px, st.bid, st)
        pos_d = virt_pos.get(dual, 0)
        self._ensure_limit(exchange, dual, pos_d, "ask", "ask", ask_px, st.ask, st)

        if st.ask.order_id is not None:
            st.prev_ask_outstanding_vol = self._volume_for_order_id(
                exchange, dual, st.ask.order_id
            )
        else:
            st.prev_ask_outstanding_vol = None

    def _bootstrap(self, exchange: Any) -> None:
        print("=" * 60)
        print("  Dual-listing limit quoter (DUAL around MAIN mid)")
        print("=" * 60)
        self._start_time = time.time()
        self._last_status = 0.0
        self._loop_count = 0
        self._pair_states.clear()

        self._all_assets = sorted({s for pair in self.DUAL_PAIRS for s in pair})
        instruments = exchange.get_instruments()
        self._instrument_meta = {
            aid: self._resolve_instrument(instruments, aid) for aid in self._all_assets
        }
        cols = list(HEADER)
        self._tick_frames = {aid: pd.DataFrame(columns=cols) for aid in self._all_assets}

        print(
            f"  Instruments: {self._all_assets}  "
            f"quote_diff={self._quote_diff}  increment={self._quote_increment}  "
            f"stale_ticks={self._cancel_after_market_trades}  vol={self._quote_volume}"
        )

        if self._csv_warm_start:
            self._load_csv_warm_start()

    def _normalize_csv_to_header(self, df: pd.DataFrame) -> pd.DataFrame:
        for c in HEADER:
            if c not in df.columns:
                df[c] = np.nan
        out = df[list(HEADER)].copy()
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
            print(f"  [CSV WARN] {instrument_id}: normalize failed: {e}")
            return pd.DataFrame(columns=list(HEADER))
        norm["timestamp"] = pd.to_numeric(norm["timestamp"], errors="coerce")
        for c in ("bid_price", "bid_volume", "ask_price", "ask_volume", "mid", "spread", "last_trade_price"):
            if c in norm.columns:
                norm[c] = pd.to_numeric(norm[c], errors="coerce")
        norm = norm.dropna(subset=["timestamp", "mid"], how="any")
        if norm.empty:
            return pd.DataFrame(columns=list(HEADER))
        return norm.sort_values("timestamp").tail(self.TICK_HISTORY_MAX_ROWS).reset_index(drop=True)

    def _load_csv_warm_start(self) -> None:
        for aid in self._all_assets:
            df = self._read_instrument_csv(aid)
            if not df.empty:
                self._tick_frames[aid] = df
                print(f"  [CSV] Loaded {len(df)} row(s) for {aid}")

    def start(self, exchange: Any) -> None:
        self._bootstrap(exchange)

    def run(
        self,
        exchange: Any,
        *,
        quote_diff: Optional[float] = None,
        increment: Optional[float] = None,
        cancel_after_market_trades: Optional[int] = None,
        quote_volume: Optional[int] = None,
    ) -> None:
        """
        Blocking quote loop. Override tuning without reconstructing ``Trader``::

            Trader().run(
                exchange,
                quote_diff=0.5,
                increment=0.1,
                cancel_after_market_trades=50,
                quote_volume=10,
            )
        """
        if quote_diff is not None:
            self._quote_diff = float(quote_diff)
        if increment is not None:
            self._quote_increment = float(increment)
        if cancel_after_market_trades is not None:
            self._cancel_after_market_trades = int(cancel_after_market_trades)
        if quote_volume is not None:
            self._quote_volume = int(quote_volume)
        self.start(exchange)
        while True:
            self.step(exchange)

    def step(self, exchange: Any) -> None:
        self._throttle_step_rate()
        # Avoid touching the client when the session is down (prevents reconnect spam).
        if not exchange.is_connected():
            return
        if not self._all_assets:
            self._bootstrap(exchange)
        try:
            self._iteration(exchange)
        except Exception as e:  # pragma: no cover
            print(f"  [LOOP ERR] {e}")

    def _iteration(self, exchange: Any) -> None:
        now = time.time()
        elapsed = now - self._start_time
        self._loop_count += 1

        if not exchange.is_connected():
            return

        books: Dict[str, Any] = {}
        for asset in self._all_assets:
            try:
                books[asset] = exchange.get_last_price_book(asset)
            except Exception:
                pass

        try:
            positions = exchange.get_positions()
        except Exception:
            return
        virt_pos: Dict[str, int] = dict(positions)

        if now - self._last_status > 15:
            try:
                pnl = exchange.get_pnl()
                rps = len(self._action_ts)
                print(f"\n[{elapsed:6.0f}s] PnL={pnl:,.2f}  RPS={rps}  Pos={dict(positions)}")
            except Exception:
                pass
            self._last_status = now

        for main, dual in self.DUAL_PAIRS:
            self._process_pair(exchange, main, dual, books, virt_pos)

        self.poll_tick_history(exchange, now, books)

    def poll_tick_history(self, exchange: Any, ts: float, books: Dict[str, Any]) -> None:
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

    @staticmethod
    def replay_dual_listing(
        main_mids: np.ndarray,
        dual_mids: np.ndarray,
        *,
        z_threshold: float = 2.0,
        z_window: int = Z_SCORE_WINDOW,
        z_std_eps: float = Z_STD_EPS,
        z_std_floor: float = Z_ROLLING_STD_FLOOR,
    ) -> List[Dict[str, Any]]:
        """Offline rolling z on spread (unchanged signature for notebooks)."""
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
            sig_eff = max(sig, z_std_floor, z_std_eps)
            z = (s_now - mu) / sig_eff
            if z >= z_threshold:
                events.append({"i": i, "kind": "short_spread", "spread": s_now, "z": z})
            elif z <= -z_threshold:
                events.append({"i": i, "kind": "long_spread", "spread": s_now, "z": z})

        return events


def main() -> None:
    from optibook.synchronous_client import Exchange as _Exchange

    exchange = _Exchange()
    exchange.connect()
    Trader().run(
        exchange,
        quote_diff=0.5,
        increment=0.1,
        cancel_after_market_trades=50,
    )


if __name__ == "__main__":
    main()
