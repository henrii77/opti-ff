"""
OMNI-ENGINE V4 — Optibook FutureFocus-style multi-strategy trader (OOP).

Strategies:
  1. Dual-listing parity (NVDA / NVDA_DUAL, NVO / NVO_DUAL)
  2. ETF basket vs fair value (OB5X_ETF)
  2b. ETF creation / redemption basket arb
  3. Multi-expiry futures vs index fair value
  3b. Calendar spreads between futures
  4. Linear-regression pairs (stat-arb)

Safety: rolling rate limit, position caps, emergency shed, warm-up, vol scaling.

Offline: use :meth:`Trader.replay_dual_listing` with aligned mid series for plots/tests.
"""

from __future__ import annotations

import datetime
import math
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np

# Optional live client (notebook / tests can import Trader without Optibook)
try:
    from optibook.synchronous_client import Exchange
except ImportError:  # pragma: no cover
    Exchange = Any  # type: ignore[misc, assignment]


class Trader:
    """Stateful trading loop + small offline replay helper for dual listings."""

    # --- Instruments ---------------------------------------------------------
    INDEX_STOCKS = ["AMZN", "JPM", "NVDA", "XOM", "NVO"]
    INDEX_WEIGHTS = {
        "AMZN": 953.21,
        "JPM": 129.25,
        "NVDA": 908.06,
        "XOM": 2245.39,
        "NVO": 124.78,
    }
    ETF_TICKER = "OB5X_ETF"
    ETF_C = 2.50
    ETF_M = 0.25
    DUAL_PAIRS = [
        ("NVDA", "NVDA_DUAL"),
        ("NVO", "NVO_DUAL"),
    ]
    STAT_PAIRS = [
        ("AMZN", "XOM"),
        ("NVDA", "NVO"),
    ]
    FALLBACK_EXPIRIES = {
        "OB5X_202606_F": 1 / 12,
        "OB5X_202609_F": 4 / 12,
        "OB5X_202612_F": 7 / 12,
    }
    RISK_FREE_RATE = 0.03

    # --- Parameters --------------------------------------------------------
    TICK_SIZE = 0.10
    MAX_POSITION = 99
    SAFE_POSITION = 80
    LOOP_SLEEP = 0.04
    RATE_LIMIT = 21

    WARMUP_SECONDS = 300
    RECALC_INTERVAL = 60

    DUAL_THRESHOLD = 0.10
    ETF_THRESHOLD = 0.15
    FUT_THRESHOLD = 0.15
    STAT_ENTRY_SPREAD = 0.40
    EXEC_BUFFER = 1.005

    BASE_VOLUME = 10
    EMERGENCY_THRESHOLD = 95

    VOL_ALPHA = 0.20
    VOL_LOOKBACK = 50
    VOL_K = 8.0
    MIN_VOL_SCALE = 0.5
    MAX_VOL_SCALE = 2.5

    OLS_MIN_POINTS = 10
    OLS_WINDOW = 60
    OLS_CORR_MIN = 0.50

    def __init__(self) -> None:
        self._action_ts: Deque[float] = deque()
        self._start_time = 0.0
        self._last_status = 0.0
        self._last_recalc = 0.0
        self._futures_expiries: Dict[str, float] = {}
        self._history: Dict[str, Deque[float]] = {}
        self._vol_state: Dict[str, float] = {}
        self._all_assets: List[str] = []
        self._ols_params: Dict[Tuple[str, str], Dict[str, Any]] = {
            pair: {"beta": 1.0, "alpha": 0.0, "ready": False} for pair in self.STAT_PAIRS
        }
        self._loop_count = 0

    # --- Static / pure helpers ---------------------------------------------
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

    def calc_index(self, mids: Dict[str, float]) -> float:
        total = sum(self.INDEX_WEIGHTS[s] * mids[s] for s in self.INDEX_STOCKS if s in mids)
        return total / 1000.0

    def update_vol(self, symbol: str) -> float:
        h = self._history[symbol]
        if len(h) < 3:
            return 0.0
        rets = np.diff(np.array(h, dtype=float))
        raw = float(np.std(rets))
        prev = self._vol_state.get(symbol, raw)
        v = self.VOL_ALPHA * raw + (1.0 - self.VOL_ALPHA) * prev
        self._vol_state[symbol] = v
        return v

    def vol_scale_factor(self, v: float) -> float:
        return max(self.MIN_VOL_SCALE, min(self.MAX_VOL_SCALE, 1.0 + v * self.VOL_K))

    def discover_futures(self, exchange: Any) -> Dict[str, float]:
        try:
            instruments = exchange.get_instruments()
        except Exception:
            return self.FALLBACK_EXPIRIES.copy()

        result: Dict[str, float] = {}
        for iid, _inst in instruments.items():
            if "OB5X" in iid and iid.endswith("_F"):
                tau = self.FALLBACK_EXPIRIES.get(iid, 0.25)
                try:
                    parts = iid.split("_")
                    yyyymm = parts[1]
                    year = int(yyyymm[:4])
                    month = int(yyyymm[4:])
                    now = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)
                    expiry_dt = datetime.datetime(year, month, 1)
                    delta = (expiry_dt - now).days
                    tau = max(delta / 365.0, 1 / 365.0)
                except Exception:
                    pass
                result[iid] = tau

        return result if result else self.FALLBACK_EXPIRIES.copy()

    def _bootstrap(self, exchange: Any) -> None:
        print("=" * 60)
        print("  OMNI-ENGINE V4  —  Starting up")
        print("=" * 60)
        self._start_time = time.time()
        self._last_status = 0.0
        self._last_recalc = 0.0
        self._futures_expiries = self.discover_futures(exchange)
        print(f"  Live futures discovered: {list(self._futures_expiries.keys())}")

        self._all_assets = list(
            set(
                list(self.INDEX_STOCKS)
                + [self.ETF_TICKER]
                + [d for pair in self.DUAL_PAIRS for d in pair]
                + [s for pair in self.STAT_PAIRS for s in pair]
                + list(self._futures_expiries.keys())
            )
        )
        self._history = {a: deque(maxlen=self.VOL_LOOKBACK) for a in self._all_assets}
        self._vol_state = {}
        self._ols_params = {
            pair: {"beta": 1.0, "alpha": 0.0, "ready": False} for pair in self.STAT_PAIRS
        }
        self._loop_count = 0

    def _recalc(self, exchange: Any) -> None:
        self._futures_expiries = self.discover_futures(exchange)
        for f in self._futures_expiries:
            if f not in self._history:
                self._history[f] = deque(maxlen=self.VOL_LOOKBACK)
                if f not in self._all_assets:
                    self._all_assets.append(f)

        win = self.OLS_WINDOW
        for s1, s2 in self.STAT_PAIRS:
            h1 = list(self._history.get(s1, ()))
            h2 = list(self._history.get(s2, ()))
            m = min(len(h1), len(h2), win)
            if m < self.OLS_MIN_POINTS:
                self._ols_params[(s1, s2)]["ready"] = False
                print(f"  [OLS SKIP] {s1}/{s2} need ≥{self.OLS_MIN_POINTS} pts, have {m}")
                continue
            arr1 = np.array(h1[-m:], dtype=float)
            arr2 = np.array(h2[-m:], dtype=float)
            correlation = float(np.corrcoef(arr1, arr2)[0, 1])
            if abs(correlation) > self.OLS_CORR_MIN:
                beta, alpha = np.polyfit(arr2, arr1, 1)
                self._ols_params[(s1, s2)] = {"beta": float(beta), "alpha": float(alpha), "ready": True}
                print(f"  [OLS] {s1}/{s2} β={beta:.4f} r={correlation:.2f}")
            else:
                self._ols_params[(s1, s2)]["ready"] = False
                print(f"  [OLS PAUSED] {s1}/{s2} r={correlation:.2f} (too weak)")

        self._last_recalc = time.time()

    def run(self, exchange: Any) -> None:
        """Main 25 Hz loop (blocks forever)."""
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
                    self._history[asset].append(m)
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

        if now - self._last_recalc > self.RECALC_INTERVAL:
            self._recalc(exchange)

        X_t: Optional[float]
        if all(s in mids_snap for s in self.INDEX_STOCKS):
            X_t = self.calc_index(mids_snap)
        else:
            X_t = None

        self._strategy_dual(books, mids_snap, virt_pos, exchange)
        self._strategy_etf_simple(books, mids_snap, virt_pos, exchange, X_t)
        self._strategy_etf_basket(books, mids_snap, virt_pos, exchange, X_t)
        self._strategy_futures_fair(books, mids_snap, virt_pos, exchange, X_t)
        self._strategy_futures_calendar(books, mids_snap, virt_pos, exchange, X_t)
        self._strategy_stat_arb(books, mids_snap, virt_pos, exchange, elapsed)
        self._strategy_emergency_shed(books, positions, exchange)

    # --- Strategies ---------------------------------------------------------

    def _strategy_dual(
        self,
        books: Dict[str, Any],
        mids_snap: Dict[str, float],
        virt_pos: Dict[str, int],
        exchange: Any,
    ) -> None:
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
            spread_key = f"{main}_{dual}_spread"
            if spread_key not in self._history:
                self._history[spread_key] = deque(maxlen=self.VOL_LOOKBACK)
            self._history[spread_key].append(spread)

            vol = self.update_vol(dual)
            scale = self.vol_scale_factor(vol)
            threshold = self.DUAL_THRESHOLD * scale * self.EXEC_BUFFER
            vol_size = max(1, int(self.BASE_VOLUME / scale))

            pos_m = virt_pos.get(main, 0)
            pos_d = virt_pos.get(dual, 0)

            spread_hist = np.array(self._history[spread_key], dtype=float)
            if len(spread_hist) >= 10:
                spread_mean = float(np.mean(spread_hist))
                spread_std = float(np.std(spread_hist))
            else:
                spread_mean = 0.0
                spread_std = 0.0

            main_hist = np.array(self._history.get(main, []), dtype=float)
            dual_hist = np.array(self._history.get(dual, []), dtype=float)
            lagging = False
            if len(main_hist) >= 5 and len(dual_hist) >= 5:
                main_move = float(main_hist[-1] - main_hist[-5])
                dual_move = float(dual_hist[-1] - dual_hist[-5])
                if main_move > threshold and abs(dual_move) < abs(main_move) * 0.5:
                    lagging = True
                elif main_move < -threshold and abs(dual_move) < abs(main_move) * 0.5:
                    lagging = True

            if spread > threshold:
                v_m = self.safe_vol(pos_m, vol_size, "ask")
                v_d = self.safe_vol(pos_d, vol_size, "bid")
                v = min(v_m, v_d)
                if v > 0 and self.can_trade(2):
                    print(
                        f"  [DUAL ENTER SHORT SPREAD] {main}-{dual}  "
                        f"spread={spread:.3f}  mean={spread_mean:.3f}"
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

            elif -spread > threshold:
                v_m = self.safe_vol(pos_m, vol_size, "bid")
                v_d = self.safe_vol(pos_d, vol_size, "ask")
                v = min(v_m, v_d)
                if v > 0 and self.can_trade(2):
                    print(
                        f"  [DUAL ENTER LONG SPREAD ] {main}-{dual}  "
                        f"spread={spread:.3f}  mean={spread_mean:.3f}"
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

            elif abs(spread) < threshold * 0.3 and (pos_m != 0 or pos_d != 0):
                if pos_m < 0 and bk_m.asks and self.can_trade(1):
                    v = self.safe_vol(pos_m, min(10, abs(pos_m)), "bid")
                    if v > 0:
                        exchange.insert_order(
                            main, price=bk_m.asks[0].price, volume=v, side="bid", order_type="ioc"
                        )
                        self.log_actions(1)
                        virt_pos[main] = pos_m + v
                        print(f"  [DUAL EXIT MAIN] {main} spread={spread:.3f}")
                elif pos_m > 0 and bk_m.bids and self.can_trade(1):
                    v = self.safe_vol(pos_m, min(10, pos_m), "ask")
                    if v > 0:
                        exchange.insert_order(
                            main, price=bk_m.bids[0].price, volume=v, side="ask", order_type="ioc"
                        )
                        self.log_actions(1)
                        virt_pos[main] = pos_m - v
                        print(f"  [DUAL EXIT MAIN] {main} spread={spread:.3f}")
                if pos_d > 0 and bk_d.bids and self.can_trade(1):
                    v = self.safe_vol(pos_d, min(10, pos_d), "ask")
                    if v > 0:
                        exchange.insert_order(
                            dual, price=bk_d.bids[0].price, volume=v, side="ask", order_type="ioc"
                        )
                        self.log_actions(1)
                        virt_pos[dual] = pos_d - v
                        print(f"  [DUAL EXIT DUAL] {dual} spread={spread:.3f}")
                elif pos_d < 0 and bk_d.asks and self.can_trade(1):
                    v = self.safe_vol(pos_d, min(10, abs(pos_d)), "bid")
                    if v > 0:
                        exchange.insert_order(
                            dual, price=bk_d.asks[0].price, volume=v, side="bid", order_type="ioc"
                        )
                        self.log_actions(1)
                        virt_pos[dual] = pos_d + v
                        print(f"  [DUAL EXIT DUAL] {dual} spread={spread:.3f}")

            if lagging and len(main_hist) >= 5:
                main_move = float(main_hist[-1] - main_hist[-5])
                if main_move > 0:
                    v = self.safe_vol(pos_d, vol_size, "bid")
                    if v > 0 and self.can_trade(1):
                        print(f"  [DUAL LAG  BUY ] {dual} main_move={main_move:.3f}")
                        exchange.insert_order(
                            dual, price=bk_d.asks[0].price, volume=v, side="bid", order_type="ioc"
                        )
                        self.log_actions(1)
                        virt_pos[dual] = pos_d + v
                else:
                    v = self.safe_vol(pos_d, vol_size, "ask")
                    if v > 0 and self.can_trade(1):
                        print(f"  [DUAL LAG  SELL] {dual} main_move={main_move:.3f}")
                        exchange.insert_order(
                            dual, price=bk_d.bids[0].price, volume=v, side="ask", order_type="ioc"
                        )
                        self.log_actions(1)
                        virt_pos[dual] = pos_d - v

    def _strategy_etf_simple(
        self,
        books: Dict[str, Any],
        mids_snap: Dict[str, float],
        virt_pos: Dict[str, int],
        exchange: Any,
        X_t: Optional[float],
    ) -> None:
        if X_t is None:
            return
        etf_fair = self.ETF_C + self.ETF_M * X_t
        bk_etf = books.get(self.ETF_TICKER)
        if not (bk_etf and bk_etf.bids and bk_etf.asks):
            return
        etf_bid = bk_etf.bids[0].price
        etf_ask = bk_etf.asks[0].price
        etf_mid = (etf_bid + etf_ask) / 2.0

        vol = self.update_vol(self.ETF_TICKER)
        scale = self.vol_scale_factor(vol)
        threshold = self.ETF_THRESHOLD * scale * self.EXEC_BUFFER
        vol_size = max(1, int(self.BASE_VOLUME / scale))
        pos_etf = virt_pos.get(self.ETF_TICKER, 0)
        mis = etf_mid - etf_fair

        if mis > threshold:
            v = self.safe_vol(pos_etf, vol_size, "ask")
            if v > 0 and self.can_trade(1):
                print(f"  [ETF  SELL] fair={etf_fair:.3f}  market={etf_mid:.3f}  mis={mis:.3f}")
                exchange.insert_order(
                    self.ETF_TICKER, price=etf_bid, volume=v, side="ask", order_type="ioc"
                )
                self.log_actions(1)
                virt_pos[self.ETF_TICKER] = pos_etf - v
        elif -mis > threshold:
            v = self.safe_vol(pos_etf, vol_size, "bid")
            if v > 0 and self.can_trade(1):
                print(f"  [ETF  BUY ] fair={etf_fair:.3f}  market={etf_mid:.3f}  mis={mis:.3f}")
                exchange.insert_order(
                    self.ETF_TICKER, price=etf_ask, volume=v, side="bid", order_type="ioc"
                )
                self.log_actions(1)
                virt_pos[self.ETF_TICKER] = pos_etf + v

    def _strategy_etf_basket(
        self,
        books: Dict[str, Any],
        mids_snap: Dict[str, float],
        virt_pos: Dict[str, int],
        exchange: Any,
        X_t: Optional[float],
    ) -> None:
        if X_t is None:
            return
        bk_etf = books.get(self.ETF_TICKER)
        if not (bk_etf and bk_etf.bids and bk_etf.asks):
            return
        etf_fair = self.ETF_C + self.ETF_M * X_t
        etf_bid = bk_etf.bids[0].price
        etf_ask = bk_etf.asks[0].price

        vol = self.update_vol(self.ETF_TICKER)
        scale = self.vol_scale_factor(vol)
        cr_thresh = 0.20 * scale * self.EXEC_BUFFER
        cr_vol = max(1, int(self.BASE_VOLUME / scale))
        pos_etf = virt_pos.get(self.ETF_TICKER, 0)

        basket = {s: round(self.ETF_M * self.INDEX_WEIGHTS[s] / 1000.0 * cr_vol) for s in self.INDEX_STOCKS}
        if sum(basket.values()) <= 0:
            return
        if not all(books.get(s) and books[s].bids and books[s].asks for s in self.INDEX_STOCKS):
            return

        basket_buy_cost = sum(books[s].asks[0].price * basket[s] for s in self.INDEX_STOCKS)
        basket_sell_proc = sum(books[s].bids[0].price * basket[s] for s in self.INDEX_STOCKS)

        creation_profit = (etf_bid * cr_vol) - basket_buy_cost - (self.ETF_C * cr_vol)
        if creation_profit > cr_thresh:
            n_orders = 1 + len(self.INDEX_STOCKS)
            etf_v = self.safe_vol(pos_etf, cr_vol, "ask")
            basket_ok = all(
                self.safe_vol(virt_pos.get(s, 0), basket[s], "bid") >= basket[s] for s in self.INDEX_STOCKS
            )
            if etf_v > 0 and basket_ok and self.can_trade(n_orders):
                print(f"  [CREATION ] profit={creation_profit:.3f}  sell ETF @ {etf_bid:.2f}  buy basket")
                exchange.insert_order(
                    self.ETF_TICKER, price=etf_bid, volume=etf_v, side="ask", order_type="ioc"
                )
                self.log_actions(1)
                virt_pos[self.ETF_TICKER] = pos_etf - etf_v
                for s in self.INDEX_STOCKS:
                    if basket[s] > 0:
                        v_s = self.safe_vol(virt_pos.get(s, 0), basket[s], "bid")
                        if v_s > 0:
                            exchange.insert_order(
                                s, price=books[s].asks[0].price, volume=v_s, side="bid", order_type="ioc"
                            )
                            self.log_actions(1)
                            virt_pos[s] = virt_pos.get(s, 0) + v_s

        redemption_profit = basket_sell_proc + (self.ETF_C * cr_vol) - (etf_ask * cr_vol)
        if redemption_profit > cr_thresh:
            n_orders = 1 + len(self.INDEX_STOCKS)
            etf_v = self.safe_vol(pos_etf, cr_vol, "bid")
            basket_ok = all(
                self.safe_vol(virt_pos.get(s, 0), basket[s], "ask") >= basket[s] for s in self.INDEX_STOCKS
            )
            if etf_v > 0 and basket_ok and self.can_trade(n_orders):
                print(f"  [REDEMPTN ] profit={redemption_profit:.3f}  buy ETF @ {etf_ask:.2f}  sell basket")
                exchange.insert_order(
                    self.ETF_TICKER, price=etf_ask, volume=etf_v, side="bid", order_type="ioc"
                )
                self.log_actions(1)
                virt_pos[self.ETF_TICKER] = pos_etf + etf_v
                for s in self.INDEX_STOCKS:
                    if basket[s] > 0:
                        v_s = self.safe_vol(virt_pos.get(s, 0), basket[s], "ask")
                        if v_s > 0:
                            exchange.insert_order(
                                s, price=books[s].bids[0].price, volume=v_s, side="ask", order_type="ioc"
                            )
                            self.log_actions(1)
                            virt_pos[s] = virt_pos.get(s, 0) - v_s

    def _strategy_futures_fair(
        self,
        books: Dict[str, Any],
        mids_snap: Dict[str, float],
        virt_pos: Dict[str, int],
        exchange: Any,
        X_t: Optional[float],
    ) -> None:
        if X_t is None:
            return
        for fut, tau in self._futures_expiries.items():
            bk_f = books.get(fut)
            if not bk_f or not bk_f.bids or not bk_f.asks:
                continue
            fut_fair = X_t * math.exp(self.RISK_FREE_RATE * tau)
            fut_bid = bk_f.bids[0].price
            fut_ask = bk_f.asks[0].price
            fut_mid = (fut_bid + fut_ask) / 2.0

            vol = self.update_vol(fut)
            scale = self.vol_scale_factor(vol)
            threshold = self.FUT_THRESHOLD * scale * self.EXEC_BUFFER
            vol_size = max(1, int(self.BASE_VOLUME / scale))
            basis = fut_mid - fut_fair
            pos_f = virt_pos.get(fut, 0)

            if basis > threshold:
                v = self.safe_vol(pos_f, vol_size, "ask")
                if v > 0 and self.can_trade(1):
                    print(f"  [FUT  SELL] {fut}  fair={fut_fair:.3f}  market={fut_mid:.3f}  basis={basis:.3f}")
                    exchange.insert_order(fut, price=fut_bid, volume=v, side="ask", order_type="ioc")
                    self.log_actions(1)
                    virt_pos[fut] = pos_f - v
            elif -basis > threshold:
                v = self.safe_vol(pos_f, vol_size, "bid")
                if v > 0 and self.can_trade(1):
                    print(f"  [FUT  BUY ] {fut}  fair={fut_fair:.3f}  market={fut_mid:.3f}  basis={basis:.3f}")
                    exchange.insert_order(fut, price=fut_ask, volume=v, side="bid", order_type="ioc")
                    self.log_actions(1)
                    virt_pos[fut] = pos_f + v

    def _strategy_futures_calendar(
        self,
        books: Dict[str, Any],
        mids_snap: Dict[str, float],
        virt_pos: Dict[str, int],
        exchange: Any,
        X_t: Optional[float],
    ) -> None:
        if X_t is None:
            return
        fut_list = [
            (f, t)
            for f, t in self._futures_expiries.items()
            if books.get(f) and books[f].bids and books[f].asks
        ]
        for i in range(len(fut_list)):
            for j in range(i + 1, len(fut_list)):
                f1, t1 = fut_list[i]
                f2, t2 = fut_list[j]
                if t1 > t2:
                    f1, t1, f2, t2 = f2, t2, f1, t1
                bk1 = books[f1]
                bk2 = books[f2]
                mid1 = (bk1.bids[0].price + bk1.asks[0].price) / 2.0
                mid2 = (bk2.bids[0].price + bk2.asks[0].price) / 2.0
                fair1 = X_t * math.exp(self.RISK_FREE_RATE * t1)
                fair2 = X_t * math.exp(self.RISK_FREE_RATE * t2)
                market_cs = mid2 - mid1
                fair_cs = fair2 - fair1
                cs_mis = market_cs - fair_cs

                vol = self.update_vol(f1)
                scale = self.vol_scale_factor(vol)
                threshold = self.FUT_THRESHOLD * scale * self.EXEC_BUFFER
                pos1 = virt_pos.get(f1, 0)
                pos2 = virt_pos.get(f2, 0)
                vol_size = max(1, int(self.BASE_VOLUME / scale))

                if cs_mis > threshold and self.can_trade(2):
                    v1 = self.safe_vol(pos1, vol_size, "bid")
                    v2 = self.safe_vol(pos2, vol_size, "ask")
                    v = min(v1, v2)
                    if v > 0:
                        print(f"  [CAL SELL FAR] {f2} vs {f1}  cs_mis={cs_mis:.3f}")
                        exchange.insert_order(f1, price=bk1.asks[0].price, volume=v, side="bid", order_type="ioc")
                        exchange.insert_order(f2, price=bk2.bids[0].price, volume=v, side="ask", order_type="ioc")
                        self.log_actions(2)
                        virt_pos[f1] = pos1 + v
                        virt_pos[f2] = pos2 - v
                elif -cs_mis > threshold and self.can_trade(2):
                    v1 = self.safe_vol(pos1, vol_size, "ask")
                    v2 = self.safe_vol(pos2, vol_size, "bid")
                    v = min(v1, v2)
                    if v > 0:
                        print(f"  [CAL BUY  FAR] {f2} vs {f1}  cs_mis={cs_mis:.3f}")
                        exchange.insert_order(f1, price=bk1.bids[0].price, volume=v, side="ask", order_type="ioc")
                        exchange.insert_order(f2, price=bk2.asks[0].price, volume=v, side="bid", order_type="ioc")
                        self.log_actions(2)
                        virt_pos[f1] = pos1 - v
                        virt_pos[f2] = pos2 + v

    def _strategy_stat_arb(
        self,
        books: Dict[str, Any],
        mids_snap: Dict[str, float],
        virt_pos: Dict[str, int],
        exchange: Any,
        elapsed: float,
    ) -> None:
        if elapsed <= self.WARMUP_SECONDS:
            return
        for s1, s2 in self.STAT_PAIRS:
            params = self._ols_params.get((s1, s2), {})
            if not params.get("ready"):
                continue
            p1 = mids_snap.get(s1)
            p2 = mids_snap.get(s2)
            if p1 is None or p2 is None:
                continue
            bk1 = books.get(s1)
            bk2 = books.get(s2)
            if not (bk1 and bk2 and bk1.bids and bk1.asks and bk2.bids and bk2.asks):
                continue
            beta = params["beta"]
            alpha = params["alpha"]
            spread = p1 - (beta * p2 + alpha)

            vol1 = self.update_vol(s1)
            scale = self.vol_scale_factor(vol1)
            threshold = self.STAT_ENTRY_SPREAD * scale * self.EXEC_BUFFER
            vol_size = max(1, int(self.BASE_VOLUME / scale))
            pos1 = virt_pos.get(s1, 0)
            pos2 = virt_pos.get(s2, 0)

            if spread > threshold and self.can_trade(2):
                v1 = self.safe_vol(pos1, vol_size, "ask")
                v2 = self.safe_vol(pos2, max(1, int(vol_size * abs(beta))), "bid")
                if v1 > 0 and v2 > 0:
                    print(f"  [STAT SELL] {s1}/{s2}  spread={spread:.3f}  β={beta:.3f}")
                    exchange.insert_order(s1, price=bk1.bids[0].price, volume=v1, side="ask", order_type="ioc")
                    exchange.insert_order(s2, price=bk2.asks[0].price, volume=v2, side="bid", order_type="ioc")
                    self.log_actions(2)
                    virt_pos[s1] = pos1 - v1
                    virt_pos[s2] = pos2 + v2
            elif -spread > threshold and self.can_trade(2):
                v1 = self.safe_vol(pos1, vol_size, "bid")
                v2 = self.safe_vol(pos2, max(1, int(vol_size * abs(beta))), "ask")
                if v1 > 0 and v2 > 0:
                    print(f"  [STAT BUY ] {s1}/{s2}  spread={spread:.3f}  β={beta:.3f}")
                    exchange.insert_order(s1, price=bk1.asks[0].price, volume=v1, side="bid", order_type="ioc")
                    exchange.insert_order(s2, price=bk2.bids[0].price, volume=v2, side="ask", order_type="ioc")
                    self.log_actions(2)
                    virt_pos[s1] = pos1 + v1
                    virt_pos[s2] = pos2 - v2

    def _strategy_emergency_shed(
        self, books: Dict[str, Any], positions: Dict[str, int], exchange: Any
    ) -> None:
        for stock, amt in positions.items():
            if abs(amt) < self.EMERGENCY_THRESHOLD:
                continue
            bk = books.get(stock)
            if not bk:
                continue
            if amt > 0 and bk.bids and self.can_trade(1):
                print(f"  !!! EMERGENCY SHED SELL {stock} pos={amt} !!!")
                exchange.insert_order(stock, price=bk.bids[0].price, volume=10, side="ask", order_type="ioc")
                self.log_actions(1)
            elif amt < 0 and bk.asks and self.can_trade(1):
                print(f"  !!! EMERGENCY SHED BUY  {stock} pos={amt} !!!")
                exchange.insert_order(stock, price=bk.asks[0].price, volume=10, side="bid", order_type="ioc")
                self.log_actions(1)

    # --- Offline replay (CSV / notebook plots) ------------------------------

    @staticmethod
    def replay_dual_listing(
        main_mids: np.ndarray,
        dual_mids: np.ndarray,
        *,
        dual_threshold: float = DUAL_THRESHOLD,
        exec_buffer: float = EXEC_BUFFER,
        vol_alpha: float = VOL_ALPHA,
        vol_k: float = VOL_K,
        min_scale: float = MIN_VOL_SCALE,
        max_scale: float = MAX_VOL_SCALE,
    ) -> List[Dict[str, Any]]:
        """
        Replay dual-listing *entry* logic on aligned mid arrays (no positions / exits).

        Returns a list of events with keys: ``i``, ``kind`` in
        ``{"short_spread", "long_spread", "lag_buy", "lag_sell"}``.
        """
        m = min(len(main_mids), len(dual_mids))
        main_mids = np.asarray(main_mids[:m], dtype=float)
        dual_mids = np.asarray(dual_mids[:m], dtype=float)
        spread = main_mids - dual_mids

        hist_main: Deque[float] = deque(maxlen=Trader.VOL_LOOKBACK)
        hist_dual: Deque[float] = deque(maxlen=Trader.VOL_LOOKBACK)
        hist_spread: Deque[float] = deque(maxlen=Trader.VOL_LOOKBACK)
        vol_state: Dict[str, float] = {}

        def ewma_vol(sym: str, hist: Deque[float]) -> float:
            h = list(hist)
            if len(h) < 3:
                return 0.0
            rets = np.diff(np.array(h, dtype=float))
            raw = float(np.std(rets))
            prev = vol_state.get(sym, raw)
            v = vol_alpha * raw + (1.0 - vol_alpha) * prev
            vol_state[sym] = v
            return v

        def scale_from_vol(v: float) -> float:
            return max(min_scale, min(max_scale, 1.0 + v * vol_k))

        events: List[Dict[str, Any]] = []
        for i in range(m):
            hist_main.append(float(main_mids[i]))
            hist_dual.append(float(dual_mids[i]))
            hist_spread.append(float(spread[i]))

            vol = ewma_vol("dual", hist_dual)
            scale = scale_from_vol(vol)
            thr = dual_threshold * scale * exec_buffer

            main_hist = np.array(hist_main, dtype=float)
            dual_hist = np.array(hist_dual, dtype=float)
            lagging = False
            if len(main_hist) >= 5 and len(dual_hist) >= 5:
                main_move = float(main_hist[-1] - main_hist[-5])
                dual_move = float(dual_hist[-1] - dual_hist[-5])
                if main_move > thr and abs(dual_move) < abs(main_move) * 0.5:
                    lagging = True
                elif main_move < -thr and abs(dual_move) < abs(main_move) * 0.5:
                    lagging = True

            s_now = float(spread[i])
            if s_now > thr:
                events.append({"i": i, "kind": "short_spread", "spread": s_now, "threshold": thr})
            elif -s_now > thr:
                events.append({"i": i, "kind": "long_spread", "spread": s_now, "threshold": thr})
            if lagging and len(main_hist) >= 5:
                mm = float(main_hist[-1] - main_hist[-5])
                if mm > 0:
                    events.append({"i": i, "kind": "lag_buy", "spread": s_now, "threshold": thr})
                else:
                    events.append({"i": i, "kind": "lag_sell", "spread": s_now, "threshold": thr})
        return events


def main() -> None:
    from optibook.synchronous_client import Exchange

    exchange = Exchange()
    Trader().run(exchange)


if __name__ == "__main__":
    main()
