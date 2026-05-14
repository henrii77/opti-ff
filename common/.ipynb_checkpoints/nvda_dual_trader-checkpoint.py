"""
NVDA / NVDA_DUAL cross-book arbitrage using Optibook's synchronous Exchange client.

Spread logic is isolated in SpreadAnalyzer for unit testing; execution is in OrderExecutor.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Literal, Optional

if TYPE_CHECKING:
    from optibook.synchronous_client import Exchange


NVDA_ID = "NVDA"
NVDA_DUAL_ID = "NVDA_DUAL"


class ArbitrageDirection(Enum):
    BUY_NVDA_SELL_DUAL = "buy_nvda_sell_dual"
    BUY_DUAL_SELL_NVDA = "buy_dual_sell_nvda"


@dataclass(frozen=True)
class TopOfBook:
    bid_price: Optional[float]
    bid_volume: Optional[int]
    ask_price: Optional[float]
    ask_volume: Optional[int]


@dataclass(frozen=True)
class ArbitrageOpportunity:
    direction: ArbitrageDirection
    volume: int
    edge_per_share: float
    buy_instrument_id: str
    buy_price: float
    sell_instrument_id: str
    sell_price: float


@dataclass
class TraderConfig:
    min_edge: float = 0.0
    max_order_volume: int = 10
    poll_interval: float = 1.0
    primary_leg: Literal["buy_first", "sell_first"] = "buy_first"


def top_of_book_from_exchange_book(book) -> TopOfBook:
    """Build TopOfBook from exchange.get_last_price_book(...) result."""
    bid_price = book.bids[0].price if book.bids else None
    bid_volume = book.bids[0].volume if book.bids else None
    ask_price = book.asks[0].price if book.asks else None
    ask_volume = book.asks[0].volume if book.asks else None
    return TopOfBook(
        bid_price=bid_price,
        bid_volume=bid_volume,
        ask_price=ask_price,
        ask_volume=ask_volume,
    )


class SpreadAnalyzer:
    """Pure logic: given NVDA and NVDA_DUAL tops of book, yield an opportunity or None."""

    @staticmethod
    def analyze(
        nvda: TopOfBook,
        dual: TopOfBook,
        cfg: TraderConfig,
    ) -> Optional[ArbitrageOpportunity]:
        # Direction 1: buy NVDA @ ask, sell NVDA_DUAL @ bid — edge = dual_bid - nvda_ask
        opp1: Optional[ArbitrageOpportunity] = None
        if (
            dual.bid_price is not None
            and nvda.ask_price is not None
            and dual.bid_volume is not None
            and nvda.ask_volume is not None
        ):
            edge1 = dual.bid_price - nvda.ask_price
            if edge1 > cfg.min_edge:
                vol = min(nvda.ask_volume, dual.bid_volume, cfg.max_order_volume)
                if vol >= 1:
                    opp1 = ArbitrageOpportunity(
                        direction=ArbitrageDirection.BUY_NVDA_SELL_DUAL,
                        volume=int(vol),
                        edge_per_share=edge1,
                        buy_instrument_id=NVDA_ID,
                        buy_price=nvda.ask_price,
                        sell_instrument_id=NVDA_DUAL_ID,
                        sell_price=dual.bid_price,
                    )

        # Direction 2: buy NVDA_DUAL @ ask, sell NVDA @ bid — edge = nvda_bid - dual_ask
        opp2: Optional[ArbitrageOpportunity] = None
        if (
            nvda.bid_price is not None
            and dual.ask_price is not None
            and nvda.bid_volume is not None
            and dual.ask_volume is not None
        ):
            edge2 = nvda.bid_price - dual.ask_price
            if edge2 > cfg.min_edge:
                vol = min(dual.ask_volume, nvda.bid_volume, cfg.max_order_volume)
                if vol >= 1:
                    opp2 = ArbitrageOpportunity(
                        direction=ArbitrageDirection.BUY_DUAL_SELL_NVDA,
                        volume=int(vol),
                        edge_per_share=edge2,
                        buy_instrument_id=NVDA_DUAL_ID,
                        buy_price=dual.ask_price,
                        sell_instrument_id=NVDA_ID,
                        sell_price=nvda.bid_price,
                    )

        if opp1 is None:
            return opp2
        if opp2 is None:
            return opp1
        # Both valid: prefer larger per-share edge (then larger total notional as tie-break)
        if opp1.edge_per_share > opp2.edge_per_share:
            return opp1
        if opp2.edge_per_share > opp1.edge_per_share:
            return opp2
        tie1 = opp1.edge_per_share * opp1.volume
        tie2 = opp2.edge_per_share * opp2.volume
        return opp1 if tie1 >= tie2 else opp2


class OrderExecutor:
    """Sends paired IOC orders for an opportunity (not atomic; leg risk applies)."""

    def __init__(self, exchange: "Exchange") -> None:
        self._exchange = exchange

    def execute(self, opp: ArbitrageOpportunity, cfg: TraderConfig) -> None:
        buy_first = cfg.primary_leg == "buy_first"
        buy_kwargs = dict(
            instrument_id=opp.buy_instrument_id,
            price=opp.buy_price,
            volume=opp.volume,
            side="bid",
            order_type="ioc",
        )
        sell_kwargs = dict(
            instrument_id=opp.sell_instrument_id,
            price=opp.sell_price,
            volume=opp.volume,
            side="ask",
            order_type="ioc",
        )
        if buy_first:
            self._exchange.insert_order(**buy_kwargs)
            self._exchange.insert_order(**sell_kwargs)
        else:
            self._exchange.insert_order(**sell_kwargs)
            self._exchange.insert_order(**buy_kwargs)


class NvdaDualTrader:
    def __init__(
        self,
        exchange: "Exchange",
        cfg: Optional[TraderConfig] = None,
        nvda_id: str = NVDA_ID,
        dual_id: str = NVDA_DUAL_ID,
    ) -> None:
        self._exchange = exchange
        self.cfg = cfg or TraderConfig()
        self._nvda_id = nvda_id
        self._dual_id = dual_id
        self._analyzer = SpreadAnalyzer()
        self._executor = OrderExecutor(exchange)

    def connect(self) -> None:
        self._exchange.connect()

    def _fetch_books(self) -> tuple[TopOfBook, TopOfBook]:
        nvda_book = self._exchange.get_last_price_book(self._nvda_id)
        dual_book = self._exchange.get_last_price_book(self._dual_id)
        return (
            top_of_book_from_exchange_book(nvda_book),
            top_of_book_from_exchange_book(dual_book),
        )

    def step(self) -> Optional[ArbitrageOpportunity]:
        nvda_tob, dual_tob = self._fetch_books()
        opp = self._analyzer.analyze(nvda_tob, dual_tob, self.cfg)
        if opp is None:
            return None
        self._executor.execute(opp, self.cfg)
        return opp

    def run_forever(self, poll_interval: Optional[float] = None) -> None:
        interval = (
            poll_interval if poll_interval is not None else self.cfg.poll_interval
        )
        while True:
            loop_start = time.time()
            self.step()
            elapsed = time.time() - loop_start
            if elapsed < interval:
                time.sleep(interval - elapsed)


def main() -> None:
    from optibook.synchronous_client import Exchange

    cfg = TraderConfig()
    exchange = Exchange()
    trader = NvdaDualTrader(exchange, cfg)
    trader.connect()
    print(
        f"Connected. NVDA/NVDA_DUAL arb — min_edge={cfg.min_edge}, "
        f"max_vol={cfg.max_order_volume}, poll={cfg.poll_interval}s. Ctrl+C to stop."
    )
    try:
        trader.run_forever()
    except KeyboardInterrupt:
        print("\nTrader stopped.")


if __name__ == "__main__":
    main()
