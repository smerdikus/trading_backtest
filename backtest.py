"""
Minimal OHLC backtest engine (long + short) with SL/TP intrabar execution.

Key ideas we agreed on:
- Long/short is inferred from SL/TP ordering:
    S = sign(tp - sl)  ->  +1 long (tp>sl), -1 short (tp<sl)
- Use "signed space" to unify logic:
    p' = S * p
    pmin = min(S*low, S*high), pmax = max(S*low, S*high)
    SL hit: pmin <= sl'
    TP hit: pmax >= tp'
- One position at a time.
- Entry is "market" on bar close or next bar open (or optional explicit entry price in `buy`).

Input bars: list[dict] with at least:
  time, open, high, low, close
  buy (bool/int/float)   # signal; if numeric > 0 => explicit entry price
  sl (float), tp (float)

Output:
  trades: list[Trade]
  final_bankroll: float
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Literal, Optional, Tuple
import math


@dataclass(frozen=True)
class Trade:
    side: int  # +1 long, -1 short
    entry_time: Any
    exit_time: Any
    entry: float
    exit: float
    sl: float
    tp: float
    qty: float
    pnl: float
    win: bool
    reason: Literal["sl", "tp", "eod"]


@dataclass
class _Position:
    S: int
    entry_time: Any
    entry_raw: float
    entry_s: float
    sl_raw: float
    tp_raw: float
    sl_s: float
    tp_s: float


class Backtester:
    def __init__(
        self,
        *,
        qty: float = 1.0,
        entry_mode: Literal["close", "next_open"] = "close",
        both_hit: Literal["sl_first", "tp_first"] = "sl_first",
        time_col: str = "time",
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        buy_col: str = "buy",
        sl_col: str = "sl",
        tp_col: str = "tp",
    ):
        if qty <= 0:
            raise ValueError("qty must be > 0")
        self.qty = float(qty)
        self.entry_mode = entry_mode
        self.both_hit = both_hit

        self.time_col = time_col
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.buy_col = buy_col
        self.sl_col = sl_col
        self.tp_col = tp_col

    # -------- core helpers (signed-space) --------
    def _signed_range(self, bar: Dict[str, Any], S: int) -> Tuple[float, float]:
        lo = float(bar[self.low_col])
        hi = float(bar[self.high_col])
        a = S * lo
        b = S * hi
        return (a, b) if a <= b else (b, a)  # pmin, pmax

    def _choose_reason(self, hit_sl: bool, hit_tp: bool) -> Literal["sl", "tp"]:
        if hit_sl and hit_tp:
            return "sl" if self.both_hit == "sl_first" else "tp"
        return "sl" if hit_sl else "tp"

    def _is_explicit_price(self, x: Any) -> bool:
        # allow numeric entry in buy (e.g., 123.45); exclude bools
        if isinstance(x, bool) or x is None:
            return False
        try:
            v = float(x)
            return math.isfinite(v) and v > 0
        except Exception:
            return False

    # -------- order/position logic --------
    def _make_position(self, bars: List[Dict[str, Any]], i: int) -> Optional[_Position]:
        bar = bars[i]

        buy = bar.get(self.buy_col, False)
        if not buy:
            return None

        sl = bar.get(self.sl_col, None)
        tp = bar.get(self.tp_col, None)
        if sl is None or tp is None:
            return None

        sl = float(sl)
        tp = float(tp)

        diff = tp - sl
        if diff == 0:
            return None
        S = 1 if diff > 0 else -1  # +1 long, -1 short

        # entry price
        if self._is_explicit_price(buy):
            entry_raw = float(buy)
            entry_time = bar[self.time_col]
        else:
            if self.entry_mode == "close":
                entry_raw = float(bar[self.close_col])
                entry_time = bar[self.time_col]
            else:  # next_open
                if i + 1 >= len(bars):
                    return None
                entry_raw = float(bars[i + 1][self.open_col])
                entry_time = bars[i + 1][self.time_col]

        entry_s = S * entry_raw
        sl_s = S * sl
        tp_s = S * tp

        # unified validity check: SL' < entry' < TP'
        if not (sl_s < entry_s < tp_s):
            return None

        return _Position(
            S=S,
            entry_time=entry_time,
            entry_raw=entry_raw,
            entry_s=entry_s,
            sl_raw=sl,
            tp_raw=tp,
            sl_s=sl_s,
            tp_s=tp_s,
        )

    def _try_exit(self, bars: List[Dict[str, Any]], i: int, pos: _Position) -> Optional[Tuple[str, float]]:
        bar = bars[i]
        pmin, pmax = self._signed_range(bar, pos.S)

        hit_sl = (pmin <= pos.sl_s)
        hit_tp = (pmax >= pos.tp_s)
        if not (hit_sl or hit_tp):
            return None

        reason = self._choose_reason(hit_sl, hit_tp)
        exit_s = pos.sl_s if reason == "sl" else pos.tp_s
        exit_raw = exit_s / pos.S
        return reason, float(exit_raw)

    # -------- public API --------
    def run(self, bars: List[Dict[str, Any]], bankroll: float) -> Tuple[List[Trade], float]:
        if bankroll <= 0:
            raise ValueError("bankroll must be > 0")
        if not bars:
            return [], float(bankroll)

        # minimal schema check (first row)
        required = [self.time_col, self.open_col, self.high_col, self.low_col, self.close_col]
        for k in required:
            if k not in bars[0]:
                raise ValueError(f"Missing required column '{k}' in bars[0]")

        trades: List[Trade] = []
        pos: Optional[_Position] = None

        for i in range(len(bars)):
            bar = bars[i]

            # 1) Exit has priority
            if pos is not None:
                out = self._try_exit(bars, i, pos)
                if out is not None:
                    reason, exit_raw = out
                    pnl = (exit_raw - pos.entry_raw) * self.qty * pos.S
                    bankroll += pnl

                    trades.append(
                        Trade(
                            side=pos.S,
                            entry_time=pos.entry_time,
                            exit_time=bar[self.time_col],
                            entry=pos.entry_raw,
                            exit=exit_raw,
                            sl=pos.sl_raw,
                            tp=pos.tp_raw,
                            qty=self.qty,
                            pnl=pnl,
                            win=pnl > 0,
                            reason=reason,  # "sl" or "tp"
                        )
                    )
                    pos = None
                    continue  # don't open a new trade in the same bar

            # 2) Entry (only if flat)
            if pos is None:
                pos = self._make_position(bars, i)

        # 3) EOD close for open position
        if pos is not None:
            last = bars[-1]
            exit_raw = float(last[self.close_col])
            pnl = (exit_raw - pos.entry_raw) * self.qty * pos.S
            bankroll += pnl
            trades.append(
                Trade(
                    side=pos.S,
                    entry_time=pos.entry_time,
                    exit_time=last[self.time_col],
                    entry=pos.entry_raw,
                    exit=exit_raw,
                    sl=pos.sl_raw,
                    tp=pos.tp_raw,
                    qty=self.qty,
                    pnl=pnl,
                    win=pnl > 0,
                    reason="eod",
                )
            )

        return trades, float(bankroll)

    @staticmethod
    def trades_to_dicts(trades: List[Trade]) -> List[Dict[str, Any]]:
        return [asdict(t) for t in trades]


if __name__ == "__main__":
    # Tiny demo
    bars = [
        {"time": "t0", "open": 100, "high": 101, "low":  99, "close": 100, "buy": False, "sl": None, "tp": None},
        {"time": "t1", "open": 100, "high": 102, "low":  98, "close": 101, "buy": True,  "sl": 95,   "tp": 110},  # long signal (tp>sl)
        {"time": "t2", "open": 101, "high": 110, "low":  90, "close": 104, "buy": False, "sl": None, "tp": None},
        {"time": "t3", "open": 104, "high": 111, "low": 103, "close": 110, "buy": False, "sl": None, "tp": None},  # TP hit
    ]

    bt = Backtester(qty=1.0, entry_mode="close", both_hit="sl_first")
    trades, final_bankroll = bt.run(bars, bankroll=10_000)

    print("Final bankroll:", final_bankroll)
    for t in trades:
        print(t)
