from __future__ import annotations
import math
import pandas as pd
import numpy as np


class Backtester:
    """
    Simple 1-position-at-a-time backtester for SL/TP bracket trades.

    Signals are read from `buy_col` and can represent either:
      - a boolean signal (use entry at close/next open depending on `entry_mode`), or
      - an explicit entry price (if `buy_col` is a valid positive number).

    Parameters
    ----------
    entry_mode : {"close", "next_open"}
        If `buy_col` is just a boolean signal, choose whether the entry is at the same-bar close
        or at the next bar's open.
    both_hit : {"sl_first", "tp_first"}
        Tie-break rule when a bar's range touches both SL and TP.
    """

    def __init__(self, entry_mode: str = "close", both_hit: str = "sl_first"):
        self.entry_mode = entry_mode
        self.both_hit = both_hit


    @staticmethod
    def _signed_range(row, S: int, *, low_col: str, high_col: str):
        # Use a sign-flip trick so long/short checks become the same comparisons.
        a = S * float(getattr(row, low_col))
        b = S * float(getattr(row, high_col))
        return (a, b) if a <= b else (b, a)  # pmin, pmax

    def _choose_reason(self, hit_sl: bool, hit_tp: bool) -> str:
        # If both are hit in one bar, pick the configured tie-break.
        if hit_sl and hit_tp:
            return "sl" if self.both_hit == "sl_first" else "tp"
        return "sl" if hit_sl else "tp"

    def _open_position(
        self,
        i: int,
        row,
        d: pd.DataFrame,
        *,
        time_col: str,
        open_col: str,
        close_col: str,
        buy_col: str,
        sl_col: str,
        tp_col: str,
    ):
        if not bool(getattr(row, buy_col)):
            return None

        sl = getattr(row, sl_col)
        tp = getattr(row, tp_col)
        if pd.isna(sl) or pd.isna(tp):
            return None
        sl = float(sl)
        tp = float(tp)
    
        S = np.sign(tp - sl)
        if S == 0:
            return None

        # Entry is always at close or next bar open (no explicit-price mode).
        if self.entry_mode == "close":
            entry = float(getattr(row, close_col))
            entry_time = getattr(row, time_col)
        else:
            if i + 1 >= len(d):
                return None
            entry = float(d.loc[i + 1, open_col])
            entry_time = d.loc[i + 1, time_col]
    
        entry_s = S * entry
        sl_s = S * sl
        tp_s = S * tp
    
        if not (sl_s < entry_s < tp_s):
            return None
    
        denom = (entry - sl)
        if denom == 0:
            return None
    
        return {
            "S": S,
            "entry": entry,
            "entry_time": entry_time,
            "denom": denom,
            "sl": sl,
            "tp": tp,
            "sl_s": sl_s,
            "tp_s": tp_s,
        }


    def _try_exit(
        self,
        row,
        pos: dict,
        *,
        time_col: str,
        low_col: str,
        high_col: str,
    ):
        S = pos["S"]
        pmin, pmax = self._signed_range(row, S, low_col=low_col, high_col=high_col)
        hit_sl = (pmin <= pos["sl_s"])
        hit_tp = (pmax >= pos["tp_s"])
        if not (hit_sl or hit_tp):
            return None

        reason = self._choose_reason(hit_sl, hit_tp)
        exit_s = pos["sl_s"] if reason == "sl" else pos["tp_s"]
        exit_price = exit_s / S
        exit_time = getattr(row, time_col)
        return reason, float(exit_price), exit_time

    def run(
        self,
        df: pd.DataFrame,
        *,
        time_col="time",
        open_col="open",
        high_col="high",
        low_col="low",
        close_col="close",
        buy_col="buy",
        sl_col="sl",
        tp_col="tp",
    ) -> pd.DataFrame:
        """
        Simulate trades over OHLCV-like bars with optional explicit entry prices.

        The input must contain the columns specified by the *_col parameters.
        At most one position is held at a time:
          - exit is checked first on each bar (SL/TP within the bar range),
          - then a new position may be opened if flat.

        Returns
        -------
        pd.DataFrame
            One row per completed trade with columns:
            ["time", "won", "sl", "tp", "r_mult", "reason"].
        """
        required = [time_col, open_col, high_col, low_col, close_col, buy_col, sl_col, tp_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        d = df.reset_index(drop=True)

        trades = []
        pos = None  # dict with S, entry, denom, sl, tp, sl_s, tp_s, entry_time

        for i, row in enumerate(d.itertuples(index=False)):
            # EXIT first (so you can't "re-enter" on the same bar after hitting a bracket)
            if pos is not None:
                out = self._try_exit(
                    row, pos,
                    time_col=time_col,
                    low_col=low_col,
                    high_col=high_col,
                )
                if out is not None:
                    reason, exit_price, exit_time = out
                    # R-multiple: normalized by the initial risk (entry - SL)
                    r_mult = (exit_price - pos["entry"]) / pos["denom"]
                    trades.append(
                        {
                            "time": exit_time,
                            "won": bool(r_mult > 0),
                            "sl": pos["sl"],
                            "tp": pos["tp"],
                            "r_mult": float(r_mult),
                            "reason": reason,
                        }
                    )
                    pos = None
                    continue

            # ENTRY if flat
            if pos is None:
                pos = self._open_position(
                    i, row, d,
                    time_col=time_col,
                    open_col=open_col,
                    close_col=close_col,
                    buy_col=buy_col,
                    sl_col=sl_col,
                    tp_col=tp_col,
                )

        # End-of-data close: mark-to-market the last bar close if still in a position.
        if pos is not None:
            last_close = float(d.loc[len(d) - 1, close_col])
            last_time = d.loc[len(d) - 1, time_col]
            r_mult = (last_close - pos["entry"]) / pos["denom"]
            trades.append(
                {
                    "time": last_time,
                    "won": bool(r_mult > 0),
                    "sl": pos["sl"],
                    "tp": pos["tp"],
                    "r_mult": float(r_mult),
                    "reason": "eod",
                }
            )

        return pd.DataFrame(trades, columns=["time", "won", "sl", "tp", "r_mult", "reason"])

    @staticmethod
    def evaluate(trades_df: pd.DataFrame, initial_bankroll: float, risk_fraction: float = 0.01) -> pd.DataFrame:
        """
        Convert a trade list (with R-multiples) into an equity curve via compounding.

        Each trade updates bankroll by:
            bankroll *= (1 + risk_fraction * r_mult)

        Returns
        -------
        pd.DataFrame
            Columns: ["time", "bankroll"] aligned to each trade's close time.
        """
        if trades_df is None or len(trades_df) == 0:
            return pd.DataFrame(columns=["time", "bankroll"])

        m = trades_df["r_mult"].astype(float).to_numpy()
        factors = 1.0 + float(risk_fraction) * m
        bankroll = float(initial_bankroll) * pd.Series(factors).cumprod()

        return pd.DataFrame({"time": trades_df["time"].values, "bankroll": bankroll.values})


if __name__ == "__main__":
    bars = [
        {"time": "t0", "open": 100, "high": 101, "low":  99, "close": 100, "buy": False, "sl": None, "tp": None},
        {"time": "t1", "open": 100, "high": 102, "low":  98, "close": 101, "buy": True,  "sl": 95,   "tp": 110},  # long signal
        {"time": "t2", "open": 101, "high": 110, "low":  90, "close": 104, "buy": False, "sl": None, "tp": None},  # long SL+TP -> SL first
        {"time": "t3", "open": 104, "high": 106, "low": 103, "close": 104, "buy": False, "sl": None, "tp": None},
        {"time": "t4", "open": 104, "high": 108, "low": 100, "close": 104, "buy": True,  "sl": 114,  "tp": 84},   # short signal
        {"time": "t5", "open": 104, "high": 109, "low":  80, "close":  92, "buy": False, "sl": None, "tp": None},  # short TP
        {"time": "t6", "open":  92, "high":  95, "low":  90, "close":  94, "buy": False, "sl": None, "tp": None},
    ]
    df = pd.DataFrame(bars)

    bt = Backtester(entry_mode="close", both_hit="sl_first")
    trades = bt.run(df)
    print("TRADES\n", trades)

    curve = bt.evaluate(trades, initial_bankroll=10_000, risk_fraction=0.01)
    print("\nCURVE\n", curve)
