from __future__ import annotations
import math
import pandas as pd


class Backtester:
    """
    Pandas backtest engine:
    - long/short inferred from SL/TP ordering: S = sign(tp - sl)
    - intrabar SL/TP using high/low in signed space
    - 1 position at a time
    - entry = close | next_open OR explicit entry price if buy is numeric (>0)
    - run(df) returns trades_df with: time, won, sl, tp, r_mult, reason
    - evaluate(trades_df) reconstructs bankroll curve from r_mult only
    """

    def __init__(self, entry_mode: str = "close", both_hit: str = "sl_first"):
        self.entry_mode = entry_mode
        self.both_hit = both_hit

    @staticmethod
    def _sgn(x: float) -> int:
        return 1 if x > 0 else (-1 if x < 0 else 0)

    @staticmethod
    def _is_price(x) -> bool:
        # numeric positive, but not bool
        if x is None:
            return False
        if isinstance(x, bool):
            return False
        try:
            v = float(x)
            return math.isfinite(v) and v > 0
        except Exception:
            return False

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
        d = df.reset_index(drop=True)

        trades = []
        pos = None  # dict with S, entry, denom, sl, tp, sl_s, tp_s, entry_time

        def signed_range(row, S: int):
            a = S * float(getattr(row, low_col))
            b = S * float(getattr(row, high_col))
            return (a, b) if a <= b else (b, a)

        def choose_reason(hit_sl: bool, hit_tp: bool) -> str:
            if hit_sl and hit_tp:
                return "sl" if self.both_hit == "sl_first" else "tp"
            return "sl" if hit_sl else "tp"

        def open_position(i: int, row):
            buy = getattr(row, buy_col)
            if not bool(buy):
                return None

            sl = getattr(row, sl_col)
            tp = getattr(row, tp_col)
            if pd.isna(sl) or pd.isna(tp):
                return None

            sl = float(sl)
            tp = float(tp)

            S = self._sgn(tp - sl)  # +1 long, -1 short
            if S == 0:
                return None

            # entry price
            if self._is_price(buy):
                entry = float(buy)
                entry_time = getattr(row, time_col)
            else:
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

            # unified validity: SL' < entry' < TP'
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

        def try_exit(i: int, row, pos):
            S = pos["S"]
            pmin, pmax = signed_range(row, S)
            hit_sl = (pmin <= pos["sl_s"])
            hit_tp = (pmax >= pos["tp_s"])
            if not (hit_sl or hit_tp):
                return None
            reason = choose_reason(hit_sl, hit_tp)
            exit_s = pos["sl_s"] if reason == "sl" else pos["tp_s"]
            exit_price = exit_s / S
            exit_time = getattr(row, time_col)
            return reason, float(exit_price), exit_time

        # iterate rows (stateful engine)
        for i, row in enumerate(d.itertuples(index=False)):
            # EXIT first
            if pos is not None:
                out = try_exit(i, row, pos)
                if out is not None:
                    reason, exit_price, exit_time = out
                    r_mult = (exit_price - pos["entry"]) / pos["denom"]
                    trades.append(
                        {
                            "time": exit_time,
                            "won": bool(r_mult > 0),
                            "sl": pos["sl"],
                            "tp": pos["tp"],
                            "r_mult": float(r_mult),
                            "reason": reason,  # sl|tp
                        }
                    )
                    pos = None
                    continue

            # ENTRY if flat
            if pos is None:
                pos = open_position(i, row)

        # EOD close
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
        bankroll_{k+1} = bankroll_k * (1 + risk_fraction * r_mult_k)
        => bankroll = initial * cumprod(1 + risk_fraction * r_mult)
        """
        if trades_df is None or len(trades_df) == 0:
            return pd.DataFrame(columns=["time", "bankroll"])

        m = trades_df["r_mult"].astype(float).to_numpy()
        factors = 1.0 + float(risk_fraction) * m
        bankroll = float(initial_bankroll) * pd.Series(factors).cumprod()

        out = pd.DataFrame({"time": trades_df["time"].values, "bankroll": bankroll.values})
        return out


if __name__ == "__main__":
    bars = [
        {"time": "t0", "open": 100, "high": 101, "low":  99, "close": 100, "buy": False, "sl": None, "tp": None},
        {"time": "t1", "open": 100, "high": 102, "low":  98, "close": 101, "buy": True,  "sl": 95,   "tp": 110},  # long signal
        {"time": "t2", "open": 101, "high": 110, "low":  90, "close": 104, "buy": False, "sl": None, "tp": None},  # long SL+TP hit -> SL first
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

