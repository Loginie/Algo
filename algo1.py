#!/usr/bin/env python3
"""
backtest_tvdatafeed_backtesting_ema12_200_with_reports.py

Requirements:
  pip install backtesting tvdatafeed pandas pandas-ta matplotlib

Produces:
  - per-trade CSV: bt_outputs/<symbol>_trades_normalized.csv
  - backtest stats CSV: bt_outputs/<symbol>_stats.csv
  - summary CSV: bt_outputs/<symbol>_summary.csv
  - equity curve PNG: bt_outputs/<symbol>_equity_static.png
  - drawdown PNG: bt_outputs/<symbol>_drawdown.png
  - PnL histogram PNG: bt_outputs/<symbol>_pnl_hist.png
"""
import os
import warnings
from math import floor
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt
from tvDatafeed import TvDatafeed, Interval
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

warnings.filterwarnings("ignore")

# ----------------------------- CONFIG (edit these) -----------------------------
TV_USERNAME = ""  # TradingView username (optional)
TV_PASSWORD = ""  # TradingView password (optional)

SYMBOLS_CSV = "nifty500_constituents.csv"
TARGET_SYMBOL = "MRF"    # set to empty "" to run all or use SYMBOLS fallback
SYMBOLS = ["MRF"]

TV_INTERVAL = Interval.in_4_hour
N_BARS = 7000  # ~3 years of 4h bars

EMA_FAST = 12
EMA_SLOW = 200

STOP_LOSS_PCT = 0.03
TAKE_PROFIT_PCT = 0.09

RISK_PER_TRADE = 0.25

STARTING_CASH = 100_000
COMMISSION = 0.002
SPREAD = 0.001

MINIMUM_BARS = 100
OUTPUT_DIR = "bt_outputs"
# -----------------------------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ----------------------------- Helpers & Fetcher --------------------------------
def init_tv() -> TvDatafeed:
    try:
        if TV_USERNAME and TV_PASSWORD:
            tv = TvDatafeed(TV_USERNAME, TV_PASSWORD)
        else:
            tv = TvDatafeed()
            print("Warning: No TradingView credentials supplied. Data requests may be limited.")
        return tv
    except Exception as e:
        print("Error initializing TvDatafeed:", e)
        return TvDatafeed()


tv = init_tv()


def parse_symbol_exchange(symbol: str) -> Tuple[str, str]:
    if ":" in symbol:
        exchange, sym = symbol.split(":", 1)
        return sym.strip(), exchange.strip()
    return symbol.strip(), "NSE"


def fetch_stock_data(symbol: str, n_bars: int = N_BARS, exchange: Optional[str] = None) -> Optional[pd.DataFrame]:
    try:
        sym, parsed_exchange = parse_symbol_exchange(symbol)
        use_exchange = exchange or parsed_exchange
        df = tv.get_hist(symbol=sym, exchange=use_exchange, interval=TV_INTERVAL, n_bars=n_bars)
        if df is None or df.empty:
            print(f"[fetch] No data for {symbol} ({sym}@{use_exchange})")
            return None
        df = df.copy()
        # Map common lower-case cols to desired names
        lower_cols = {c.lower(): c for c in df.columns}
        col_map = {}
        for want, names in [("Open", ["open"]), ("High", ["high"]), ("Low", ["low"]), ("Close", ["close"]), ("Volume", ["volume"])]:
            for n in names:
                if n in lower_cols:
                    col_map[lower_cols[n]] = want
                    break
        df = df.rename(columns=col_map)
        if not all(c in df.columns for c in ["Open", "High", "Low", "Close"]):
            print(f"[fetch] Missing OHLC for {symbol}: cols={list(df.columns)}")
            return None
        if "Volume" not in df.columns:
            df["Volume"] = pd.NA
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index(ascending=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Open", "High", "Low", "Close"])
        return df
    except Exception as e:
        print(f"[fetch] Error fetching {symbol}: {e}")
        return None


def load_symbols_from_csv(path: str) -> List[str]:
    df = pd.read_csv(path)
    for col in ["Symbol", "SYMBOL", "symbol"]:
        if col in df.columns:
            return df[col].astype(str).str.strip().tolist()
    return df.iloc[:, 0].astype(str).str.strip().tolist()


# ----------------------------- Strategy -----------------------------------------
class Ema12Ema200Strategy(Strategy):
    def init(self):
        self.ema12 = self.I(lambda s: ta.ema(s, length=EMA_FAST), self.data.Close)
        self.ema200 = self.I(lambda s: ta.ema(s, length=EMA_SLOW), self.data.Close)

    def next(self):
        # Not enough data
        if pd.isna(self.ema12[-1]) or pd.isna(self.ema200[-1]):
            return

        # Long entry
        if crossover(self.ema12, self.ema200):
            if self.position:
                self.position.close()
            entry_price = self.data.Close[-1]
            equity = getattr(self, "equity", STARTING_CASH)
            money_at_risk = equity * RISK_PER_TRADE
            shares = max(1, floor(money_at_risk / (entry_price * STOP_LOSS_PCT)))
            try:
                self.buy(size=shares, sl=STOP_LOSS_PCT, tp=TAKE_PROFIT_PCT)
            except TypeError:
                self.buy(size=shares)

        # Short entry
        elif crossover(self.ema200, self.ema12):
            if self.position:
                self.position.close()
            entry_price = self.data.Close[-1]
            equity = getattr(self, "equity", STARTING_CASH)
            money_at_risk = equity * RISK_PER_TRADE
            shares = max(1, floor(money_at_risk / (entry_price * STOP_LOSS_PCT)))
            try:
                self.sell(size=shares, sl=STOP_LOSS_PCT, tp=TAKE_PROFIT_PCT)
            except TypeError:
                self.sell(size=shares)


# ----------------------------- Reporting Helpers -------------------------------
def normalize_trades_dataframe(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Accepts backtesting._trades DataFrame (various schema across versions)
    and returns a normalized DataFrame with these columns:
    ['Entry Time','Exit Time','Entry Price','Exit Price','Size','Side','Gross PnL','Net PnL','Return %','Duration Bars']
    """
    if trades_df is None or trades_df.empty:
        return pd.DataFrame(columns=['Entry Time', 'Exit Time', 'Entry Price', 'Exit Price',
                                     'Size', 'Side', 'Gross PnL', 'Net PnL', 'Return %', 'Duration Bars'])
    df = trades_df.copy()

    # Standardize common columns: try several names
    def col_choice(candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None

    # possible names in different backtesting.py versions
    en_time_col = col_choice(['Entry Time', 'EntryTime', 'EntryDate', 'Entry'])
    ex_time_col = col_choice(['Exit Time', 'ExitTime', 'ExitDate', 'Exit'])
    en_price_col = col_choice(['Entry Price', 'EntryPrice', 'Entry_Price', 'EntryPrice'])
    ex_price_col = col_choice(['Exit Price', 'ExitPrice', 'Exit_Price', 'ExitPrice'])
    size_col = col_choice(['Size', 'size'])
    side_col = col_choice(['Side', 'side', 'Direction'])
    profit_col = col_choice(['Profit', 'PnL', 'pnl', 'Profit INR', 'Profit ()', 'Profit (%)', 'Return'])

    # Some versions store 'Return [%]' etc inside stats but trades may have 'Return (%)'
    # Build normalized columns
    # Entry/Exit time: try index if none found
    entry_times = df[en_time_col] if en_time_col else (df.index if hasattr(df.index, 'to_series') else pd.Series([pd.NaT]*len(df)))
    exit_times = df[ex_time_col] if ex_time_col else pd.Series([pd.NaT]*len(df))

    entry_prices = df[en_price_col] if en_price_col else pd.Series([np.nan]*len(df))
    exit_prices = df[ex_price_col] if ex_price_col else pd.Series([np.nan]*len(df))
    sizes = df[size_col] if size_col else pd.Series([np.nan]*len(df))
    sides = df[side_col] if side_col else pd.Series([np.nan]*len(df))

    # Compute gross pnl if not present:
    if profit_col and profit_col in df.columns:
        gross_pnl = df[profit_col]
    else:
        # derive
        gross_pnl = pd.Series(np.nan, index=df.index)
        # long: (exit-entry)*size ; short: (entry-exit)*size
        for idx in df.index:
            ep = entry_prices.loc[idx] if idx in entry_prices.index else entry_prices.iloc[idx] if len(entry_prices)>idx else np.nan
            xp = exit_prices.loc[idx] if idx in exit_prices.index else exit_prices.iloc[idx] if len(exit_prices)>idx else np.nan
            s = sizes.loc[idx] if idx in sizes.index else (sizes.iloc[idx] if len(sizes)>idx else np.nan)
            sd = sides.loc[idx] if idx in sides.index else (sides.iloc[idx] if len(sides)>idx else np.nan)
            try:
                if pd.isna(ep) or pd.isna(xp) or pd.isna(s):
                    gross = np.nan
                else:
                    if isinstance(sd, str) and sd.lower().startswith('s'):
                        # short
                        gross = (ep - xp) * float(s)
                    else:
                        # assume long
                        gross = (xp - ep) * float(s)
            except Exception:
                gross = np.nan
            gross_pnl.loc[idx] = gross

    # Net PnL: some trades df includes commission & slippage applied already; we can't reliably subtract commission unless columns exist.
    # If the trades df has 'Fees' or 'Commission', use them; else set Net = Gross
    fees_col = col_choice(['Fees', 'Fee', 'Commission', 'Commissions'])
    if fees_col and fees_col in df.columns:
        net_pnl = gross_pnl - df[fees_col].fillna(0)
    else:
        net_pnl = gross_pnl

    # Return %
    return_pct_col = col_choice(['Return (%)', 'Return %', 'ReturnPct', 'Return', 'Return (%)'])
    if return_pct_col and return_pct_col in df.columns:
        return_pct = df[return_pct_col]
    else:
        # compute as gross_pnl / (entry_price * size)
        with np.errstate(divide='ignore', invalid='ignore'):
            return_pct = gross_pnl / (entry_prices.astype(float) * sizes.astype(float)) * 100

    # Duration in bars if entry/exit indices present:
    duration = pd.Series(np.nan, index=df.index)
    try:
        # Some trade tables include 'Entry Time' and 'Exit Time' as timestamps; else use index differences if integer index
        for idx in df.index:
            et = entry_times.loc[idx] if idx in entry_times.index else pd.NaT
            xt = exit_times.loc[idx] if idx in exit_times.index else pd.NaT
            # If they are timestamps, count number of bars between them (approx)
            if pd.notna(et) and pd.notna(xt):
                try:
                    # if et/xt already timestamps, compute delta in minutes and divide by timeframe
                    delta = pd.to_datetime(xt) - pd.to_datetime(et)
                    # 4h bars => convert to number of 4h bars
                    duration.loc[idx] = max(1, int(np.ceil(delta / pd.Timedelta(hours=4))))
                except Exception:
                    duration.loc[idx] = np.nan
            else:
                duration.loc[idx] = np.nan
    except Exception:
        duration = pd.Series(np.nan, index=df.index)

    # Build normalized DataFrame
    norm = pd.DataFrame({
        'Entry Time': pd.to_datetime(entry_times).astype('datetime64[ns]'),
        'Exit Time': pd.to_datetime(exit_times).astype('datetime64[ns]'),
        'Entry Price': pd.to_numeric(entry_prices, errors='coerce'),
        'Exit Price': pd.to_numeric(exit_prices, errors='coerce'),
        'Size': pd.to_numeric(sizes, errors='coerce'),
        'Side': sides.astype(str),
        'Gross PnL': pd.to_numeric(gross_pnl, errors='coerce'),
        'Net PnL': pd.to_numeric(net_pnl, errors='coerce'),
        'Return %': pd.to_numeric(return_pct, errors='coerce'),
        'Duration Bars': duration
    }, index=df.index)

    # Reset index to simple integer index for CSV clarity
    norm = norm.reset_index(drop=True)
    return norm


def compute_drawdown(equity_series: pd.Series) -> pd.DataFrame:
    """
    Returns DataFrame with columns: Equity, RunningMax, Drawdown (absolute), DrawdownPct
    """
    eq = equity_series.copy().fillna(method='ffill').fillna(method='bfill')
    running_max = eq.cummax()
    drawdown = eq - running_max
    drawdown_pct = (eq / running_max - 1) * 100
    dd = pd.DataFrame({
        'Equity': eq,
        'RunningMax': running_max,
        'Drawdown': drawdown,
        'DrawdownPct': drawdown_pct
    })
    return dd


# ----------------------------- Backtesting Routines ------------------------------
def run_single_backtest(symbol: str, data: pd.DataFrame):
    bt = Backtest(data, Ema12Ema200Strategy, cash=STARTING_CASH, commission=COMMISSION, spread=SPREAD)
    stats = bt.run()
    symbol_safe = symbol.replace("/", "_").replace(":", "_")
    # Save stats
    try:
        stats.to_csv(os.path.join(OUTPUT_DIR, f"{symbol_safe}_stats.csv"))
    except Exception:
        try:
            pd.Series(stats).to_csv(os.path.join(OUTPUT_DIR, f"{symbol_safe}_stats.csv"))
        except Exception:
            pass
    # Extract raw trades (various versions store it differently)
    trades_raw = getattr(bt, "_trades", None)
    if (trades_raw is None or getattr(trades_raw, "empty", False)) and isinstance(stats, dict) and "_trades" in stats:
        trades_raw = stats["_trades"]
    # Normalize and save trades CSV
    trades_norm = normalize_trades_dataframe(trades_raw)
    trades_path = os.path.join(OUTPUT_DIR, f"{symbol_safe}_trades_normalized.csv")
    trades_norm.to_csv(trades_path, index=False)
    return stats, bt, trades_raw, trades_norm


def analyze_and_save_reports(symbol: str, bt, trades_norm: pd.DataFrame, stats):
    symbol_safe = symbol.replace("/", "_").replace(":", "_")

    # 1) Per-trade CSV already saved in run_single_backtest; ensure it's present
    trades_path = os.path.join(OUTPUT_DIR, f"{symbol_safe}_trades_normalized.csv")

    # 2) Equity curve: try to extract from bt
    eq_df = None
    try:
        eq_df = getattr(bt, "_equity_curve", None) or getattr(bt, "equity_curve", None)
    except Exception:
        eq_df = None

    # Fallback: check stats for _equity_curve
    if (eq_df is None or (hasattr(eq_df, "empty") and eq_df.empty)) and hasattr(stats, "__getitem__") and "_equity_curve" in stats:
        eq_df = stats["_equity_curve"]

    equity_series = None
    if eq_df is not None and "Equity" in eq_df:
        equity_series = eq_df["Equity"]
    elif isinstance(eq_df, pd.Series):
        equity_series = eq_df
    else:
        # Could not find equity curve; create one from cumulative trade PnL applied to starting cash if trades exist
        if not trades_norm.empty and 'Net PnL' in trades_norm.columns:
            cum = trades_norm['Net PnL'].cumsum().rename('Equity') + STARTING_CASH
            equity_series = cum
        else:
            equity_series = None

    # Plot equity curve
    if equity_series is not None:
        plt.figure(figsize=(12, 5))
        plt.plot(equity_series.index, equity_series.values)
        plt.title(f"Equity Curve - {symbol}")
        plt.xlabel("Index")
        plt.ylabel("Equity (INR)")
        plt.grid(True)
        plt.tight_layout()
        eq_path = os.path.join(OUTPUT_DIR, f"{symbol_safe}_equity_static.png")
        plt.savefig(eq_path)
        plt.close()
    else:
        print("No equity series available to plot for", symbol)

    # Drawdown plot
    if equity_series is not None:
        dd = compute_drawdown(equity_series)
        plt.figure(figsize=(12, 4))
        plt.fill_between(dd.index, dd['DrawdownPct'], step='pre')
        plt.title(f"Drawdown (%) - {symbol}")
        plt.xlabel("Index")
        plt.ylabel("Drawdown (%)")
        plt.grid(True)
        plt.tight_layout()
        dd_path = os.path.join(OUTPUT_DIR, f"{symbol_safe}_drawdown.png")
        plt.savefig(dd_path)
        plt.close()
    else:
        dd_path = None

    # PnL histogram
    if not trades_norm.empty and 'Net PnL' in trades_norm.columns:
        pnl = trades_norm['Net PnL'].dropna()
        if not pnl.empty:
            plt.figure(figsize=(8, 5))
            plt.hist(pnl, bins=30, edgecolor='black')
            plt.title(f"Per-trade Net PnL Histogram - {symbol}")
            plt.xlabel("Net PnL (INR)")
            plt.ylabel("Count")
            plt.tight_layout()
            pnl_hist_path = os.path.join(OUTPUT_DIR, f"{symbol_safe}_pnl_hist.png")
            plt.savefig(pnl_hist_path)
            plt.close()
        else:
            pnl_hist_path = None
    else:
        pnl_hist_path = None

    # Summary metrics
    total_trades = int(len(trades_norm))
    wins = int((trades_norm['Net PnL'] > 0).sum()) if 'Net PnL' in trades_norm.columns else 0
    losses = int((trades_norm['Net PnL'] <= 0).sum()) if 'Net PnL' in trades_norm.columns else 0
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0.0
    net_profit = float(trades_norm['Net PnL'].sum()) if 'Net PnL' in trades_norm.columns else 0.0
    avg_pnl = float(trades_norm['Net PnL'].mean()) if 'Net PnL' in trades_norm.columns and total_trades>0 else 0.0
    max_dd_pct = None
    if equity_series is not None:
        dd_df = compute_drawdown(equity_series)
        max_dd_pct = float(dd_df['DrawdownPct'].min()) if not dd_df.empty else 0.0

    summary = {
        'Symbol': symbol,
        'Total Trades': total_trades,
        'Wins': wins,
        'Losses': losses,
        'Win Rate (%)': round(win_rate, 2),
        'Net Profit (INR)': round(net_profit, 2),
        'Avg Net PnL (INR)': round(avg_pnl, 2),
        'Max Drawdown (%)': round(max_dd_pct, 2) if max_dd_pct is not None else None
    }
    summary_df = pd.DataFrame([summary])
    summary_path = os.path.join(OUTPUT_DIR, f"{symbol_safe}_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # Return paths for convenience
    return {
        'trades_csv': os.path.join(OUTPUT_DIR, f"{symbol_safe}_trades_normalized.csv"),
        'stats_csv': os.path.join(OUTPUT_DIR, f"{symbol_safe}_stats.csv"),
        'summary_csv': summary_path,
        'equity_png': eq_path if equity_series is not None else None,
        'drawdown_png': dd_path,
        'pnl_hist_png': pnl_hist_path
    }


# ----------------------------- Main ---------------------------------------------
def main():
    # Build symbol list
    if SYMBOLS_CSV and os.path.exists(SYMBOLS_CSV):
        symbols_all = load_symbols_from_csv(SYMBOLS_CSV)
        print(f"Loaded {len(symbols_all)} symbols from {SYMBOLS_CSV}")
    else:
        symbols_all = SYMBOLS.copy()
        print("Symbols CSV missing; using SYMBOLS fallback list")

    if TARGET_SYMBOL:
        matches = [s for s in symbols_all if s.strip().upper() == TARGET_SYMBOL.strip().upper()]
        if matches:
            symbols = [matches[0]]
            print(f"Target symbol found in CSV: {matches[0]}")
        else:
            symbols = [TARGET_SYMBOL.strip()]
            print(f"Target symbol not found in CSV; using provided TARGET_SYMBOL: {TARGET_SYMBOL}")
    else:
        symbols = symbols_all

    if not symbols:
        raise RuntimeError("No symbols to run.")

    for i, symbol in enumerate(symbols):
        print(f"\n--- Running backtest for {symbol} ({i+1}/{len(symbols)}) ---")
        data = fetch_stock_data(symbol)
        if data is None or len(data) < MINIMUM_BARS:
            print("  Skipped (insufficient data).")
            continue
        stats, bt, trades_raw, trades_norm = run_single_backtest(symbol, data)
        report_paths = analyze_and_save_reports(symbol, bt, trades_norm, stats)

        # Print summary to console
        summary = pd.read_csv(report_paths['summary_csv'])
        row = summary.iloc[0]
        print(f"Results for {symbol}:")
        print(f"  Total trades : {row['Total Trades']}")
        print(f"  Wins         : {row['Wins']}")
        print(f"  Losses       : {row['Losses']}")
        print(f"  Win rate     : {row['Win Rate (%)']:.2f}%")
        print(f"  Net profit   : {row['Net Profit (INR)']:.2f}")
        print("Saved files:")
        for k, p in report_paths.items():
            if p:
                print(f"  {k}: {p}")

    print("\n--- All done. Outputs in", OUTPUT_DIR, "---")


if __name__ == "__main__":
    main()

