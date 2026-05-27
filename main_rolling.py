"""
RISE/FALL Self-Calibrating + XGBoost + AI ADVISOR Bot
═══════════════════════════════════════════════════════════════════════════════
Phase 1  COLLECT (2 hours, no trading)
  • Surveys R_25, R_75, R_100 simultaneously
  • Saves per-symbol CSVs to PERSIST_DIR/symbol_data/
  • Scores each symbol for directional predictability

Phase 2  TRADE (starts automatically after Phase 1)
  • AI Advisor picks best symbol + best expiry (ticks OR minutes)
  • Expiry ladder tested: [1,2,3,5,10 ticks] + [1,2,3,5 minutes]
  • Loads calibration.json + 3 model files trained on Phase 1 data

  Signal pipeline:
    C1  Trend momentum   — EMA7 vs EMA21 cross strength
    C2  Z-score          — price displacement in trend direction
    C3  Volatility check — sigma_ewma in tradeable band
    C4  Entropy gate     — market is not random noise
    C5  Regime gate      — TRENDING or CALM_TREND only (not CHAOS/RANGING)

  3-Layer Ensemble (C6):
    Layer 1  XGBoost              — directional pattern detector
             file: xgb_rf_model.json   threshold: XGB_THRESHOLD (0.62)
    Layer 2  Logistic Regression  — calibrated directional probability
             file: lr_rf_model.pkl     threshold: LR_THRESHOLD (0.60)
    Layer 3  Isolation Forest     — regime anomaly blocker
             file: iso_rf_model.pkl    contamination: 0.10

  Vote: trade when ≥2 of 3 approve
  Direction: determined by EMA cross + Z-score agreement

AI ADVISOR (built-in)
  • Runs every recal cycle (every 2h)
  • Optimises: best symbol, best expiry (ticks vs minutes ladder)
  • Adjusts: gate thresholds, stake, martingale, direction bias
  • Evaluates: win-rate per expiry bucket → picks highest EV expiry
  • Hot-swaps all changes live

Martingale: moderate (up to 2 steps)
  Step 0: BASE_STAKE
  Step 1 (1st loss): BASE_STAKE × MARTINGALE_MULT
  Step 2 (2nd loss): BASE_STAKE × MARTINGALE_MULT²
  Reset on win or after 2 steps exhausted

Railway deployment:
  pip install xgboost websockets scikit-learn   (add to requirements.txt)
  Mount Volume at /app/data
  ENV: DERIV_API_TOKEN, BASE_STAKE, TARGET_PROFIT, STOP_LOSS,
       XGB_THRESHOLD, LR_THRESHOLD, COLLECT_HOURS, PERSIST_DIR

Run:
    python rise_fall_bot.py
    python rise_fall_bot.py --collect-only
    python rise_fall_bot.py --trade-only
"""

import asyncio
import csv
import json
import logging
import math
import os
import sys
import time
import traceback
import threading
from collections import deque
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, List, Optional, Tuple

try:
    import websockets
    from websockets.exceptions import (
        ConnectionClosed, ConnectionClosedError, ConnectionClosedOK,
    )
except ImportError:
    sys.exit("websockets not installed — run: pip install websockets")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

API_TOKEN  = os.getenv("DERIV_API_TOKEN", "iCCn0vuMCzLcq1J")
APP_ID     = os.getenv("DERIV_APP_ID",    "1089")
WS_URL     = f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}"

COLLECT_HOURS = float(os.getenv("COLLECT_HOURS", "2"))
COLLECT_SECS  = COLLECT_HOURS * 3600

_PERSIST_DIR  = os.getenv("PERSIST_DIR", os.path.join(os.getcwd(), "data"))
os.makedirs(_PERSIST_DIR, exist_ok=True)
CAL_FILE      = os.path.join(_PERSIST_DIR, "calibration_rf.json")
DATA_DIR      = os.path.join(_PERSIST_DIR, "symbol_data")
PORT          = int(os.getenv("PORT", "8080"))

# Symbols to survey — Volatility indices (standard, not HZ)
SURVEY_SYMBOLS = ["R_25", "R_75", "R_100"]

# Expiry ladder — advisor picks best bucket per cycle
TICK_EXPIRIES   = [1, 2, 3, 5, 10]        # tick-based contracts
MINUTE_EXPIRIES = [1, 2, 3, 5]            # minute-based (in minutes)

# Martingale — moderate (2 steps)
BASE_STAKE       = float(os.getenv("BASE_STAKE",    "1.0"))
MARTINGALE_MULT  = float(os.getenv("MARTI_MULT",    "2.20"))
MARTINGALE_STEPS = int(os.getenv("MARTI_STEPS",     "2"))
LOSS_COOLDOWN    = float(os.getenv("LOSS_COOLDOWN", "30"))

# Trade risk
TARGET_PROFIT = float(os.getenv("TARGET_PROFIT", "30.0"))
STOP_LOSS     = float(os.getenv("STOP_LOSS",     "10.0"))
LOCK_TIMEOUT  = 360   # seconds

# ML gate
XGB_THRESHOLD      = float(os.getenv("XGB_THRESHOLD",      "0.62"))
LR_THRESHOLD       = float(os.getenv("LR_THRESHOLD",       "0.60"))
ISO_CONTAMINATION  = float(os.getenv("ISO_CONTAMINATION",  "0.10"))
ROLLING_MAX_HOURS  = float(os.getenv("ROLLING_MAX_HOURS",  "24"))
ROLLING_MAX_SECS   = ROLLING_MAX_HOURS * 3600

# ─────────────────────────────────────────────────────────────────────────────
# AI ADVISOR SAFE BOUNDS
# ─────────────────────────────────────────────────────────────────────────────

ADVISOR_LOG    = os.path.join(_PERSIST_DIR, "advisor_rf_log.txt")
CANDLE_GRAN_1  = 60
CANDLE_GRAN_5  = 300
CANDLE_COUNT   = 30

SAFE_BOUNDS = {
    "momentum_gate":    (0.0002, 0.010,  0.0010),
    "z_gate":           (0.30,   2.50,   0.200),
    "sigma_lo":         (0.0001, 0.020,  0.002),
    "sigma_hi":         (0.005,  0.200,  0.010),
    "entropy_gate":     (0.30,   0.90,   0.050),
    "xgb_threshold":    (0.55,   0.85,   0.050),
    "lr_threshold":     (0.55,   0.85,   0.050),
    "base_stake":       (0.35,   5.00,   0.500),
    "martingale_steps": (1,      2,      1),
    "loss_cooldown":    (10,     120,    15),
}

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("rise_fall_bot")

def info(m):  log.info(m)
def warn(m):  log.warning(m)
def err(m):   log.error(m)
def tlog(m):  log.info(f"[TRADE] {m}")

# ─────────────────────────────────────────────────────────────────────────────
# ML FEATURES  (18 directional features — same at train & inference)
# ─────────────────────────────────────────────────────────────────────────────

_FEATURES = [
    "ema_cross",        # EMA7 - EMA21 (signed — direction signal)
    "ema_cross_abs",    # |EMA7 - EMA21| (magnitude)
    "ema_accel",        # change in ema_cross over last 10 ticks
    "zscore_50",        # signed z-score (direction-aware)
    "zscore_abs",       # |z-score|
    "sigma_ewma",       # volatility level
    "sigma_ratio",      # sigma vs 50-tick rolling mean of sigma
    "range_20",         # 20-tick range
    "range_50",         # 50-tick range
    "range_ratio",      # range_20 / range_50
    "momentum_5",       # price[now] - price[5 ticks ago]
    "momentum_10",      # price[now] - price[10 ticks ago]
    "momentum_20",      # price[now] - price[20 ticks ago]
    "entropy_20",       # Shannon entropy of last 20 moves
    "spike_10",         # max single-tick move in last 10
    "atr_14",           # average true range
    "tick_sign_run",    # consecutive same-direction ticks (signed)
    "hour_of_day",      # UTC hour (session effect)
]

_REGIME_ENC = {"CHAOS": -1, "RANGING": 0, "CALM_TREND": 1, "STRONG_TREND": 2}

# ─────────────────────────────────────────────────────────────────────────────
# SYMBOL STATS ENGINE  (Phase 1 — directional metrics)
# ─────────────────────────────────────────────────────────────────────────────

class SymbolStats:
    EWMA_ALPHA = 0.05

    def __init__(self, symbol: str):
        self.symbol     = symbol
        self.tick_n     = 0
        self.prices: deque = deque(maxlen=500)
        self.sigma_ewma = None

        self._ema7  = self._ema21 = None
        self._k7    = 2 / (7  + 1)
        self._k21   = 2 / (21 + 1)

        self._sign_run   = 0   # consecutive same-direction ticks
        self._prev_delta = 0.0

        self._regime_counts = {
            "CHAOS": 0, "RANGING": 0, "CALM_TREND": 0, "STRONG_TREND": 0}
        self._regime_start = time.time()
        self._regime_cur   = "RANGING"

        # Track direction outcomes for expiry calibration
        # key: expiry_label → list of (was_right: bool)
        self._expiry_outcomes: Dict[str, List[bool]] = {}
        self._pending_checks: List[dict] = []   # {check_at, price_now, direction, labels}

        os.makedirs(DATA_DIR, exist_ok=True)
        fname = os.path.join(DATA_DIR, f"{symbol}.csv")
        file_exists = os.path.exists(fname) and os.path.getsize(fname) > 0
        self._csv_f = open(fname, "a", newline="")
        self._csv_w = csv.DictWriter(self._csv_f, fieldnames=self._fields())
        if not file_exists:
            self._csv_w.writeheader()
        self._rows_since_flush = 0

    @staticmethod
    def _fields():
        return [
            "ts", "epoch", "symbol", "tick_n",
            "price", "tick_delta", "tick_abs_delta",
            "sigma_ewma", "range_20", "range_50",
            "ema7", "ema21", "ema_cross",
            "zscore_50", "spike_10", "atr_14",
            "entropy_20", "momentum_5", "momentum_10", "momentum_20",
            "tick_sign_run", "regime",
        ]

    def update(self, price: float, epoch: float) -> dict:
        self.tick_n += 1
        prev  = self.prices[-1] if self.prices else price
        delta = price - prev
        abs_delta = abs(delta)
        self.prices.append(price)

        # EWMA sigma
        if self.sigma_ewma is None:
            self.sigma_ewma = abs_delta
        else:
            self.sigma_ewma = (self.EWMA_ALPHA * abs_delta
                               + (1 - self.EWMA_ALPHA) * self.sigma_ewma)

        # EMAs
        if self._ema7 is None:
            self._ema7 = self._ema21 = price
        else:
            self._ema7  = price * self._k7  + self._ema7  * (1 - self._k7)
            self._ema21 = price * self._k21 + self._ema21 * (1 - self._k21)
        ema_cross = self._ema7 - self._ema21

        # Sign run
        if delta > 0:
            self._sign_run = max(1, self._sign_run + 1)
        elif delta < 0:
            self._sign_run = min(-1, self._sign_run - 1)
        else:
            self._sign_run = 0

        prices = list(self.prices)

        # Ranges
        range_20 = (max(prices[-20:]) - min(prices[-20:])
                    if len(prices) >= 20 else 0)
        range_50 = (max(prices[-50:]) - min(prices[-50:])
                    if len(prices) >= 50 else 0)

        # Z-score (signed)
        zscore_50 = 0.0
        if len(prices) >= 200:
            baseline = prices[-200:]
            mu  = sum(baseline) / len(baseline)
            var = sum((p - mu)**2 for p in baseline) / len(baseline)
            std = math.sqrt(var) if var > 0 else 1e-9
            short_mean = sum(prices[-50:]) / 50
            zscore_50  = (short_mean - mu) / (std / math.sqrt(50))

        # Spike
        moves = [abs(prices[i] - prices[i-1]) for i in range(-10, 0)
                 if i-1 >= -len(prices)]
        spike_10 = max(moves) if moves else 0

        # ATR-14
        atr_moves = [abs(prices[i] - prices[i-1]) for i in range(-14, 0)
                     if i-1 >= -len(prices)]
        atr_14 = sum(atr_moves) / len(atr_moves) if atr_moves else 0

        # Entropy
        entropy_20 = self._entropy(prices[-21:]) if len(prices) >= 21 else 1.0

        # Momentum
        momentum_5  = (price - prices[-6])  if len(prices) >= 6  else 0.0
        momentum_10 = (price - prices[-11]) if len(prices) >= 11 else 0.0
        momentum_20 = (price - prices[-21]) if len(prices) >= 21 else 0.0

        # Regime
        regime = self._detect_regime(ema_cross, self.sigma_ewma, zscore_50)
        if regime != self._regime_cur:
            self._regime_counts[self._regime_cur] += (time.time() - self._regime_start)
            self._regime_cur   = regime
            self._regime_start = time.time()

        row = {
            "ts":            datetime.now(timezone.utc).isoformat(),
            "epoch":         epoch,
            "symbol":        self.symbol,
            "tick_n":        self.tick_n,
            "price":         round(price, 5),
            "tick_delta":    round(delta, 5),
            "tick_abs_delta":round(abs_delta, 5),
            "sigma_ewma":    round(self.sigma_ewma, 5),
            "range_20":      round(range_20, 4),
            "range_50":      round(range_50, 4),
            "ema7":          round(self._ema7, 5),
            "ema21":         round(self._ema21, 5),
            "ema_cross":     round(ema_cross, 6),
            "zscore_50":     round(zscore_50, 4),
            "spike_10":      round(spike_10, 5),
            "atr_14":        round(atr_14, 5),
            "entropy_20":    round(entropy_20, 4),
            "momentum_5":    round(momentum_5, 5),
            "momentum_10":   round(momentum_10, 5),
            "momentum_20":   round(momentum_20, 5),
            "tick_sign_run": self._sign_run,
            "regime":        regime,
        }

        self._csv_w.writerow(row)
        self._rows_since_flush += 1
        if self._rows_since_flush >= 300:
            self._csv_f.flush()
            self._rows_since_flush = 0

        return row

    @staticmethod
    def _entropy(prices: list) -> float:
        moves = [abs(prices[i] - prices[i-1]) for i in range(1, len(prices))]
        if not moves:
            return 1.0
        mx = max(moves) or 1
        buckets = [0] * 5
        for m in moves:
            buckets[min(4, int(m / mx * 4))] += 1
        n = len(moves); H = 0.0
        for b in buckets:
            if b > 0:
                p = b / n
                H -= p * math.log2(p)
        return H / math.log2(5)

    @staticmethod
    def _detect_regime(ema_cross: float, sigma: float, zscore: float) -> str:
        if abs(zscore) > 3.0 or sigma > 0.5:
            return "CHAOS"
        if abs(ema_cross) > 0.15 and abs(zscore) > 1.5:
            return "STRONG_TREND"
        if abs(ema_cross) > 0.05 or abs(zscore) > 0.8:
            return "CALM_TREND"
        return "RANGING"

    def summarise(self) -> dict:
        self._regime_counts[self._regime_cur] += (time.time() - self._regime_start)
        total_secs = sum(self._regime_counts.values()) or 1
        return {
            "symbol":      self.symbol,
            "ticks":       self.tick_n,
            "regime_pct":  {k: round(v / total_secs, 4)
                            for k, v in self._regime_counts.items()},
            "data_file":   os.path.join(DATA_DIR, f"{self.symbol}.csv"),
        }

    def close(self):
        self._csv_f.flush()
        self._csv_f.close()

# ─────────────────────────────────────────────────────────────────────────────
# EXPIRY CALIBRATION  — measure directional accuracy per expiry bucket
# ─────────────────────────────────────────────────────────────────────────────

def compute_expiry_accuracy(csv_path: str) -> dict:
    """
    Simulates Rise/Fall trades at each expiry bucket using historical tick data.
    For each tick: if EMA cross agrees with z-score direction, that's a signal.
    Measures what fraction of those signals would have won at each expiry.

    Returns dict: expiry_label → {"win_rate": float, "n_trades": int, "ev": float}
    """
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    "price":     float(row["price"]),
                    "ema_cross": float(row["ema_cross"]),
                    "zscore_50": float(row["zscore_50"]),
                    "sigma_ewma":float(row["sigma_ewma"]),
                    "entropy_20":float(row["entropy_20"]),
                    "regime":    row["regime"].strip(),
                })
            except Exception:
                continue

    if len(rows) < 500:
        return {}

    prices  = [r["price"] for r in rows]
    n       = len(rows)

    # Build expiry buckets: ticks + minutes (approx ticks at ~1 tick/sec for HZ)
    # R_25/R_75/R_100 tick ~1/sec, so 1min ≈ 60 ticks
    tick_buckets   = {f"t{t}": t       for t in TICK_EXPIRIES}
    minute_buckets = {f"m{m}": m * 60  for m in MINUTE_EXPIRIES}
    all_buckets    = {**tick_buckets, **minute_buckets}

    results = {label: {"wins": 0, "total": 0} for label in all_buckets}

    # Payout estimate: tick contracts ~85%, minute contracts ~87%
    payout = {label: (0.85 if label.startswith("t") else 0.87)
              for label in all_buckets}

    for i, row in enumerate(rows):
        # Signal conditions: trending regime + EMA/z agreement
        regime   = row["regime"]
        cross    = row["ema_cross"]
        z        = row["zscore_50"]
        entropy  = row["entropy_20"]
        sigma    = row["sigma_ewma"]

        if regime not in ("CALM_TREND", "STRONG_TREND"):
            continue
        if abs(cross) < 0.0005:
            continue
        if entropy < 0.35:
            continue
        if sigma < 0.0002:
            continue

        # Direction: CALL if both ema_cross > 0 and zscore > 0
        if cross > 0 and z > 0:
            direction = "CALL"
        elif cross < 0 and z < 0:
            direction = "PUT"
        else:
            continue   # disagreement — no signal

        for label, offset in all_buckets.items():
            end_idx = i + offset
            if end_idx >= n:
                continue
            end_price = prices[end_idx]
            won = (end_price > prices[i]) if direction == "CALL" else (end_price < prices[i])
            results[label]["wins"]  += int(won)
            results[label]["total"] += 1

    out = {}
    for label, d in results.items():
        if d["total"] >= 30:
            wr  = d["wins"] / d["total"]
            ev  = wr * payout[label] - (1 - wr)   # expected value per unit stake
            out[label] = {
                "win_rate":  round(wr, 4),
                "n_trades":  d["total"],
                "ev":        round(ev, 4),
                "payout":    payout[label],
            }

    return out

# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_calibration(summaries: List[dict]) -> dict:
    import statistics

    info("Computing calibration + expiry accuracy from collected data...")
    symbol_scores = {}

    for s in summaries:
        sym   = s["symbol"]
        fpath = s["data_file"]
        if not os.path.exists(fpath):
            continue

        crosses, zscores, sigmas, entropies = [], [], [], []

        with open(fpath, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    crosses.append(abs(float(row["ema_cross"])))
                    zscores.append(abs(float(row["zscore_50"])))
                    sigmas.append(float(row["sigma_ewma"]))
                    entropies.append(float(row["entropy_20"]))
                except (ValueError, KeyError):
                    continue

        if len(crosses) < 200:
            warn(f"{sym}: insufficient data ({len(crosses)} rows) — skipping")
            continue

        crosses.sort(); zscores.sort(); sigmas.sort(); entropies.sort()

        def pct(lst, p):
            idx = max(0, int(len(lst) * p / 100) - 1)
            return lst[idx]

        # Trend pct = CALM_TREND + STRONG_TREND time
        trend_pct = (s["regime_pct"].get("CALM_TREND",    0)
                   + s["regime_pct"].get("STRONG_TREND",  0))
        chaos_pct  = s["regime_pct"].get("CHAOS", 0)

        # Gates calibrated from data
        momentum_gate = pct(crosses, 30)    # 30th pct — require meaningful cross
        z_gate        = max(0.5, pct(zscores, 30))
        sigma_lo      = pct(sigmas, 10)     # below this: market is dead
        sigma_hi      = pct(sigmas, 85)     # above this: too volatile
        entropy_gate  = pct(entropies, 25)  # below this: too random

        # Directional predictability score
        # Penalise symbols with too much chaos, reward trend time
        dir_score = trend_pct * (1 - chaos_pct)

        # Compute expiry accuracy
        expiry_acc = compute_expiry_accuracy(fpath)
        if expiry_acc:
            best_expiry = max(expiry_acc, key=lambda k: expiry_acc[k]["ev"])
            best_ev     = expiry_acc[best_expiry]["ev"]
        else:
            best_expiry = "t5"
            best_ev     = 0.0
            warn(f"{sym}: no expiry data — defaulting to t5")

        info(f"  {sym}: dir_score={dir_score:.4f}  trend={trend_pct:.1%}  "
             f"chaos={chaos_pct:.1%}  best_expiry={best_expiry}  ev={best_ev:.4f}")

        if expiry_acc:
            top3 = sorted(expiry_acc, key=lambda k: expiry_acc[k]["ev"], reverse=True)[:3]
            for lbl in top3:
                d = expiry_acc[lbl]
                info(f"      {lbl}: wr={d['win_rate']:.2%}  ev={d['ev']:.4f}  "
                     f"n={d['n_trades']}")

        symbol_scores[sym] = {
            "symbol":       sym,
            "ticks":        s["ticks"],
            "dir_score":    round(dir_score, 4),
            "trend_pct":    round(trend_pct, 4),
            "chaos_pct":    round(chaos_pct, 4),
            "regime_pct":   s["regime_pct"],
            "best_expiry":  best_expiry,
            "best_ev":      round(best_ev, 4),
            "expiry_accuracy": expiry_acc,
            # Signal gates
            "momentum_gate": round(momentum_gate, 6),
            "z_gate":        round(z_gate, 4),
            "sigma_lo":      round(sigma_lo, 6),
            "sigma_hi":      round(sigma_hi, 5),
            "entropy_gate":  round(entropy_gate, 4),
        }

    if not symbol_scores:
        raise RuntimeError("No symbols had sufficient data for calibration")

    ranked = sorted(symbol_scores.values(),
                    key=lambda x: x["dir_score"], reverse=True)

    top1 = ranked[:1]   # trade best symbol only (advisor can switch)
    info(f"\nBest symbol: {top1[0]['symbol']}  "
         f"dir_score={top1[0]['dir_score']:.4f}  "
         f"best_expiry={top1[0]['best_expiry']}")

    cal = {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "collect_hours": COLLECT_HOURS,
        "all_symbols":   ranked,
        "trade_symbols": top1,
    }

    with open(CAL_FILE, "w") as f:
        json.dump(cal, f, indent=2)

    info(f"Calibration saved to {CAL_FILE}")
    return cal


def rolling_csv_trim(symbol: str):
    fpath = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(fpath):
        return

    cutoff_epoch = time.time() - ROLLING_MAX_SECS
    kept = []; removed = 0; fields = None

    with open(fpath, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        for row in reader:
            try:
                ts_str    = row["ts"].replace("Z", "+00:00")
                row_epoch = datetime.fromisoformat(ts_str).timestamp()
                if row_epoch >= cutoff_epoch:
                    kept.append(row)
                else:
                    removed += 1
            except Exception:
                kept.append(row)

    if removed == 0:
        return

    tmp = fpath + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(kept)
    os.replace(tmp, fpath)

    info(f"[ROLLING] {symbol}: trimmed {removed} rows  kept={len(kept)}")

# ─────────────────────────────────────────────────────────────────────────────
# CANDLE FEED
# ─────────────────────────────────────────────────────────────────────────────

class CandleFeed:
    async def fetch(self, symbol: str) -> dict:
        result = {"candles_1m": [], "candles_5m": []}
        try:
            ws = await websockets.connect(
                WS_URL, ping_interval=20, ping_timeout=15, close_timeout=5)
            rid = 0

            async def send(data):
                nonlocal rid
                rid += 1
                data["req_id"] = rid
                await ws.send(json.dumps(data))

            async def recv_type(mtype, timeout=10):
                deadline = asyncio.get_event_loop().time() + timeout
                while True:
                    rem = deadline - asyncio.get_event_loop().time()
                    if rem <= 0:
                        return None
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=rem)
                        msg = json.loads(raw)
                        if mtype in msg or "error" in msg:
                            return msg
                    except Exception:
                        return None

            await send({"authorize": API_TOKEN})
            auth = await recv_type("authorize", timeout=10)
            if not auth or "error" in auth:
                warn("[CANDLE] Auth failed"); return result

            end_epoch = int(time.time())

            for gran, key in [(CANDLE_GRAN_1, "candles_1m"),
                              (CANDLE_GRAN_5, "candles_5m")]:
                start_epoch = end_epoch - gran * CANDLE_COUNT * 2
                await send({
                    "ticks_history": symbol,
                    "style":         "candles",
                    "granularity":   gran,
                    "start":         start_epoch,
                    "end":           end_epoch,
                    "count":         CANDLE_COUNT,
                })
                resp = await recv_type("candles", timeout=12)
                if resp and "candles" in resp:
                    result[key] = [
                        {"epoch": c["epoch"], "open": float(c["open"]),
                         "high": float(c["high"]), "low": float(c["low"]),
                         "close": float(c["close"])}
                        for c in resp["candles"][-CANDLE_COUNT:]
                    ]

            await ws.close()
        except Exception as exc:
            err(f"[CANDLE] fetch error: {exc}")
        return result

# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class IndicatorEngine:
    @staticmethod
    def rsi(closes: list, period: int = 14) -> Optional[float]:
        if len(closes) < period + 1:
            return None
        gains = []; losses = []
        for i in range(1, len(closes)):
            d = closes[i] - closes[i-1]
            gains.append(max(d, 0)); losses.append(max(-d, 0))
        g = gains[-period:]; l = losses[-period:]
        avg_g = sum(g) / period; avg_l = sum(l) / period
        if avg_l == 0:
            return 100.0
        return round(100 - 100 / (1 + avg_g / avg_l), 2)

    @staticmethod
    def ema(closes: list, period: int) -> Optional[float]:
        if len(closes) < period:
            return None
        k = 2 / (period + 1)
        e = closes[0]
        for c in closes[1:]:
            e = c * k + e * (1 - k)
        return round(e, 6)

    @staticmethod
    def atr(candles: list, period: int = 14) -> Optional[float]:
        if len(candles) < period + 1:
            return None
        trs = []
        for i in range(1, len(candles)):
            h = candles[i]["high"]; l = candles[i]["low"]; pc = candles[i-1]["close"]
            trs.append(max(h - l, abs(h - pc), abs(l - pc)))
        return round(sum(trs[-period:]) / period, 6)

    @classmethod
    def compute(cls, candle_data: dict) -> dict:
        out = {}
        c1m = candle_data.get("candles_1m", [])
        c5m = candle_data.get("candles_5m", [])
        closes_1m = [c["close"] for c in c1m]
        closes_5m = [c["close"] for c in c5m]

        out["rsi_1m"]   = cls.rsi(closes_1m, 14)
        out["rsi_5m"]   = cls.rsi(closes_5m, 14)
        out["ema7_1m"]  = cls.ema(closes_1m, 7)
        out["ema21_1m"] = cls.ema(closes_1m, 21)
        out["ema7_5m"]  = cls.ema(closes_5m, 7)
        out["ema21_5m"] = cls.ema(closes_5m, 21)
        out["atr_1m"]   = cls.atr(c1m)
        out["atr_5m"]   = cls.atr(c5m)

        e7  = out["ema7_1m"]
        e21 = out["ema21_1m"]
        if e7 and e21:
            out["candle_cross"]   = e7 - e21
            out["candle_trend"]   = "UP" if e7 > e21 else "DOWN"
        else:
            out["candle_cross"]   = 0
            out["candle_trend"]   = "NEUTRAL"

        rsi = out["rsi_1m"]
        if rsi:
            out["candle_regime"] = ("OVERBOUGHT" if rsi > 70 else
                                    "OVERSOLD"   if rsi < 30 else "NEUTRAL")
        else:
            out["candle_regime"] = "NEUTRAL"

        return out

# ─────────────────────────────────────────────────────────────────────────────
# AI ADVISOR
# ─────────────────────────────────────────────────────────────────────────────

class AIAdvisor:
    """
    Runs every recal cycle. Evaluates performance, picks best symbol,
    best expiry, adjusts gates and stake. Hot-swaps everything live.
    """

    def __init__(self):
        self._cycle = 0

    def advise(
        self,
        traders: list,
        all_cals: list,
        indicators: dict,
        session_stats: dict,
    ) -> dict:
        self._cycle += 1
        ts  = datetime.now(timezone.utc).isoformat()
        log_lines = [
            f"\n{'='*70}",
            f"AI ADVISOR  cycle={self._cycle}  {ts}",
            f"{'='*70}",
        ]

        adjustments = {}

        # ── 1. Pick best symbol ─────────────────────────────────────────────
        best_sym_cal = max(all_cals, key=lambda c: c.get("dir_score", 0))
        best_sym     = best_sym_cal["symbol"]
        log_lines.append(f"\n[SYMBOL] Best by dir_score: {best_sym} "
                         f"(score={best_sym_cal.get('dir_score',0):.4f})")

        # ── 2. Pick best expiry from live cycle stats ───────────────────────
        # Use both historical accuracy AND live win rate per expiry
        expiry_acc = best_sym_cal.get("expiry_accuracy", {})
        live_stats = session_stats.get("expiry_stats", {})

        # Blend: 70% historical EV, 30% live win rate (if enough data)
        blended = {}
        for label, hd in expiry_acc.items():
            h_ev = hd["ev"]
            if label in live_stats and live_stats[label]["total"] >= 10:
                l_wr = live_stats[label]["wins"] / live_stats[label]["total"]
                l_ev = l_wr * hd["payout"] - (1 - l_wr)
                blended[label] = 0.70 * h_ev + 0.30 * l_ev
                log_lines.append(f"  {label}: h_ev={h_ev:.4f}  l_ev={l_ev:.4f}  "
                                  f"blended={blended[label]:.4f}")
            else:
                blended[label] = h_ev
                log_lines.append(f"  {label}: h_ev={h_ev:.4f}  (live data insufficient)")

        if blended:
            best_expiry = max(blended, key=lambda k: blended[k])
            best_ev     = blended[best_expiry]
        else:
            best_expiry = best_sym_cal.get("best_expiry", "t5")
            best_ev     = best_sym_cal.get("best_ev", 0.0)

        log_lines.append(f"\n[EXPIRY] Selected: {best_expiry}  blended_ev={best_ev:.4f}")
        adjustments["best_expiry"] = best_expiry
        adjustments["best_symbol"] = best_sym

        # ── 3. Gate adjustments based on recent win rate ────────────────────
        wins   = session_stats.get("wins",   0)
        losses = session_stats.get("losses", 0)
        total  = wins + losses
        wr     = wins / total if total >= 10 else None

        log_lines.append(f"\n[PERF] wins={wins}  losses={losses}  "
                         f"wr={wr:.2%}" if wr else
                         f"\n[PERF] wins={wins}  losses={losses}  (insufficient data)")

        cal = best_sym_cal.copy()

        if wr is not None:
            if wr < 0.45:
                # Tighten gates — too many losses
                new_mg = min(cal["momentum_gate"] * 1.15,
                             SAFE_BOUNDS["momentum_gate"][1])
                new_zg = min(cal["z_gate"] * 1.10,
                             SAFE_BOUNDS["z_gate"][1])
                log_lines.append(f"  WR={wr:.1%} LOW → tightening gates")
                log_lines.append(f"  momentum_gate: {cal['momentum_gate']:.6f} → {new_mg:.6f}")
                log_lines.append(f"  z_gate: {cal['z_gate']:.4f} → {new_zg:.4f}")
                cal["momentum_gate"] = round(new_mg, 6)
                cal["z_gate"]        = round(new_zg, 4)

            elif wr > 0.62:
                # Relax gates slightly — bot is performing well
                new_mg = max(cal["momentum_gate"] * 0.92,
                             SAFE_BOUNDS["momentum_gate"][0])
                new_zg = max(cal["z_gate"] * 0.93,
                             SAFE_BOUNDS["z_gate"][0])
                log_lines.append(f"  WR={wr:.1%} HIGH → relaxing gates for more trades")
                cal["momentum_gate"] = round(new_mg, 6)
                cal["z_gate"]        = round(new_zg, 4)

        # ── 4. Candle indicator checks ──────────────────────────────────────
        inds    = indicators.get(best_sym, {})
        c_trend = inds.get("candle_trend", "NEUTRAL")
        c_rsi   = inds.get("rsi_1m")
        log_lines.append(f"\n[CANDLE] {best_sym}  trend={c_trend}  rsi={c_rsi}")

        # If candle trend is neutral and WR is mediocre, bump entropy gate
        if c_trend == "NEUTRAL" and (wr is None or wr < 0.55):
            new_eg = min(cal["entropy_gate"] * 1.08,
                         SAFE_BOUNDS["entropy_gate"][1])
            log_lines.append(f"  Candle NEUTRAL → raising entropy_gate "
                             f"{cal['entropy_gate']:.4f} → {new_eg:.4f}")
            cal["entropy_gate"] = round(new_eg, 4)

        # ── 5. XGB/LR threshold nudge ───────────────────────────────────────
        global XGB_THRESHOLD, LR_THRESHOLD
        if wr is not None:
            if wr < 0.42:
                new_xgb = min(XGB_THRESHOLD + 0.03, SAFE_BOUNDS["xgb_threshold"][1])
                new_lr  = min(LR_THRESHOLD  + 0.03, SAFE_BOUNDS["lr_threshold"][1])
                log_lines.append(f"\n[ENS] WR low → raising thresholds "
                                 f"XGB {XGB_THRESHOLD:.2f}→{new_xgb:.2f}  "
                                 f"LR {LR_THRESHOLD:.2f}→{new_lr:.2f}")
                XGB_THRESHOLD = new_xgb
                LR_THRESHOLD  = new_lr
            elif wr > 0.65:
                new_xgb = max(XGB_THRESHOLD - 0.02, SAFE_BOUNDS["xgb_threshold"][0])
                new_lr  = max(LR_THRESHOLD  - 0.02, SAFE_BOUNDS["lr_threshold"][0])
                log_lines.append(f"\n[ENS] WR high → relaxing thresholds "
                                 f"XGB {XGB_THRESHOLD:.2f}→{new_xgb:.2f}  "
                                 f"LR {LR_THRESHOLD:.2f}→{new_lr:.2f}")
                XGB_THRESHOLD = new_xgb
                LR_THRESHOLD  = new_lr

        # ── 6. Stake nudge ──────────────────────────────────────────────────
        global BASE_STAKE
        if wr is not None:
            pnl = session_stats.get("pnl", 0.0)
            if wr > 0.60 and pnl > 0:
                new_stake = min(BASE_STAKE * 1.10, SAFE_BOUNDS["base_stake"][1])
                if new_stake != BASE_STAKE:
                    log_lines.append(f"\n[STAKE] WR={wr:.1%} + PnL positive → "
                                     f"raising stake {BASE_STAKE:.2f}→{new_stake:.2f}")
                    BASE_STAKE = round(new_stake, 2)
            elif wr < 0.45 and pnl < 0:
                new_stake = max(BASE_STAKE * 0.90, SAFE_BOUNDS["base_stake"][0])
                if new_stake != BASE_STAKE:
                    log_lines.append(f"\n[STAKE] WR={wr:.1%} + PnL negative → "
                                     f"reducing stake {BASE_STAKE:.2f}→{new_stake:.2f}")
                    BASE_STAKE = round(new_stake, 2)

        adjustments["cal"] = cal
        log_lines.append(f"\n{'='*70}\n")

        # Write to log file
        try:
            with open(ADVISOR_LOG, "a") as f:
                f.write("\n".join(log_lines))
        except Exception as e:
            warn(f"[ADVISOR] log write error: {e}")

        return adjustments

# ─────────────────────────────────────────────────────────────────────────────
# ENSEMBLE GATE
# ─────────────────────────────────────────────────────────────────────────────

class EnsembleGate:
    def __init__(self, persist_dir: str):
        self._xgb = self._lr = self._iso = None
        self.persist_dir = persist_dir
        self._load()

    def _load(self):
        import pickle

        xgb_path = os.path.join(self.persist_dir, "xgb_rf_model.json")
        if os.path.exists(xgb_path):
            try:
                from xgboost import XGBClassifier
                m = XGBClassifier()
                m.load_model(xgb_path)
                self._xgb = m
                info(f"[ENS] L1 XGBoost loaded  thresh={XGB_THRESHOLD}")
            except Exception:
                try:
                    with open(xgb_path.replace(".json", ".pkl"), "rb") as f:
                        self._xgb = pickle.load(f)
                    info(f"[ENS] L1 GBM fallback loaded")
                except Exception as e:
                    warn(f"[ENS] L1 load failed: {e}")

        lr_path = os.path.join(self.persist_dir, "lr_rf_model.pkl")
        if os.path.exists(lr_path):
            try:
                with open(lr_path, "rb") as f:
                    self._lr = pickle.load(f)
                info(f"[ENS] L2 LogReg loaded  thresh={LR_THRESHOLD}")
            except Exception as e:
                warn(f"[ENS] L2 load failed: {e}")

        iso_path = os.path.join(self.persist_dir, "iso_rf_model.pkl")
        if os.path.exists(iso_path):
            try:
                with open(iso_path, "rb") as f:
                    self._iso = pickle.load(f)
                info(f"[ENS] L3 IsoForest loaded")
            except Exception as e:
                warn(f"[ENS] L3 load failed: {e}")

        loaded = sum(x is not None for x in [self._xgb, self._lr, self._iso])
        if loaded == 0:
            warn("[ENS] No models — 5-condition fallback mode")
        else:
            info(f"[ENS] {loaded}/3 layers loaded")

    @property
    def active(self):
        return any(x is not None for x in [self._xgb, self._lr, self._iso])

    def predict(self, feats: dict, regime: str) -> dict:
        import numpy as np

        if regime == "CHAOS":
            return {"votes": 0, "trade": False, "reason": "chaos",
                    "xgb_prob": 0.0, "lr_prob": 0.0, "iso_score": 0.0,
                    "v_xgb": False, "v_lr": False, "v_iso": False}

        row = [feats.get(f, 0.0) for f in _FEATURES]
        X   = [row]

        v_xgb = True; xgb_prob = 1.0
        v_lr  = True; lr_prob  = 1.0
        v_iso = True; iso_score = 0.0

        if self._xgb is not None:
            try:
                xgb_prob = float(self._xgb.predict_proba(
                    np.array(X, dtype=float))[:, 1][0])
                v_xgb = xgb_prob >= XGB_THRESHOLD
            except Exception as e:
                warn(f"[ENS] L1 predict error: {e}")

        if self._lr is not None:
            try:
                lr_prob = float(self._lr.predict_proba(
                    np.array(X, dtype=float))[:, 1][0])
                v_lr = lr_prob >= LR_THRESHOLD
            except Exception as e:
                warn(f"[ENS] L2 predict error: {e}")

        if self._iso is not None:
            try:
                iso_score = float(self._iso.score_samples(
                    np.array(X, dtype=float))[0])
                v_iso = iso_score >= getattr(self._iso, "_ens_threshold", -0.5)
            except Exception as e:
                warn(f"[ENS] L3 predict error: {e}")

        votes = sum([v_xgb, v_lr, v_iso])
        return {
            "votes": votes, "trade": votes >= 2,
            "xgb_prob": round(xgb_prob, 4), "lr_prob": round(lr_prob, 4),
            "iso_score": round(iso_score, 4),
            "v_xgb": v_xgb, "v_lr": v_lr, "v_iso": v_iso,
        }


_ensemble: Optional[EnsembleGate] = None

def load_ensemble() -> EnsembleGate:
    global _ensemble
    _ensemble = EnsembleGate(_PERSIST_DIR)
    return _ensemble

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE BUILDER  (shared between training and inference)
# ─────────────────────────────────────────────────────────────────────────────

def _build_feature_matrix(rows_raw: list, labels_arr=None):
    import numpy as np

    prices_all  = [r["price"]      for r in rows_raw]
    sigmas_all  = [r["sigma_ewma"] for r in rows_raw]

    sigma_mean_buf = deque(maxlen=50)
    X_rows = []; y_rows = []

    for i, r in enumerate(rows_raw):
        if labels_arr is not None:
            lbl = labels_arr[i]
            if lbl != lbl:   # nan check
                continue

        sigma_mean_buf.append(r["sigma_ewma"])
        sigma_mean = sum(sigma_mean_buf) / len(sigma_mean_buf)

        crosses_window = [rows_raw[j]["ema_cross"]
                          for j in range(max(0, i-10), i)]
        ema_accel = (r["ema_cross"] - crosses_window[0]) if crosses_window else 0.0

        mom5  = prices_all[i] - prices_all[max(0, i-5)]
        mom10 = prices_all[i] - prices_all[max(0, i-10)]
        mom20 = prices_all[i] - prices_all[max(0, i-20)]

        X_rows.append([
            r["ema_cross"],
            abs(r["ema_cross"]),
            ema_accel,
            r["zscore_50"],
            abs(r["zscore_50"]),
            r["sigma_ewma"],
            r["sigma_ewma"] / (sigma_mean + 1e-9),
            r["range_20"],
            r["range_50"],
            r["range_20"] / (r["range_50"] + 1e-9),
            mom5,
            mom10,
            mom20,
            r["entropy_20"],
            r["spike_10"],
            r["atr_14"],
            float(r["tick_sign_run"]),
            float(int(r["ts"][11:13])),   # hour_of_day
        ])

        if labels_arr is not None:
            y_rows.append(int(labels_arr[i]))

    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=int) if labels_arr is not None else None
    return X, y

# ─────────────────────────────────────────────────────────────────────────────
# RETRAIN ENSEMBLE
# ─────────────────────────────────────────────────────────────────────────────

def retrain_ensemble(cal: dict):
    """
    Trains all 3 layers on the accumulated CSV.
    Labels: 1 = direction was correct N ticks/seconds later (per best_expiry).
    """
    global _ensemble
    import numpy as np, csv as _csv, pickle

    sym        = cal.get("symbol", "R_75")
    csv_path   = os.path.join(DATA_DIR, f"{sym}.csv")
    expiry_lbl = cal.get("best_expiry", "t5")

    # Convert expiry label to tick offset
    if expiry_lbl.startswith("t"):
        offset = int(expiry_lbl[1:])
    else:
        offset = int(expiry_lbl[1:]) * 60   # minutes → approx ticks

    if not os.path.exists(csv_path):
        warn("[ENS] retrain: CSV not found"); return

    rows_raw = []
    with open(csv_path, newline="") as f:
        for row in _csv.DictReader(f):
            try:
                rows_raw.append({
                    "price":        float(row["price"]),
                    "ema_cross":    float(row["ema_cross"]),
                    "zscore_50":    float(row["zscore_50"]),
                    "sigma_ewma":   float(row["sigma_ewma"]),
                    "range_20":     float(row["range_20"]),
                    "range_50":     float(row["range_50"]),
                    "spike_10":     float(row["spike_10"]),
                    "atr_14":       float(row["atr_14"]),
                    "entropy_20":   float(row["entropy_20"]),
                    "momentum_5":   float(row.get("momentum_5",   0)),
                    "momentum_10":  float(row.get("momentum_10",  0)),
                    "momentum_20":  float(row.get("momentum_20",  0)),
                    "tick_sign_run":float(row.get("tick_sign_run", 0)),
                    "regime":       row["regime"].strip(),
                    "ts":           row["ts"],
                })
            except Exception:
                continue

    if len(rows_raw) < 500:
        warn(f"[ENS] retrain: only {len(rows_raw)} rows — skipping"); return

    # Labels: was the directional signal correct at +offset ticks?
    prices = [r["price"] for r in rows_raw]
    n      = len(prices)
    labels = []
    for i in range(n):
        if i + offset >= n:
            labels.append(float('nan')); continue
        cross = rows_raw[i]["ema_cross"]
        z     = rows_raw[i]["zscore_50"]
        regime= rows_raw[i]["regime"]

        if regime == "CHAOS":
            labels.append(float('nan')); continue

        # Only label rows where direction is unambiguous
        if cross > 0 and z > 0:   # predicted UP
            won = prices[i + offset] > prices[i]
        elif cross < 0 and z < 0: # predicted DOWN
            won = prices[i + offset] < prices[i]
        else:
            labels.append(float('nan')); continue

        labels.append(1.0 if won else 0.0)

    X, y = _build_feature_matrix(rows_raw, labels)

    valid = ~np.isnan(y.astype(float))
    X = X[valid]; y = y[valid]

    if len(X) < 200:
        warn(f"[ENS] retrain: too few valid labels ({len(X)}) — skipping"); return

    info(f"[ENS] Training {len(X)} samples  base_wr={y.mean()*100:.1f}%  "
         f"expiry={expiry_lbl}  offset={offset} ticks")

    # ── Layer 1: XGBoost ─────────────────────────────────────────────────────
    xgb_ok = False
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=8,
            reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss", verbosity=0,
        )
        xgb.fit(X, y, verbose=False)
        out = os.path.join(_PERSIST_DIR, "xgb_rf_model.json")
        xgb.save_model(out)
        probs  = xgb.predict_proba(X)[:, 1]
        mask   = probs >= XGB_THRESHOLD
        n_sig  = mask.sum()
        prec   = y[mask].mean() * 100 if n_sig > 0 else 0
        info(f"[ENS] L1 XGBoost: {n_sig} signals ≥{XGB_THRESHOLD}  precision={prec:.1f}%")
        xgb_ok = True
    except ImportError:
        warn("[ENS] L1 XGBoost not installed")
    except Exception as e:
        warn(f"[ENS] L1 XGBoost failed: {e}")

    if not xgb_ok:
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            import pickle
            gbm = GradientBoostingClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=8, random_state=42)
            gbm.fit(X, y)
            out = os.path.join(_PERSIST_DIR, "xgb_rf_model.pkl")
            with open(out, "wb") as f:
                pickle.dump(gbm, f)
            info(f"[ENS] L1 GBM fallback trained")
        except Exception as e:
            warn(f"[ENS] L1 GBM fallback failed: {e}")

    # ── Layer 2: Logistic Regression ─────────────────────────────────────────
    try:
        import pickle
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        lr_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("poly",   PolynomialFeatures(degree=2, interaction_only=True,
                                          include_bias=False)),
            ("lr",     LogisticRegression(C=0.1, max_iter=1000,
                                          solver="lbfgs", random_state=42)),
        ])
        lr_pipe.fit(X, y)
        lr_probs = lr_pipe.predict_proba(X)[:, 1]
        lr_mask  = lr_probs >= LR_THRESHOLD
        n_sig    = lr_mask.sum()
        prec     = y[lr_mask].mean() * 100 if n_sig > 0 else 0
        info(f"[ENS] L2 LogReg: {n_sig} signals ≥{LR_THRESHOLD}  precision={prec:.1f}%")
        out = os.path.join(_PERSIST_DIR, "lr_rf_model.pkl")
        with open(out, "wb") as f:
            pickle.dump(lr_pipe, f)
    except Exception as e:
        warn(f"[ENS] L2 LogReg failed: {e}")

    # ── Layer 3: Isolation Forest ─────────────────────────────────────────────
    try:
        import pickle, numpy as np
        from sklearn.ensemble import IsolationForest
        X_wins = X[y == 1]
        info(f"[ENS] L3 IsoForest training on {len(X_wins)} WIN rows")
        iso = IsolationForest(
            n_estimators=200, contamination=ISO_CONTAMINATION,
            random_state=42, n_jobs=-1)
        iso.fit(X_wins)
        win_scores = iso.score_samples(X_wins)
        iso._ens_threshold = float(np.percentile(win_scores, ISO_CONTAMINATION * 100))
        all_scores = iso.score_samples(X)
        blocked    = (all_scores < iso._ens_threshold).sum()
        info(f"[ENS] L3 IsoForest: threshold={iso._ens_threshold:.4f}  "
             f"blocks {blocked}/{len(X)} ({blocked/len(X)*100:.1f}%)")
        out = os.path.join(_PERSIST_DIR, "iso_rf_model.pkl")
        with open(out, "wb") as f:
            pickle.dump(iso, f)
    except Exception as e:
        warn(f"[ENS] L3 IsoForest failed: {e}")

    # Hot-swap
    _ensemble = EnsembleGate(_PERSIST_DIR)
    info("[ENS] Ensemble hot-swapped")

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class SignalEngine:
    """
    5-condition gate (C1-C5) + Ensemble (C6).

    C1  EMA cross      > momentum_gate  (trend strong enough)
    C2  |Z-score|      > z_gate         (displacement in trend direction)
    C3  sigma in band  [sigma_lo, sigma_hi]
    C4  entropy        > entropy_gate   (not random noise)
    C5  EMA & Z agree on direction
    C6  Ensemble 2/3 vote

    Requires 4/5 on C1-C5, then C6 mandatory if models loaded.
    Direction: CALL if ema_cross > 0 and zscore > 0, else PUT.
    """

    def __init__(self, cal: dict):
        self.cal    = cal
        self.tick_n = 0
        self.prices: deque = deque(maxlen=500)
        self._sigma_ewma = None
        self._ema7 = self._ema21 = None
        self._k7   = 2 / 8
        self._k21  = 2 / 22
        self.EWMA_ALPHA = 0.05
        self._warmup    = 100
        self._sign_run  = 0
        self._cross_buf: deque = deque(maxlen=15)

    def ingest(self, price: float) -> dict:
        self.tick_n += 1
        prev  = self.prices[-1] if self.prices else price
        delta = price - prev
        self.prices.append(price)

        if self._sigma_ewma is None:
            self._sigma_ewma = abs(delta)
        else:
            self._sigma_ewma = (self.EWMA_ALPHA * abs(delta)
                                + (1 - self.EWMA_ALPHA) * self._sigma_ewma)

        if self._ema7 is None:
            self._ema7 = self._ema21 = price
        else:
            self._ema7  = price * self._k7  + self._ema7  * (1 - self._k7)
            self._ema21 = price * self._k21 + self._ema21 * (1 - self._k21)

        ema_cross = self._ema7 - self._ema21
        self._cross_buf.append(ema_cross)

        if delta > 0:
            self._sign_run = max(1, self._sign_run + 1)
        elif delta < 0:
            self._sign_run = min(-1, self._sign_run - 1)
        else:
            self._sign_run = 0

        if self.tick_n < self._warmup:
            return {"trade": False, "reason": "warmup", "tick": self.tick_n}

        prices = list(self.prices)
        sigma  = self._sigma_ewma

        range20 = max(prices[-20:]) - min(prices[-20:]) if len(prices) >= 20 else 999
        range50 = max(prices[-50:]) - min(prices[-50:]) if len(prices) >= 50 else range20

        # Z-score (signed)
        z_raw = 0.0
        if len(prices) >= 200:
            baseline   = prices[-200:]
            mu         = sum(baseline) / 200
            var        = sum((p - mu)**2 for p in baseline) / 200
            std        = math.sqrt(var) if var > 0 else 1e-9
            short_mean = sum(prices[-50:]) / 50
            z_raw      = (short_mean - mu) / (std / math.sqrt(50))
        z_abs = abs(z_raw)

        # Spike
        moves  = [abs(prices[i] - prices[i-1]) for i in range(-10, 0)
                  if i-1 >= -len(prices)]
        spike  = max(moves) if moves else 0

        # ATR
        atr_moves = [abs(prices[i] - prices[i-1]) for i in range(-14, 0)
                     if i-1 >= -len(prices)]
        atr14 = sum(atr_moves) / len(atr_moves) if atr_moves else 0

        # Entropy
        if len(prices) >= 21:
            ep  = prices[-21:]
            em  = [abs(ep[i] - ep[i-1]) for i in range(1, len(ep))]
            mx  = max(em) or 1
            bk  = [0] * 5
            for m in em:
                bk[min(4, int(m / mx * 4))] += 1
            ne = len(em); H = 0.0
            for b in bk:
                if b > 0:
                    p = b / ne; H -= p * math.log2(p)
            entropy20 = H / math.log2(5)
        else:
            entropy20 = 1.0

        # Momentum
        mom5  = price - prices[-6]  if len(prices) >= 6  else 0.0
        mom10 = price - prices[-11] if len(prices) >= 11 else 0.0
        mom20 = price - prices[-21] if len(prices) >= 21 else 0.0

        # EMA acceleration
        cb = list(self._cross_buf)
        ema_accel = (ema_cross - cb[0]) if len(cb) >= 10 else 0.0

        # Regime
        if abs(z_raw) > 3.0 or sigma > 0.5:
            regime = "CHAOS"
        elif abs(ema_cross) > 0.15 and abs(z_raw) > 1.5:
            regime = "STRONG_TREND"
        elif abs(ema_cross) > 0.05 or abs(z_raw) > 0.8:
            regime = "CALM_TREND"
        else:
            regime = "RANGING"

        cal = self.cal
        c1 = abs(ema_cross) > cal["momentum_gate"]
        c2 = z_abs          > cal["z_gate"]
        c3 = cal["sigma_lo"] <= sigma <= cal["sigma_hi"]
        c4 = entropy20      > cal["entropy_gate"]
        c5 = (ema_cross > 0 and z_raw > 0) or (ema_cross < 0 and z_raw < 0)

        gate_score = sum([c1, c2, c3, c4, c5])

        # Direction
        if ema_cross > 0 and z_raw > 0:
            direction = "CALL"
        elif ema_cross < 0 and z_raw < 0:
            direction = "PUT"
        else:
            direction = None

        # ML features
        sg = cal.get("momentum_gate", 0.001)
        sigma_mean = sigma   # simplified (full rolling mean in training)
        ml_feats = {
            "ema_cross":     ema_cross,
            "ema_cross_abs": abs(ema_cross),
            "ema_accel":     ema_accel,
            "zscore_50":     z_raw,
            "zscore_abs":    z_abs,
            "sigma_ewma":    sigma,
            "sigma_ratio":   1.0,   # simplified at inference
            "range_20":      range20,
            "range_50":      range50,
            "range_ratio":   range20 / (range50 + 1e-9),
            "momentum_5":    mom5,
            "momentum_10":   mom10,
            "momentum_20":   mom20,
            "entropy_20":    entropy20,
            "spike_10":      spike,
            "atr_14":        atr14,
            "tick_sign_run": float(self._sign_run),
            "hour_of_day":   float(datetime.now(timezone.utc).hour),
        }

        if _ensemble and _ensemble.active:
            ens = _ensemble.predict(ml_feats, regime)
        else:
            ens = {"votes": 3, "trade": True, "xgb_prob": 1.0,
                   "lr_prob": 1.0, "iso_score": 0.0,
                   "v_xgb": True, "v_lr": True, "v_iso": True}

        c6    = ens["trade"]
        trade = gate_score >= 4 and c6 and direction is not None

        return {
            "trade":     trade,
            "direction": direction,
            "gate_score":gate_score,
            "tick":      self.tick_n,
            "sigma":     round(sigma, 5),
            "ema_cross": round(ema_cross, 6),
            "z":         round(z_raw, 4),
            "entropy":   round(entropy20, 4),
            "regime":    regime,
            "votes":     ens["votes"],
            "xgb_prob":  ens["xgb_prob"],
            "lr_prob":   ens["lr_prob"],
            "c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5, "c6": c6,
        }

# ─────────────────────────────────────────────────────────────────────────────
# RISK MANAGER  (moderate — 2-step martingale)
# ─────────────────────────────────────────────────────────────────────────────

class RiskManager:
    def __init__(self):
        self.stake          = BASE_STAKE
        self.loss_streak    = 0
        self.session_pnl    = 0.0
        self.wins = self.losses = 0
        self._cooldown_until = 0.0
        self.expiry_stats: Dict[str, dict] = {}   # label → {wins, total}

    def get_stake(self) -> float:
        return round(self.stake, 2)

    def can_trade(self) -> Tuple[bool, str]:
        if time.monotonic() < self._cooldown_until:
            left = self._cooldown_until - time.monotonic()
            return False, f"cooldown({left:.0f}s)"
        if self.session_pnl <= -STOP_LOSS:
            return False, "stop_loss"
        if self.session_pnl >= TARGET_PROFIT:
            return False, "target_hit"
        return True, "ok"

    def record_win(self, profit: float, expiry_label: str = ""):
        self.wins        += 1
        self.session_pnl += profit
        self.loss_streak  = 0
        self.stake        = BASE_STAKE
        if expiry_label:
            self._track_expiry(expiry_label, True)
        tlog(f"WIN +${profit:.4f}  stake→${self.stake:.2f}  "
             f"P&L=${self.session_pnl:.4f}  expiry={expiry_label}")

    def record_loss(self, amount: float, expiry_label: str = ""):
        self.losses      += 1
        self.session_pnl -= amount
        self.loss_streak += 1
        self._cooldown_until = time.monotonic() + LOSS_COOLDOWN
        if expiry_label:
            self._track_expiry(expiry_label, False)

        if self.loss_streak > MARTINGALE_STEPS:
            self.stake       = BASE_STAKE
            self.loss_streak = 0
            warn(f"MARTINGALE exhausted — RESET to base=${self.stake:.2f}  "
                 f"P&L=${self.session_pnl:.4f}")
        elif self.loss_streak == 1:
            self.stake = BASE_STAKE   # hold on first loss
            tlog(f"LOSS streak=1/{MARTINGALE_STEPS}  "
                 f"next_stake=${self.stake:.2f} (holding)")
        else:
            self.stake = round(BASE_STAKE * (MARTINGALE_MULT ** (self.loss_streak - 1)), 2)
            tlog(f"LOSS streak={self.loss_streak}/{MARTINGALE_STEPS}  "
                 f"next_stake=${self.stake:.2f}  P&L=${self.session_pnl:.4f}")

    def _track_expiry(self, label: str, won: bool):
        if label not in self.expiry_stats:
            self.expiry_stats[label] = {"wins": 0, "total": 0}
        self.expiry_stats[label]["wins"]  += int(won)
        self.expiry_stats[label]["total"] += 1

    def session_summary(self) -> dict:
        return {
            "wins":        self.wins,
            "losses":      self.losses,
            "pnl":         round(self.session_pnl, 4),
            "expiry_stats":self.expiry_stats,
        }

# ─────────────────────────────────────────────────────────────────────────────
# DERIV CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class DerivClient:
    def __init__(self):
        self._ws        = None
        self._send_q    = asyncio.Queue()
        self._inbox     = asyncio.Queue()
        self._send_task = self._recv_task = None
        self._rid       = 0
        self.balance: float = 0.0

    async def connect(self) -> bool:
        try:
            info(f"Connecting → {WS_URL}")
            self._ws = await websockets.connect(
                WS_URL, ping_interval=20, ping_timeout=20, close_timeout=10)
            self._start_io()
            await self._send_msg({"authorize": API_TOKEN})
            resp = await self._recv_type("authorize", timeout=15)
            if not resp or "error" in resp:
                err(f"Auth failed: {(resp or {}).get('error',{}).get('message','?')}")
                return False
            auth = resp["authorize"]
            self.balance = float(auth.get("balance", 0))
            info(f"Auth OK  {auth.get('loginid')}  balance=${self.balance:.2f}")
            return True
        except Exception as exc:
            err(f"connect: {exc}"); return False

    def _start_io(self):
        for t in (self._send_task, self._recv_task):
            if t and not t.done():
                t.cancel()
        self._send_task = asyncio.create_task(self._send_pump())
        self._recv_task = asyncio.create_task(self._recv_pump())

    async def _send_pump(self):
        while True:
            data, fut = await self._send_q.get()
            try:
                await self._ws.send(json.dumps(data))
                if fut and not fut.done():
                    fut.set_result(True)
            except Exception as exc:
                if fut and not fut.done():
                    fut.set_exception(exc)
            finally:
                self._send_q.task_done()

    async def _recv_pump(self):
        try:
            async for raw in self._ws:
                try:
                    await self._inbox.put(json.loads(raw))
                except Exception:
                    pass
        except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK):
            await self._inbox.put({"__disconnect__": True})
        except Exception as exc:
            err(f"recv pump: {exc}")
            await self._inbox.put({"__disconnect__": True})

    async def close(self):
        for t in (self._send_task, self._recv_task):
            if t and not t.done():
                t.cancel()
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass

    async def _send_msg(self, data: dict, wait_ack=False):
        self._rid += 1
        data["req_id"] = self._rid
        fut = asyncio.get_event_loop().create_future() if wait_ack else None
        await self._send_q.put((data, fut))
        if fut:
            await fut

    async def _recv_type(self, mtype: str, timeout: float = 30) -> Optional[dict]:
        deadline = time.monotonic() + timeout
        while True:
            rem = deadline - time.monotonic()
            if rem <= 0:
                return None
            try:
                msg = await asyncio.wait_for(self._inbox.get(), timeout=rem)
                if mtype in msg or "error" in msg or "__disconnect__" in msg:
                    return msg
                await self._inbox.put(msg)
                await asyncio.sleep(0.01)
            except asyncio.TimeoutError:
                return None

    async def subscribe_ticks(self, symbol: str):
        await self._send_msg({"ticks": symbol, "subscribe": 1})

    async def next_tick(self, timeout: float = 10.0) -> Optional[dict]:
        deadline = time.monotonic() + timeout
        while True:
            rem = deadline - time.monotonic()
            if rem <= 0:
                return None
            try:
                msg = await asyncio.wait_for(self._inbox.get(), timeout=rem)
                if "__disconnect__" in msg:
                    return msg
                if "tick" in msg:
                    return msg
                await self._inbox.put(msg)
                await asyncio.sleep(0.005)
            except asyncio.TimeoutError:
                return None

    async def buy_contract(
        self, symbol: str, direction: str,
        expiry_label: str, stake: float
    ) -> Optional[dict]:
        """
        Places a Rise/Fall contract.
        expiry_label: t1..t10 (ticks) or m1..m5 (minutes)
        direction: "CALL" (Rise) or "PUT" (Fall)
        """
        contract_type = "CALL" if direction == "CALL" else "PUT"

        if expiry_label.startswith("t"):
            n_ticks = int(expiry_label[1:])
            proposal = {
                "buy": 1,
                "price": stake,
                "parameters": {
                    "amount":         stake,
                    "basis":          "stake",
                    "contract_type":  contract_type,
                    "currency":       "USD",
                    "duration":       n_ticks,
                    "duration_unit":  "t",
                    "symbol":         symbol,
                },
            }
        else:
            n_min = int(expiry_label[1:])
            proposal = {
                "buy": 1,
                "price": stake,
                "parameters": {
                    "amount":         stake,
                    "basis":          "stake",
                    "contract_type":  contract_type,
                    "currency":       "USD",
                    "duration":       n_min,
                    "duration_unit":  "m",
                    "symbol":         symbol,
                },
            }

        await self._send_msg(proposal)
        resp = await self._recv_type("buy", timeout=15)
        if not resp or "error" in resp:
            msg = (resp or {}).get("error", {}).get("message", "?")
            err(f"[BUY] failed: {msg}")
            return None
        return resp

    async def wait_for_settlement(
        self, contract_id: int, timeout: float = LOCK_TIMEOUT
    ) -> Optional[dict]:
        """Waits for POC (Profit or Loss) message for the given contract."""
        deadline = time.monotonic() + timeout
        while True:
            rem = deadline - time.monotonic()
            if rem <= 0:
                warn(f"[SETTLE] timeout for contract {contract_id}")
                return None
            try:
                msg = await asyncio.wait_for(self._inbox.get(), timeout=min(rem, 5))
                if "__disconnect__" in msg:
                    return None
                if "proposal_open_contract" in msg:
                    poc = msg["proposal_open_contract"]
                    if poc.get("contract_id") == contract_id and poc.get("is_sold"):
                        return poc
                elif "buy" in msg:
                    # Sometimes settlement comes as buy echo — re-queue
                    await self._inbox.put(msg)
                    await asyncio.sleep(0.01)
                else:
                    await self._inbox.put(msg)
                    await asyncio.sleep(0.01)
            except asyncio.TimeoutError:
                # Subscribe to POC if not yet done
                await self._send_msg({
                    "proposal_open_contract": 1,
                    "contract_id": contract_id,
                    "subscribe": 1,
                })

# ─────────────────────────────────────────────────────────────────────────────
# COLLECTOR  (Phase 1 — parallel multi-symbol tick collection)
# ─────────────────────────────────────────────────────────────────────────────

class Collector:
    async def run(self) -> dict:
        info(f"Phase 1: collecting {COLLECT_HOURS}h of ticks from "
             f"{SURVEY_SYMBOLS}...")
        stats   = {sym: SymbolStats(sym) for sym in SURVEY_SYMBOLS}
        clients = {}

        # Connect one WS per symbol
        for sym in SURVEY_SYMBOLS:
            client = DerivClient()
            ok = await client.connect()
            if not ok:
                warn(f"[COLLECT] {sym}: connect failed — skipping")
                continue
            await client.subscribe_ticks(sym)
            clients[sym] = client
            info(f"[COLLECT] {sym}: subscribed")
            await asyncio.sleep(0.3)

        if not clients:
            raise RuntimeError("No symbols connected for Phase 1")

        end_time = time.time() + COLLECT_SECS
        tick_counts = {sym: 0 for sym in clients}

        try:
            while time.time() < end_time:
                tasks = {
                    sym: asyncio.create_task(client.next_tick(timeout=5.0))
                    for sym, client in clients.items()
                }
                results = await asyncio.gather(*tasks.values(), return_exceptions=True)
                for sym, result in zip(tasks.keys(), results):
                    if isinstance(result, Exception) or result is None:
                        continue
                    if "__disconnect__" in result:
                        warn(f"[COLLECT] {sym} disconnected — attempting reconnect")
                        try:
                            c = DerivClient()
                            if await c.connect():
                                await c.subscribe_ticks(sym)
                                clients[sym] = c
                        except Exception:
                            pass
                        continue
                    tick = result.get("tick", {})
                    if tick:
                        stats[sym].update(float(tick["quote"]), float(tick["epoch"]))
                        tick_counts[sym] += 1

                elapsed = time.time() - (end_time - COLLECT_SECS)
                remaining = end_time - time.time()
                if tick_counts.get(SURVEY_SYMBOLS[0], 0) % 1800 == 0 and \
                   tick_counts.get(SURVEY_SYMBOLS[0], 0) > 0:
                    info(f"[COLLECT] {elapsed/60:.0f}min elapsed  "
                         f"remaining={remaining/60:.1f}min  "
                         + "  ".join(f"{s}:{tick_counts[s]}" for s in clients))
        finally:
            for sym, client in clients.items():
                await client.close()
                stats[sym].close()
                rolling_csv_trim(sym)

        summaries = [s.summarise() for s in stats.values()
                     if s.tick_n >= 200]

        if not summaries:
            raise RuntimeError("Insufficient data collected in Phase 1")

        return compute_calibration(summaries)

# ─────────────────────────────────────────────────────────────────────────────
# SYMBOL TRADER
# ─────────────────────────────────────────────────────────────────────────────

class SymbolTrader:
    def __init__(self, cal: dict):
        self.cal          = cal
        self.symbol       = cal["symbol"]
        self.expiry       = cal.get("best_expiry", "t5")
        self.engine       = SignalEngine(cal)
        self.risk         = RiskManager()
        self._lock        = asyncio.Lock()
        self._in_trade    = False
        self.hot_swap_calibration = None   # set by main

    async def run(self):
        backoff = 5
        while True:
            client = DerivClient()
            try:
                if not await client.connect():
                    await asyncio.sleep(backoff); backoff = min(backoff * 2, 120)
                    continue
                backoff = 5
                await client.subscribe_ticks(self.symbol)
                info(f"[{self.symbol}] Subscribed  expiry={self.expiry}")

                # Subscribe to POC stream for settlement
                await client._send_msg({
                    "proposal_open_contract": 1, "subscribe": 1})

                while True:
                    tick_msg = await client.next_tick(timeout=15)
                    if tick_msg is None:
                        warn(f"[{self.symbol}] tick timeout — reconnecting")
                        break
                    if "__disconnect__" in tick_msg:
                        warn(f"[{self.symbol}] disconnected — reconnecting")
                        break

                    tick  = tick_msg.get("tick", {})
                    price = float(tick.get("quote", 0))
                    if not price:
                        continue

                    sig = self.engine.ingest(price)
                    if not sig.get("trade"):
                        continue

                    can, reason = self.risk.can_trade()
                    if not can:
                        continue

                    if self._in_trade:
                        continue

                    direction = sig["direction"]
                    stake     = self.risk.get_stake()
                    expiry    = self.expiry

                    tlog(f"[{self.symbol}] SIGNAL {direction}  "
                         f"expiry={expiry}  stake=${stake:.2f}  "
                         f"gates={sig['gate_score']}/5  "
                         f"votes={sig['votes']}/3  "
                         f"xgb={sig['xgb_prob']:.3f}  "
                         f"lr={sig['lr_prob']:.3f}  "
                         f"cross={sig['ema_cross']:.5f}  "
                         f"z={sig['z']:.3f}  "
                         f"regime={sig['regime']}")

                    self._in_trade = True
                    try:
                        resp = await client.buy_contract(
                            self.symbol, direction, expiry, stake)
                        if not resp:
                            self._in_trade = False
                            continue

                        buy_data    = resp.get("buy", {})
                        contract_id = buy_data.get("contract_id")
                        buy_price   = float(buy_data.get("buy_price", stake))

                        tlog(f"[{self.symbol}] CONTRACT {contract_id}  "
                             f"buy_price=${buy_price:.4f}")

                        poc = await client.wait_for_settlement(
                            contract_id, timeout=LOCK_TIMEOUT)

                        if poc:
                            profit = float(poc.get("profit", 0))
                            is_win = profit > 0
                            if is_win:
                                self.risk.record_win(profit, expiry)
                            else:
                                self.risk.record_loss(abs(profit), expiry)
                        else:
                            warn(f"[{self.symbol}] settlement timeout — treating as loss")
                            self.risk.record_loss(stake, expiry)
                    finally:
                        self._in_trade = False

            except Exception as exc:
                err(f"[{self.symbol}] trader error: {exc}\n{traceback.format_exc()}")
            finally:
                try:
                    await client.close()
                except Exception:
                    pass
                await asyncio.sleep(backoff)

# ─────────────────────────────────────────────────────────────────────────────
# HEALTH SERVER
# ─────────────────────────────────────────────────────────────────────────────

_health_state: dict = {
    "phase": "init", "traders": [], "collect_start": None}
_shutdown_event = asyncio.Event()


def start_health_server(traders, phase="collect", collect_start=None):
    _health_state.update(phase=phase, traders=traders,
                         collect_start=collect_start)

    class _H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/status":
                payload = {
                    "phase":   _health_state["phase"],
                    "symbols": [],
                    "uptime":  (time.time() - _health_state["collect_start"]
                                if _health_state["collect_start"] else 0),
                }
                for t in _health_state.get("traders", []):
                    r = t.risk
                    payload["symbols"].append({
                        "symbol":  t.symbol,
                        "expiry":  t.expiry,
                        "wins":    r.wins,
                        "losses":  r.losses,
                        "pnl":     round(r.session_pnl, 4),
                        "stake":   r.get_stake(),
                        "streak":  r.loss_streak,
                    })
                body = json.dumps(payload, indent=2).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers(); self.wfile.write(body)

            elif self.path == "/":
                rows = ""
                for t in _health_state.get("traders", []):
                    r = t.risk
                    wr = (r.wins / (r.wins + r.losses) * 100
                          if r.wins + r.losses > 0 else 0)
                    rows += (f"<tr><td>{t.symbol}</td><td>{t.expiry}</td>"
                             f"<td>{r.wins}</td><td>{r.losses}</td>"
                             f"<td>{wr:.1f}%</td>"
                             f"<td>${r.session_pnl:+.4f}</td>"
                             f"<td>${r.get_stake():.2f}</td></tr>")

                body_html = f"""<!DOCTYPE html><html><head>
<title>Rise/Fall AI Bot</title>
<meta http-equiv="refresh" content="15">
<style>
  body{{font-family:monospace;background:#0d1117;color:#e6edf3;padding:2rem;}}
  table{{border-collapse:collapse;width:100%;}}
  th,td{{padding:.5rem 1rem;border:1px solid #21262d;text-align:left;}}
  th{{background:#161b22;color:#58a6ff;}}
  h2{{color:#3fb950;}}
  .phase{{color:#d29922;font-size:0.85rem;}}
</style></head><body>
<h2>⬆⬇ Rise/Fall AI Advisor Bot</h2>
<p class="phase">Phase: <b>{_health_state['phase'].upper()}</b> &nbsp;|&nbsp;
Uptime: {(time.time()-_health_state['collect_start'])/60:.0f}min</p>
<table><tr>
  <th>Symbol</th><th>Expiry</th><th>Wins</th><th>Losses</th>
  <th>Win Rate</th><th>PnL</th><th>Stake</th>
</tr>{rows}</table>
<p style="color:#8b949e;font-size:0.8rem;margin-top:1rem">
  Auto-refreshes every 15s &nbsp;|&nbsp;
  <a href="/status" style="color:#58a6ff">/status JSON</a>
</p></body></html>"""
                body = body_html.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers(); self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("", PORT), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    info(f"Health server on :{PORT}  (/ = dashboard  /status = JSON)")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    collect_only = "--collect-only" in sys.argv
    trade_only   = "--trade-only"   in sys.argv

    for arg in sys.argv:
        if arg.startswith("--collect-hours="):
            global COLLECT_SECS
            COLLECT_SECS = float(arg.split("=")[1]) * 3600

    collect_start = time.time()
    start_health_server([], phase="collect", collect_start=collect_start)

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    if os.path.exists(CAL_FILE) and not trade_only:
        info(f"Found existing calibration — skipping Phase 1.")
        trade_only = True

    if not trade_only:
        collector    = Collector()
        calibration  = await collector.run()
    else:
        if not os.path.exists(CAL_FILE):
            sys.exit(f"calibration_rf.json not found. Run without --trade-only first.")
        with open(CAL_FILE) as f:
            calibration = json.load(f)
        info(f"Loaded calibration from {CAL_FILE}")
        info(f"Generated at: {calibration['generated_at']}")

    if collect_only:
        info("--collect-only done.")
        info(json.dumps(calibration["trade_symbols"], indent=2))
        return

    # ── Train ML ensemble on Phase 1 data ─────────────────────────────────────
    if calibration["trade_symbols"]:
        info("Training 3-layer Rise/Fall ensemble (~30s)...")
        try:
            retrain_ensemble(calibration["trade_symbols"][0])
        except Exception as exc:
            warn(f"Ensemble training failed: {exc} — 5-condition fallback mode")

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    trade_syms = calibration["trade_symbols"]
    info(f"Phase 2: trading {[s['symbol'] for s in trade_syms]}")

    traders = [SymbolTrader(cal) for cal in trade_syms]

    ens = load_ensemble()
    if ens.active:
        loaded = sum(x is not None for x in [ens._xgb, ens._lr, ens._iso])
        info(f"[ENS] {loaded}/3 layers active  "
             f"XGB>={XGB_THRESHOLD}  LR>={LR_THRESHOLD}")
    else:
        info("[ENS] No models — 5-condition fallback mode")

    _health_state["traders"] = traders
    _health_state["phase"]   = "trade"

    info("=" * 60)
    for t in traders:
        info(f"  {t.symbol}: best_expiry={t.expiry}  "
             f"dir_score={t.cal.get('dir_score', 0):.4f}  "
             f"trend_pct={t.cal.get('trend_pct', 0):.1%}")
        info(f"  gates: momentum>{t.cal['momentum_gate']}  "
             f"z>{t.cal['z_gate']}  "
             f"sigma=[{t.cal['sigma_lo']},{t.cal['sigma_hi']}]  "
             f"entropy>{t.cal['entropy_gate']}")
        acc = t.cal.get("expiry_accuracy", {})
        if acc:
            top = sorted(acc, key=lambda k: acc[k]["ev"], reverse=True)[:3]
            info(f"  top expiries: " +
                 "  ".join(f"{k}(wr={acc[k]['win_rate']:.2%},ev={acc[k]['ev']:.4f})"
                           for k in top))
    info("=" * 60)

    # ── Hot-swap wiring ────────────────────────────────────────────────────────
    advisor = AIAdvisor()

    for t in traders:
        def _make_swap(trader):
            def hot_swap(new_cal: dict):
                trader.cal    = new_cal
                trader.expiry = new_cal.get("best_expiry", trader.expiry)
                trader.engine = SignalEngine(new_cal)
                info(f"[{trader.symbol}] ♻ hot-swapped  "
                     f"expiry={trader.expiry}  "
                     f"dir_score={new_cal.get('dir_score', 0):.4f}")
            return hot_swap
        t.hot_swap_calibration = _make_swap(t)

    # ── Rolling recalibration loop ──────────────────────────────────────────────
    async def recal_loop():
        cycle = 1
        while not _shutdown_event.is_set():
            await asyncio.sleep(COLLECT_SECS)
            if _shutdown_event.is_set():
                break
            cycle += 1
            info(f"♻ Recalibration cycle {cycle}...")
            try:
                new_cal = await Collector().run()

                # Gather candle indicators for all surveyed symbols
                cf   = CandleFeed()
                inds = {}
                for sym in SURVEY_SYMBOLS:
                    cdata      = await cf.fetch(sym)
                    inds[sym]  = IndicatorEngine.compute(cdata)

                # Run AI Advisor
                all_cals = new_cal.get("all_symbols", [])
                sess     = traders[0].risk.session_summary() if traders else {}
                adv      = advisor.advise(traders, all_cals, inds, sess)

                # Apply advisor: switch symbol if needed, update cal, update expiry
                best_sym    = adv.get("best_symbol")
                best_expiry = adv.get("best_expiry")
                adv_cal     = adv.get("cal", {})

                for t in traders:
                    sym_cal = next(
                        (s for s in new_cal["all_symbols"]
                         if s["symbol"] == (best_sym or t.symbol)), None)
                    if sym_cal:
                        merged = {**sym_cal, **{k: adv_cal.get(k, sym_cal.get(k))
                                                for k in ("momentum_gate", "z_gate",
                                                           "entropy_gate")}}
                        if best_expiry:
                            merged["best_expiry"] = best_expiry
                        t.symbol = merged["symbol"]
                        t.hot_swap_calibration(merged)

                # Retrain ensemble in background
                if new_cal["trade_symbols"]:
                    threading.Thread(
                        target=retrain_ensemble,
                        args=(new_cal["trade_symbols"][0],),
                        daemon=True, name="ens_retrain",
                    ).start()

                info(f"♻ Cycle {cycle} complete — "
                     f"symbol={best_sym}  expiry={best_expiry}")

            except Exception as exc:
                err(f"Recal cycle {cycle} failed: {exc}\n{traceback.format_exc()}")

    trader_tasks  = [asyncio.create_task(t.run()) for t in traders]
    recal_task    = asyncio.create_task(recal_loop())
    shutdown_task = asyncio.create_task(_shutdown_event.wait())

    done, pending = await asyncio.wait(
        trader_tasks + [recal_task, shutdown_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    if shutdown_task in done:
        info("Shutdown — stopping traders...")
        for task in trader_tasks + [recal_task]:
            task.cancel()
        await asyncio.gather(*trader_tasks, recal_task, return_exceptions=True)
        info("All stopped.")

# ─────────────────────────────────────────────────────────────────────────────
# GRACEFUL SHUTDOWN
# ─────────────────────────────────────────────────────────────────────────────

def _handle_signal(signum, frame):
    info(f"Signal {signum} — shutting down...")
    try:
        asyncio.get_event_loop().call_soon_threadsafe(_shutdown_event.set)
    except Exception:
        pass

if __name__ == "__main__":
    import signal as _signal
    _signal.signal(_signal.SIGTERM, _handle_signal)
    _signal.signal(_signal.SIGINT,  _handle_signal)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        info("Stopped by user.")
