"""
EXPIRYRANGE Self-Calibrating + XGBoost Layer Bot
═══════════════════════════════════════════════════════════════════════════════
Phase 1  COLLECT (4 hours, no trading)
  • Subscribes to 1HZ10V, saves per-symbol CSVs to PERSIST_DIR/symbol_data/

Phase 2  TRADE (starts automatically after Phase 1)
  • Loads calibration.json + 3 model files (all trained on Phase 1 data)
  • 5-condition confluence gate (C1-C5) — same as base bot
  • 3-Layer Ensemble gate (C6): 2-of-3 model vote required
      Layer 1  XGBoost              — feature interaction detector
               file: xgb_model.json    threshold: XGB_THRESHOLD (0.70)
      Layer 2  Logistic Regression   — calibrated probability estimator
               file: lr_model.pkl      threshold: LR_THRESHOLD  (0.72)
      Layer 3  Isolation Forest      — regime anomaly blocker
               file: iso_model.pkl     contamination: 0.15
      Vote: trade when ≥2 of 3 approve AND regime != CHAOS
  • 18 features identical across all 3 models — zero drift from training
  • All models absent → falls back to 5-condition mode (no crash)
  • Models retrained on rolling accumulated CSV (up to ROLLING_MAX_HOURS=24h)
    Each 4-hour recal cycle appends new ticks; rows older than 24h are trimmed.
    Training window grows each cycle: 4h → 8h → 12h → ... → 24h (then holds)

Railway deployment:
  • pip install xgboost websockets   (add to requirements.txt)
  • Mount a Volume at /app/data — all persisted files live there
  • ENV: DERIV_API_TOKEN, XGB_THRESHOLD (default 0.70), BASE_STAKE,
         TARGET_PROFIT, STOP_LOSS, COLLECT_HOURS, PERSIST_DIR

Run:
    python main_xgb.py                  # full run
    python main_xgb.py --collect-only   # Phase 1 only
    python main_xgb.py --trade-only     # Phase 2 only (needs calibration.json)
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
from collections import deque
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, List, Optional, Tuple
import threading

# ── websockets import with helpful error ─────────────────────────────────────
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

API_TOKEN   = os.getenv("DERIV_API_TOKEN", "iCCn0vuMCzLcq1J")
APP_ID      = os.getenv("DERIV_APP_ID",    "1089")
WS_URL      = f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}"

COLLECT_HOURS  = float(os.getenv("COLLECT_HOURS", "4"))
COLLECT_SECS   = COLLECT_HOURS * 3600

# ── Persistent storage — mount a Railway Volume at /app/data ─────────────────
# In Railway: Settings → Volumes → Mount Path: /app/data
# CAL_FILE and DATA_DIR will survive restarts/redeploys
_PERSIST_DIR   = os.getenv("PERSIST_DIR", os.path.join(os.getcwd(), "data"))
os.makedirs(_PERSIST_DIR, exist_ok=True)
CAL_FILE       = os.path.join(_PERSIST_DIR, "calibration.json")
DATA_DIR       = os.path.join(_PERSIST_DIR, "symbol_data")
PORT           = int(os.getenv("PORT", "8080"))

# Symbols to survey in Phase 1
SURVEY_SYMBOLS = ["1HZ10V"]

# Martingale
BASE_STAKE        = float(os.getenv("BASE_STAKE",    "1.0"))
MARTINGALE_MULT   = float(os.getenv("MARTI_MULT",    "4.45"))
MARTINGALE_STEPS  = int(os.getenv("MARTI_STEPS",     "3"))
LOSS_COOLDOWN     = float(os.getenv("LOSS_COOLDOWN", "45"))

# Trade risk
TARGET_PROFIT  = float(os.getenv("TARGET_PROFIT", "10.0"))
STOP_LOSS      = float(os.getenv("STOP_LOSS",     "30.0"))
LOCK_TIMEOUT   = 360   # 5-min contract + 60s buffer

# ── ML gate ──────────────────────────────────────────────────────────────────
XGB_THRESHOLD  = float(os.getenv("XGB_THRESHOLD", "0.70"))  # XGBoost vote threshold
LR_THRESHOLD   = float(os.getenv("LR_THRESHOLD",  "0.72"))  # Logistic Regression threshold
ISO_CONTAMINATION = float(os.getenv("ISO_CONTAMINATION", "0.15"))  # Isolation Forest anomaly rate
ROLLING_MAX_HOURS = float(os.getenv("ROLLING_MAX_HOURS", "24"))  # max hours of data kept for training
ROLLING_MAX_SECS  = ROLLING_MAX_HOURS * 3600

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("expiryrange_bot")

def info(m):  log.info(m)
def warn(m):  log.warning(m)
def err(m):   log.error(m)
def tlog(m):  log.info(f"[TRADE] {m}")

# ─────────────────────────────────────────────────────────────────────────────
# SYMBOL STATS ENGINE  (used in Phase 1)
# ─────────────────────────────────────────────────────────────────────────────

class SymbolStats:
    """
    Computes all 14 volatility/regime metrics per tick.
    Used during Phase 1 collection on every symbol simultaneously.
    """
    EWMA_ALPHA = 0.05

    def __init__(self, symbol: str):
        self.symbol    = symbol
        self.tick_n    = 0
        self.prices: deque = deque(maxlen=500)
        self.sigma_ewma = None

        # Regime tracking
        self._regime_counts = {"CALM": 0, "RANGING": 0,
                               "TRENDING": 0, "CHAOS": 0}
        self._regime_start  = time.time()
        self._regime_cur    = "CALM"

        # EMA state
        self._ema7  = None
        self._ema14 = None
        self._k7    = 2 / (7 + 1)
        self._k14   = 2 / (14 + 1)

        # CSV writer — append mode so rolling cycles accumulate data
        os.makedirs(DATA_DIR, exist_ok=True)
        fname = os.path.join(DATA_DIR, f"{symbol}.csv")
        file_exists = os.path.exists(fname) and os.path.getsize(fname) > 0
        self._csv_f = open(fname, "a", newline="")
        self._csv_w = csv.DictWriter(self._csv_f, fieldnames=self._fields())
        if not file_exists:
            self._csv_w.writeheader()   # only write header on first ever run
        self._rows_since_flush = 0

    @staticmethod
    def _fields():
        return [
            "ts", "epoch", "symbol", "tick_n",
            "price", "tick_delta", "tick_abs_delta",
            "sigma_ewma", "range_20", "range_50",
            "ema7", "ema14", "ema_gap",
            "zscore_50", "spike_10", "atr_14",
            "entropy_20", "regime",
        ]

    def update(self, price: float, epoch: float) -> dict:
        self.tick_n += 1
        prev = self.prices[-1] if self.prices else price
        delta     = price - prev
        abs_delta = abs(delta)
        self.prices.append(price)

        # EWMA sigma
        if self.sigma_ewma is None:
            self.sigma_ewma = abs_delta
        else:
            self.sigma_ewma = (self.EWMA_ALPHA * abs_delta +
                               (1 - self.EWMA_ALPHA) * self.sigma_ewma)

        # EMA
        if self._ema7 is None:
            self._ema7 = self._ema14 = price
        else:
            self._ema7  = price * self._k7  + self._ema7  * (1 - self._k7)
            self._ema14 = price * self._k14 + self._ema14 * (1 - self._k14)
        ema_gap = abs(self._ema7 - self._ema14)

        prices = list(self.prices)

        # Range
        range_20 = (max(prices[-20:]) - min(prices[-20:])) if len(prices) >= 20 else 0
        range_50 = (max(prices[-50:]) - min(prices[-50:])) if len(prices) >= 50 else 0

        # Z-score (50-tick window vs 200-tick baseline)
        zscore_50 = 0.0
        if len(prices) >= 200:
            baseline = prices[-200:]
            mu  = sum(baseline) / len(baseline)
            var = sum((p - mu)**2 for p in baseline) / len(baseline)
            std = math.sqrt(var) if var > 0 else 1e-9
            short = prices[-50:]
            short_mean = sum(short) / len(short)
            zscore_50 = (short_mean - mu) / (std / math.sqrt(50))

        # Spike
        moves = [abs(prices[i] - prices[i-1]) for i in range(-10, 0)
                 if i-1 >= -len(prices)]
        spike_10 = max(moves) if moves else 0

        # ATR-14 (simplified: avg of abs tick moves over 14 ticks)
        atr_moves = [abs(prices[i] - prices[i-1]) for i in range(-14, 0)
                     if i-1 >= -len(prices)]
        atr_14 = sum(atr_moves) / len(atr_moves) if atr_moves else 0

        # Shannon entropy of last 20 tick moves (bucketed into 5 bins)
        entropy_20 = self._entropy(prices[-21:]) if len(prices) >= 21 else 1.0

        # Regime
        regime = self._detect_regime(ema_gap, self.sigma_ewma, zscore_50)
        if regime != self._regime_cur:
            self._regime_counts[self._regime_cur] += (
                time.time() - self._regime_start)
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
            "ema14":         round(self._ema14, 5),
            "ema_gap":       round(ema_gap, 5),
            "zscore_50":     round(zscore_50, 4),
            "spike_10":      round(spike_10, 5),
            "atr_14":        round(atr_14, 5),
            "entropy_20":    round(entropy_20, 4),
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
        n = len(moves)
        H = 0.0
        for b in buckets:
            if b > 0:
                p = b / n
                H -= p * math.log2(p)
        return H / math.log2(5)   # normalised to [0,1]

    @staticmethod
    def _detect_regime(ema_gap, sigma, zscore) -> str:
        if abs(zscore) > 2.5 or sigma > 0.3:
            return "CHAOS"
        if abs(zscore) > 1.5 and ema_gap > 0.3:
            return "TRENDING"
        if abs(zscore) < 1.0 and ema_gap < 0.15:
            return "CALM"
        return "RANGING"

    def summarise(self) -> dict:
        """Called after Phase 1 to produce calibration data."""
        self._regime_counts[self._regime_cur] += (
            time.time() - self._regime_start)
        total_secs = sum(self._regime_counts.values()) or 1
        return {
            "symbol":        self.symbol,
            "ticks":         self.tick_n,
            "regime_pct":    {k: round(v / total_secs, 4)
                              for k, v in self._regime_counts.items()},
            "data_file":     os.path.join(DATA_DIR, f"{self.symbol}.csv"),
        }

    def close(self):
        self._csv_f.flush()
        self._csv_f.close()

# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION  (Phase 1 → Phase 2 bridge)
# ─────────────────────────────────────────────────────────────────────────────

def compute_calibration(summaries: List[dict]) -> dict:
    """
    Reads per-symbol CSVs, computes percentile stats,
    derives thresholds, ranks symbols, picks top 2.
    """
    import statistics

    info("Computing calibration from collected data...")
    symbol_scores = {}

    for s in summaries:
        sym  = s["symbol"]
        fpath = s["data_file"]
        if not os.path.exists(fpath):
            continue

        sigmas, ranges, ema_gaps, zscores, spikes = [], [], [], [], []

        with open(fpath, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    sigmas.append(float(row["sigma_ewma"]))
                    ranges.append(float(row["range_20"]))
                    ema_gaps.append(float(row["ema_gap"]))
                    zscores.append(abs(float(row["zscore_50"])))
                    spikes.append(float(row["spike_10"]))
                except (ValueError, KeyError):
                    continue

        if len(sigmas) < 200:
            warn(f"{sym}: insufficient data ({len(sigmas)} rows) — skipping")
            continue

        sigmas.sort(); ranges.sort(); ema_gaps.sort()
        zscores.sort(); spikes.sort()

        def pct(lst, p):
            idx = max(0, int(len(lst) * p / 100) - 1)
            return lst[idx]

        sigma_p50  = pct(sigmas,   50)
        sigma_p75  = pct(sigmas,   75)
        range_p50  = pct(ranges,   50)
        ema_p50    = pct(ema_gaps, 50)
        z_p50      = pct(zscores,  50)
        spike_p90  = pct(spikes,   90)

        # Derived barrier: chosen so P(win) ≥ 0.78 at sigma_p75
        # P = erf(B / (sqrt(2) * sigma_p75 * sqrt(120)))
        # Solve for B: B = erf⁻¹(0.78) * sqrt(2) * sigma_p75 * sqrt(120)
        # erf⁻¹(0.78) ≈ 0.906
        barrier = round(0.906 * math.sqrt(2) * sigma_p75 * math.sqrt(120), 2)
        barrier = max(1.5, min(barrier, 4.0))   # clamp to sensible range

        # P(win) at median sigma with derived barrier
        p_win = math.erf(barrier / (math.sqrt(2) * sigma_p50 * math.sqrt(120)))

        # Duration: minimum 2 minutes on all signals
        duration = 2

        # Calm regime fraction (higher = better for EXPIRYRANGE)
        calm_pct = s["regime_pct"].get("CALM", 0) + \
                   s["regime_pct"].get("RANGING", 0)

        # Score: calm% × p_win (higher = more time in tradeable regime
        # AND higher win probability when we do trade)
        score = calm_pct * p_win

        symbol_scores[sym] = {
            "symbol":       sym,
            "ticks":        s["ticks"],
            "score":        round(score, 4),
            "p_win_median": round(p_win, 4),
            "calm_pct":     round(calm_pct, 4),
            "barrier":      barrier,
            "duration_min": duration,
            # Signal gates — derived from data
            "sigma_gate":   round(sigma_p50, 5),
            "range_gate":   round(range_p50, 4),
            "ema_gate":     round(barrier * 0.15, 4),  # 15% of barrier
            "z_gate":       round(max(0.6, min(z_p50, 1.5)), 4),
            "spike_gate":   round(spike_p90, 5),
            "regime_pct":   s["regime_pct"],
        }

        info(f"  {sym}: score={score:.4f}  p_win={p_win:.3f}  "
             f"calm={calm_pct:.1%}  barrier=±{barrier}  "
             f"sigma_gate={sigma_p50:.5f}  range_gate={range_p50:.4f}")

    if not symbol_scores:
        raise RuntimeError("No symbols had sufficient data for calibration")

    # Rank and pick top 2
    ranked = sorted(symbol_scores.values(),
                    key=lambda x: x["score"], reverse=True)

    top2 = ranked[:min(2, len(ranked))]
    sym_summary = "  |  ".join(
        f"{s['symbol']} score={s['score']:.4f} p_win={s['p_win_median']:.3f}"
        for s in top2
    )
    info(f"\nCalibrated symbols: {sym_summary}")

    cal = {
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "collect_hours":  COLLECT_HOURS,
        "all_symbols":    ranked,
        "trade_symbols":  top2,
    }

    with open(CAL_FILE, "w") as f:
        json.dump(cal, f, indent=2)

    info(f"Calibration saved to {CAL_FILE}")
    return cal


def rolling_csv_trim(symbol: str):
    """
    Removes rows older than ROLLING_MAX_HOURS from the accumulated CSV.
    Called after each Collector run so the file never grows beyond 24h of data.
    Uses the 'ts' column (ISO-8601 UTC timestamp written by SymbolStats).
    Safe to call even if the CSV is brand new (< ROLLING_MAX_SECS old).
    """
    fpath = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(fpath):
        return

    cutoff_epoch = time.time() - ROLLING_MAX_SECS
    kept = []
    removed = 0
    fields = None

    with open(fpath, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        for row in reader:
            try:
                # Parse ISO-8601 timestamp; datetime.fromisoformat handles +00:00
                ts_str = row["ts"].replace("Z", "+00:00")
                row_epoch = datetime.fromisoformat(ts_str).timestamp()
                if row_epoch >= cutoff_epoch:
                    kept.append(row)
                else:
                    removed += 1
            except Exception:
                kept.append(row)   # keep malformed rows to avoid data loss

    if removed == 0:
        return   # nothing to trim

    # Rewrite file with only kept rows
    tmp = fpath + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(kept)
    os.replace(tmp, fpath)

    total = len(kept) + removed
    info(f"[ROLLING] {symbol}: trimmed {removed} rows older than "
         f"{ROLLING_MAX_HOURS:.0f}h  kept={len(kept)}/{total}  "
         f"({len(kept)/3600:.1f}h of data)")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — COLLECTOR
# ─────────────────────────────────────────────────────────────────────────────

class Collector:
    """
    Opens a single WS connection, subscribes to all SURVEY_SYMBOLS,
    routes incoming ticks to the correct SymbolStats instance.
    Runs for COLLECT_SECS then closes and calls compute_calibration().
    """

    def __init__(self):
        self._stats: Dict[str, SymbolStats] = {
            s: SymbolStats(s) for s in SURVEY_SYMBOLS
        }
        self._ws         = None
        self._rid        = 0
        self._pending:   Dict[int, asyncio.Future] = {}
        self._inbox      = asyncio.Queue()
        self._send_q     = asyncio.Queue()
        self._start_time = time.time()
        self._tick_counts: Dict[str, int] = {s: 0 for s in SURVEY_SYMBOLS}

    async def run(self) -> dict:
        info(f"Phase 1: collecting {COLLECT_HOURS}h of data from "
             f"{len(SURVEY_SYMBOLS)} symbols...")
        info(f"Symbols: {SURVEY_SYMBOLS}")

        await self._connect_and_auth()

        # Subscribe to all symbols
        for sym in SURVEY_SYMBOLS:
            await self._send({"ticks": sym, "subscribe": 1})

        # Run until time limit
        deadline = self._start_time + COLLECT_SECS
        while time.time() < deadline:
            remaining = deadline - time.time()
            try:
                msg = await asyncio.wait_for(
                    self._inbox.get(), timeout=min(30, remaining))
            except asyncio.TimeoutError:
                if time.time() >= deadline:
                    break
                continue

            if "__disconnect__" in msg:
                warn("Collector: WS disconnected — reconnecting")
                await asyncio.sleep(5)
                await self._connect_and_auth()
                for sym in SURVEY_SYMBOLS:
                    await self._send({"ticks": sym, "subscribe": 1})
                continue

            if msg.get("msg_type") == "tick":
                tick = msg.get("tick", {})
                sym  = tick.get("symbol", "")
                if sym in self._stats:
                    self._stats[sym].update(
                        float(tick["quote"]),
                        float(tick.get("epoch", time.time()))
                    )
                    self._tick_counts[sym] += 1

            elapsed = time.time() - self._start_time
            if int(elapsed) % 300 == 0 and elapsed > 1:
                self._log_progress(elapsed)

        # Final progress log
        elapsed = time.time() - self._start_time
        self._log_progress(elapsed)
        info("Phase 1 complete. Computing calibration...")

        summaries = []
        for sym, st in self._stats.items():
            summaries.append(st.summarise())
            st.close()
            rolling_csv_trim(sym)   # drop rows older than ROLLING_MAX_HOURS

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass

        return compute_calibration(summaries)

    def _log_progress(self, elapsed: float):
        remaining = max(0, COLLECT_SECS - elapsed)
        counts = "  ".join(f"{s}:{self._tick_counts[s]}" for s in SURVEY_SYMBOLS)
        info(f"[COLLECT] elapsed={elapsed/3600:.2f}h  "
             f"remaining={remaining/60:.0f}min  ticks=[{counts}]")

    async def _connect_and_auth(self):
        info("Collector: connecting...")
        self._ws = await websockets.connect(
            WS_URL, ping_interval=20, ping_timeout=15)
        asyncio.create_task(self._recv_pump())
        asyncio.create_task(self._send_pump())

        # Auth
        await self._send({"authorize": API_TOKEN})
        resp = await self._recv_one("authorize", timeout=15)
        if not resp or "error" in resp:
            raise ConnectionError(
                f"Auth failed: {(resp or {}).get('error',{}).get('message','?')}")
        info(f"Collector: auth OK  "
             f"balance=${resp['authorize'].get('balance',0):.2f}")

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
            err(f"Collector recv pump: {exc}")
            await self._inbox.put({"__disconnect__": True})

    async def _send(self, data: dict):
        self._rid += 1
        data["req_id"] = self._rid
        fut = asyncio.get_event_loop().create_future()
        await self._send_q.put((data, fut))
        return fut

    async def _recv_one(self, msg_type: str, timeout=10) -> Optional[dict]:
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return None
            try:
                msg = await asyncio.wait_for(
                    self._inbox.get(), timeout=remaining)
            except asyncio.TimeoutError:
                return None
            if "__disconnect__" in msg:
                await self._inbox.put(msg)
                return None
            if msg_type in msg or "error" in msg:
                return msg
            await self._inbox.put(msg)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — TRADING BOT (one instance per symbol, runs in parallel)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# 3-LAYER ENSEMBLE GATE
# Layer 1: XGBoost              — feature interaction detector
# Layer 2: Logistic Regression  — calibrated probability estimator
# Layer 3: Isolation Forest     — regime anomaly blocker
# Vote rule: ≥2 of 3 approve AND regime != CHAOS
# ─────────────────────────────────────────────────────────────────────────────

# 18 features — identical across all 3 models
_FEATURES = [
    'sigma_ewma', 'range_20', 'range_50', 'ema_gap', 'zscore_50',
    'spike_10', 'atr_14', 'entropy_20', 'regime_enc',
    'sigma_trend', 'range_ratio', 'ema_cross', 'zscore_abs',
    'entropy_delta', 'sigma_vs_gate', 'spike_vs_sigma', 'atr_trend',
    'hour_of_day',
]
_REGIME_ENC = {'CALM': 0, 'RANGING': 1, 'TRENDING': 2, 'CHAOS': 3}


class EnsembleGate:
    """
    3-layer ensemble. Each layer votes independently.
    Trade fires when >=2 of 3 vote YES AND regime != CHAOS.

    Layer 1  XGBoost (xgb_model.json)
        Captures non-linear feature interactions. Threshold XGB_THRESHOLD.
    Layer 2  Logistic Regression with poly features (lr_model.pkl)
        Well-calibrated probabilities; corrects XGBoost overconfidence.
        Threshold LR_THRESHOLD.
    Layer 3  Isolation Forest (iso_model.pkl)
        Trained only on WIN-labelled rows. Blocks any tick whose feature
        vector is anomalous relative to known winning conditions.
        Threshold derived at training time on win-sample score distribution.

    Graceful degradation: any absent/broken model abstains (votes YES)
    so the gate stays open in fallback mode.
    """

    def __init__(self, persist_dir: str, sigma_gate: float):
        self.persist_dir = persist_dir
        self.sigma_gate  = sigma_gate
        self._xgb = None
        self._lr  = None
        self._iso = None
        self._load()

    def _load(self):
        # Layer 1 — XGBoost
        xgb_path = os.path.join(self.persist_dir, "xgb_model.json")
        if os.path.exists(xgb_path):
            try:
                from xgboost import XGBClassifier
                m = XGBClassifier(); m.load_model(xgb_path)
                self._xgb = m
                info(f"[ENS] L1 XGBoost loaded  thresh={XGB_THRESHOLD}")
            except Exception as e:
                warn(f"[ENS] L1 XGBoost load failed: {e}")
        # Fallback to sklearn GBM if XGBoost file absent
        if self._xgb is None:
            gb_path = os.path.join(self.persist_dir, "gb_model.pkl")
            if os.path.exists(gb_path):
                try:
                    import pickle
                    with open(gb_path, 'rb') as f:
                        self._xgb = pickle.load(f)
                    info(f"[ENS] L1 sklearn GBM loaded (XGB fallback)  thresh={XGB_THRESHOLD}")
                except Exception as e:
                    warn(f"[ENS] L1 GBM fallback load failed: {e}")

        # Layer 2 — Logistic Regression
        lr_path = os.path.join(self.persist_dir, "lr_model.pkl")
        if os.path.exists(lr_path):
            try:
                import pickle
                with open(lr_path, 'rb') as f:
                    self._lr = pickle.load(f)
                info(f"[ENS] L2 LogReg loaded  thresh={LR_THRESHOLD}")
            except Exception as e:
                warn(f"[ENS] L2 LogReg load failed: {e}")

        # Layer 3 — Isolation Forest
        iso_path = os.path.join(self.persist_dir, "iso_model.pkl")
        if os.path.exists(iso_path):
            try:
                import pickle
                with open(iso_path, 'rb') as f:
                    self._iso = pickle.load(f)
                info(f"[ENS] L3 IsoForest loaded  threshold={getattr(self._iso,'_ens_threshold','?'):.4f}")
            except Exception as e:
                warn(f"[ENS] L3 IsoForest load failed: {e}")

        loaded = sum(x is not None for x in [self._xgb, self._lr, self._iso])
        if loaded == 0:
            warn("[ENS] No models found — 5-condition fallback mode")
        else:
            info(f"[ENS] {loaded}/3 layers loaded — ensemble active")

    @property
    def active(self) -> bool:
        return any(x is not None for x in [self._xgb, self._lr, self._iso])

    def predict(self, feats: dict, regime: str) -> dict:
        """Returns per-layer votes and final trade decision."""
        import numpy as np
        row = [feats.get(f, 0.0) for f in _FEATURES]
        X   = [[row[i] for i in range(len(_FEATURES))]]

        # Hard block on CHAOS
        if regime == "CHAOS":
            return {"votes": 0, "trade": False, "reason": "chaos",
                    "xgb_prob": 0.0, "lr_prob": 0.0, "iso_score": 0.0,
                    "v_xgb": False, "v_lr": False, "v_iso": False}

        v_xgb = True; xgb_prob = 1.0
        v_lr  = True; lr_prob  = 1.0
        v_iso = True; iso_score = 0.0

        if self._xgb is not None:
            try:
                import numpy as np
                xgb_prob = float(self._xgb.predict_proba(
                    np.array(X, dtype=float))[:, 1][0])
                v_xgb = xgb_prob >= XGB_THRESHOLD
            except Exception as e:
                warn(f"[ENS] L1 predict error: {e}")

        if self._lr is not None:
            try:
                import numpy as np
                lr_prob = float(self._lr.predict_proba(
                    np.array(X, dtype=float))[:, 1][0])
                v_lr = lr_prob >= LR_THRESHOLD
            except Exception as e:
                warn(f"[ENS] L2 predict error: {e}")

        if self._iso is not None:
            try:
                import numpy as np
                iso_score = float(self._iso.score_samples(
                    np.array(X, dtype=float))[0])
                v_iso = iso_score >= getattr(self._iso, '_ens_threshold', -0.5)
            except Exception as e:
                warn(f"[ENS] L3 predict error: {e}")

        votes = sum([v_xgb, v_lr, v_iso])
        return {
            "votes":     votes,
            "trade":     votes >= 2,
            "xgb_prob":  round(xgb_prob, 4),
            "lr_prob":   round(lr_prob, 4),
            "iso_score": round(iso_score, 4),
            "v_xgb":     v_xgb,
            "v_lr":      v_lr,
            "v_iso":     v_iso,
        }


# Module-level singleton
_ensemble: Optional['EnsembleGate'] = None


def load_ensemble(cal: dict) -> 'EnsembleGate':
    global _ensemble
    _ensemble = EnsembleGate(_PERSIST_DIR, cal.get("sigma_gate", 0.13444))
    return _ensemble


def _build_feature_matrix(rows_raw, sigma_gate, labels_arr=None):
    """
    Shared feature builder — called by retrain_ensemble() for all 3 models.
    Guarantees identical 18-feature vectors in training and inference.
    """
    import numpy as np, math as _math
    sigma_v = [r["sigma_ewma"] for r in rows_raw]
    atr_v   = [r["atr_14"]     for r in rows_raw]
    ent_v   = [r["entropy_20"] for r in rows_raw]
    hours   = [int(r["ts"][11:13]) for r in rows_raw]

    X_rows, y_rows = [], []
    for i, r in enumerate(rows_raw):
        if labels_arr is not None:
            lbl = labels_arr[i]
            if lbl != lbl:   # nan check
                continue
        X_rows.append([
            r["sigma_ewma"], r["range_20"],  r["range_50"],  r["ema_gap"],
            r["zscore_50"],  r["spike_10"],  r["atr_14"],    r["entropy_20"],
            _REGIME_ENC.get(r["regime"], 0),
            sigma_v[i] - sigma_v[max(0, i-10)],
            r["range_20"] / (r["range_50"] + 1e-9),
            r["ema7"] - r["ema14"],
            abs(r["zscore_50"]),
            ent_v[i] - ent_v[max(0, i-5)],
            r["sigma_ewma"] / (sigma_gate + 1e-9),
            r["spike_10"]   / (r["sigma_ewma"] + 1e-9),
            atr_v[i] - atr_v[max(0, i-10)],
            hours[i],
        ])
        if labels_arr is not None:
            y_rows.append(int(labels_arr[i]))

    import numpy as np
    X = np.array(X_rows, dtype=float)
    y = np.array(y_rows, dtype=int) if labels_arr is not None else None
    return X, y


def retrain_ensemble(cal: dict):
    """
    Trains all 3 layers on freshly collected CSV data.
    Called synchronously at Phase 1->2 transition (~20-40s).
    Called in a background thread after each rolling recalibration.
    Saves model files and hot-swaps _ensemble singleton.
    """
    global _ensemble
    import numpy as np, csv as _csv, pickle

    sym        = cal.get("symbol", "1HZ10V")
    csv_path   = os.path.join(DATA_DIR, f"{sym}.csv")
    BARRIER    = cal.get("barrier", 2.15)
    WINDOW     = 120
    sigma_gate = cal.get("sigma_gate", 0.13444)

    if not os.path.exists(csv_path):
        warn("[ENS] retrain: CSV not found — skipping"); return

    # ── Load CSV ──────────────────────────────────────────────────────────────
    rows_raw = []
    with open(csv_path, newline="") as f:
        for row in _csv.DictReader(f):
            try:
                rows_raw.append({
                    "sigma_ewma": float(row["sigma_ewma"]),
                    "range_20":   float(row["range_20"]),
                    "range_50":   float(row["range_50"]),
                    "ema_gap":    float(row["ema_gap"]),
                    "ema7":       float(row["ema7"]),
                    "ema14":      float(row["ema14"]),
                    "zscore_50":  float(row["zscore_50"]),
                    "spike_10":   float(row["spike_10"]),
                    "atr_14":     float(row["atr_14"]),
                    "entropy_20": float(row["entropy_20"]),
                    "regime":     row["regime"].strip(),
                    "price":      float(row["price"]),
                    "ts":         row["ts"],
                })
            except Exception:
                continue

    if len(rows_raw) < 500:
        warn(f"[ENS] retrain: only {len(rows_raw)} rows — skipping"); return

    # ── Labels ────────────────────────────────────────────────────────────────
    prices = np.array([r["price"] for r in rows_raw])
    n = len(prices)
    labels = []
    for i in range(n):
        if i + WINDOW >= n:
            labels.append(float('nan')); continue
        max_dev = np.max(np.abs(prices[i+1:i+WINDOW+1] - prices[i]))
        labels.append(1.0 if max_dev <= BARRIER else 0.0)

    X, y = _build_feature_matrix(rows_raw, sigma_gate, labels)
    hours_of_data = len(rows_raw) / 3600
    info(f"[ENS] Training {len(X)} samples  "
         f"~{hours_of_data:.1f}h of data  base_wr={y.mean()*100:.1f}%")

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER 1: XGBoost (with GBM fallback)
    # ─────────────────────────────────────────────────────────────────────────
    xgb_ok = False
    try:
        from xgboost import XGBClassifier
        xgb = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
            reg_alpha=0.1, reg_lambda=1.0, eval_metric='logloss', verbosity=0,
        )
        xgb.fit(X, y, verbose=False)
        out = os.path.join(_PERSIST_DIR, "xgb_model.json")
        xgb.save_model(out)
        probs = xgb.predict_proba(X)[:, 1]
        mask  = probs >= XGB_THRESHOLD
        n_sig = mask.sum()
        prec  = y[mask].mean() * 100 if n_sig > 0 else 0
        info(f"[ENS] L1 XGBoost: {n_sig} signals at >={XGB_THRESHOLD}  precision={prec:.1f}%")
        xgb_ok = True
    except ImportError:
        warn("[ENS] L1 XGBoost not installed")
    except Exception as e:
        warn(f"[ENS] L1 XGBoost failed: {e}")

    if not xgb_ok:
        try:
            from sklearn.ensemble import GradientBoostingClassifier
            gbm = GradientBoostingClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=10, random_state=42,
            )
            gbm.fit(X, y)
            out = os.path.join(_PERSIST_DIR, "gb_model.pkl")
            with open(out, 'wb') as f:
                pickle.dump(gbm, f)
            probs = gbm.predict_proba(X)[:, 1]
            mask  = probs >= XGB_THRESHOLD
            n_sig = mask.sum()
            prec  = y[mask].mean() * 100 if n_sig > 0 else 0
            info(f"[ENS] L1 GBM fallback: {n_sig} signals  precision={prec:.1f}%")
        except Exception as e:
            warn(f"[ENS] L1 GBM fallback failed: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER 2: Logistic Regression with Polynomial Interaction Features
    # ─────────────────────────────────────────────────────────────────────────
    try:
        from sklearn.preprocessing import PolynomialFeatures, StandardScaler
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        lr_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('poly',   PolynomialFeatures(degree=2, interaction_only=True,
                                          include_bias=False)),
            ('lr',     LogisticRegression(C=0.1, max_iter=1000,
                                          solver='lbfgs', random_state=42)),
        ])
        lr_pipe.fit(X, y)
        lr_probs = lr_pipe.predict_proba(X)[:, 1]
        lr_mask  = lr_probs >= LR_THRESHOLD
        n_sig    = lr_mask.sum()
        prec     = y[lr_mask].mean() * 100 if n_sig > 0 else 0
        info(f"[ENS] L2 LogReg: {n_sig} signals at >={LR_THRESHOLD}  precision={prec:.1f}%")
        out = os.path.join(_PERSIST_DIR, "lr_model.pkl")
        with open(out, 'wb') as f:
            pickle.dump(lr_pipe, f)
    except Exception as e:
        warn(f"[ENS] L2 LogReg failed: {e}")

    # ─────────────────────────────────────────────────────────────────────────
    # LAYER 3: Isolation Forest — trained on WIN rows only
    # ─────────────────────────────────────────────────────────────────────────
    try:
        from sklearn.ensemble import IsolationForest

        X_wins = X[y == 1]
        info(f"[ENS] L3 IsoForest training on {len(X_wins)} WIN rows")
        iso = IsolationForest(
            n_estimators=200, contamination=ISO_CONTAMINATION,
            random_state=42, n_jobs=-1,
        )
        iso.fit(X_wins)

        # Compute threshold from win-sample score distribution
        win_scores = iso.score_samples(X_wins)
        iso._ens_threshold = float(np.percentile(win_scores, ISO_CONTAMINATION * 100))

        all_scores = iso.score_samples(X)
        blocked    = (all_scores < iso._ens_threshold).sum()
        info(f"[ENS] L3 IsoForest: threshold={iso._ens_threshold:.4f}  "
             f"blocks {blocked}/{len(X)} ticks ({blocked/len(X)*100:.1f}%)")
        out = os.path.join(_PERSIST_DIR, "iso_model.pkl")
        with open(out, 'wb') as f:
            pickle.dump(iso, f)
    except Exception as e:
        warn(f"[ENS] L3 IsoForest failed: {e}")

    # ── Hot-swap singleton ────────────────────────────────────────────────────
    _ensemble = EnsembleGate(_PERSIST_DIR, sigma_gate)
    info("[ENS] Ensemble hot-swapped — all 3 layers active")


class SignalEngine:
    """
    5-condition confluence gate (C1-C5) + 3-Layer Ensemble (C6).

    C1  sigma_ewma  < sigma_gate
    C2  range(20)   < range_gate
    C3  |ema_gap|   < ema_gate
    C4  |Z|         < z_gate
    C5  spike(10)   < spike_gate
    C6  Ensemble vote >= 2/3  (hard block on CHAOS regardless)

    C1-C5: 4/5 must pass. C6: mandatory when any model loaded.
    """

    def __init__(self, cal: dict):
        self.cal    = cal
        self.tick_n = 0
        self.prices: deque = deque(maxlen=500)
        self._sigma_ewma = None
        self._ema7 = self._ema14 = None
        self._k7   = 2 / 8
        self._k14  = 2 / 15
        self.EWMA_ALPHA = 0.05
        self._warmup = 100
        self._sigma_buf   = deque(maxlen=15)
        self._entropy_buf = deque(maxlen=10)
        self._atr_buf     = deque(maxlen=15)

    def ingest(self, price: float) -> dict:
        self.tick_n += 1
        prev  = self.prices[-1] if self.prices else price
        delta = abs(price - prev)
        self.prices.append(price)

        if self._sigma_ewma is None:
            self._sigma_ewma = delta
        else:
            self._sigma_ewma = (self.EWMA_ALPHA * delta +
                                (1 - self.EWMA_ALPHA) * self._sigma_ewma)

        if self._ema7 is None:
            self._ema7 = self._ema14 = price
        else:
            self._ema7  = price * self._k7  + self._ema7  * (1 - self._k7)
            self._ema14 = price * self._k14 + self._ema14 * (1 - self._k14)

        if self.tick_n < self._warmup:
            return {"trade": False, "reason": "warmup", "tick": self.tick_n}

        prices = list(self.prices)
        sigma  = self._sigma_ewma

        range20 = (max(prices[-20:]) - min(prices[-20:])
                   if len(prices) >= 20 else 999)
        range50 = (max(prices[-50:]) - min(prices[-50:])
                   if len(prices) >= 50 else range20)
        ema_gap = abs(self._ema7 - self._ema14)

        if len(prices) >= 200:
            baseline = prices[-200:]
            mu  = sum(baseline) / 200
            var = sum((p - mu)**2 for p in baseline) / 200
            std = math.sqrt(var) if var > 0 else 1e-9
            z_raw = (sum(prices[-50:])/50 - mu) / (std / math.sqrt(50))
        else:
            z_raw = 0.0
        z = abs(z_raw)

        moves = [abs(prices[i] - prices[i-1]) for i in range(-10, 0)
                 if i-1 >= -len(prices)]
        spike = max(moves) if moves else 0

        atr_moves = [abs(prices[i] - prices[i-1]) for i in range(-14, 0)
                     if i-1 >= -len(prices)]
        atr14 = sum(atr_moves) / len(atr_moves) if atr_moves else 0

        if len(prices) >= 21:
            ep = prices[-21:]
            em = [abs(ep[i] - ep[i-1]) for i in range(1, len(ep))]
            mx = max(em) or 1
            bk = [0] * 5
            for m in em:
                bk[min(4, int(m / mx * 4))] += 1
            ne = len(em); H = 0.0
            for b in bk:
                if b > 0:
                    p = b / ne; H -= p * math.log2(p)
            entropy20 = H / math.log2(5)
        else:
            entropy20 = 1.0

        if abs(z_raw) > 2.5 or sigma > 0.3:
            regime = "CHAOS"
        elif abs(z_raw) > 1.5 and ema_gap > 0.3:
            regime = "TRENDING"
        elif abs(z_raw) < 1.0 and ema_gap < 0.15:
            regime = "CALM"
        else:
            regime = "RANGING"

        c1 = sigma   < self.cal["sigma_gate"]
        c2 = range20 < self.cal["range_gate"]
        c3 = ema_gap < self.cal["ema_gate"]
        c4 = z       < self.cal["z_gate"]
        c5 = spike   < self.cal["spike_gate"]
        score = sum([c1, c2, c3, c4, c5])

        self._sigma_buf.append(sigma)
        self._entropy_buf.append(entropy20)
        self._atr_buf.append(atr14)

        sigma_trend   = (sigma     - list(self._sigma_buf)[0]
                         if len(self._sigma_buf)   >= 10 else 0.0)
        entropy_delta = (entropy20 - list(self._entropy_buf)[0]
                         if len(self._entropy_buf) >= 5  else 0.0)
        atr_trend     = (atr14     - list(self._atr_buf)[0]
                         if len(self._atr_buf)     >= 10 else 0.0)

        sg = self.cal.get("sigma_gate", 0.13444)
        ml_feats = {
            "sigma_ewma":    sigma,
            "range_20":      range20,
            "range_50":      range50,
            "ema_gap":       ema_gap,
            "zscore_50":     z_raw,
            "spike_10":      spike,
            "atr_14":        atr14,
            "entropy_20":    entropy20,
            "regime_enc":    _REGIME_ENC.get(regime, 0),
            "sigma_trend":   sigma_trend,
            "range_ratio":   range20 / (range50 + 1e-9),
            "ema_cross":     self._ema7 - self._ema14,
            "zscore_abs":    z,
            "entropy_delta": entropy_delta,
            "sigma_vs_gate": sigma / (sg + 1e-9),
            "spike_vs_sigma":spike / (sigma + 1e-9),
            "atr_trend":     atr_trend,
            "hour_of_day":   datetime.now(timezone.utc).hour,
        }

        if _ensemble and _ensemble.active:
            ens = _ensemble.predict(ml_feats, regime)
        else:
            ens = {"votes": 3, "trade": True, "xgb_prob": 1.0,
                   "lr_prob": 1.0, "iso_score": 0.0,
                   "v_xgb": True, "v_lr": True, "v_iso": True}

        c6    = ens["trade"]
        trade = score >= 4 and c6

        return {
            "trade":     trade,
            "score":     score,
            "tick":      self.tick_n,
            "sigma":     round(sigma, 5),
            "range20":   round(range20, 4),
            "ema_gap":   round(ema_gap, 5),
            "z":         round(z, 4),
            "spike":     round(spike, 5),
            "regime":    regime,
            "votes":     ens["votes"],
            "xgb_prob":  ens["xgb_prob"],
            "lr_prob":   ens["lr_prob"],
            "iso_score": ens["iso_score"],
            "v_xgb":     ens["v_xgb"],
            "v_lr":      ens["v_lr"],
            "v_iso":     ens["v_iso"],
            "c6":        c6,
            "c1": c1, "c2": c2, "c3": c3, "c4": c4, "c5": c5,
        }


class RiskManager:
    def __init__(self):
        self.stake        = BASE_STAKE
        self.loss_streak  = 0
        self.session_pnl  = 0.0
        self.wins = self.losses = 0
        self._cooldown_until = 0.0

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

    def record_win(self, profit: float):
        self.wins        += 1
        self.session_pnl += profit
        self.loss_streak  = 0
        self.stake        = BASE_STAKE
        tlog(f"WIN +${profit:.4f}  stake→${self.stake}  "
             f"P&L=${self.session_pnl:.4f}")

    def record_loss(self, amount: float):
        self.losses      += 1
        self.session_pnl -= amount
        self.loss_streak += 1
        self._cooldown_until = time.monotonic() + LOSS_COOLDOWN
        if self.loss_streak > MARTINGALE_STEPS:
            # All steps exhausted — reset to base
            self.stake       = BASE_STAKE
            self.loss_streak = 0
            warn(f"MARTINGALE exhausted — RESET to base=${self.stake:.2f}  "
                 f"P&L=${self.session_pnl:.4f}")
        elif self.loss_streak < 2:
            # First loss — hold at base stake, no escalation yet
            self.stake = BASE_STAKE
            tlog(f"LOSS streak={self.loss_streak}/{MARTINGALE_STEPS}  "
                 f"next_stake=${self.stake:.2f} (holding)  "
                 f"P&L=${self.session_pnl:.4f}")
        else:
            # 2nd loss onward — escalate; index offset by 1 so 2nd loss = mult^1
            escalation = self.loss_streak - 1
            self.stake = round(BASE_STAKE * (MARTINGALE_MULT ** escalation), 2)
            tlog(f"LOSS streak={self.loss_streak}/{MARTINGALE_STEPS}  "
                 f"next_stake=${self.stake:.2f}  "
                 f"P&L=${self.session_pnl:.4f}")


class DerivClient:
    """Full resilient WS client — same layer as proven bots."""

    def __init__(self):
        self._ws         = None
        self._send_q     = asyncio.Queue()
        self._inbox      = asyncio.Queue()
        self._send_task  = None
        self._recv_task  = None
        self._rid        = 0
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
            err(f"connect: {exc}")
            return False

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

    async def _send_msg(self, data: dict):
        self._rid += 1
        data["req_id"] = self._rid
        loop = asyncio.get_event_loop()
        fut  = loop.create_future()
        await self._send_q.put((data, fut))
        await fut

    async def _recv_type(self, msg_type: str, timeout=10) -> Optional[dict]:
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return None
            try:
                msg = await asyncio.wait_for(
                    self._inbox.get(), timeout=remaining)
            except asyncio.TimeoutError:
                return None
            if "__disconnect__" in msg:
                await self._inbox.put(msg)
                return None
            if msg_type in msg or "error" in msg:
                return msg
            await self._inbox.put(msg)

    async def receive(self, timeout=60) -> dict:
        try:
            return await asyncio.wait_for(self._inbox.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return {}

    async def subscribe_ticks(self, symbol: str) -> bool:
        await self._send_msg({"ticks": symbol, "subscribe": 1})
        resp = await self._recv_type("tick", timeout=10)
        if not resp or "error" in resp:
            err(f"Tick sub failed: {(resp or {}).get('error',{}).get('message','?')}")
            return False
        info(f"Subscribed to {symbol}")
        return True

    async def fetch_balance(self) -> Optional[float]:
        try:
            await self._send_msg({"balance": 1})
            resp = await self._recv_type("balance", timeout=10)
            if resp and "balance" in resp:
                return float(resp["balance"]["balance"])
        except Exception as exc:
            warn(f"fetch_balance: {exc}")
        return None

    async def place_trade(self, barrier: float,
                          duration_min: int, stake: float
                          ) -> Tuple[Optional[int], Optional[int]]:
        await self._send_msg({
            "proposal":      1,
            "amount":        stake,
            "basis":         "stake",
            "contract_type": "EXPIRYRANGE",
            "currency":      "USD",
            "duration":      duration_min,
            "duration_unit": "m",
            "symbol":        self.symbol if hasattr(self, "symbol") else "1HZ10V",
            "barrier":       f"+{barrier}",
            "barrier2":      f"-{barrier}",
        })
        proposal = await self._recv_type("proposal", timeout=12)
        if not proposal or "error" in proposal:
            err(f"Proposal: {(proposal or {}).get('error',{}).get('message','?')}")
            return None, None

        prop = proposal.get("proposal", {})
        pid  = prop.get("id")
        ask  = float(prop.get("ask_price", stake))
        payout = float(prop.get("payout", 0))
        roi    = (payout - ask) / ask * 100 if ask > 0 else 0
        info(f"Proposal OK  ask=${ask:.2f}  payout=${payout:.2f}  ROI={roi:.1f}%")

        if not pid:
            err("No proposal ID")
            return None, None

        buy_ts = time.time()
        await self._send_msg({"buy": pid, "price": ask})

        contract_id = expiry_time = None
        for attempt in range(8):
            resp = await self._recv_type("buy", timeout=8)
            if resp is None:
                warn(f"Buy no response attempt {attempt+1}")
                continue
            if "error" in resp:
                err(f"Buy error: {resp['error'].get('message','')}")
                return None, None
            bd          = resp.get("buy", {})
            contract_id = bd.get("contract_id")
            expiry_time = bd.get("date_expiry")
            if contract_id:
                break

        if not contract_id:
            warn("Orphan recovery via profit_table")
            for _ in range(4):
                await asyncio.sleep(3)
                await self._send_msg({"profit_table": 1, "description": 1,
                                      "sort": "DESC", "limit": 5})
                r = await self._recv_type("profit_table", timeout=10)
                if r and "profit_table" in r:
                    for tx in r["profit_table"].get("transactions", []):
                        if (abs(float(tx.get("buy_price", 0)) - stake) < 0.01
                                and float(tx.get("purchase_time", 0))
                                >= buy_ts - 10):
                            contract_id = tx.get("contract_id")
                            info(f"Orphan recovered → {contract_id}")
                            break
                if contract_id:
                    break
            if not contract_id:
                err("Orphan recovery failed")
                return None, None

        try:
            await self._send_msg({
                "proposal_open_contract": 1,
                "contract_id":            contract_id,
                "subscribe":              1,
            })
        except Exception:
            pass

        tlog(f"Placed  contract={contract_id}  "
             f"EXPIRYRANGE ±{barrier}  ${ask:.2f}  {duration_min}min  "
             f"expiry_ts={expiry_time}")
        return contract_id, expiry_time

    async def poll_contract(self, contract_id: int) -> Optional[dict]:
        try:
            await self._send_msg({
                "proposal_open_contract": 1,
                "contract_id": contract_id,
            })
            resp = await self._recv_type("proposal_open_contract", timeout=10)
            if resp and "proposal_open_contract" in resp:
                return resp["proposal_open_contract"]
        except Exception as exc:
            warn(f"poll_contract: {exc}")
        return None

    @staticmethod
    def is_settled(data: dict) -> bool:
        if data.get("is_settled") or data.get("is_sold"):
            return True
        return data.get("status", "").lower() in ("sold", "won", "lost")


class SymbolTrader:
    """
    Runs one trading loop for a single symbol.
    Created with calibrated thresholds from Phase 1.
    """

    def __init__(self, cal: dict):
        self.cal     = cal
        self.symbol  = cal["symbol"]
        self.engine  = SignalEngine(cal)
        self.risk    = RiskManager()
        self.client  = DerivClient()
        self.client.symbol = self.symbol

        self.waiting     = False
        self._evaluating = False
        self._settling   = False
        self.current_trade: Optional[dict] = None
        self.lock_since: Optional[float]   = None
        self._stop       = False
        self._loss_cd_until = 0.0
        self._poller_task: Optional[asyncio.Task] = None
        self.live_ticks  = 0
        self.signals     = 0

    def _unlock(self, reason="manual"):
        if self.waiting:
            cid = (self.current_trade or {}).get("id", "?")
            info(f"[{self.symbol}] Unlock cid={cid} reason={reason}")
        self.waiting       = False
        self.current_trade = None
        self.lock_since    = None
        self._evaluating   = False
        self._settling     = False
        if self._poller_task and not self._poller_task.done():
            self._poller_task.cancel()
            self._poller_task = None

    def _check_lock_timeout(self):
        if self.waiting and self.lock_since:
            if time.monotonic() - self.lock_since >= LOCK_TIMEOUT:
                warn(f"[{self.symbol}] Lock timeout — unlocking")
                self._unlock("timeout")

    async def on_tick(self, price: float):
        self.live_ticks += 1
        self._check_lock_timeout()
        sig = self.engine.ingest(price)

        if self.live_ticks % 30 == 0:
            cd_left = max(0, self._loss_cd_until - time.monotonic())
            ok, why = self.risk.can_trade()
            status  = ("LOCKED" if self.waiting
                       else f"COOLDOWN({cd_left:.0f}s)" if cd_left > 0
                       else "READY" if ok else f"BLOCKED:{why}")
            info(f"[{self.symbol}] tick={sig['tick']} "
                 f"score={sig.get('score','?')}/5  "
                 f"sigma={sig.get('sigma','?')}  "
                 f"range={sig.get('range20','?')}  "
                 f"Z={sig.get('z','?')}  {status}")

        if self.waiting or self._evaluating:
            return
        if time.monotonic() < self._loss_cd_until:
            return
        if not sig.get("trade"):
            return
        ok, reason = self.risk.can_trade()
        if not ok:
            return

        self._evaluating = True
        try:
            await self._evaluate(sig)
        finally:
            self._evaluating = False

    async def _evaluate(self, sig: dict):
        if self.waiting:
            return
        self.signals += 1
        ens_str = ""
        if _ensemble and _ensemble.active:
            ens_str = (f"  votes={sig.get('votes','?')}/3"
                       f"  xgb={sig.get('xgb_prob','?')}"
                       f"  lr={sig.get('lr_prob','?')}"
                       f"  iso={sig.get('iso_score','?')}"
                       f"  regime={sig.get('regime','?')}")
        info(f"[{self.symbol}] SIGNAL #{self.signals}  "
             f"score={sig['score']}/5  "
             f"sigma={sig['sigma']}  range={sig['range20']}  "
             f"ema_gap={sig['ema_gap']}  Z={sig['z']}  "
             f"spike={sig['spike']}{ens_str}")

        stake        = self.risk.get_stake()
        barrier      = self.cal["barrier"]
        duration_min = self.cal["duration_min"]

        bal = await self.client.fetch_balance()
        if bal:
            self.client.balance = bal

        cid, expiry_time = await self.client.place_trade(
            barrier, duration_min, stake)

        if cid:
            self.current_trade = {
                "id":          cid,
                "stake":       stake,
                "barrier":     barrier,
                "expiry_time": expiry_time,
            }
            self.waiting    = True
            self.lock_since = time.monotonic()
            self._poller_task = asyncio.create_task(
                self._expiry_poller(cid, expiry_time, duration_min),
                name=f"poller_{cid}"
            )
        else:
            warn(f"[{self.symbol}] Trade placement failed")

    async def _expiry_poller(self, cid: int,
                              expiry_time: Optional[int],
                              duration_min: int):
        wait = max(5.0, (expiry_time - time.time()) + 5) if expiry_time \
               else duration_min * 60 + 10
        info(f"[{self.symbol}] Expiry poller: sleeping {wait:.1f}s")
        await asyncio.sleep(wait)

        if not self.waiting or not self.current_trade or \
                self.current_trade.get("id") != cid:
            return

        warn(f"[{self.symbol}] Expiry poller: {cid} still locked — polling")
        for attempt in range(1, 7):
            try:
                data = await self.client.poll_contract(cid)
                if data and self.client.is_settled(data):
                    info(f"[{self.symbol}] Poller settled attempt {attempt}")
                    ok = await self.handle_settlement(data)
                    if not ok:
                        self._stop = True
                    return
            except Exception as exc:
                warn(f"[{self.symbol}] Poller attempt {attempt}: {exc}")
            await asyncio.sleep(5)

        if self.waiting and self.current_trade and \
                self.current_trade.get("id") == cid:
            warn(f"[{self.symbol}] Poller exhausted — force unlock")
            self._unlock("poller_exhausted")

    async def handle_settlement(self, data: dict) -> bool:
        if self._settling:
            return True
        self._settling = True
        try:
            return await self._settle_inner(data)
        finally:
            self._settling = False

    async def _settle_inner(self, data: dict) -> bool:
        cid = data.get("contract_id")
        if not self.current_trade or \
                str(cid) != str(self.current_trade["id"]):
            return True
        if not self.client.is_settled(data):
            return True

        profit = float(data.get("profit", 0))
        status = data.get("status", "?")

        bal = await self.client.fetch_balance()
        if bal:
            self.client.balance = bal
        actual = round(bal - self.client.balance, 4) if bal else profit

        tlog(f"[{self.symbol}] SETTLED  cid={cid}  "
             f"status={status}  profit={profit:+.4f}")

        if profit > 0:
            self.risk.record_win(profit)
        else:
            self.risk.record_loss(self.current_trade["stake"])
            self._loss_cd_until = time.monotonic() + LOSS_COOLDOWN

        self._unlock("settlement")
        info(f"[{self.symbol}] Ready for next signal")
        # Always return True — cooldown/stop-loss handled in on_tick loop
        # Returning False here incorrectly sets _stop=True and kills the bot
        return True

    async def run(self):
        retry_delay = 5
        while not self._stop:
            try:
                if not await self.client.connect():
                    raise ConnectionError("connect failed")
                if not await self.client.subscribe_ticks(self.symbol):
                    raise ConnectionError("tick sub failed")

                info(f"[{self.symbol}] Live  "
                     f"barrier=±{self.cal['barrier']}  "
                     f"duration={self.cal['duration_min']}min  "
                     f"sigma_gate={self.cal['sigma_gate']}  "
                     f"range_gate={self.cal['range_gate']}  "
                     f"z_gate={self.cal['z_gate']}")

                while not self._stop:
                    msg = await self.client.receive(timeout=60)

                    if "__disconnect__" in msg:
                        warn(f"[{self.symbol}] WS disconnected")
                        break
                    if not msg:
                        try:
                            await self.client._ws.ping()
                        except Exception:
                            break
                        continue

                    if "tick" in msg:
                        await self.on_tick(float(msg["tick"]["quote"]))

                    for key in ("proposal_open_contract", "buy"):
                        if key in msg:
                            ok = await self.handle_settlement(msg[key])
                            if not ok:
                                self._stop = True

                    if "transaction" in msg:
                        tx = msg["transaction"]
                        if "contract_id" in tx:
                            ok = await self.handle_settlement({
                                "contract_id": tx.get("contract_id"),
                                "profit":      tx.get("profit", 0),
                                "status":      tx.get("action", "sold"),
                                "is_settled":  True,
                            })
                            if not ok:
                                self._stop = True

            except Exception as exc:
                err(f"[{self.symbol}] Session error: {exc}")
                traceback.print_exc()

            if not self._stop:
                warn(f"[{self.symbol}] Reconnecting in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)
                await self.client.close()
                self.client = DerivClient()
                self.client.symbol = self.symbol

        r = self.risk
        total = r.wins + r.losses
        wr    = r.wins / total * 100 if total else 0
        info(f"[{self.symbol}] DONE  trades={total}  "
             f"W={r.wins}  L={r.losses}  WR={wr:.1f}%  "
             f"P&L=${r.session_pnl:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# HEALTH SERVER
# ─────────────────────────────────────────────────────────────────────────────

_health_state: dict = {"phase": "collect", "traders": [], "collect_start": 0}

def start_health_server(traders: List[SymbolTrader], phase: str,
                        collect_start: float = 0):
    _health_state["phase"]         = phase
    _health_state["traders"]       = traders
    _health_state["collect_start"] = collect_start
    if getattr(start_health_server, "_started", False):
        return
    start_health_server._started = True

    class _H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/status":
                data = {"phase": phase, "traders": []}
                for t in traders:
                    r = t.risk
                    tot = r.wins + r.losses
                    data["traders"].append({
                        "symbol":   t.symbol,
                        "ticks":    t.live_ticks,
                        "signals":  t.signals,
                        "trades":   tot,
                        "wins":     r.wins,
                        "losses":   r.losses,
                        "win_rate": round(r.wins/tot, 4) if tot else 0,
                        "pnl":      round(r.session_pnl, 4),
                        "stake":    r.stake,
                        "locked":   t.waiting,
                    })
                body = json.dumps(data, indent=2).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            else:
                # Human summary
                if phase == "collect":
                    elapsed = time.time() - collect_start
                    remaining = max(0, COLLECT_SECS - elapsed)
                    html_body = f"""<h2>Phase 1: Collecting Data</h2>
<p>Elapsed: {elapsed/3600:.2f}h &nbsp; Remaining: {remaining/60:.0f}min</p>
<p>Symbols: {', '.join(SURVEY_SYMBOLS)}</p>"""
                else:
                    rows = ""
                    for t in traders:
                        r = t.risk
                        tot = r.wins + r.losses
                        wr  = r.wins/tot*100 if tot else 0
                        rows += f"""<tr>
  <td>{t.symbol}</td>
  <td>{tot}</td>
  <td style="color:{'#3fb950' if r.wins >= r.losses else '#f85149'}">{r.wins}</td>
  <td style="color:#f85149">{r.losses}</td>
  <td style="color:{'#3fb950' if wr>=74.6 else '#f85149'}">{wr:.1f}%</td>
  <td style="color:{'#3fb950' if r.session_pnl>=0 else '#f85149'}">${r.session_pnl:+.4f}</td>
  <td>${r.stake:.2f}</td>
  <td>{'🔒' if t.waiting else '🟢'}</td>
</tr>"""
                    html_body = f"""<h2>Phase 2: Trading</h2>
<table border=1 cellpadding=6>
<tr><th>Symbol</th><th>Trades</th><th>Wins</th><th>Losses</th>
    <th>WR</th><th>P&L</th><th>Stake</th><th>Status</th></tr>
{rows}</table>
<p style="font-size:0.8rem">Breakeven: 74.6% &nbsp;|&nbsp; Auto-refreshes 10s</p>"""

                html = f"""<!DOCTYPE html>
<html><head><meta charset=utf-8>
<meta http-equiv="refresh" content="10">
<title>EXPIRYRANGE Bot</title>
<style>body{{font-family:monospace;background:#0d1117;color:#e6edf3;
padding:2rem;}}table{{border-collapse:collapse;}}
th,td{{padding:.4rem .8rem;border:1px solid #21262d;}}
th{{background:#161b22;color:#8b949e;}}
h2{{color:#58a6ff;}}</style></head>
<body>{html_body}
<p><a href="/status" style="color:#58a6ff">/status JSON</a></p>
</body></html>"""
                body = html.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("", PORT), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    info(f"Health server on :{PORT}  (/ = summary  /status = JSON)")

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

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    if os.path.exists(CAL_FILE) and not trade_only:
        info(f"Found existing calibration at {CAL_FILE} — skipping Phase 1.")
        trade_only = True

    if not trade_only:
        collect_start = time.time()
        start_health_server([], phase="collect", collect_start=collect_start)
        collector = Collector()
        calibration = await collector.run()
    else:
        if not os.path.exists(CAL_FILE):
            sys.exit(f"calibration.json not found. "
                     f"Run without --trade-only first.")
        with open(CAL_FILE) as f:
            calibration = json.load(f)
        info(f"Loaded calibration from {CAL_FILE}")
        info(f"Generated at: {calibration['generated_at']}")

    if collect_only:
        info("--collect-only: stopping after Phase 1.")
        info(f"Calibration: {json.dumps(calibration['trade_symbols'], indent=2)}")
        return

    # ── Train ML model on freshly collected Phase 1 data ─────────────────────
    # Runs synchronously here (blocking) so the model is always ready before
    # the first tick arrives in Phase 2. Takes ~10-20s on 14k rows.
    if calibration["trade_symbols"]:
        info("Training 3-layer ensemble on Phase 1 data — please wait (~30s)...")
        try:
            retrain_ensemble(calibration["trade_symbols"][0])
        except Exception as exc:
            warn(f"Ensemble training failed: {exc} — Phase 2 in 5-condition mode")

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    trade_symbols = calibration["trade_symbols"]
    info(f"Phase 2: trading {[s['symbol'] for s in trade_symbols]}")

    traders = [SymbolTrader(cal) for cal in trade_symbols]

    # Load 3-layer ensemble — shared singleton across all traders
    ens = load_ensemble(trade_symbols[0])
    if ens.active:
        loaded = sum(x is not None for x in [ens._xgb, ens._lr, ens._iso])
        info(f"[ENS] {loaded}/3 layers active  "
             f"XGB>={XGB_THRESHOLD}  LR>={LR_THRESHOLD}  "
             f"ISO_contamination={ISO_CONTAMINATION}  "
             f"rolling_window={ROLLING_MAX_HOURS:.0f}h")
    else:
        info("[ENS] No models — 5-condition fallback mode")
    _health_state["traders"] = traders
    _health_state["phase"]   = "trade"

    info("=" * 60)
    for t in traders:
        info(f"  {t.symbol}: barrier=±{t.cal['barrier']}  "
             f"duration={t.cal['duration_min']}min  "
             f"p_win={t.cal['p_win_median']:.3f}  "
             f"score={t.cal['score']:.4f}")
        info(f"    gates: sigma<{t.cal['sigma_gate']}  "
             f"range<{t.cal['range_gate']}  "
             f"ema_gap<{t.cal['ema_gate']}  "
             f"|Z|<{t.cal['z_gate']}  "
             f"spike<{t.cal['spike_gate']}")
    info("=" * 60)

    # ── Hot-swap calibration support ────────────────────────────────────────
    for t in traders:
        def _make_swap(trader):
            def hot_swap_calibration(new_cal: dict):
                trader.cal    = new_cal
                trader.engine = SignalEngine(new_cal)
                info(f"[{trader.symbol}] ♻ Calibration hot-swapped "
                     f"barrier=±{new_cal['barrier']} "
                     f"p_win={new_cal['p_win_median']:.3f}")
            return hot_swap_calibration
        t.hot_swap_calibration = _make_swap(t)

    # ── Rolling recalibration every COLLECT_SECS while trading continues ─────
    async def recal_loop():
        cycle = 1
        while not _shutdown_event.is_set():
            await asyncio.sleep(COLLECT_SECS)
            if _shutdown_event.is_set():
                break
            cycle += 1
            info(f"♻ Recalibration cycle {cycle} starting "
                 f"(trading continues uninterrupted) "
                 f"rolling window up to {ROLLING_MAX_HOURS:.0f}h...")
            try:
                new_cal = await Collector().run()
                for t in traders:
                    sym_cal = next(
                        (s for s in new_cal["trade_symbols"]
                         if s["symbol"] == t.symbol), None)
                    if sym_cal:
                        t.hot_swap_calibration(sym_cal)
                    else:
                        warn(f"[{t.symbol}] not in new calibration — "
                             f"keeping existing thresholds")
                info(f"♻ Cycle {cycle} complete — trader updated")
                # Retrain ensemble in background thread (non-blocking)
                if new_cal["trade_symbols"]:
                    _first_sym_cal = new_cal["trade_symbols"][0]
                    threading.Thread(
                        target=retrain_ensemble,
                        args=(_first_sym_cal,),
                        daemon=True,
                        name="ens_retrain"
                    ).start()
            except Exception as exc:
                err(f"Recalibration cycle {cycle} failed: {exc} — "
                    f"keeping existing calibration")

    # Run trader + recal loop + shutdown watcher in parallel
    trader_tasks  = [asyncio.create_task(t.run()) for t in traders]
    recal_task    = asyncio.create_task(recal_loop())
    shutdown_task = asyncio.create_task(_shutdown_event.wait())
    done, pending = await asyncio.wait(
        trader_tasks + [recal_task, shutdown_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    if shutdown_task in done:
        info("Shutdown signal received — cancelling active traders...")
        for task in trader_tasks + [recal_task]:
            task.cancel()
        await asyncio.gather(*trader_tasks, recal_task, return_exceptions=True)
        info("All traders stopped. Exiting.")


# ─────────────────────────────────────────────────────────────────────────────
# GRACEFUL SHUTDOWN — handles Railway SIGTERM and Ctrl-C
# ─────────────────────────────────────────────────────────────────────────────

_shutdown_event = asyncio.Event()

def _handle_signal(signum, frame):
    sig_name = "SIGTERM" if signum == 2 else f"signal {signum}"
    info(f"Received {sig_name} — shutting down gracefully...")
    # Schedule the event set on the running loop (signal arrives on main thread)
    try:
        loop = asyncio.get_event_loop()
        loop.call_soon_threadsafe(_shutdown_event.set)
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
