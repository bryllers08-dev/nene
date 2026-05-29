"""
EXPIRYRANGE LSTM-Only Bot
═══════════════════════════════════════════════════════════════════════════════
No live data collection. Trains on CSV stored in Supabase Storage bucket.

Startup sequence
  1. Download latest 1HZ10V.csv from Supabase bucket  (falls back to local)
  2. Compute calibration from that CSV (barrier, sigma_gate, duration)
  3. Train LSTM on that data  (~2-3 min)
  4. Start trading immediately — LSTM is the sole gate

LSTM gate
  • Sequence:  60 ticks x 18 features
  • Fires when P(win) >= LSTM_THRESHOLD  (default 0.72)
  • If idle > MAX_IDLE_MINUTES threshold relaxes by 0.05 steps (floor 0.60)
  • Resets to default on every win
  • Every RETRAIN_HOURS: re-downloads CSV from Supabase + retrains LSTM

All regimes included — CALM, CHAOS, RANGING, TRENDING.
The LSTM learns which patterns produce wins. No hard regime filters.

Supabase Storage layout
  bucket : Supabot
  file   : 1HZ10V.csv   (upload your existing CSV here once to seed)
  Built-in TickLogger subscribes to 1HZ10V on a 2nd WS connection,
  computes all 18 features per tick, and uploads the full growing CSV
  to Supabase every 5 minutes automatically. No external collector needed.
  Every 4hr retrain downloads the latest CSV and retrains the LSTM.

ENV vars
  DERIV_API_TOKEN       your Deriv WS token
  DERIV_APP_ID          default 1089
  SUPABASE_URL          https://qtkjixwiahghisqqxrcz.supabase.co
  SUPABASE_KEY          your service_role key
  SUPABASE_BUCKET       Supabot
  DATA_DIR              local folder to save downloaded CSV  (default ./data/symbol_data)
  PERSIST_DIR           where model + calibration are saved  (default ./data)
  BASE_STAKE            default 1.0
  TARGET_PROFIT         default 45.0
  STOP_LOSS             default 10.0
  LSTM_THRESHOLD        default 0.72
  LSTM_THRESHOLD_FLOOR  default 0.60
  LSTM_SEQ_LEN          default 60
  MAX_IDLE_MINUTES      default 30
  RETRAIN_HOURS         default 4
  UPLOAD_INTERVAL_SECS  default 300  (5 minutes)

Run
  python main_lstm_bot.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import asyncio, csv, json, logging, math, os, pickle, signal, sys
import threading, time, traceback
from collections   import deque
from datetime      import datetime, timezone
from http.server   import HTTPServer, BaseHTTPRequestHandler
from typing        import Dict, List, Optional, Tuple

try:
    import websockets
    from websockets.exceptions import (
        ConnectionClosed, ConnectionClosedError, ConnectionClosedOK,
    )
except ImportError:
    sys.exit("Run: pip install websockets")

try:
    import numpy as np
except ImportError:
    sys.exit("Run: pip install numpy")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (all overridable via ENV)
# ─────────────────────────────────────────────────────────────────────────────
API_TOKEN = os.getenv("DERIV_API_TOKEN", "")
APP_ID    = os.getenv("DERIV_APP_ID",    "1089")
WS_URL    = f"wss://ws.binaryws.com/websockets/v3?app_id={APP_ID}"

# Supabase Storage
SUPABASE_URL    = os.getenv("SUPABASE_URL",    "https://qtkjixwiahghisqqxrcz.supabase.co")
SUPABASE_KEY    = os.getenv("SUPABASE_KEY",    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InF0a2ppeHdpYWhnaGlzcXF4cmN6Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3OTI2ODU4NSwiZXhwIjoyMDk0ODQ0NTg1fQ.hvrhbSZ7KjY2YCeWOI1qlWILB7ugdovd6cFPdto-E4E")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "Supabot")

_PERSIST_DIR = os.getenv("PERSIST_DIR", os.path.join(os.getcwd(), "data"))
DATA_DIR     = os.getenv("DATA_DIR",    os.path.join(_PERSIST_DIR, "symbol_data"))
CAL_FILE     = os.path.join(_PERSIST_DIR, "calibration.json")
ADVISOR_LOG  = os.path.join(_PERSIST_DIR, "advisor_log.txt")
PORT         = int(os.getenv("PORT", "8080"))

os.makedirs(_PERSIST_DIR, exist_ok=True)
os.makedirs(DATA_DIR,     exist_ok=True)

# Trading
BASE_STAKE       = float(os.getenv("BASE_STAKE",       "1.0"))
MARTINGALE_MULT  = float(os.getenv("MARTI_MULT",       "4.45"))
MARTINGALE_STEPS = int(  os.getenv("MARTI_STEPS",      "3"))
LOSS_COOLDOWN    = float(os.getenv("LOSS_COOLDOWN",    "45"))
TARGET_PROFIT    = float(os.getenv("TARGET_PROFIT",    "45.0"))
STOP_LOSS        = float(os.getenv("STOP_LOSS",        "10.0"))
LOCK_TIMEOUT     = 360

# LSTM
LSTM_THRESHOLD       = float(os.getenv("LSTM_THRESHOLD",       "0.72"))
LSTM_THRESHOLD_FLOOR = float(os.getenv("LSTM_THRESHOLD_FLOOR", "0.60"))
LSTM_SEQ_LEN         = int(  os.getenv("LSTM_SEQ_LEN",         "60"))
MAX_IDLE_MINUTES     = float(os.getenv("MAX_IDLE_MINUTES",     "30"))
RETRAIN_HOURS        = float(os.getenv("RETRAIN_HOURS",        "4"))
RETRAIN_SECS         = RETRAIN_HOURS * 3600

# Advisor candles
CANDLE_GRAN_1 = 60
CANDLE_GRAN_5 = 300
CANDLE_COUNT  = 20

SAFE_BOUNDS = {
    "barrier":           (1.50, 4.00, 0.25),
    "lstm_threshold":    (0.60, 0.90, 0.05),
    "base_stake":        (0.35, 2.00, 0.35),
    "martingale_steps":  (1,    3,    1),
    "loss_cooldown":     (10,   120,  15),
}

# 18 features — must match columns in your CSV exactly
_FEATURES = [
    "sigma_ewma", "range_20", "range_50", "ema_gap", "zscore_50",
    "spike_10", "atr_14", "entropy_20", "regime_enc",
    "sigma_trend", "range_ratio", "ema_cross", "zscore_abs",
    "entropy_delta", "sigma_vs_gate", "spike_vs_sigma", "atr_trend",
    "hour_of_day",
]
_REGIME_ENC = {"CALM": 0, "RANGING": 1, "TRENDING": 2, "CHAOS": 3}

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log  = logging.getLogger("lstm_bot")
info = log.info
warn = log.warning
err  = log.error
def tlog(m): log.info(f"[TRADE] {m}")

# ─────────────────────────────────────────────────────────────────────────────
# CALIBRATION  — derived from local CSV, no live collection
# ─────────────────────────────────────────────────────────────────────────────

def supabase_download(symbol: str) -> bool:
    """
    Downloads symbol CSV from Supabase Storage bucket into DATA_DIR.
    Tries supabase-py first, falls back to raw urllib.
    Returns True on success, False on failure (bot uses local CSV as fallback).
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    dest        = os.path.join(DATA_DIR, f"{symbol}.csv")
    remote_path = f"{symbol}.csv"
    info(f"[SUPABASE] Downloading {remote_path} from bucket={SUPABASE_BUCKET} ...")

    # ── Try supabase-py ───────────────────────────────────────────────────────
    try:
        from supabase import create_client
        sb  = create_client(SUPABASE_URL, SUPABASE_KEY)
        raw = sb.storage.from_(SUPABASE_BUCKET).download(remote_path)
        with open(dest, "wb") as f:
            f.write(raw)
        rows = sum(1 for _ in open(dest)) - 1
        info(f"[SUPABASE] OK via supabase-py  rows={rows}  -> {dest}")
        return True
    except ImportError:
        warn("[SUPABASE] supabase-py not installed — trying urllib")
    except Exception as exc:
        warn(f"[SUPABASE] supabase-py failed: {exc} — trying urllib")

    # ── Fallback: raw urllib ──────────────────────────────────────────────────
    try:
        import urllib.request
        url = (f"{SUPABASE_URL}/storage/v1/object/"
               f"{SUPABASE_BUCKET}/{remote_path}")
        req = urllib.request.Request(
            url,
            headers={
                "apikey":        SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
            }
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = resp.read()
        with open(dest, "wb") as f:
            f.write(data)
        rows = data.count(b"\n") - 1
        info(f"[SUPABASE] OK via urllib  rows={rows}  -> {dest}")
        return True
    except Exception as exc:
        err(f"[SUPABASE] urllib failed: {exc}")
        if os.path.exists(dest):
            rows = sum(1 for _ in open(dest)) - 1
            warn(f"[SUPABASE] Using existing local file  rows={rows}")
            return True
        return False


def scan_data_dir() -> List[str]:
    """Return list of symbol names that have a CSV in DATA_DIR."""
    found = []
    if not os.path.isdir(DATA_DIR):
        return found
    for fname in os.listdir(DATA_DIR):
        if fname.endswith(".csv"):
            found.append(fname[:-4])   # strip .csv → symbol name
    return found


def compute_calibration(symbols: List[str]) -> dict:
    """
    Read each symbol's CSV, compute volatility percentiles,
    derive barrier + sigma_gate, write calibration.json.
    """
    import statistics as _stats
    info("Computing calibration from local CSVs...")
    symbol_scores = {}

    for sym in symbols:
        fpath = os.path.join(DATA_DIR, f"{sym}.csv")
        if not os.path.exists(fpath):
            warn(f"  {sym}: file not found — skipping"); continue

        sigmas, ema_gaps, zscores = [], [], []
        regime_counts = {k: 0 for k in _REGIME_ENC}
        ticks = 0

        with open(fpath, newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                warn(f"  {sym}: empty CSV — skipping"); continue
            for row in reader:
                try:
                    sigmas.append(float(row["sigma_ewma"]))
                    ema_gaps.append(float(row["ema_gap"]))
                    zscores.append(abs(float(row["zscore_50"])))
                    r = row.get("regime", "CHAOS").strip()
                    if r in regime_counts:
                        regime_counts[r] += 1
                    ticks += 1
                except (ValueError, KeyError):
                    continue

        if ticks < LSTM_SEQ_LEN + 200:
            warn(f"  {sym}: only {ticks} rows (need {LSTM_SEQ_LEN+200}) — skipping")
            continue

        sigmas.sort(); zscores.sort()

        def pct(lst, p):
            return lst[max(0, int(len(lst) * p / 100) - 1)]

        sigma_p50 = pct(sigmas, 50)
        sigma_p75 = pct(sigmas, 75)
        total_r   = sum(regime_counts.values()) or 1

        barrier = round(0.906 * math.sqrt(2) * sigma_p75 * math.sqrt(120), 2)
        barrier = max(1.5, min(barrier, 4.0))
        p_win   = math.erf(barrier / (math.sqrt(2) * sigma_p50 * math.sqrt(120)))
        calm_pct = ((regime_counts.get("CALM", 0) +
                     regime_counts.get("RANGING", 0)) / total_r)
        score    = calm_pct * p_win

        regime_pct = {k: round(v / total_r, 4)
                      for k, v in regime_counts.items()}

        symbol_scores[sym] = {
            "symbol":       sym,
            "ticks":        ticks,
            "score":        round(score, 4),
            "p_win_median": round(p_win, 4),
            "calm_pct":     round(calm_pct, 4),
            "barrier":      barrier,
            "duration_min": 2,
            "sigma_gate":   round(sigma_p50, 5),
            "regime_pct":   regime_pct,
        }
        info(f"  {sym}: ticks={ticks}  score={score:.4f}  "
             f"barrier=±{barrier}  sigma_gate={sigma_p50:.5f}  "
             f"p_win={p_win:.3f}")

    if not symbol_scores:
        sys.exit(
            f"No usable CSVs found in {DATA_DIR}.\n"
            f"Make sure your 1HZ10V.csv (or other symbol CSVs) are in that folder."
        )

    ranked = sorted(symbol_scores.values(),
                    key=lambda x: x["score"], reverse=True)
    top    = ranked[:min(2, len(ranked))]

    cal = {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "source":        "local_csv",
        "all_symbols":   ranked,
        "trade_symbols": top,
    }
    with open(CAL_FILE, "w") as f:
        json.dump(cal, f, indent=2)
    info(f"Calibration saved → {CAL_FILE}")
    return cal

# ─────────────────────────────────────────────────────────────────────────────
# LSTM GATE
# ─────────────────────────────────────────────────────────────────────────────

class LSTMGate:
    """
    Trains on the local symbol CSV.
    Single binary output: P(price stays within ±barrier over 120 ticks).
    All regimes are included — no CHAOS filter during training.
    """

    MODEL_FILE  = os.path.join(_PERSIST_DIR, "lstm_model.keras")
    SCALER_FILE = os.path.join(_PERSIST_DIR, "lstm_scaler.pkl")

    def __init__(self):
        self._model  = None
        self._scaler = None
        self._lock   = threading.Lock()
        self._load_existing()

    # ── Load from disk if available ───────────────────────────────────────────
    def _load_existing(self):
        if os.path.exists(self.MODEL_FILE) and os.path.exists(self.SCALER_FILE):
            try:
                import tensorflow as tf
                with self._lock:
                    self._model = tf.keras.models.load_model(self.MODEL_FILE)
                    with open(self.SCALER_FILE, "rb") as f:
                        self._scaler = pickle.load(f)
                info("[LSTM] Loaded saved model from disk — gate ready immediately")
            except Exception as exc:
                warn(f"[LSTM] Could not load saved model: {exc} — will retrain")
                self._model = self._scaler = None

    @property
    def active(self) -> bool:
        return self._model is not None and self._scaler is not None

    # ── Build training sequences from CSV rows ────────────────────────────────
    @staticmethod
    def _build_sequences(rows: list, sigma_gate: float,
                         barrier: float, window: int = 120):
        SEQ    = LSTM_SEQ_LEN
        n      = len(rows)
        prices = np.array([r["price"]       for r in rows], dtype=float)
        sig_v  = np.array([r["sigma_ewma"]  for r in rows], dtype=float)
        atr_v  = np.array([r["atr_14"]      for r in rows], dtype=float)
        ent_v  = np.array([r["entropy_20"]  for r in rows], dtype=float)

        feat = np.zeros((n, 18), dtype=float)
        for i, r in enumerate(rows):
            sigma_trend   = sig_v[i] - sig_v[max(0, i - 10)]
            entropy_delta = ent_v[i] - ent_v[max(0, i - 5)]
            atr_trend     = atr_v[i] - atr_v[max(0, i - 10)]
            feat[i] = [
                r["sigma_ewma"],
                r["range_20"],
                r["range_50"],
                r["ema_gap"],
                r["zscore_50"],
                r["spike_10"],
                r["atr_14"],
                r["entropy_20"],
                float(_REGIME_ENC.get(r["regime"], 0)),
                sigma_trend,
                r["range_20"] / (r["range_50"] + 1e-9),
                r["ema7"]  - r["ema14"],
                abs(r["zscore_50"]),
                entropy_delta,
                r["sigma_ewma"] / (sigma_gate + 1e-9),
                r["spike_10"]   / (r["sigma_ewma"] + 1e-9),
                atr_trend,
                float(int(r["ts"][11:13])),   # hour-of-day from ISO ts
            ]

        # Binary label: 1 = price stays within ±barrier for next `window` ticks
        labels = np.full(n, np.nan)
        for i in range(n - window):
            max_dev  = np.max(np.abs(prices[i+1:i+window+1] - prices[i]))
            labels[i] = 1.0 if max_dev <= barrier else 0.0

        X_list, y_list = [], []
        for i in range(SEQ, n - window):
            if not np.isnan(labels[i]):
                X_list.append(feat[i - SEQ: i])
                y_list.append(labels[i])

        if not X_list:
            return None, None
        return (np.array(X_list, dtype=np.float32),
                np.array(y_list,  dtype=np.float32))

    # ── Train ─────────────────────────────────────────────────────────────────
    def train(self, cal: dict):
        """
        Blocking train — call from a background thread for retrains.
        cal keys used: symbol, barrier, sigma_gate
        """
        import tensorflow as tf
        from sklearn.preprocessing import StandardScaler

        sym        = cal.get("symbol", "1HZ10V")
        barrier    = cal.get("barrier",    2.15)
        sigma_gate = cal.get("sigma_gate", 0.13444)
        csv_path   = os.path.join(DATA_DIR, f"{sym}.csv")

        if not os.path.exists(csv_path):
            warn(f"[LSTM] {csv_path} not found — cannot train"); return

        # ── Load CSV ──────────────────────────────────────────────────────────
        rows = []
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    rows.append({
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

        min_rows = LSTM_SEQ_LEN + 200
        if len(rows) < min_rows:
            warn(f"[LSTM] Only {len(rows)} rows — need {min_rows}"); return

        hours = len(rows) / 3600
        info(f"[LSTM] Building sequences — {len(rows)} rows (~{hours:.1f}h)  "
             f"barrier=±{barrier}")

        X, y = self._build_sequences(rows, sigma_gate, barrier)
        if X is None or len(X) < 100:
            warn("[LSTM] Not enough valid sequences"); return

        info(f"[LSTM] {len(X)} sequences  base_wr={y.mean()*100:.1f}%  "
             f"shape={X.shape}")

        # ── Scale (fit on train split only — chronological 80/20) ─────────────
        N, T, F  = X.shape
        split    = int(N * 0.8)
        scaler   = StandardScaler()
        X_flat   = X.reshape(-1, F)
        X_flat   = scaler.fit_transform(X_flat)
        X_scaled = X_flat.reshape(N, T, F)

        X_train, y_train = X_scaled[:split], y[:split]
        X_val,   y_val   = X_scaled[split:], y[split:]

        # ── Architecture ──────────────────────────────────────────────────────
        tf.keras.backend.clear_session()
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(T, F)),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        # Class weights — handle win/loss imbalance
        pos  = y_train.sum()
        neg  = len(y_train) - pos
        tot  = len(y_train)
        cw   = {0: tot / (2 * neg + 1e-9), 1: tot / (2 * pos + 1e-9)}

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5,
                restore_best_weights=True, verbose=0),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5,
                patience=3, min_lr=1e-5, verbose=0),
        ]

        info("[LSTM] Training started...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=64,
            class_weight=cw,
            callbacks=callbacks,
            verbose=0,
        )

        epochs_run = len(history.history.get("val_loss", []))
        val_loss   = min(history.history.get("val_loss",     [999]))
        val_acc    = max(history.history.get("val_accuracy", [0]))

        # Precision at threshold on val set
        val_probs = model.predict(X_val, verbose=0).flatten()
        mask      = val_probs >= LSTM_THRESHOLD
        n_sig     = mask.sum()
        prec      = y_val[mask].mean() * 100 if n_sig > 0 else 0.0

        info(f"[LSTM] Done  epochs={epochs_run}  val_loss={val_loss:.4f}  "
             f"val_acc={val_acc:.1%}  "
             f"signals@{LSTM_THRESHOLD}={n_sig}  precision={prec:.1f}%")

        # ── Save + hot-swap ───────────────────────────────────────────────────
        model.save(self.MODEL_FILE)
        with open(self.SCALER_FILE, "wb") as f:
            pickle.dump(scaler, f)

        with self._lock:
            self._model  = model
            self._scaler = scaler
        info("[LSTM] Model hot-swapped — gate ACTIVE")

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(self, sequence: np.ndarray) -> float:
        """
        sequence: np.ndarray shape (LSTM_SEQ_LEN, 18)
        Returns P(win) in [0,1], or -1.0 if model not ready.
        """
        if not self.active:
            return -1.0
        try:
            import tensorflow as tf
            with self._lock:
                scaler = self._scaler
                model  = self._model
            N, F     = sequence.shape
            flat     = scaler.transform(sequence.reshape(-1, F))
            inp      = flat.reshape(1, N, F).astype(np.float32)
            return round(float(model.predict(inp, verbose=0)[0][0]), 4)
        except Exception as exc:
            warn(f"[LSTM] predict error: {exc}")
            return -1.0


# Module-level singleton
_lstm_gate: Optional[LSTMGate] = None

def get_lstm_gate() -> LSTMGate:
    global _lstm_gate
    if _lstm_gate is None:
        _lstm_gate = LSTMGate()
    return _lstm_gate

def retrain_lstm(cal: dict):
    """Run in a background thread."""
    global _lstm_gate
    if _lstm_gate is None:
        _lstm_gate = LSTMGate()
    try:
        _lstm_gate.train(cal)
    except Exception as exc:
        err(f"[LSTM] retrain failed: {exc}")
        traceback.print_exc()

# ─────────────────────────────────────────────────────────────────────────────
# CANDLE FEED
# ─────────────────────────────────────────────────────────────────────────────

class CandleFeed:
    async def fetch(self, symbol: str) -> dict:
        result = {"candles_1m": [], "candles_5m": []}
        try:
            ws  = await websockets.connect(WS_URL, ping_interval=20,
                                           ping_timeout=15, close_timeout=5)
            rid = 0

            async def send(d):
                nonlocal rid; rid += 1; d["req_id"] = rid
                await ws.send(json.dumps(d))

            async def recv_type(mt, timeout=10):
                dl = asyncio.get_event_loop().time() + timeout
                while True:
                    rem = dl - asyncio.get_event_loop().time()
                    if rem <= 0: return None
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=rem)
                        msg = json.loads(raw)
                        if mt in msg or "error" in msg: return msg
                    except Exception: return None

            await send({"authorize": API_TOKEN})
            auth = await recv_type("authorize", timeout=10)
            if not auth or "error" in auth:
                warn("[CANDLE] Auth failed"); return result

            end_ep = int(time.time())
            for gran, key in [(CANDLE_GRAN_1,"candles_1m"),(CANDLE_GRAN_5,"candles_5m")]:
                await send({"ticks_history": symbol, "style": "candles",
                            "granularity": gran,
                            "start": end_ep - gran * CANDLE_COUNT * 2,
                            "end": end_ep, "count": CANDLE_COUNT})
                resp = await recv_type("candles", timeout=12)
                if resp and "candles" in resp:
                    result[key] = [{"epoch": c["epoch"],
                                    "open":  float(c["open"]),
                                    "high":  float(c["high"]),
                                    "low":   float(c["low"]),
                                    "close": float(c["close"])}
                                   for c in resp["candles"][-CANDLE_COUNT:]]
            await ws.close()
        except Exception as exc:
            err(f"[CANDLE] {exc}")
        return result

# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class IndicatorEngine:
    @staticmethod
    def rsi(closes, period=14):
        if len(closes) < period + 1: return None
        gains = [max(closes[i]-closes[i-1], 0) for i in range(1,len(closes))]
        losses= [max(closes[i-1]-closes[i], 0) for i in range(1,len(closes))]
        ag = sum(gains[-period:]) / period
        al = sum(losses[-period:]) / period
        return round(100 - 100/(1+ag/al), 2) if al else 100.0

    @staticmethod
    def bollinger(closes, period=20, std_dev=2.0):
        if len(closes) < period: return None
        w   = closes[-period:]
        mid = sum(w)/period
        std = math.sqrt(sum((x-mid)**2 for x in w)/period)
        up  = mid + std_dev*std; lo = mid - std_dev*std
        return {"upper": round(up,5), "mid": round(mid,5),
                "lower": round(lo,5),
                "width": round((up-lo)/(mid+1e-9),6),
                "pos":   round((closes[-1]-lo)/(up-lo+1e-9),4)}

    @staticmethod
    def ema_cross(closes):
        if len(closes) < 14:
            return {"ema7":None,"ema14":None,"cross":"neutral","gap":0.0}
        k7=2/8; k14=2/15; e7=closes[0]; e14=closes[0]
        for c in closes[1:]:
            e7=c*k7+e7*(1-k7); e14=c*k14+e14*(1-k14)
        gap = e7-e14
        return {"ema7":round(e7,5),"ema14":round(e14,5),
                "cross":"bullish" if gap>0 else ("bearish" if gap<0 else "neutral"),
                "gap":round(gap,6)}

    @staticmethod
    def atr(candles, period=14):
        if len(candles) < period+1: return None
        trs=[]
        for i in range(1,len(candles)):
            h=candles[i]["high"]; l=candles[i]["low"]; pc=candles[i-1]["close"]
            trs.append(max(h-l,abs(h-pc),abs(l-pc)))
        return round(sum(trs[-period:])/period,6)

    @classmethod
    def compute(cls, candle_data):
        c1m = candle_data.get("candles_1m",[])
        c5m = candle_data.get("candles_5m",[])
        cl1 = [c["close"] for c in c1m]
        cl5 = [c["close"] for c in c5m]
        out = {
            "rsi_14_1m": cls.rsi(cl1), "rsi_14_5m": cls.rsi(cl5),
            "bb_1m": cls.bollinger(cl1), "bb_5m": cls.bollinger(cl5),
            "ema_1m": cls.ema_cross(cl1), "ema_5m": cls.ema_cross(cl5),
            "atr_1m": cls.atr(c1m), "atr_5m": cls.atr(c5m),
        }
        rsi = out["rsi_14_1m"]; bb = out["bb_1m"]; ema = out["ema_1m"]
        if rsi and bb:
            if   bb["width"] < 0.0005:                         regime="COMPRESSED"
            elif rsi > 70 and bb["pos"] > 0.85:                regime="OVERBOUGHT"
            elif rsi < 30 and bb["pos"] < 0.15:                regime="OVERSOLD"
            elif bb["width"] > 0.003 and ema["cross"]!="neutral": regime="TRENDING"
            elif bb["width"] < 0.0015 and 40<=(rsi or 50)<=60:   regime="CALM"
            else:                                               regime="RANGING"
        else:
            regime = "UNKNOWN"
        out["market_regime"] = regime
        return out

# ─────────────────────────────────────────────────────────────────────────────
# AI ADVISOR
# ─────────────────────────────────────────────────────────────────────────────

class AIAdvisor:
    BREAKEVEN_WR = 0.746
    MIN_TRADES   = 5

    def __init__(self):
        self._cycle            = 0
        self._last_adj         = {}
        self._consecutive_hold = 0

    def advise(self, context: dict) -> dict:
        self._cycle  = context.get("cycle", self._cycle + 1)
        traders      = context.get("traders", [])
        cal          = context.get("calibration", {})
        indicators   = context.get("indicators", {})

        total   = sum(t.risk.wins + t.risk.losses for t in traders)
        wins    = sum(t.risk.wins                  for t in traders)
        pnl     = sum(t.risk.session_pnl           for t in traders)
        streak  = max((t.risk.loss_streak for t in traders), default=0)
        wr      = wins/total if total > 0 else None

        lstm_conf   = context.get("lstm_conf_mean")
        lstm_idle   = context.get("lstm_idle_mins", 0)
        lstm_thresh = context.get("lstm_threshold", LSTM_THRESHOLD)

        reasoning=[]; adj={}; layer="HOLD"

        # L1 EMERGENCY
        if streak >= MARTINGALE_STEPS:
            reasoning.append(f"L1-EMERGENCY: loss_streak={streak}. "
                             f"Raising threshold.")
            adj["lstm_threshold"] = min(lstm_thresh+0.05,
                                        SAFE_BOUNDS["lstm_threshold"][1])
            layer="L1_EMERGENCY"

        if pnl < -(STOP_LOSS*0.5) and total >= self.MIN_TRADES:
            reasoning.append(f"L1-EMERGENCY: P&L=${pnl:.2f}. "
                             f"Raising threshold + resetting stake.")
            adj["lstm_threshold"] = min(lstm_thresh+0.05,
                                        SAFE_BOUNDS["lstm_threshold"][1])
            adj["base_stake"] = BASE_STAKE; layer="L1_EMERGENCY"

        # L2 PERFORMANCE
        if layer=="HOLD" and total>=self.MIN_TRADES and wr is not None:
            if wr < self.BREAKEVEN_WR - 0.05:
                reasoning.append(f"L2-PERFORMANCE: WR={wr:.1%} >5% below breakeven.")
                adj["lstm_threshold"]=min(lstm_thresh+0.04,
                                          SAFE_BOUNDS["lstm_threshold"][1])
                layer="L2_PERFORMANCE"
            elif wr < self.BREAKEVEN_WR:
                reasoning.append(f"L2-PERFORMANCE: WR={wr:.1%} marginally below.")
                adj["lstm_threshold"]=min(lstm_thresh+0.02,
                                          SAFE_BOUNDS["lstm_threshold"][1])
                layer="L2_PERFORMANCE"

        # L3 MARKET
        if layer=="HOLD" and indicators:
            regime = indicators.get("market_regime","UNKNOWN")
            if regime in ("TRENDING","COMPRESSED"):
                reasoning.append(f"L3-MARKET: regime={regime}. Widening barrier.")
                adj["barrier"] = min(cal.get("barrier",2.15)*1.10,
                                     SAFE_BOUNDS["barrier"][1])
                layer="L3_MARKET"

        # L4 LSTM HEALTH
        if layer=="HOLD":
            if lstm_idle >= MAX_IDLE_MINUTES:
                new_t = max(lstm_thresh-0.05, LSTM_THRESHOLD_FLOOR)
                if new_t < lstm_thresh:
                    reasoning.append(f"L4-LSTM_HEALTH: idle {lstm_idle:.0f}min. "
                                     f"Relaxing threshold {lstm_thresh:.2f}→{new_t:.2f}.")
                    adj["lstm_threshold"]=new_t; layer="L4_LSTM_HEALTH"
                else:
                    reasoning.append(f"L4-LSTM_HEALTH: idle {lstm_idle:.0f}min "
                                     f"but already at floor {LSTM_THRESHOLD_FLOOR}.")
            elif lstm_conf is not None and lstm_conf > 0:
                reasoning.append(f"L4-LSTM_HEALTH: conf={lstm_conf:.3f}  "
                                 f"thresh={lstm_thresh:.2f}  idle={lstm_idle:.0f}min. OK.")

        # L5 FINE-TUNE
        if layer=="HOLD" and total>=self.MIN_TRADES and wr and wr>=self.BREAKEVEN_WR:
            self._consecutive_hold += 1
            if self._consecutive_hold >= 3:
                cur_b = cal.get("barrier",2.15)
                if cur_b < SAFE_BOUNDS["barrier"][1] - 0.25:
                    reasoning.append(f"L5-FINE_TUNE: {self._consecutive_hold} "
                                     f"HOLD cycles WR={wr:.1%}. Nudging barrier.")
                    adj["barrier"]=cur_b+0.10; layer="L5_FINE_TUNE"
                    self._consecutive_hold=0
        else:
            self._consecutive_hold=0

        # L6 HOLD
        if layer=="HOLD":
            if total < self.MIN_TRADES:
                reasoning.append(f"L6-HOLD: {total} trades — not enough data.")
            else:
                reasoning.append(f"L6-HOLD: WR={wr:.1%}  P&L=${pnl:.2f}  "
                                 f"streak={streak}. All good.")

        # Apply safe bounds
        applied={}; rejected={}
        for key, proposed in adj.items():
            if key not in SAFE_BOUNDS: continue
            lo,hi,ms = SAFE_BOUNDS[key]
            cur = cal.get(key, globals().get(key.upper(), proposed))
            if isinstance(proposed, float):
                d = proposed - cur
                v = round(max(lo, min(hi, cur + max(-ms, min(ms,d)))), 5)
            else:
                d = proposed - cur
                v = max(int(lo), min(int(hi), int(cur)+max(-int(ms),min(int(ms),int(d)))))
            if v == cur: rejected[key]=f"no change (proposed={proposed})"
            else:        applied[key]={"from":cur,"to":v}

        self._last_adj = {k:v["to"] for k,v in applied.items()}
        return {"cycle":self._cycle,"layer":layer,"reasoning":reasoning,
                "applied":applied,"rejected":rejected,
                "context_summary":{"trades":total,"win_rate":round(wr,4) if wr else None,
                                   "session_pnl":round(pnl,4),"max_streak":streak,
                                   "market_regime":indicators.get("market_regime","?"),
                                   "lstm_conf":lstm_conf,"lstm_idle_min":lstm_idle,
                                   "lstm_threshold":lstm_thresh}}

    def write_log(self, result: dict):
        sep   = "═"*70
        lines = [f"\n{sep}",
                 f"CYCLE {result['cycle']}  |  "
                 f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
                 f"LAYER: {result['layer']}", sep, "CONTEXT:"]
        for k,v in result["context_summary"].items():
            lines.append(f"  {k:<22} {v}")
        lines.append("\nREASONING:")
        for r in result["reasoning"]: lines.append(f"  • {r}")
        if result["applied"]:
            lines.append("\nADJUSTMENTS APPLIED:")
            for k,v in result["applied"].items():
                lines.append(f"  ✓ {k:<22} {v['from']} → {v['to']}")
        else:
            lines.append("\nADJUSTMENTS APPLIED: none")
        if result["rejected"]:
            lines.append("\nADJUSTMENTS REJECTED:")
            for k,v in result["rejected"].items():
                lines.append(f"  ✗ {k:<22} {v}")
        lines.append(sep)
        block = "\n".join(lines)
        try:
            with open(ADVISOR_LOG,"a", encoding="utf-8") as f: f.write(block+"\n")
        except Exception as exc:
            warn(f"[ADVISOR] log write failed: {exc}")
        info(block)


_advisor = AIAdvisor()

# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL ENGINE  — LSTM-only gate
# ─────────────────────────────────────────────────────────────────────────────

class SignalEngine:
    """
    Maintains a rolling 60-tick feature buffer.
    Calls LSTMGate.predict() as the SOLE gate.
    No rule-based filters. All regimes allowed.
    """

    EWMA_ALPHA = 0.05

    def __init__(self, cal: dict):
        self.cal    = cal
        self.tick_n = 0
        self.prices: deque = deque(maxlen=500)
        self._sigma_ewma = None
        self._ema7 = self._ema14 = None
        self._k7   = 2/8;  self._k14 = 2/15
        self._warmup = 100

        self._sigma_buf:   deque = deque(maxlen=15)
        self._entropy_buf: deque = deque(maxlen=10)
        self._atr_buf:     deque = deque(maxlen=15)
        self._seq_buf:     deque = deque(maxlen=LSTM_SEQ_LEN)   # (60, 18) buffer

        self._last_signal_time   = time.time()
        self._lstm_threshold     = LSTM_THRESHOLD
        self._conf_history: deque = deque(maxlen=200)

    def reset_threshold(self):
        if self._lstm_threshold < LSTM_THRESHOLD:
            info(f"[SIG] Win — threshold reset "
                 f"{self._lstm_threshold:.2f} → {LSTM_THRESHOLD:.2f}")
        self._lstm_threshold = LSTM_THRESHOLD

    @property
    def idle_minutes(self) -> float:
        return (time.time() - self._last_signal_time) / 60.0

    @property
    def mean_confidence(self) -> float:
        if not self._conf_history: return 0.0
        return round(sum(self._conf_history)/len(self._conf_history), 4)

    def ingest(self, price: float) -> dict:
        self.tick_n += 1
        prev  = self.prices[-1] if self.prices else price
        delta = abs(price - prev)
        self.prices.append(price)

        # EWMA sigma
        if self._sigma_ewma is None: self._sigma_ewma = delta
        else:
            self._sigma_ewma = (self.EWMA_ALPHA*delta +
                                (1-self.EWMA_ALPHA)*self._sigma_ewma)

        # EMA 7 / 14
        if self._ema7 is None: self._ema7 = self._ema14 = price
        else:
            self._ema7  = price*self._k7  + self._ema7 *(1-self._k7)
            self._ema14 = price*self._k14 + self._ema14*(1-self._k14)

        if self.tick_n < self._warmup:
            return {"trade":False,"reason":"warmup","tick":self.tick_n}

        prices   = list(self.prices)
        sigma    = self._sigma_ewma
        ema_gap  = abs(self._ema7 - self._ema14)

        range20  = (max(prices[-20:])-min(prices[-20:])) if len(prices)>=20 else 999
        range50  = (max(prices[-50:])-min(prices[-50:])) if len(prices)>=50 else range20

        if len(prices) >= 200:
            bl  = prices[-200:]; mu=sum(bl)/200
            std = math.sqrt(sum((p-mu)**2 for p in bl)/200) or 1e-9
            z   = (sum(prices[-50:])/50 - mu) / (std/math.sqrt(50))
        else:
            z = 0.0

        moves  = [abs(prices[i]-prices[i-1]) for i in range(-10,0) if i-1>=-len(prices)]
        spike  = max(moves) if moves else 0
        atrmvs = [abs(prices[i]-prices[i-1]) for i in range(-14,0) if i-1>=-len(prices)]
        atr14  = sum(atrmvs)/len(atrmvs) if atrmvs else 0

        if len(prices)>=21:
            ep=prices[-21:]; em=[abs(ep[i]-ep[i-1]) for i in range(1,len(ep))]
            mx=max(em) or 1; bk=[0]*5
            for m in em: bk[min(4,int(m/mx*4))]+=1
            ne=len(em); H=0.0
            for b in bk:
                if b>0: p=b/ne; H-=p*math.log2(p)
            ent20 = H/math.log2(5)
        else:
            ent20 = 1.0

        if   abs(z)>2.5 or sigma>0.3:              regime="CHAOS"
        elif abs(z)>1.5 and ema_gap>0.3:           regime="TRENDING"
        elif abs(z)<1.0 and ema_gap<0.15:          regime="CALM"
        else:                                       regime="RANGING"

        self._sigma_buf.append(sigma)
        self._entropy_buf.append(ent20)
        self._atr_buf.append(atr14)

        sg            = self.cal.get("sigma_gate", 0.13444)
        sigma_trend   = sigma - list(self._sigma_buf)[0]   if len(self._sigma_buf)>=10   else 0.0
        entropy_delta = ent20 - list(self._entropy_buf)[0] if len(self._entropy_buf)>=5  else 0.0
        atr_trend     = atr14 - list(self._atr_buf)[0]     if len(self._atr_buf)>=10     else 0.0

        feat_vec = np.array([
            sigma, range20, range50, ema_gap, z,
            spike, atr14, ent20,
            float(_REGIME_ENC.get(regime, 0)),
            sigma_trend,
            range20/(range50+1e-9),
            self._ema7 - self._ema14,
            abs(z),
            entropy_delta,
            sigma/(sg+1e-9),
            spike/(sigma+1e-9),
            atr_trend,
            float(datetime.now(timezone.utc).hour),
        ], dtype=np.float32)

        self._seq_buf.append(feat_vec)

        if len(self._seq_buf) < LSTM_SEQ_LEN:
            return {"trade":False,"reason":"seq_warmup","tick":self.tick_n,
                    "regime":regime,"lstm_prob":-1.0,"threshold":self._lstm_threshold}

        # ── LSTM inference ────────────────────────────────────────────────────
        gate  = get_lstm_gate()
        seq   = np.array(list(self._seq_buf), dtype=np.float32)
        prob  = gate.predict(seq)
        if prob >= 0: self._conf_history.append(prob)

        trade = gate.active and (prob >= self._lstm_threshold)
        if trade: self._last_signal_time = time.time()

        return {
            "trade":      trade,
            "tick":       self.tick_n,
            "regime":     regime,
            "lstm_prob":  prob,
            "threshold":  self._lstm_threshold,
            "lstm_active":gate.active,
            "sigma":      round(sigma,5),
            "range20":    round(range20,4),
            "ema_gap":    round(ema_gap,5),
            "z":          round(abs(z),4),
            "spike":      round(spike,5),
        }

# ─────────────────────────────────────────────────────────────────────────────
# RISK MANAGER
# ─────────────────────────────────────────────────────────────────────────────

class RiskManager:
    def __init__(self):
        self.stake           = BASE_STAKE
        self.loss_streak     = 0
        self.session_pnl     = 0.0
        self.wins = self.losses = 0
        self._cooldown_until = 0.0

    def get_stake(self) -> float:
        return round(self.stake, 2)

    def can_trade(self) -> Tuple[bool, str]:
        if time.monotonic() < self._cooldown_until:
            return False, f"cooldown({self._cooldown_until-time.monotonic():.0f}s)"
        if self.session_pnl <= -STOP_LOSS:   return False, "stop_loss"
        if self.session_pnl >= TARGET_PROFIT: return False, "target_hit"
        return True, "ok"

    def record_win(self, profit: float, engine: Optional["SignalEngine"] = None):
        self.wins        += 1
        self.session_pnl += profit
        self.loss_streak  = 0
        self.stake        = BASE_STAKE
        if engine: engine.reset_threshold()
        tlog(f"WIN +${profit:.4f}  stake→${self.stake}  P&L=${self.session_pnl:.4f}")

    def record_loss(self, amount: float):
        self.losses      += 1
        self.session_pnl -= amount
        self.loss_streak += 1
        self._cooldown_until = time.monotonic() + LOSS_COOLDOWN
        if self.loss_streak > MARTINGALE_STEPS:
            self.stake       = BASE_STAKE
            self.loss_streak = 0
            warn(f"MARTINGALE exhausted — reset stake=${self.stake}  "
                 f"P&L=${self.session_pnl:.4f}")
        elif self.loss_streak < 2:
            self.stake = BASE_STAKE
            tlog(f"LOSS streak={self.loss_streak}/{MARTINGALE_STEPS}  "
                 f"stake=${self.stake:.2f} (holding)  P&L=${self.session_pnl:.4f}")
        else:
            self.stake = round(BASE_STAKE*(MARTINGALE_MULT**(self.loss_streak-1)), 2)
            tlog(f"LOSS streak={self.loss_streak}/{MARTINGALE_STEPS}  "
                 f"stake=${self.stake:.2f}  P&L=${self.session_pnl:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# DERIV CLIENT
# ─────────────────────────────────────────────────────────────────────────────

class DerivClient:
    def __init__(self):
        self._ws = self._send_task = self._recv_task = None
        self._send_q  = asyncio.Queue()
        self._inbox   = asyncio.Queue()
        self._rid     = 0
        self.balance  = 0.0

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
            if t and not t.done(): t.cancel()
        self._send_task = asyncio.create_task(self._send_pump())
        self._recv_task = asyncio.create_task(self._recv_pump())

    async def _send_pump(self):
        while True:
            data, fut = await self._send_q.get()
            try:
                await self._ws.send(json.dumps(data))
                if fut and not fut.done(): fut.set_result(True)
            except Exception as exc:
                if fut and not fut.done(): fut.set_exception(exc)
            finally:
                self._send_q.task_done()

    async def _recv_pump(self):
        try:
            async for raw in self._ws:
                try: await self._inbox.put(json.loads(raw))
                except Exception: pass
        except (ConnectionClosed, ConnectionClosedError, ConnectionClosedOK):
            await self._inbox.put({"__disconnect__": True})
        except Exception as exc:
            err(f"recv pump: {exc}")
            await self._inbox.put({"__disconnect__": True})

    async def close(self):
        for t in (self._send_task, self._recv_task):
            if t and not t.done(): t.cancel()
        if self._ws:
            try: await self._ws.close()
            except Exception: pass

    async def _send_msg(self, data: dict):
        self._rid += 1; data["req_id"] = self._rid
        loop = asyncio.get_event_loop()
        fut  = loop.create_future()
        await self._send_q.put((data, fut)); await fut

    async def _recv_type(self, msg_type: str, timeout=10) -> Optional[dict]:
        dl = asyncio.get_event_loop().time() + timeout
        while True:
            rem = dl - asyncio.get_event_loop().time()
            if rem <= 0: return None
            try:
                msg = await asyncio.wait_for(self._inbox.get(), timeout=rem)
            except asyncio.TimeoutError: return None
            if "__disconnect__" in msg:
                await self._inbox.put(msg); return None
            if msg_type in msg or "error" in msg: return msg
            await self._inbox.put(msg)

    async def receive(self, timeout=60) -> dict:
        try: return await asyncio.wait_for(self._inbox.get(), timeout=timeout)
        except asyncio.TimeoutError: return {}

    async def subscribe_ticks(self, symbol: str) -> bool:
        await self._send_msg({"ticks": symbol, "subscribe": 1})
        resp = await self._recv_type("tick", timeout=10)
        if not resp or "error" in resp:
            err(f"Tick sub failed: {(resp or {}).get('error',{}).get('message','?')}")
            return False
        info(f"Subscribed ticks: {symbol}"); return True

    async def place_trade(self, symbol: str, barrier: float,
                          duration_min: int, stake: float
                          ) -> Tuple[Optional[int], Optional[int]]:
        await self._send_msg({
            "proposal": 1, "amount": stake, "basis": "stake",
            "contract_type": "EXPIRYRANGE", "currency": "USD",
            "duration": duration_min, "duration_unit": "m",
            "symbol": symbol,
            "barrier": f"+{barrier}", "barrier2": f"-{barrier}",
        })
        prop = await self._recv_type("proposal", timeout=12)
        if not prop or "error" in prop:
            err(f"Proposal: {(prop or {}).get('error',{}).get('message','?')}")
            return None, None
        pd   = prop.get("proposal", {})
        pid  = pd.get("id")
        ask  = float(pd.get("ask_price", stake))
        pout = float(pd.get("payout", 0))
        roi  = (pout-ask)/ask*100 if ask else 0
        info(f"Proposal OK  ask=${ask:.2f}  payout=${pout:.2f}  ROI={roi:.1f}%")
        if not pid: err("No proposal ID"); return None, None

        buy_ts = time.time()
        await self._send_msg({"buy": pid, "price": ask})
        cid = exp = None
        for attempt in range(8):
            resp = await self._recv_type("buy", timeout=8)
            if resp is None: warn(f"Buy no response #{attempt+1}"); continue
            if "error" in resp:
                err(f"Buy error: {resp['error'].get('message','')}"); return None, None
            bd = resp.get("buy", {})
            cid = bd.get("contract_id"); exp = bd.get("date_expiry")
            if cid: break

        # Orphan recovery
        if not cid:
            warn("Orphan recovery via profit_table")
            for _ in range(4):
                await asyncio.sleep(3)
                await self._send_msg({"profit_table":1,"description":1,
                                      "sort":"DESC","limit":5})
                r = await self._recv_type("profit_table", timeout=10)
                if r and "profit_table" in r:
                    for tx in r["profit_table"].get("transactions",[]):
                        if (abs(float(tx.get("buy_price",0))-stake)<0.01
                                and float(tx.get("purchase_time",0))>=buy_ts-10):
                            cid = tx.get("contract_id")
                            info(f"Orphan recovered → {cid}"); break
                if cid: break
            if not cid: err("Orphan recovery failed"); return None, None

        try:
            await self._send_msg({"proposal_open_contract":1,
                                   "contract_id":cid,"subscribe":1})
        except Exception: pass
        tlog(f"Placed  cid={cid}  EXPIRYRANGE ±{barrier}  ${ask:.2f}  {duration_min}min")
        return cid, exp

    async def poll_contract(self, cid: int) -> Optional[dict]:
        try:
            await self._send_msg({"proposal_open_contract":1,"contract_id":cid})
            resp = await self._recv_type("proposal_open_contract", timeout=10)
            if resp and "proposal_open_contract" in resp:
                return resp["proposal_open_contract"]
        except Exception as exc:
            warn(f"poll_contract: {exc}")
        return None

    @staticmethod
    def is_settled(data: dict) -> bool:
        if data.get("is_settled") or data.get("is_sold"): return True
        return data.get("status","").lower() in ("sold","won","lost")

# ─────────────────────────────────────────────────────────────────────────────
# SYMBOL TRADER
# ─────────────────────────────────────────────────────────────────────────────

class SymbolTrader:
    def __init__(self, cal: dict):
        self.cal    = cal
        self.symbol = cal["symbol"]
        self.engine = SignalEngine(cal)
        self.risk   = RiskManager()
        self.client = DerivClient()

        self.waiting     = False
        self._evaluating = False
        self._settling   = False
        self.current_trade: Optional[dict] = None
        self.lock_since:    Optional[float] = None
        self._stop          = False
        self._poller_task:  Optional[asyncio.Task] = None
        self.live_ticks     = 0
        self.signals        = 0
        self.hot_swap_calibration = lambda new_cal: None

    def _unlock(self, reason="manual"):
        if self.waiting:
            cid = (self.current_trade or {}).get("id","?")
            info(f"[{self.symbol}] Unlock cid={cid} reason={reason}")
        self.waiting = self._evaluating = self._settling = False
        self.current_trade = self.lock_since = None
        if self._poller_task and not self._poller_task.done():
            self._poller_task.cancel(); self._poller_task = None

    def _check_lock_timeout(self):
        if self.waiting and self.lock_since:
            if time.time()-self.lock_since > LOCK_TIMEOUT:
                warn(f"[{self.symbol}] Lock timeout — forcing unlock")
                self._unlock("timeout")

    async def on_tick(self, price: float):
        self.live_ticks += 1
        self._check_lock_timeout()
        if self.waiting or self._evaluating: return

        sig = self.engine.ingest(price)

        if self.live_ticks % 500 == 0:
            gate = get_lstm_gate()
            info(f"[{self.symbol}] tick={self.live_ticks}  "
                 f"regime={sig.get('regime','?')}  "
                 f"prob={sig.get('lstm_prob',-1):.4f}  "
                 f"thresh={sig.get('threshold',LSTM_THRESHOLD):.2f}  "
                 f"active={gate.active}  idle={self.engine.idle_minutes:.1f}min")

        if not sig.get("trade"): return
        ok, reason = self.risk.can_trade()
        if not ok:
            info(f"[{self.symbol}] Signal blocked: {reason}"); return

        self.signals    += 1
        self._evaluating = True
        stake = self.risk.get_stake()
        tlog(f"[{self.symbol}] SIGNAL  "
             f"prob={sig['lstm_prob']:.4f} >= {sig['threshold']:.2f}  "
             f"regime={sig['regime']}  stake=${stake}")

        try:
            cid, exp = await self.client.place_trade(
                self.symbol, self.cal["barrier"],
                self.cal["duration_min"], stake)
        except Exception as exc:
            err(f"[{self.symbol}] place_trade: {exc}")
            self._evaluating = False; return

        if not cid:
            self._evaluating = False; return

        self.waiting     = True
        self._evaluating = False
        self.lock_since  = time.time()
        self.current_trade = {"id":cid,"stake":stake,
                               "expiry":exp,"barrier":self.cal["barrier"]}
        self._poller_task = asyncio.create_task(self._expiry_poller(cid))

    async def _expiry_poller(self, cid: int):
        await asyncio.sleep(self.cal["duration_min"]*60 + 15)
        if not self.waiting or not self.current_trade or \
                self.current_trade.get("id") != cid: return
        warn(f"[{self.symbol}] Poller: {cid} still locked — polling")
        for attempt in range(1,7):
            try:
                data = await self.client.poll_contract(cid)
                if data and self.client.is_settled(data):
                    ok = await self.handle_settlement(data)
                    if not ok: self._stop=True
                    return
            except Exception as exc:
                warn(f"[{self.symbol}] Poller #{attempt}: {exc}")
            await asyncio.sleep(5)
        if self.waiting and self.current_trade and \
                self.current_trade.get("id") == cid:
            warn(f"[{self.symbol}] Poller exhausted — force unlock")
            self._unlock("poller_exhausted")

    async def handle_settlement(self, data: dict) -> bool:
        if self._settling: return True
        self._settling = True
        try:   return await self._settle_inner(data)
        finally: self._settling = False

    async def _settle_inner(self, data: dict) -> bool:
        cid = data.get("contract_id")
        if not self.current_trade or \
                str(cid) != str(self.current_trade["id"]): return True
        if not self.client.is_settled(data): return True
        profit = float(data.get("profit", 0))
        status = data.get("status","?")
        tlog(f"[{self.symbol}] SETTLED  cid={cid}  "
             f"status={status}  profit={profit:+.4f}")
        if profit > 0:
            self.risk.record_win(profit, engine=self.engine)
        else:
            self.risk.record_loss(self.current_trade["stake"])
        self._unlock("settlement")
        info(f"[{self.symbol}] Ready for next signal")
        return True

    async def run(self):
        retry_delay = 5
        while not self._stop:
            try:
                if not await self.client.connect():
                    raise ConnectionError("connect failed")
                if not await self.client.subscribe_ticks(self.symbol):
                    raise ConnectionError("tick sub failed")
                info(f"[{self.symbol}] LIVE  barrier=±{self.cal['barrier']}  "
                     f"dur={self.cal['duration_min']}min  "
                     f"thresh={self.engine._lstm_threshold:.2f}")

                while not self._stop:
                    msg = await self.client.receive(timeout=60)
                    if "__disconnect__" in msg:
                        warn(f"[{self.symbol}] WS disconnected"); break
                    if not msg:
                        try: await self.client._ws.ping()
                        except Exception: break
                        continue
                    if "tick" in msg:
                        await self.on_tick(float(msg["tick"]["quote"]))
                    for key in ("proposal_open_contract","buy"):
                        if key in msg:
                            ok = await self.handle_settlement(msg[key])
                            if not ok: self._stop=True
                    if "transaction" in msg:
                        tx = msg["transaction"]
                        if "contract_id" in tx:
                            ok = await self.handle_settlement({
                                "contract_id": tx.get("contract_id"),
                                "profit":      tx.get("profit",0),
                                "status":      tx.get("action","sold"),
                                "is_settled":  True,
                            })
                            if not ok: self._stop=True

            except Exception as exc:
                err(f"[{self.symbol}] session error: {exc}")
                traceback.print_exc()

            if not self._stop:
                warn(f"[{self.symbol}] Reconnecting in {retry_delay}s...")
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay*2, 60)
                await self.client.close()
                self.client = DerivClient()

        r = self.risk; total = r.wins+r.losses
        wr = r.wins/total*100 if total else 0
        info(f"[{self.symbol}] DONE  trades={total}  W={r.wins}  L={r.losses}  "
             f"WR={wr:.1f}%  P&L=${r.session_pnl:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# HEALTH SERVER
# ─────────────────────────────────────────────────────────────────────────────

_health: dict = {"traders": []}

def start_health_server(traders: List[SymbolTrader]):
    _health["traders"] = traders
    if getattr(start_health_server, "_started", False): return
    start_health_server._started = True

    class _H(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/status":
                gate = get_lstm_gate()
                data = {"lstm_active": gate.active, "traders": []}
                for t in _health["traders"]:
                    r = t.risk; tot = r.wins+r.losses
                    data["traders"].append({
                        "symbol":    t.symbol, "ticks": t.live_ticks,
                        "signals":   t.signals, "trades": tot,
                        "wins":      r.wins, "losses": r.losses,
                        "win_rate":  round(r.wins/tot,4) if tot else 0,
                        "pnl":       round(r.session_pnl,4),
                        "stake":     r.stake, "locked": t.waiting,
                        "lstm_prob": t.engine.mean_confidence,
                        "threshold": t.engine._lstm_threshold,
                        "idle_min":  round(t.engine.idle_minutes,1),
                    })
                body = json.dumps(data, indent=2).encode()
                self.send_response(200)
                self.send_header("Content-Type","application/json")
                self.end_headers(); self.wfile.write(body)
            else:
                gate = get_lstm_gate()
                lstm_st = ("<span style='color:#3fb950'>ACTIVE</span>"
                           if gate.active else
                           "<span style='color:#f85149'>TRAINING…</span>")
                rows = ""
                for t in _health["traders"]:
                    r = t.risk; tot = r.wins+r.losses
                    wr = r.wins/tot*100 if tot else 0
                    rows += (
                        f"<tr>"
                        f"<td>{t.symbol}</td><td>{tot}</td>"
                        f"<td style='color:{('#3fb950' if r.wins>=r.losses else '#f85149')}'>{r.wins}</td>"
                        f"<td style='color:#f85149'>{r.losses}</td>"
                        f"<td style='color:{('#3fb950' if wr>=74.6 else '#f85149')}'>{wr:.1f}%</td>"
                        f"<td style='color:{('#3fb950' if r.session_pnl>=0 else '#f85149')}'>${r.session_pnl:+.4f}</td>"
                        f"<td>${r.stake:.2f}</td>"
                        f"<td>{t.engine._lstm_threshold:.2f}</td>"
                        f"<td>{t.engine.mean_confidence:.3f}</td>"
                        f"<td>{t.engine.idle_minutes:.1f}m</td>"
                        f"<td>{'🔒' if t.waiting else '🟢'}</td>"
                        f"</tr>"
                    )
                html = (
                    f"<!DOCTYPE html><html><head><meta charset=utf-8>"
                    f"<meta http-equiv='refresh' content='10'>"
                    f"<title>LSTM Bot</title>"
                    f"<style>body{{font-family:monospace;background:#0d1117;"
                    f"color:#e6edf3;padding:2rem;}}"
                    f"table{{border-collapse:collapse;}}"
                    f"th,td{{padding:.4rem .8rem;border:1px solid #21262d;}}"
                    f"th{{background:#161b22;color:#8b949e;}}"
                    f"h2{{color:#58a6ff;}}</style></head><body>"
                    f"<h2>LSTM Bot  |  Gate: {lstm_st}</h2>"
                    f"<table border=1 cellpadding=6>"
                    f"<tr><th>Symbol</th><th>Trades</th><th>Wins</th>"
                    f"<th>Losses</th><th>WR</th><th>P&L</th>"
                    f"<th>Stake</th><th>Threshold</th>"
                    f"<th>Avg Conf</th><th>Idle</th><th>Status</th></tr>"
                    f"{rows}</table>"
                    f"<p style='font-size:.8rem'>Breakeven 74.6% &nbsp;|&nbsp; "
                    f"auto-refresh 10s</p>"
                    f"<p><a href='/status' style='color:#58a6ff'>/status JSON</a></p>"
                    f"</body></html>"
                )
                body = html.encode()
                self.send_response(200)
                self.send_header("Content-Type","text/html; charset=utf-8")
                self.end_headers(); self.wfile.write(body)

        def log_message(self, *a): pass

    srv = HTTPServer(("", PORT), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    info(f"Health server → http://localhost:{PORT}   /status for JSON")


# ─────────────────────────────────────────────────────────────────────────────
# SUPABASE UPLOAD
# ─────────────────────────────────────────────────────────────────────────────

def supabase_upload(symbol: str) -> bool:
    """
    Uploads the full local symbol CSV to Supabase Storage (overwrites).
    Called every UPLOAD_INTERVAL_SECS by the TickLogger.
    Small file (~15-20MB) uploads in under 2 seconds on a normal connection.
    Returns True on success, False on failure (non-fatal — next upload retries).
    """
    src_path    = os.path.join(DATA_DIR, f"{symbol}.csv")
    remote_path = f"{symbol}.csv"

    if not os.path.exists(src_path):
        warn(f"[UPLOAD] {src_path} not found — skipping"); return False

    size_kb = os.path.getsize(src_path) / 1024

    # ── Try supabase-py ───────────────────────────────────────────────────────
    try:
        from supabase import create_client
        sb = create_client(SUPABASE_URL, SUPABASE_KEY)
        with open(src_path, "rb") as f:
            data = f.read()
        sb.storage.from_(SUPABASE_BUCKET).upload(
            remote_path, data,
            file_options={"content-type": "text/csv", "upsert": "true"},
        )
        info(f"[UPLOAD] {symbol}.csv uploaded  {size_kb:.0f}KB  -> Supabase/{SUPABASE_BUCKET}")
        return True
    except ImportError:
        pass   # fall through to urllib
    except Exception as exc:
        warn(f"[UPLOAD] supabase-py upload failed: {exc} — trying urllib")

    # ── Fallback: raw urllib ──────────────────────────────────────────────────
    try:
        import urllib.request
        with open(src_path, "rb") as f:
            data = f.read()
        url = (f"{SUPABASE_URL}/storage/v1/object/"
               f"{SUPABASE_BUCKET}/{remote_path}")
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={
                "apikey":        SUPABASE_KEY,
                "Authorization": f"Bearer {SUPABASE_KEY}",
                "Content-Type":  "text/csv",
                "x-upsert":      "true",
            }
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            _ = resp.read()
        info(f"[UPLOAD] {symbol}.csv uploaded via urllib  {size_kb:.0f}KB")
        return True
    except Exception as exc:
        err(f"[UPLOAD] urllib upload failed: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# TICK LOGGER
# ─────────────────────────────────────────────────────────────────────────────

UPLOAD_INTERVAL_SECS = int(os.getenv("UPLOAD_INTERVAL_SECS", "300"))   # 5 minutes


class TickLogger:
    """
    Runs on its own WS connection (separate from the trading connection).
    Subscribes to 1HZ10V, computes all 18 features per tick, appends to the
    local CSV, and uploads the full CSV to Supabase every UPLOAD_INTERVAL_SECS.

    Feature computation is identical to the original SymbolStats engine so the
    rows it writes are indistinguishable from existing CSV rows — the LSTM
    retrains on a seamless, ever-growing dataset.
    """

    EWMA_ALPHA = 0.05

    def __init__(self, symbol: str):
        self.symbol    = symbol
        self._stop     = False

        # Rolling state (mirrors SymbolStats)
        self._tick_n     = 0
        self._prices     = deque(maxlen=500)
        self._sigma_ewma = None
        self._ema7       = None
        self._ema14      = None
        self._k7         = 2 / (7 + 1)
        self._k14        = 2 / (14 + 1)

        # In-memory buffer flushed every UPLOAD_INTERVAL_SECS
        self._buffer: List[dict] = []
        self._last_upload        = time.time()

        # CSV path
        self._csv_path = os.path.join(DATA_DIR, f"{symbol}.csv")
        self._fields   = [
            "ts", "epoch", "symbol", "tick_n",
            "price", "tick_delta", "tick_abs_delta",
            "sigma_ewma", "range_20", "range_50",
            "ema7", "ema14", "ema_gap",
            "zscore_50", "spike_10", "atr_14",
            "entropy_20", "regime",
        ]

    def _compute(self, price: float, epoch: float) -> dict:
        self._tick_n += 1
        prev      = self._prices[-1] if self._prices else price
        delta     = price - prev
        abs_delta = abs(delta)
        self._prices.append(price)

        # EWMA sigma
        if self._sigma_ewma is None:
            self._sigma_ewma = abs_delta
        else:
            self._sigma_ewma = (self.EWMA_ALPHA * abs_delta +
                                (1 - self.EWMA_ALPHA) * self._sigma_ewma)

        # EMA 7 / 14
        if self._ema7 is None:
            self._ema7 = self._ema14 = price
        else:
            self._ema7  = price * self._k7  + self._ema7  * (1 - self._k7)
            self._ema14 = price * self._k14 + self._ema14 * (1 - self._k14)

        ema_gap = abs(self._ema7 - self._ema14)
        prices  = list(self._prices)

        range_20 = (max(prices[-20:]) - min(prices[-20:])
                    if len(prices) >= 20 else 0.0)
        range_50 = (max(prices[-50:]) - min(prices[-50:])
                    if len(prices) >= 50 else 0.0)

        # Z-score
        zscore_50 = 0.0
        if len(prices) >= 200:
            bl  = prices[-200:]
            mu  = sum(bl) / 200
            var = sum((p - mu) ** 2 for p in bl) / 200
            std = math.sqrt(var) if var > 0 else 1e-9
            zscore_50 = (sum(prices[-50:]) / 50 - mu) / (std / math.sqrt(50))

        moves    = [abs(prices[i] - prices[i-1]) for i in range(-10, 0)
                    if i - 1 >= -len(prices)]
        spike_10 = max(moves) if moves else 0.0

        atr_mvs = [abs(prices[i] - prices[i-1]) for i in range(-14, 0)
                   if i - 1 >= -len(prices)]
        atr_14  = sum(atr_mvs) / len(atr_mvs) if atr_mvs else 0.0

        # Shannon entropy
        if len(prices) >= 21:
            ep  = prices[-21:]
            em  = [abs(ep[i] - ep[i-1]) for i in range(1, len(ep))]
            mx  = max(em) or 1
            bk  = [0] * 5
            for m in em:
                bk[min(4, int(m / mx * 4))] += 1
            ne  = len(em); H = 0.0
            for b in bk:
                if b > 0:
                    p = b / ne; H -= p * math.log2(p)
            entropy_20 = H / math.log2(5)
        else:
            entropy_20 = 1.0

        # Regime
        if abs(zscore_50) > 2.5 or self._sigma_ewma > 0.3:
            regime = "CHAOS"
        elif abs(zscore_50) > 1.5 and ema_gap > 0.3:
            regime = "TRENDING"
        elif abs(zscore_50) < 1.0 and ema_gap < 0.15:
            regime = "CALM"
        else:
            regime = "RANGING"

        return {
            "ts":             datetime.now(timezone.utc).isoformat(),
            "epoch":          epoch,
            "symbol":         self.symbol,
            "tick_n":         self._tick_n,
            "price":          round(price, 5),
            "tick_delta":     round(delta, 5),
            "tick_abs_delta": round(abs_delta, 5),
            "sigma_ewma":     round(self._sigma_ewma, 5),
            "range_20":       round(range_20, 4),
            "range_50":       round(range_50, 4),
            "ema7":           round(self._ema7, 5),
            "ema14":          round(self._ema14, 5),
            "ema_gap":        round(ema_gap, 5),
            "zscore_50":      round(zscore_50, 4),
            "spike_10":       round(spike_10, 5),
            "atr_14":         round(atr_14, 5),
            "entropy_20":     round(entropy_20, 4),
            "regime":         regime,
        }

    def _flush_buffer(self):
        """Append buffered rows to local CSV."""
        if not self._buffer:
            return
        file_exists = os.path.exists(self._csv_path)
        try:
            with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._fields)
                if not file_exists:
                    writer.writeheader()
                writer.writerows(self._buffer)
            count = len(self._buffer)
            self._buffer.clear()
            return count
        except Exception as exc:
            err(f"[TICKLOG] flush failed: {exc}")
            return 0

    def _upload_cycle(self):
        """Flush buffer to CSV then upload to Supabase. Runs in thread."""
        count = self._flush_buffer()
        if count:
            info(f"[TICKLOG] Flushed {count} rows to local CSV")
        supabase_upload(self.symbol)

    async def run(self):
        """
        Async loop: subscribe to ticks, compute features, buffer rows.
        Every UPLOAD_INTERVAL_SECS flush + upload in a background thread
        so it never blocks tick ingestion.
        """
        info(f"[TICKLOG] Starting tick logger for {self.symbol} "
             f"(upload every {UPLOAD_INTERVAL_SECS//60}min)")
        retry_delay = 5

        while not self._stop:
            ws = None
            try:
                ws  = await websockets.connect(
                    WS_URL, ping_interval=20, ping_timeout=20, close_timeout=10)
                rid = 0

                async def send(d):
                    nonlocal rid; rid += 1; d["req_id"] = rid
                    await ws.send(json.dumps(d))

                # Auth
                await send({"authorize": API_TOKEN})
                deadline = asyncio.get_event_loop().time() + 15
                authed   = False
                while asyncio.get_event_loop().time() < deadline:
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=5)
                        msg = json.loads(raw)
                        if "authorize" in msg:
                            authed = True; break
                        if "error" in msg:
                            err(f"[TICKLOG] Auth error: {msg['error']}"); break
                    except asyncio.TimeoutError:
                        break

                if not authed:
                    warn("[TICKLOG] Auth failed — retrying"); raise ConnectionError

                # Subscribe
                await send({"ticks": self.symbol, "subscribe": 1})
                info(f"[TICKLOG] Subscribed to {self.symbol}")
                retry_delay = 5

                async for raw in ws:
                    if self._stop: break
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        continue

                    if "tick" not in msg:
                        continue

                    tick  = msg["tick"]
                    price = float(tick["quote"])
                    epoch = float(tick.get("epoch", time.time()))
                    row   = self._compute(price, epoch)
                    self._buffer.append(row)

                    # Every UPLOAD_INTERVAL_SECS — flush + upload
                    now = time.time()
                    if now - self._last_upload >= UPLOAD_INTERVAL_SECS:
                        self._last_upload = now
                        threading.Thread(
                            target=self._upload_cycle,
                            daemon=True,
                            name="tick_upload",
                        ).start()

            except (ConnectionClosed, ConnectionClosedError,
                    ConnectionClosedOK, ConnectionError):
                warn(f"[TICKLOG] WS disconnected — retrying in {retry_delay}s")
            except Exception as exc:
                err(f"[TICKLOG] error: {exc}")
            finally:
                if ws:
                    try: await ws.close()
                    except Exception: pass

            if not self._stop:
                await asyncio.sleep(retry_delay)
                retry_delay = min(retry_delay * 2, 60)

        # Final flush on stop
        self._flush_buffer()
        info(f"[TICKLOG] Stopped for {self.symbol}")

    def stop(self):
        self._stop = True

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

_shutdown_event = asyncio.Event()

async def main():
    info("=" * 60)
    info("  EXPIRYRANGE LSTM-Only Bot")
    info(f"  DATA_DIR        = {DATA_DIR}")
    info(f"  PERSIST_DIR     = {_PERSIST_DIR}")
    info(f"  SUPABASE_URL    = {SUPABASE_URL}")
    info(f"  SUPABASE_BUCKET = {SUPABASE_BUCKET}")
    info("=" * 60)

    # Step 1: download latest CSV from Supabase
    # Always download first so the LSTM trains on the latest data.
    # Falls back silently to existing local CSV if Supabase is unreachable.
    for sym in ["1HZ10V"]:
        ok = supabase_download(sym)
        if not ok:
            warn(f"[SUPABASE] Could not download {sym}.csv -- "
                 f"will use local file if it exists")

    symbols = scan_data_dir()
    if not symbols:
        sys.exit(
            f"\nNo CSV files found in {DATA_DIR}\n"
            f"Upload 1HZ10V.csv to Supabase bucket Supabot "
            f"or place it locally in {DATA_DIR} and restart."
        )
    info(f"Found symbol data: {symbols}")

    # ── Step 2: calibration ───────────────────────────────────────────────────
    # Always recompute from CSV so calibration reflects latest data
    calibration = compute_calibration(symbols)

    trade_symbols = calibration["trade_symbols"]
    if not trade_symbols:
        sys.exit("Calibration produced no tradeable symbols.")

    # ── Step 3: train LSTM ────────────────────────────────────────────────────
    primary_cal = trade_symbols[0]
    info(f"Training LSTM on {primary_cal['symbol']} data — please wait...")
    retrain_lstm(primary_cal)    # blocking — model ready before first tick

    gate = get_lstm_gate()
    if not gate.active:
        warn("LSTM gate did not activate after training — "
             "bot will start but LSTM must activate before any trades fire")

    # ── Step 4: start trading ─────────────────────────────────────────────────
    info(f"Starting traders: {[s['symbol'] for s in trade_symbols]}")
    traders = [SymbolTrader(cal) for cal in trade_symbols]
    start_health_server(traders)

    info("=" * 60)
    for t in traders:
        info(f"  {t.symbol}  barrier=±{t.cal['barrier']}  "
             f"dur={t.cal['duration_min']}min  "
             f"p_win={t.cal['p_win_median']:.3f}  "
             f"sigma_gate={t.cal['sigma_gate']:.5f}")
    if gate.active:
        info(f"  LSTM  threshold={LSTM_THRESHOLD}  "
             f"floor={LSTM_THRESHOLD_FLOOR}  "
             f"seq_len={LSTM_SEQ_LEN}  idle_fallback={MAX_IDLE_MINUTES}min")
    info("=" * 60)

    # Hot-swap support
    for t in traders:
        def _make_swap(trader):
            def swap(new_cal):
                trader.cal    = new_cal
                trader.engine = SignalEngine(new_cal)
                info(f"[{trader.symbol}] ♻ Calibration hot-swapped  "
                     f"barrier=±{new_cal['barrier']}")
            return swap
        t.hot_swap_calibration = _make_swap(t)

    # ── Rolling retrain every RETRAIN_HOURS ───────────────────────────────────
    async def retrain_loop():
        cycle = 0
        candle_feed = CandleFeed()
        while not _shutdown_event.is_set():
            await asyncio.sleep(RETRAIN_SECS)
            if _shutdown_event.is_set(): break
            cycle += 1
            info(f"♻ Retrain cycle {cycle} — recomputing calibration + "
                 f"retraining LSTM on latest CSV...")

            try:
                # Re-download latest CSV from Supabase before retraining
                for sym in ["1HZ10V"]:
                    supabase_download(sym)

                new_cal = compute_calibration(scan_data_dir())
                for t in traders:
                    sc = next((s for s in new_cal["trade_symbols"]
                               if s["symbol"] == t.symbol), None)
                    if sc: t.hot_swap_calibration(sc)

                if new_cal["trade_symbols"]:
                    threading.Thread(
                        target=retrain_lstm,
                        args=(new_cal["trade_symbols"][0],),
                        daemon=True, name="lstm_retrain").start()

                # AI Advisor
                try:
                    sym     = traders[0].symbol
                    candles = await candle_feed.fetch(sym)
                    indics  = IndicatorEngine.compute(candles)
                    engine  = traders[0].engine
                    adv_ctx = {
                        "traders":       traders,
                        "calibration":   new_cal["trade_symbols"][0]
                                         if new_cal["trade_symbols"] else {},
                        "indicators":    indics,
                        "cycle":         cycle,
                        "lstm_conf_mean":engine.mean_confidence,
                        "lstm_idle_mins":engine.idle_minutes,
                        "lstm_threshold":engine._lstm_threshold,
                    }
                    adv = _advisor.advise(adv_ctx)
                    _advisor.write_log(adv)
                    for t in traders:
                        for key, val in adv["applied"].items():
                            nv = val["to"]
                            if   key == "barrier":        t.cal["barrier"] = nv
                            elif key == "lstm_threshold": t.engine._lstm_threshold = nv
                            elif key == "base_stake":
                                global BASE_STAKE; BASE_STAKE = nv
                            elif key == "loss_cooldown":
                                global LOSS_COOLDOWN; LOSS_COOLDOWN = nv
                except Exception as exc:
                    warn(f"Advisor cycle {cycle}: {exc}")

                info(f"♻ Cycle {cycle} done")
            except Exception as exc:
                err(f"Retrain cycle {cycle} failed: {exc}")

    # ── Start tick logger (separate WS, logs + uploads every 5min) ─────────
    tick_logger      = TickLogger("1HZ10V")
    tick_logger_task = asyncio.create_task(tick_logger.run())
    info(f"[TICKLOG] Logger started — uploads every {UPLOAD_INTERVAL_SECS//60}min to {SUPABASE_BUCKET}")

    trader_tasks  = [asyncio.create_task(t.run()) for t in traders]
    retrain_task  = asyncio.create_task(retrain_loop())
    shutdown_task = asyncio.create_task(_shutdown_event.wait())

    all_tasks = trader_tasks + [retrain_task, tick_logger_task, shutdown_task]

    done, _ = await asyncio.wait(
        all_tasks,
        return_when=asyncio.FIRST_COMPLETED,
    )
    if shutdown_task in done:
        info("Shutdown — stopping all tasks...")
        tick_logger.stop()
        for task in trader_tasks + [retrain_task, tick_logger_task]:
            task.cancel()
        await asyncio.gather(*trader_tasks, retrain_task,
                             tick_logger_task, return_exceptions=True)
        info("All stopped. Goodbye.")

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def _handle_signal(signum, frame):
    info(f"Signal {signum} — shutting down...")
    try:
        asyncio.get_event_loop().call_soon_threadsafe(_shutdown_event.set)
    except Exception:
        pass

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT,  _handle_signal)
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        info("Stopped by user.")
