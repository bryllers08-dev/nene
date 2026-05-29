# EXPIRYRANGE LSTM Bot — Railway Deployment

## Files in this package
```
main_lstm_bot.py    — the bot
Dockerfile          — Railway build instructions
railway.toml        — Railway project config
requirements.txt    — Python dependencies
.dockerignore       — keeps image clean
.gitignore          — keeps repo clean
README.md           — this file
```

---

## Pre-deployment checklist

### 1. Upload your CSV to Supabase
Before deploying, make sure `1HZ10V.csv` is in your `Supabot` bucket.
The bot downloads it at startup — without it the bot cannot train.

- Go to: https://supabase.com → your project → Storage → Supabot
- Upload: `1HZ10V.csv` from your PC
- Confirm it shows as ~28MB in the bucket

### 2. Push to GitHub
```bash
git init
git add .
git commit -m "LSTM bot initial deploy"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

---

## Railway deployment steps

### Step 1 — Create new project
- Go to https://railway.app
- New Project → Deploy from GitHub repo
- Select your repo

### Step 2 — Add Volume (CRITICAL)
Without a volume the model retrains from scratch on every restart.

- Railway dashboard → your service → Volumes
- Add Volume → Mount path: `/app/data`
- This persists: lstm_model.keras, lstm_scaler.pkl, calibration.json, symbol_data/1HZ10V.csv

### Step 3 — Set environment variables
Railway dashboard → your service → Variables → Add all of these:

```
DERIV_API_TOKEN       your_deriv_token_here
SUPABASE_KEY          your_service_role_key_here
```

These are already set as defaults in the Dockerfile (no need to add unless overriding):
```
SUPABASE_URL          https://qtkjixwiahghisqqxrcz.supabase.co
SUPABASE_BUCKET       Supabot
PERSIST_DIR           /app/data
DATA_DIR              /app/data/symbol_data
PORT                  8080
BASE_STAKE            1.0
TARGET_PROFIT         45.0
STOP_LOSS             10.0
LSTM_THRESHOLD        0.72
LSTM_THRESHOLD_FLOOR  0.60
RETRAIN_HOURS         4
UPLOAD_INTERVAL_SECS  300
```

### Step 4 — Deploy
- Railway will build the Docker image (~5-10 min first time)
- Bot starts, downloads CSV from Supabase, trains LSTM (~25-30 min)
- Health server goes live at your Railway URL

### Step 5 — Check health
Visit your Railway public URL:
- `/`        → HTML dashboard (trades, win rate, P&L)
- `/status`  → JSON status

---

## What happens on startup (Railway)

```
1. Docker container starts
2. Bot downloads 1HZ10V.csv from Supabase → /app/data/symbol_data/
3. Computes calibration → /app/data/calibration.json
4. Trains LSTM (~25-30 min) → /app/data/lstm_model.keras
5. Starts trading on 1HZ10V
6. TickLogger runs in background → uploads CSV to Supabase every 5min
7. Every 4hr → re-downloads CSV, retrains LSTM, advisor adjusts params
```

## What happens on restart (Railway)

If the Volume is mounted:
```
1. Bot loads lstm_model.keras from Volume (no retraining needed)
2. Downloads fresh CSV from Supabase
3. Starts trading immediately
```

If no Volume (or first deploy):
```
1. Downloads CSV from Supabase
2. Full retrain (~25-30 min)
3. Then trades
```

---

## Railway plan recommendation
- **Hobby plan ($5/month)** — sufficient for this bot
- **512MB RAM minimum** — TensorFlow needs it for training
- **1GB RAM recommended** — comfortable headroom

---

## Troubleshooting

**Bot crashes on startup with memory error**
- Your Railway plan may have insufficient RAM
- Upgrade to a plan with at least 1GB RAM

**LSTM gate not activating**
- Check logs for training errors
- Verify 1HZ10V.csv is in Supabase bucket and has 10,000+ rows

**No trades firing**
- Normal for first 100 ticks (warmup period)
- Check /status JSON for lstm_active: true
- Check threshold vs lstm_prob in logs

**Upload failing**
- Check SUPABASE_KEY is set correctly in Railway Variables
- Verify Supabot bucket exists and is public or accessible with service_role key
