# GameSense — Deployment Guide

GameSense has two parts that deploy separately:

- **Frontend** — Next.js. Cheap/free and easy to host.
- **Backend** — FastAPI + a heavy CV/ML stack (OpenCV, MediaPipe, RF-DETR). Needs real RAM, CPU, and **persistent storage** for the SQLite database and uploaded videos.

## TL;DR recommended setup (cheapest that actually works)

| Part | Host | Cost |
| --- | --- | --- |
| Frontend | **Vercel** | Free |
| Backend | **Small VPS** (Hetzner CX22 / DigitalOcean basic), Caddy for HTTPS | ~$5–6/mo |

The backend is **not** a good fit for free tiers: Render's free 512 MB RAM will crash on the ML imports, and most free tiers wipe disk on restart (you'd lose accounts + videos). A tiny VPS with ~4 GB RAM is the realistic minimum.

> Free-but-caveated option: a Hugging Face **CPU Space** (free, ~16 GB RAM) can run the backend, but its disk is **ephemeral** — the database and videos reset on restart unless you add external storage. Only use it for a throwaway demo.

---

## 1. Frontend on Vercel (free)

1. Push the repo to GitHub.
2. On vercel.com → **New Project** → import the repo → set **Root Directory** to `frontend`.
3. Add an environment variable:
   - `NEXT_PUBLIC_API_URL = https://api.yourdomain.com` (your backend's public URL)
4. Deploy. Vercel gives you HTTPS automatically.

## 2. Backend on a VPS

On a fresh Ubuntu VPS:

```bash
# system deps for OpenCV
sudo apt update && sudo apt install -y python3.12 python3.12-venv git ffmpeg libgl1
git clone <your-repo> gamesense && cd gamesense/backend
python3.12 -m venv venv && . venv/bin/activate
pip install -r requirements.txt
```

Create `backend/.env` on the server (do **not** commit it):

```
ROBOFLOW_API_KEY=...
GAMESENSE_SECRET_KEY=<a long random value>
GAMESENSE_ALLOWED_ORIGINS=https://yourdomain.com
GAMESENSE_MEDIA_ROOT=/var/gamesense-storage
```

Run it (bind to localhost; Caddy terminates HTTPS in front):

```bash
. venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

For production, run it under a process manager so it restarts on crash/reboot — e.g. a systemd service or `pm2 start "uvicorn app.main:app --host 127.0.0.1 --port 8000" --name gamesense-api`.

## 3. HTTPS with Caddy (automatic certificates)

Install Caddy, then create `/etc/caddy/Caddyfile`:

```
api.yourdomain.com {
    reverse_proxy 127.0.0.1:8000
    request_body {
        max_size 250MB
    }
}
```

`sudo systemctl reload caddy` — Caddy fetches and renews a free Let's Encrypt certificate automatically. Point an `api` DNS A-record at the VPS IP first.

## 4. Required environment variables

| Variable | Where | Purpose |
| --- | --- | --- |
| `NEXT_PUBLIC_API_URL` | Frontend (Vercel) | Backend's public URL |
| `GAMESENSE_ALLOWED_ORIGINS` | Backend | CORS — your frontend origin(s), comma-separated |
| `GAMESENSE_SECRET_KEY` | Backend | Login-token signing key (keep stable & secret) |
| `ROBOFLOW_API_KEY` | Backend | Detection model access |
| `GAMESENSE_MEDIA_ROOT` | Backend | Where videos + the database live (use a real disk path) |

The CPU performance knobs (`GAMESENSE_MAX_PROCESSING_HEIGHT`, etc.) are optional and carry over.

## 5. Backups (don't skip this)

Schedule the backup script on the server (cron) — the database holds every account, league, and score:

```bash
# daily 3am: snapshot DB + mirror videos
0 3 * * *  cd /path/to/gamesense/backend && venv/bin/python scripts/backup_data.py --media --dest /var/gamesense-backups
```

Even better: copy `/var/gamesense-backups` off-box (e.g. to S3/Backblaze) so a server loss doesn't lose the backups too.

## 5b. Stripe billing (memberships)

GameSense has three tiers — **Free**, **Pro**, **Elite** — enforced server-side.
Billing runs on Stripe. Without Stripe configured the app still works: everyone
is Free, and the pricing page shows a "checkout isn't live" notice.

**One-time Stripe setup:**
1. In the Stripe Dashboard, create two **recurring (monthly)** Products/Prices:
   "GameSense Pro" and "GameSense Elite". Copy each **price id** (`price_...`).
2. Add a **webhook endpoint** pointing at `https://api.yourdomain.com/billing/webhook`
   and subscribe to: `checkout.session.completed`,
   `customer.subscription.created/updated/deleted`,
   `invoice.payment_succeeded`, `invoice.payment_failed`. Copy the **signing
   secret** (`whsec_...`).

**Backend env vars (add to `backend/.env`):**

| Variable | Purpose |
| --- | --- |
| `STRIPE_SECRET_KEY` | `sk_live_...` / `sk_test_...` |
| `STRIPE_WEBHOOK_SECRET` | `whsec_...` from the webhook endpoint |
| `STRIPE_PRICE_PRO` | `price_...` for the Pro plan |
| `STRIPE_PRICE_ELITE` | `price_...` for the Elite plan |
| `FRONTEND_URL` | e.g. `https://yourdomain.com` — used for checkout redirect URLs |

`pip install -r requirements.txt` now also pulls in `stripe`.

Limits/feature gating live in `backend/app/services/subscriptions.py` — the single
source of truth. Usage resets are automatic (rolling weekly/monthly windows), so
there is no cron job to run.

## 6. Scaling note

Each sprint analysis is CPU-heavy and the app processes ~2 at a time (`_cv_executor`). A few simultaneous users will queue. If usage grows, move to a bigger CPU box (or a GPU instance) and consider a proper job queue. Fine for launch; revisit when you have real traffic.
