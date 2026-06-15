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

## 6. Scaling note

Each sprint analysis is CPU-heavy and the app processes ~2 at a time (`_cv_executor`). A few simultaneous users will queue. If usage grows, move to a bigger CPU box (or a GPU instance) and consider a proper job queue. Fine for launch; revisit when you have real traffic.
