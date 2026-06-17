# GameSense backend — FastAPI + OpenCV/MediaPipe CV stack.
# Python pinned to 3.11 (inference-sdk / supervision don't support 3.13+).
FROM python:3.11-slim

# System libraries the CV stack needs at runtime:
#   libgl1, libglib2.0-0  -> OpenCV (cv2)
#   ffmpeg                -> imageio / video decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so Docker can cache this layer across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# App source.
COPY . .

# Uploaded videos + the SQLite database live here. Mount a Railway Volume at
# this path so accounts, scores, and clips survive restarts/redeploys.
ENV GAMESENSE_MEDIA_ROOT=/data
RUN mkdir -p /data

# Railway provides $PORT at runtime; default to 8000 locally.
ENV PORT=8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
