# Dockerfile

# 1. Base image
FROM python:3.10-slim

# 2. Don’t write .pyc files, and unbuffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Working directory
WORKDIR /app

# 4. Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# 5. Copy source code
COPY . .

# 6. Expose the port (override via build or runtime ARG/ENV)
ARG PORT=8000
EXPOSE ${PORT}

# 7. Launch with Gunicorn + Uvicorn worker
CMD ["gunicorn","-k","uvicorn.workers.UvicornWorker","main:app","--bind","0.0.0.0:$PORT","--workers","1"]
