# Dockerfile

# 1. Use a slim Python base image
FROM python:3.10-slim

# 2. Set a working directory
WORKDIR /app

# 3. Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the application code
COPY . .

# 5. Expose the port (Railway injects $PORT at runtime)
ARG PORT=8000
EXPOSE ${PORT}

# 6. Run with Gunicorn + Uvicorn worker
CMD ["gunicorn", 
     "-k", "uvicorn.workers.UvicornWorker", 
     "main:app", 
     "--bind", "0.0.0.0:${PORT}", 
     "--workers", "1"]
