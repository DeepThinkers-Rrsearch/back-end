# Use an official, minimal Python runtime as a parent image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Install build dependencies (if you need to compile native extensions)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Allow the port to be configured at build or runtime
ARG PORT=8000
EXPOSE ${PORT}

# Start the FastAPI application using Gunicorn with a Uvicorn worker
# Shell form is used so $PORT is picked up from the environment at runtime
CMD gunicorn -k uvicorn.workers.UvicornWorker main:app --bind 0.0.0.0:$PORT --workers 1
