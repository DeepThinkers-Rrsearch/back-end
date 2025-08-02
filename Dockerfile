# Use the Python 3.11 slim official image for better compatibility with ML libraries
# https://hub.docker.com/_/python
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y \
    graphviz \
    libgraphviz-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and change to the app directory.
WORKDIR /app

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy local code to the container image.
COPY . .

# Run the web service on container startup.
CMD ["hypercorn", "main:app", "--bind", "0.0.0.0:8000"]