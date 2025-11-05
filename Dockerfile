# Use a slim Python base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .


# Expose port 8080 for Cloud Run
EXPOSE 8080

# Set environment variables
ENV PORT=8080
ENV PYTHONPATH=/app

# Start FastAPI app using Gunicorn with Uvicorn worker
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "-k", "uvicorn.workers.UvicornWorker", "--timeout", "120", "--workers", "1", "main:app"]