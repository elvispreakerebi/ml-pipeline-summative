FROM python:3.13-slim

# Install system dependencies for librosa/soundfile
RUN apt-get update && \
    apt-get install -y --no-install-recommends libsndfile1 ffmpeg gcc g++ gfortran libopenblas-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir "setuptools<81" && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose API port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
