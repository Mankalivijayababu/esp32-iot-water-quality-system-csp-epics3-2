# Use Python 3.10 slim
FROM python:3.10-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# ---------------------------
# Install system dependencies (NO atlas, NO lapack issues)
# ---------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev \
    libblas-dev \
    liblapack-dev \
    gfortran \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------
# Install TensorFlow 2.10.0 (CPU only)
# ---------------------------
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir tensorflow-cpu==2.10.0

# ---------------------------
# Install backend Python deps
# ---------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------
# Copy backend code + model files
# ---------------------------
COPY . /app
WORKDIR /app

# ---------------------------
# Expose port
# ---------------------------
EXPOSE 5000

# ---------------------------
# Run app.py
# ---------------------------
CMD ["python", "app.py"]
