# ---------------------------
# 1. BASE IMAGE (Python 3.10 + minimal)
# ---------------------------
FROM python:3.10-slim

# ---------------------------
# 2. Install system dependencies
# ---------------------------
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    libatlas-base-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------
# 3. Set working directory
# ---------------------------
WORKDIR /app

# ---------------------------
# 4. Copy all backend files
# ---------------------------
COPY . .

# ---------------------------
# 5. Install Python dependencies
# ---------------------------
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------
# 6. Expose Render port
# ---------------------------
ENV PORT=10000
EXPOSE 10000

# ---------------------------
# 7. Start Gunicorn
# ---------------------------
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
