# Use Python 3.10 base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install all lightweight dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install TensorFlow from a FAST mirror to avoid Render timeout
RUN pip install --no-cache-dir tensorflow-cpu==2.10.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# Copy the rest of the project files
COPY . .

# Expose port 10000 (Render default for web services)
EXPOSE 10000

# Start the server using gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app"]
