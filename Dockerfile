# Base image with TensorFlow + GPU support (use tensorflow/tensorflow:2.13.0 for CPU-only)
FROM tensorflow/tensorflow:2.13.0-gpu

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV + webcam access
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libv4l-dev \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY live_fruit_classifier.py .
COPY fruit_freshness_classifier.h5 .

# Copy overlay images if they exist (optional — COPY won't fail if missing)
COPY fresh.jpeg* rotten.jpeg* ./

# Default command
CMD ["python", "live_fruit_classifier.py"]
