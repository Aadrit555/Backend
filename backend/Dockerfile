# Use a PyTorch base image with CUDA support since we need GPU for Unsloth/YOLO
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies that might be needed for ML libraries
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# (Optional) Install the heavy ML dependencies here if you want them baked in
# RUN pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# RUN pip install autogluon ultralytics

# Copy backend code
COPY backend ./backend

ENV PYTHONPATH=/app

EXPOSE 8000

# Run the FastAPI app
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]
