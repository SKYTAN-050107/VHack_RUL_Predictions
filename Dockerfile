# Dockerfile for NASA CMAPSS RUL Notebooks
# Reproducibility: pinned dependencies from requirements.txt

# Use Python 3.11 for broad scientific stack compatibility
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# System deps (LightGBM/OpenMP)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirement files
COPY requirements.txt .

# Install dependencies
# Using --no-cache-dir to keep image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Jupyter for notebooks
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
