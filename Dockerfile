# Dockerfile for NASA CMAPSS Predictive Maintenance Pipeline
# Traceability: Issue #1
# Reproducibility: Pinned Versions

# Use Python 3.12 as base image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirement file
COPY requirement.txt .

# Install dependencies
# Using --no-cache-dir to keep image size small
RUN pip install --no-cache-dir -r requirement.txt

# Copy the entire project
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Default command (run the first phase script)
CMD ["python", "pipeline/scripts/01_data_acquisition.py"]
