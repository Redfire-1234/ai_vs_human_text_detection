# Use official Python runtime as base image
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app.py .

# Copy the trained model directory
COPY bert-ai-human-model/ ./bert-ai-human-model/

# Expose port 5000 for Flask
EXPOSE 5000

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the application
CMD ["python", "app.py"]

