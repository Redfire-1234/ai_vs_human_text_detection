FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Download and cache the model from Hugging Face during build
RUN python3 << EOF
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os

print("=" * 50)
print("Downloading model from Hugging Face...")
print("=" * 50)

model_name = "Redfire-1234/bert-ai-human-model"
save_path = "./bert-ai-human-model"

# Download model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Save to local directory
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print("=" * 50)
print("Model downloaded and saved successfully!")
print(f"Model saved to: {save_path}")
print("=" * 50)
EOF

# Expose the port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:5000/health')" || exit 1

# Run the application
CMD ["python3", "-u", "app.py"]
