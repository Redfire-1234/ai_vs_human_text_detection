FROM python:3.10-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Download model from Hugging Face at build time
# This caches the model in the Docker image
RUN python -c "from transformers import AutoModelForSequenceClassification, AutoTokenizer; \
    print('Downloading model from Hugging Face...'); \
    model = AutoModelForSequenceClassification.from_pretrained('Redfire-1234/bert-ai-human-model'); \
    tokenizer = AutoTokenizer.from_pretrained('Redfire-1234/bert-ai-human-model'); \
    model.save_pretrained('./bert-ai-human-model'); \
    tokenizer.save_pretrained('./bert-ai-human-model'); \
    print('Model downloaded and saved successfully!')"

# Expose port 5000 for Flask
EXPOSE 5000

# Run the application with gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
