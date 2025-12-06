from flask import Flask, request, jsonify, render_template_string
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

app = Flask(__name__)

# Global variables to store model and tokenizer
model = None
tokenizer = None

def load_model():
    """Load the model and tokenizer when the app starts"""
    global model, tokenizer
    
    try:
        # Try to load from local directory first (for Docker builds)
        model_path = "./bert-ai-human-model"
        
        # If local model doesn't exist, load from Hugging Face
        if not os.path.exists(model_path):
            print("Local model not found, loading from Hugging Face...")
            model_path = "Redfire-1234/bert-ai-human-model"
        else:
            print("Loading model from local directory...")
        
        print(f"Loading model from: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()  # Set to evaluation mode
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

def predict_text(text):
    """Predict whether text is AI or Human generated"""
    if model is None or tokenizer is None:
        raise Exception("Model not loaded")
    
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).numpy()[0]
        predicted_class = int(torch.argmax(logits, dim=1))
    
    label_map = {0: "Human", 1: "AI"}
    
    return {
        "label": label_map[predicted_class],
        "confidence": float(probs[predicted_class]),
        "probabilities": {
            "human": float(probs[0]),
            "ai": float(probs[1])
        }
    }

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI vs Human Text Classifier</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 50px auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
        }
        textarea {
            width: 100%;
            height: 150px;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
            margin-bottom: 20px;
            box-sizing: border-box;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 12px 30px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
        }
        button:hover {
            background-color: #45a049;
        }
        button:disabled {
            background-color: #cccccc;
            cursor: not-allowed;
        }
        .result {
            margin-top: 20px;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 5px;
            display: none;
        }
        .result.show {
            display: block;
        }
        .prediction {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .human { color: #2196F3; }
        .ai { color: #FF5722; }
        .confidence-bar {
            width: 100%;
            height: 30px;
            background-color: #e0e0e0;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }
        .confidence-fill {
            height: 100%;
            background-color: #4CAF50;
            transition: width 0.3s ease;
        }
        .loading {
            text-align: center;
            color: #666;
            margin-top: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AI vs Human Text Classifier</h1>
        <p style="text-align: center; color: #666;">Enter text below to check if it was written by a human or AI</p>
        
        <textarea id="textInput" placeholder="Enter your text here..."></textarea>
        <button id="classifyBtn" onclick="classifyText()">Classify Text</button>
        <div id="loading" class="loading" style="display: none;">Analyzing...</div>
        
        <div id="result" class="result">
            <div class="prediction" id="prediction"></div>
            <p><strong>Confidence:</strong> <span id="confidence"></span></p>
            <div class="confidence-bar">
                <div class="confidence-fill" id="confidenceBar"></div>
            </div>
            <p><strong>Probabilities:</strong></p>
            <p>Human: <span id="humanProb"></span></p>
            <p>AI: <span id="aiProb"></span></p>
        </div>
    </div>

    <script>
        async function classifyText() {
            const text = document.getElementById('textInput').value;
            const btn = document.getElementById('classifyBtn');
            const loading = document.getElementById('loading');
            const resultDiv = document.getElementById('result');
            
            if (!text.trim()) {
                alert('Please enter some text!');
                return;
            }
            
            // Show loading state
            btn.disabled = true;
            loading.style.display = 'block';
            resultDiv.classList.remove('show');
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    alert('Error: ' + data.error);
                    return;
                }
                
                // Display results
                const predictionDiv = document.getElementById('prediction');
                
                predictionDiv.textContent = 'Prediction: ' + data.label;
                predictionDiv.className = 'prediction ' + data.label.toLowerCase();
                
                document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(2) + '%';
                document.getElementById('confidenceBar').style.width = (data.confidence * 100) + '%';
                document.getElementById('humanProb').textContent = (data.probabilities.human * 100).toFixed(2) + '%';
                document.getElementById('aiProb').textContent = (data.probabilities.ai * 100).toFixed(2) + '%';
                
                resultDiv.classList.add('show');
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                // Hide loading state
                btn.disabled = false;
                loading.style.display = 'none';
            }
        }
        
        // Allow Enter key in textarea (Shift+Enter for new line)
        document.getElementById('textInput').addEventListener('keydown', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                classifyText();
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Serve the web interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        
        if not text.strip():
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        result = predict_text(text)
        return jsonify(result)
    
    except Exception as e:
        print(f"Error in predict endpoint: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy', 
        'model_loaded': model is not None,
        'tokenizer_loaded': tokenizer is not None
    })

# Load model when the module is imported
print("Starting application...")
load_model()
print("Application ready!")

if __name__ == '__main__':
    # Get port from environment variable or default to 5000
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
