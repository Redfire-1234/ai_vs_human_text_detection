import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification # load the autotokenizer and fine-tune model that we saved earlier
from lime.lime_text import LimeTextExplainer

st.title("AI vs Human Text Classifier with Explainability")

text = st.text_area("Enter text to classify")

if st.button("Predict"):
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-ai-human-model")
    model = AutoModelForSequenceClassification.from_pretrained("bert-ai-human-model")
    model.eval()

    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Prediction
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
        predicted_class = int(torch.argmax(outputs.logits, dim=1))

    label_map = {0: "Human", 1: "AI"}
    st.write(f"**Prediction:** {label_map[predicted_class]}")

    # Confidence bar
    st.progress(int(probs[predicted_class]*100))
    st.write(f"**Confidence:** {probs[predicted_class]:.3f}")

    # LIME explanation
    class_names = ["Human", "AI"]
    explainer = LimeTextExplainer(class_names=class_names)

    def predict_fn(texts):
        inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=256)
        with torch.no_grad():
            logits = model(**inputs).logits
        return torch.softmax(logits, dim=1).numpy()

    st.write("**Words influencing the prediction:**")
    exp = explainer.explain_instance(text, predict_fn, num_features=10)
    st.write(exp.as_list())
