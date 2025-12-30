import streamlit as st
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "best_bert_sentiment"

@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model_and_tokenizer()
label_names = ["negative", "neutral", "positive"]

def predict_sentiment(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    pred_id = int(np.argmax(probs))
    return label_names[pred_id], probs

# ----------------- Streamlit UI -----------------

st.title("BERT Sentiment Classifier (3 Classes)")
st.write("Negative / Neutral / Positive")

user_text = st.text_area(
    "Enter a review or sentence:",
    value="This movie was absolutely amazing!",
    height=120,
)

if st.button("Analyze sentiment"):
    if user_text.strip():
        label, probs = predict_sentiment(user_text)
        st.subheader(f"Predicted sentiment: **{label}**")

        st.write("Class probabilities:")
        for name, p in zip(label_names, probs):
            st.write(f"- {name}: {p*100:.2f}%")
    else:
        st.warning("Please enter some text.")
