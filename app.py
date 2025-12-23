import torch
import numpy as np
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------- CONFIG ----------------
# Use a public, pre-trained sentiment model from Hugging Face for the demo.
# This model outputs 5 classes: 1–5 stars (very negative to very positive).
MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"  # HF sentiment model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model_and_tokenizer():
    """Load pre-trained sentiment model and tokenizer from Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return tokenizer, model


tokenizer, model = load_model_and_tokenizer()


def predict_sentiment(text: str):
    """Run the model on text and return star rating (1–5) + probabilities."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]  # shape (5,)
        pred_id = int(np.argmax(probs))  # 0..4
        stars = pred_id + 1  # 1..5
        return stars, probs


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="BERT Sentiment Classifier", page_icon="🎭")

st.title("BERT Sentiment Classifier 🎭")
st.write(
    "This demo uses a pre-trained **BERT sentiment model** from Hugging Face "
    "to rate reviews from **1 to 5 stars** (very negative → very positive)."
)

user_text = st.text_area(
    "Enter your review text:",
    height=180,
    placeholder="Type a movie or product review here...",
)

if st.button("Analyze sentiment"):
    if not user_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        stars, probs = predict_sentiment(user_text)

        st.subheader(f"Predicted sentiment: **{stars} ⭐**")

        st.write("Class probabilities:")
        for i, p in enumerate(probs, start=1):
            st.write(f"{i} ⭐: {p:.3f}")

        # Visualize positivity roughly as (stars / 5)
        st.progress(float(stars) / 5.0)
        st.caption("Progress bar approximates how positive the sentiment is.")

st.caption(
    "Demo model: `nlptown/bert-base-multilingual-uncased-sentiment` from Hugging Face. "
    "Training experiments (fine-tuning `bert-base-uncased` on IMDB) are in the notebook."
)
