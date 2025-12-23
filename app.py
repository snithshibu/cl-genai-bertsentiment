import torch
import numpy as np
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------- CONFIG ----------------
MODEL_NAME = "bert-base-uncased"  # load from Hugging Face hub
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Explicit label mapping: 0 = negative, 1 = positive
ID2LABEL = {0: "negative", 1: "positive"}
LABEL2ID = {"negative": 0, "positive": 1}


@st.cache_resource
def load_model_and_tokenizer():
    """Load BERT model and tokenizer from Hugging Face."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


tokenizer, model = load_model_and_tokenizer()


def predict_sentiment(text: str):
    """Run the model on a single text and return label + probabilities."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        pred_id = int(np.argmax(probs))
        return ID2LABEL[pred_id], probs


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="BERT Sentiment Classifier", page_icon="🎭")

st.title("BERT Sentiment Classifier 🎭")
st.write(
    "This app uses a **BERT** model to classify movie/product reviews "
    "as positive or negative."
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
        label, probs = predict_sentiment(user_text)
        if len(probs) == 2:
            neg_prob, pos_prob = probs
        else:
            neg_prob, pos_prob = None, None

        st.subheader(f"Predicted sentiment: **{label}**")

        if neg_prob is not None:
            st.write(f"Negative probability: {neg_prob:.3f}")
            st.write(f"Positive probability: {pos_prob:.3f}")
            st.progress(float(pos_prob))
            st.caption("Progress bar shows how confident the model is about 'positive'.")

st.caption("Model: `bert-base-uncased` (base model on Streamlit Cloud).")
