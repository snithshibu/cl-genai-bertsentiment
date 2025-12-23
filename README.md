# BERT Sentiment Classifier 🎭

This project fine-tunes a **BERT** model on the IMDB movie reviews dataset to classify reviews as positive or negative, and also provides a **Streamlit** web app for interactive sentiment analysis. 

## Project Structure

- `bert_sentiment.ipynb` – Jupyter notebook for:
  - Loading the IMDB dataset with `datasets.load_dataset("imdb")`
  - Preprocessing and tokenization with Hugging Face `AutoTokenizer`
  - Fine-tuning `bert-base-uncased` for binary sentiment classification
  - Evaluating accuracy, F1-score, and confusion matrix
  - Logging experiments with **Weights & Biases (W&B)** 
- `app.py` – Streamlit app used for deployment (cloud demo).
- `requirements.txt` – Main Python dependencies.

## Dataset

- **IMDB Reviews** dataset (`datasets.load_dataset("imdb")`)
- 25,000 training reviews and 25,000 test reviews.
- Binary labels: 0 = negative, 1 = positive. 

## Model & Training (Notebook)

- Base model: `bert-base-uncased` from Hugging Face Transformers. 
- Tokenization: max sequence length 128.
- Training:
  - Batch size: 16 (train), 32 (test)
  - Optimizer: AdamW, learning rate `2e-5`
  - Scheduler: linear warmup + decay
  - Epochs: 2
- Metrics on a 4k/1k IMDB subset:
  - Train accuracy: ~0.91
  - Train F1 (weighted): ~0.91
  - Test accuracy: ~0.86
  - Test F1 (weighted): ~0.86

### Weights & Biases

- Training and evaluation metrics are logged to **Weights & Biases**:
  - Per-epoch loss, accuracy, and F1.
  - Final test accuracy, F1, and confusion matrix. 

## Streamlit App (Demo)

The public Streamlit app is designed as a **demo UI** for sentiment analysis, and uses a robust pre-trained model from Hugging Face, **separate from the IMDB fine-tuned model in the notebook**.

- Deployed model: `nlptown/bert-base-multilingual-uncased-sentiment`
- Output: 1–5 star rating (very negative → very positive). 
- The app:
  - Lets you input a review.
  - Displays the predicted star rating (1–5).
  - Shows class probabilities for each star level.
  - Visualizes positivity with a progress bar.

> Note: The **notebook** shows results for the model fine-tuned on IMDB (`bert-base-uncased`), while the **Streamlit cloud demo** uses a pre-trained Hugging Face sentiment model for stable online inference.
