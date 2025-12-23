# BERT Sentiment Classifier 🎭

This project fine-tunes a **BERT** model to classify movie/product reviews as **positive** or **negative**, and exposes a simple **Streamlit** web app for interactive predictions. 

## Project Structure

- `bert_sentiment.ipynb` – Notebook for data loading, preprocessing, training, evaluation, and W&B logging.
- `app.py` – Streamlit app for interactive sentiment analysis.
- `bert_imdb_finetuned/` – Folder with fine-tuned BERT model and tokenizer weights.
- `requirements.txt` – Python dependencies needed to run the notebook and app.

## Dataset

- **IMDB Reviews** dataset (`datasets.load_dataset("imdb")`)
- 25,000 training reviews and 25,000 test reviews with binary sentiment labels (0 = negative, 1 = positive). 

## Model & Training

- Base model: `bert-base-uncased` (Hugging Face Transformers). 
- Max sequence length: 128 tokens.
- Batch size: 16 (train), 32 (test).
- Optimizer: AdamW, learning rate `2e-5`.
- Scheduler: Linear warmup + decay.
- Epochs: 2.
- Logged training and evaluation metrics with **Weights & Biases**.

### Results (on a 4k/1k subset of IMDB)

- Train accuracy: ~0.91
- Train F1 (weighted): ~0.91
- Test accuracy: ~0.86
- Test F1 (weighted): ~0.86
