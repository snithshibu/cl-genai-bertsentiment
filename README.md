# BERT Sentiment Classifier (3 Classes)

Fine-tuned **BERT** model that classifies text reviews into **Negative**, **Neutral**, or **Positive** sentiment. The project includes a training notebook (Colab), experiment tracking with **Weights & Biases**, a **Streamlit app** for interactive inference, and a saved fine‑tuned model.

***

## 1. Project Overview

This project builds and fine-tunes a **BERT-based sentiment classifier** on a 3-class sentiment dataset, then evaluates it using **accuracy**, **macro F1-score**, and a **confusion matrix**. A minimal **Streamlit web app** is provided to run the model on custom text.

**Main components:**

- Fine-tuning `bert-base-uncased` for 3-class sentiment classification.
- Dataset: `syedkhalid076/Sentiment-Analysis` (Negative / Neutral / Positive).
- Training and evaluation in a Jupyter/Colab notebook.  
- Experiment tracking with **Weights & Biases (W&B)**.
- Streamlit app (`app.py`) for interactive predictions using the fine-tuned model.

---

## 2. Dataset

**Name:** [`syedkhalid076/Sentiment-Analysis`](https://huggingface.co/datasets/syedkhalid076/Sentiment-Analysis)

**Description:**

- Source: Hugging Face Datasets Hub.  
- Language: English short texts / social-style sentences.  
- Sentiment labels:
  - `0` → Negative  
  - `1` → Neutral  
  - `2` → Positive

**Splits:**

- Train: 83,989 examples  
- Validation: 10,499 examples  
- Test: 10,499 examples

Only a subset of the training and validation splits is used for faster training (see below), but evaluation is done on the **full validation and test** sets.

***

## 3. Model and Training Setup

**Base model:** [`bert-base-uncased`](https://huggingface.co/bert-base-uncased) with a 3-class classification head.

**Architecture & labels:**

- Encoder: BERT (uncased, 12-layer).  
- Classification head: linear layer with 3 outputs.  
- Label mapping:
  - `0 → negative`  
  - `1 → neutral`  
  - `2 → positive`  

**Environment:**

- Training: Google Colab with GPU enabled.[5]
- Libraries: `torch`, `transformers`, `datasets`, `accelerate`, `scikit-learn`, `wandb`, `streamlit`.

**Training configuration (subset):**

- Train subset: **40,000** examples from the training split.  
- Validation subset: **8,000** examples from the validation split.  
- Optimizer & scheduler: defaults via `TrainingArguments`.
- Hyperparameters:
  - Learning rate: `2e-5`  
  - Batch size: `16` (per device)  
  - Epochs: `2`  
  - Weight decay: `0.01`  
  - Max sequence length: `128` tokens  
- Logging & checkpoints:
  - Evaluation every epoch (`eval_strategy="epoch"`).  
  - Best model loaded at end (`load_best_model_at_end=True`).  
  - Metrics and losses logged to **Weights & Biases** (`report_to=["wandb"]`).

**Files:**

- `notebooks/bert_sentiment_training.ipynb`: full training & evaluation notebook.  
- `best_bert_sentiment/`: directory containing the fine-tuned model and tokenizer (`config.json`, `pytorch_model.bin`, tokenizer files).  

***

## 4. Results

All metrics below are computed on the **full** validation and test splits, using the best model checkpoint. 

### 4.1 Validation and Test Metrics

- **Validation**:
  - Accuracy: ≈ **0.879**  
  - Macro F1: ≈ **0.842**

- **Test**:
  - Accuracy: ≈ **0.883**  
  - Macro F1: ≈ **0.847**

These scores indicate strong performance across all three sentiment classes, not just the majority classes.

### 4.2 Test Confusion Matrix

Confusion matrix on the test set (rows = true labels, columns = predicted labels):

| true \ pred | negative | neutral | positive |
|------------|----------|---------|----------|
| negative   | 3722     | 202     | 263      |
| neutral    | 163      | 970     | 122      |
| positive   | 265      | 218     | 4574     |

- Negative and positive are predicted very accurately.  
- Neutral is more challenging but still obtains strong performance (970 correctly classified vs 163+122 confused). 

### 4.3 Qualitative Predictions

Example predictions from the model:

- “This movie was absolutely amazing!” → **Positive** (prob ≈ 0.996).  
- “It was okay, nothing special.” → **Neutral** (prob ≈ 0.56).  
- “Terrible experience, I hated it.” → **Negative** (prob ≈ 0.99).

These examples show the model can differentiate strong positive/negative sentiment and more mild, neutral text.

***
