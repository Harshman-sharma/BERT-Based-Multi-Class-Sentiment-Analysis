# BERT-Based Multi-Class Sentiment Analysis

A transformer-based sentiment classification project using **BERT (Bidirectional Encoder Representations from Transformers)** to classify textual data into three sentiment categories: Positive, Negative, and Neutral.

The project fine-tunes a pre-trained BERT model using PyTorch and Hugging Face Transformers, evaluates its performance on a held-out test dataset, and implements model saving, reloading, and inference on new text inputs.

## Key Features

- **Text Preprocessing:** Tokenization and sequence preparation using the BERT tokenizer.
- **Transfer Learning:** Fine-tuning the pre-trained `bert-base-uncased` model for multi-class sentiment classification.
- **Multi-Class Classification:** Predicts Positive, Negative, and Neutral sentiments.
- **Model Evaluation:** Evaluates performance using Accuracy, Precision, Recall, Macro F1-score, and Weighted F1-score.
- **Baseline Comparison:** Compares BERT performance against a majority-class classification baseline.
- **Confusion Matrix:** Visualizes classification performance across sentiment categories.
- **Model Persistence:** Saves and reloads the fine-tuned model and tokenizer.
- **Inference Pipeline:** Performs sentiment predictions on previously unseen text inputs.

## Dataset

- **Dataset:** TweetEval – Sentiment
- **Source:** Hugging Face Datasets
- **Dataset Link:** [https://huggingface.co/datasets/cardiffnlp/tweet_eval](https://huggingface.co/datasets/cardiffnlp/tweet_eval)
- **Task:** Three-class sentiment classification.
- **Classes:** Negative, Neutral, Positive.
- **Splits:** Training, Validation, and Testing.

The dataset is loaded directly using the Hugging Face Datasets library.

## Model Architecture

- **Base Model:** BERT Base Uncased
- **Model:** `bert-base-uncased`
- **Framework:** PyTorch
- **Pre-trained Model Library:** Hugging Face Transformers
- **Task:** Sequence Classification
- **Output:** Three sentiment classes.

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1-score
- Weighted Precision
- Weighted Recall
- Weighted F1-score

A majority-class baseline is also implemented to establish a reference for evaluating model performance.

Actual evaluation results are generated after model training and saved in `outputs/metrics.json`.

## Technologies Used

Python, PyTorch, Hugging Face Transformers, Hugging Face Datasets, Scikit-learn, NumPy, Matplotlib, Seaborn.


## Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
cd BERT-Sentiment-Analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

Run the training script:

```bash
python src/train.py
```

The script automatically:

1. Loads the TweetEval sentiment dataset.
2. Tokenizes text using the BERT tokenizer.
3. Fine-tunes the pre-trained BERT model.
4. Evaluates the model on validation and test sets.
5. Generates evaluation metrics and a confusion matrix.
6. Saves the fine-tuned model and tokenizer.

## Inference

After training, run:

```bash
python src/predict.py
```

Enter text to obtain its predicted sentiment and confidence score.

## Project Objective

To explore transformer-based natural language processing and apply transfer learning for multi-class sentiment analysis, while developing a complete pipeline covering model fine-tuning, evaluation, persistence, and inference.
