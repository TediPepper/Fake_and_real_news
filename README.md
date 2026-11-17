# Fake and Real News Classifier

Fine-tune `bert-base-uncased` to classify article text using the Real and Fake
News dataset. This is a learning project: its output reflects patterns in the
training data and is **not independent verification that a claim is true**.

## Setup

Use Python 3.10 or later. Activate your virtual environment, then install
dependencies:

```bash
pip install -r requirements.txt
```

Place the dataset files here (they are intentionally not included in Git):

```text
data/raw/Fake.csv
data/raw/True.csv
```

The CSV files should contain a `text` and/or `title` column. The common Kaggle
Real and Fake News dataset uses these filenames and columns.

## Train and evaluate

Run from the repository root:

```bash
PYTHONPATH=src python -m fake_news_detector.train
```

The script uses 80% of the data for training (with a validation split from that
portion) and holds out 20% for testing. It reports accuracy, precision, recall,
and F1, and saves the model and test metrics under `models/fake-news-bert/`.

To adjust training, use options such as:

```bash
PYTHONPATH=src python -m fake_news_detector.train --epochs 2 --batch-size 2 --max-length 128
```

The defaults use a small batch and sequence length to reduce memory use,
especially on CPU-only machines. Increase them only if your system has enough
memory.

## Predict

After training, classify a short sample:

```bash
PYTHONPATH=src python -m fake_news_detector.predict --text "Article text goes here"
```

Or pass a UTF-8 text file:

```bash
PYTHONPATH=src python -m fake_news_detector.predict --file article.txt
```

The output labels (`FAKE` and `REAL`) mean the model's predicted dataset class,
not a trustworthy verdict about the article. Dataset artifacts, duplicate
stories, source bias, and changes over time can affect the results. Review
evidence from reliable sources before drawing conclusions.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s tests
```
