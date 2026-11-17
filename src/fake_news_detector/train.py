import argparse
import json
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

from fake_news_detector.data import load_news_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT on the Real and Fake News CSV dataset."
    )
    parser.add_argument("--fake-csv", type=Path, default=Path("data/raw/Fake.csv"))
    parser.add_argument("--real-csv", type=Path, default=Path("data/raw/True.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/fake-news-bert"))
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_dataset(frame, tokenizer, max_length: int) -> Dataset:
    dataset = Dataset.from_pandas(frame[["text", "label"]], preserve_index=False)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length)

    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def compute_metrics(prediction) -> dict[str, float]:
    predicted = np.argmax(prediction.predictions, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        prediction.label_ids,
        predicted,
        average="binary",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(prediction.label_ids, predicted)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.max_length < 1 or args.epochs <= 0:
        raise ValueError("batch size, max length, and epochs must be positive")
    set_seed(args.seed)

    data = load_news_dataset(args.fake_csv, args.real_csv)
    train_frame, test_frame = train_test_split(
        data, test_size=0.2, random_state=args.seed, stratify=data["label"]
    )
    train_frame, validation_frame = train_test_split(
        train_frame,
        test_size=0.1,
        random_state=args.seed,
        stratify=train_frame["label"],
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = make_dataset(train_frame, tokenizer, args.max_length)
    validation_dataset = make_dataset(validation_frame, tokenizer, args.max_length)
    test_dataset = make_dataset(test_frame, tokenizer, args.max_length)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        id2label={0: "FAKE", 1: "REAL"},
        label2id={"FAKE": 0, "REAL": 1},
    )

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        dataloader_pin_memory=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
    )
    trainer.train()
    test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
