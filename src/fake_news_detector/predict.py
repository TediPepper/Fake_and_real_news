import argparse
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify text with a fine-tuned fake/real news model."
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models/fake-news-bert"))
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Article text to classify")
    source.add_argument("--file", type=Path, help="Path to a UTF-8 text file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.text if args.text is not None else args.file.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError("Input text must not be empty")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to(device)
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(**inputs).logits[0], dim=-1).cpu()

    label = model.config.id2label[int(torch.argmax(probabilities))]
    scores = {
        model.config.id2label[index]: round(float(score), 4)
        for index, score in enumerate(probabilities)
    }
    print(f"Dataset label prediction: {label}")
    print(f"Model scores: {scores}")
    print("This is a dataset-based prediction, not independent fact verification.")


if __name__ == "__main__":
    main()
