from pathlib import Path

import pandas as pd


def _read_labeled_csv(path: Path, label: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "text" not in frame.columns and "title" not in frame.columns:
        raise ValueError(f"{path} must contain a 'text' or 'title' column")

    title = (
        frame["title"].fillna("").astype(str)
        if "title" in frame.columns
        else pd.Series("", index=frame.index)
    )
    body = (
        frame["text"].fillna("").astype(str)
        if "text" in frame.columns
        else pd.Series("", index=frame.index)
    )
    text = (title.str.strip() + "\n" + body.str.strip()).str.strip()
    result = pd.DataFrame({"text": text, "label": label})
    return result[result["text"].ne("")]


def load_news_dataset(fake_csv: Path, real_csv: Path) -> pd.DataFrame:
    """Load Fake.csv and True.csv, assigning fake=0 and real=1."""
    for path in (fake_csv, real_csv):
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")

    data = pd.concat(
        [_read_labeled_csv(fake_csv, 0), _read_labeled_csv(real_csv, 1)],
        ignore_index=True,
    )
    data = data.drop_duplicates(subset="text").reset_index(drop=True)
    if data.empty or data["label"].nunique() != 2:
        raise ValueError("Both files must contain at least one non-empty article")
    return data
