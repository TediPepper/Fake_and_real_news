import tempfile
import unittest
from pathlib import Path

import pandas as pd

from fake_news_detector.data import load_news_dataset


class LoadNewsDatasetTests(unittest.TestCase):
    def test_combines_title_and_body_and_assigns_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_path = root / "Fake.csv"
            real_path = root / "True.csv"
            pd.DataFrame(
                {"title": ["Fake headline"], "text": ["Fake article body"]}
            ).to_csv(fake_path, index=False)
            pd.DataFrame(
                {"title": ["Real headline"], "text": ["Real article body"]}
            ).to_csv(real_path, index=False)

            data = load_news_dataset(fake_path, real_path)

        self.assertEqual(data["text"].tolist(), [
            "Fake headline\nFake article body",
            "Real headline\nReal article body",
        ])
        self.assertEqual(data["label"].tolist(), [0, 1])

    def test_drops_empty_articles_and_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake_path = root / "Fake.csv"
            real_path = root / "True.csv"
            pd.DataFrame({"text": ["Repeated story", ""]}).to_csv(
                fake_path, index=False
            )
            pd.DataFrame({"text": ["Repeated story", "A real story"]}).to_csv(
                real_path, index=False
            )

            data = load_news_dataset(fake_path, real_path)

        self.assertEqual(data["text"].tolist(), ["Repeated story", "A real story"])
        self.assertEqual(data["label"].tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
