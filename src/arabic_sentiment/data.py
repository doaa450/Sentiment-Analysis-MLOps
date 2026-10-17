import re

import pandas as pd

from arabic_sentiment.labels import LABEL2ID

TEXT_COL, LABEL_COL = "review_description", "rating"

_URL = re.compile(r"https?://\S+")
_DIACRITICS = re.compile(r"[\u064B-\u0652\u0640]")  # tashkeel + tatweel


def clean_text(text: str) -> str:
    text = _URL.sub(" ", text)
    text = _DIACRITICS.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path).dropna(subset=[TEXT_COL, LABEL_COL])
    df["text"] = df[TEXT_COL].astype(str).map(clean_text)
    df = df[df["text"].str.len() > 0].drop_duplicates("text")
    # String labels map directly to ids
    df["label"] = df[LABEL_COL].str.strip().str.lower().map(LABEL2ID)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
    return df[["text", "label"]].reset_index(drop=True)
