"""Evaluate a Hugging Face sequence-classification model on the fixed validation split."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from arabic_sentiment.data import load_dataset

DATA_PATH = "data/Final_Data.csv"
MAX_LENGTH = 128


def get_val_split():
    """Same split as the training notebook: 20%, stratified, random_state=42."""
    df = load_dataset(DATA_PATH)
    _, val = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    return val.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="Hugging Face repo id")
    parser.add_argument("--name", required=True, help="short name for the output file")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    torch.set_num_threads(2)
    val = get_val_split()
    texts = val["text"].tolist()
    y_true = val["label"].to_numpy()

    tokenizer = AutoTokenizer.from_pretrained(args.repo)
    model = AutoModelForSequenceClassification.from_pretrained(args.repo).eval()

    # Sort by length so each batch needs little padding
    order = np.argsort([len(t) for t in texts])
    preds = np.zeros(len(texts), dtype=int)
    with torch.inference_mode():
        for start in range(0, len(order), args.batch_size):
            idx = order[start : start + args.batch_size]
            enc = tokenizer(
                [texts[j] for j in idx],
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            )
            preds[idx] = model(**enc).logits.argmax(-1).numpy()

    result = {
        "name": args.name,
        "repo": args.repo,
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, preds)),
        "f1_macro": float(f1_score(y_true, preds, average="macro")),
        "f1_neutral": float(f1_score(y_true, preds, labels=[1], average="macro")),
    }
    out = Path(f"reports/eval_{args.name}.json")
    out.write_text(json.dumps(result, indent=2))
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
