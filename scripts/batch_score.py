"""Score the validation split with the INT8 model from the BentoML store (batch job)."""

import subprocess
import sys
import time
from pathlib import Path

import bentoml
import numpy as np
import onnxruntime as ort
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from arabic_sentiment.data import load_dataset
from arabic_sentiment.labels import ID2LABEL

DATA_PATH = "data/Final_Data.csv"
OUTPUT_PATH = Path("data/scoring/output/val_scored.csv")
MODEL_TAG = "arabic_sentiment_student_int8:latest"
MAX_LENGTH = 128
BATCH_SIZE = 32


def main() -> int:
    df = load_dataset(DATA_PATH)
    _, val = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    val = val.reset_index(drop=True)

    model = bentoml.models.get(MODEL_TAG)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    session = ort.InferenceSession(
        str(Path(model.path) / "model.onnx"), opts, providers=["CPUExecutionProvider"]
    )
    tokenizer = AutoTokenizer.from_pretrained(str(Path(model.path) / "tokenizer"))

    texts = val["text"].tolist()
    order = np.argsort([len(t) for t in texts])
    pred = np.zeros(len(texts), dtype=int)
    conf = np.zeros(len(texts), dtype=float)

    start = time.perf_counter()
    for i in range(0, len(order), BATCH_SIZE):
        idx = order[i : i + BATCH_SIZE]
        enc = tokenizer(
            [texts[j] for j in idx],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="np",
        )
        feed = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        logits = session.run(["logits"], feed)[0]
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp / exp.sum(axis=-1, keepdims=True)
        pred[idx] = probs.argmax(axis=-1)
        conf[idx] = probs.max(axis=-1)
    elapsed = time.perf_counter() - start

    out = pd.DataFrame(
        {
            "text_length": [len(t) for t in texts],
            "label": [ID2LABEL[int(p)] for p in pred],
            "confidence": conf,
            "true_label": [ID2LABEL[int(v)] for v in val["label"]],
            "model_version": str(model.tag),
        }
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    sys.stdout.write(
        f"rows={len(out)} seconds={elapsed:.1f} rows_per_sec={len(out) / elapsed:.1f} "
        f"accuracy={(out['label'] == out['true_label']).mean():.4f} -> {OUTPUT_PATH}\n"
    )
    # Item 10: run the drift check after every scoring run
    return subprocess.run([sys.executable, "scripts/drift_check.py"], check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
