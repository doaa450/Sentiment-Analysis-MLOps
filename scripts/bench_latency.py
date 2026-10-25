"""Batch-1 CPU latency (tokenize + forward) for one model variant."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from arabic_sentiment.data import load_dataset

DATA_PATH = "data/Final_Data.csv"
MAX_LENGTH = 128
N_SAMPLES = 200
N_WARMUP = 20
TOKENIZER_REPO = "DoaaMoh/arabert-sentiment-student"
VARIANTS = {
    "teacher_pytorch": ("hf", "DoaaMoh/arabert-sentiment-v2"),
    "student_pytorch": ("hf", "DoaaMoh/arabert-sentiment-student"),
    "student_onnx_fp32": ("onnx", "models/onnx/student_fp32.onnx"),
    "student_onnx_int8": ("onnx", "models/onnx/student_int8.onnx"),
}


def val_texts() -> list[str]:
    """Texts sampled from the fixed validation split (same for every variant)."""
    df = load_dataset(DATA_PATH)
    _, val = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    return val.sample(N_SAMPLES, random_state=42)["text"].tolist()


def build_runner(kind: str, ref: str):
    """Return (run(text) -> predicted id, size of the model file in bytes)."""
    if kind == "hf":
        tok = AutoTokenizer.from_pretrained(ref)
        model = AutoModelForSequenceClassification.from_pretrained(ref).eval()
        size = Path(hf_hub_download(ref, "model.safetensors")).stat().st_size

        def run(text: str) -> int:
            enc = tok(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            with torch.inference_mode():
                return int(model(**enc).logits.argmax(-1))

        return run, size

    tok = AutoTokenizer.from_pretrained(TOKENIZER_REPO)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    sess = ort.InferenceSession(ref, opts, providers=["CPUExecutionProvider"])

    def run(text: str) -> int:
        enc = tok(text, truncation=True, max_length=MAX_LENGTH, return_tensors="np")
        feed = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        return int(sess.run(["logits"], feed)[0][0].argmax())

    return run, Path(ref).stat().st_size


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=list(VARIANTS))
    args = parser.parse_args()

    torch.set_num_threads(2)
    texts = val_texts()
    run, size = build_runner(*VARIANTS[args.variant])

    for text in texts[:N_WARMUP]:
        run(text)

    times_ms = []
    for text in texts:
        start = time.perf_counter()
        run(text)
        times_ms.append((time.perf_counter() - start) * 1000)

    result = {
        "variant": args.variant,
        "n": len(times_ms),
        "warmup": N_WARMUP,
        "threads": 2,
        "p50_ms": float(np.percentile(times_ms, 50)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "mean_ms": float(np.mean(times_ms)),
        "size_bytes": size,
    }
    Path(f"reports/latency_{args.variant}.json").write_text(json.dumps(result, indent=2))
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
