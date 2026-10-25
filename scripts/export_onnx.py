"""Export the distilled student to ONNX (FP32) and check parity against PyTorch."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from arabic_sentiment.data import load_dataset

REPO = "DoaaMoh/arabert-sentiment-student"
OUT = Path("models/onnx/student_fp32.onnx")
REPORT = Path("reports/onnx_parity_student.json")
DATA_PATH = "data/Final_Data.csv"
MAX_LENGTH = 128
N_PARITY = 64


class LogitsOnly(torch.nn.Module):
    """Wrap the model so the ONNX graph returns only the logits tensor."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def val_texts() -> list[str]:
    """Texts of the fixed validation split (20%, stratified, random_state=42)."""
    df = load_dataset(DATA_PATH)
    _, val = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
    return val["text"].tolist()


def export(wrapper: torch.nn.Module, example: tuple, legacy: bool) -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    io_names = {
        "input_names": ["input_ids", "attention_mask"],
        "output_names": ["logits"],
    }
    if legacy:
        axes = {
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        }
        torch.onnx.export(
            wrapper,
            example,
            str(OUT),
            dynamo=False,
            opset_version=17,
            dynamic_axes=axes,
            **io_names,
        )
    else:
        batch = torch.export.Dim("batch", min=1, max=64)
        seq = torch.export.Dim("seq", min=2, max=MAX_LENGTH)
        shapes = ({0: batch, 1: seq}, {0: batch, 1: seq})
        torch.onnx.export(
            wrapper,
            example,
            str(OUT),
            dynamo=True,
            external_data=False,
            dynamic_shapes=shapes,
            **io_names,
        )


def parity(wrapper: torch.nn.Module, tokenizer, texts: list[str]) -> dict:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    sess = ort.InferenceSession(str(OUT), opts, providers=["CPUExecutionProvider"])

    # Batches of 16 with different lengths, plus one batch of size 1
    chunks = [texts[i : i + 16] for i in range(0, len(texts), 16)] + [texts[:1]]
    max_diff, agree, total = 0.0, 0, 0
    for chunk in chunks:
        enc = tokenizer(
            chunk, truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt"
        )
        with torch.inference_mode():
            ref = wrapper(enc["input_ids"], enc["attention_mask"]).numpy()
        feed = {
            "input_ids": enc["input_ids"].numpy(),
            "attention_mask": enc["attention_mask"].numpy(),
        }
        out = sess.run(["logits"], feed)[0]
        max_diff = max(max_diff, float(np.abs(ref - out).max()))
        agree += int((ref.argmax(-1) == out.argmax(-1)).sum())
        total += len(ref)
    return {
        "samples": total,
        "max_abs_logit_diff": max_diff,
        "argmax_agreement": agree / total,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", action="store_true", help="use the TorchScript exporter")
    args = parser.parse_args()

    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(REPO)
    model = AutoModelForSequenceClassification.from_pretrained(REPO).eval()
    wrapper = LogitsOnly(model).eval()

    texts = val_texts()[:N_PARITY]
    enc = tokenizer(
        texts[:4], truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt"
    )
    export(wrapper, (enc["input_ids"], enc["attention_mask"]), args.legacy)

    result = parity(wrapper, tokenizer, texts)
    result["exporter"] = "legacy" if args.legacy else "dynamo"
    result["file"] = str(OUT)
    result["size_bytes"] = OUT.stat().st_size
    REPORT.write_text(json.dumps(result, indent=2))
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
