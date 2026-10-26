"""BentoML service for the INT8 student: ONNX Runtime on CPU with adaptive batching."""

from pathlib import Path
from typing import Any

import bentoml
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

MODEL_TAG = "arabic_sentiment_student_int8:latest"
MAX_LENGTH = 128
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}


@bentoml.service(
    workers=1,
    traffic={"timeout": 60, "concurrency": 64},
    metrics={"enabled": True},
)
class SentimentService:
    def __init__(self) -> None:
        model = bentoml.models.get(MODEL_TAG)
        self.model_version = str(model.tag)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(Path(model.path) / "model.onnx"),
            opts,
            providers=["CPUExecutionProvider"],
        )
        self.tokenizer = AutoTokenizer.from_pretrained(str(Path(model.path) / "tokenizer"))

    @bentoml.api(batchable=True, max_batch_size=2, max_latency_ms=10000)
    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        enc = self.tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
            return_tensors="np",
        )
        feed = {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        }
        logits = self.session.run(["logits"], feed)[0]
        exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp / exp.sum(axis=-1, keepdims=True)
        idx = probs.argmax(axis=-1)
        return [
            {
                "label": ID2LABEL[int(i)],
                "confidence": float(probs[n, i]),
                "model_version": self.model_version,
            }
            for n, i in enumerate(idx)
        ]
