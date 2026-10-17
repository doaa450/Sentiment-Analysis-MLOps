# src/arabic_sentiment/model.py
import logging
from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from arabic_sentiment.labels import ID2LABEL, LABEL2ID

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    model_version: str


class SentimentModel:
    """AraBERT v2 wrapper. Loads once; reuse the instance for all requests."""

    def __init__(
        self,
        model_name: str = "aubmindlab/bert-base-arabertv2",
        model_version: str = "v0.1.0",
        max_length: int = 128,
    ) -> None:
        self.model_version = model_version
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading %s on %s", model_name, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(LABEL2ID),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        self.model.to(self.device).eval()

    @torch.inference_mode()
    def predict(self, text: str) -> Prediction:
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        probs = torch.softmax(self.model(**inputs).logits, dim=-1)[0]
        idx = int(probs.argmax())
        return Prediction(ID2LABEL[idx], float(probs[idx]), self.model_version)
