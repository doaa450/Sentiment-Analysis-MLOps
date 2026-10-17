# src/arabic_sentiment/train.py
import argparse
import logging

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from arabic_sentiment.data import load_dataset
from arabic_sentiment.labels import ID2LABEL, LABEL2ID

logger = logging.getLogger(__name__)
MODEL_NAME = "aubmindlab/bert-base-arabertv2"


class EncodedDataset(torch.utils.data.Dataset):
    def __init__(self, enc, labels) -> None:
        self.enc, self.labels = enc, list(labels)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> dict:
        return {**{k: v[i] for k, v in self.enc.items()}, "labels": self.labels[i]}


class WeightedTrainer(Trainer):
    """Trainer with class-weighted CrossEntropy to handle class imbalance."""

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(outputs.logits.device))
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(p) -> dict[str, float]:
    preds = p.predictions.argmax(-1)
    return {"macro_f1": f1_score(p.label_ids, preds, average="macro")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="models/arabert-sentiment")
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    df = load_dataset(args.data)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)

    weights = compute_class_weight("balanced", classes=np.arange(3), y=train_df["label"].values)
    logger.info("class_weights=%s", weights.round(3).tolist())

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )

    def encode(d):
        return EncodedDataset(
            tok(d["text"].tolist(), truncation=True, max_length=128), d["label"].tolist()
        )

    trainer = WeightedTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=32,
            learning_rate=2e-5,
            eval_strategy="epoch",
            save_strategy="no",
            fp16=torch.cuda.is_available(),
            report_to="none",
        ),
        train_dataset=encode(train_df),
        eval_dataset=encode(val_df),
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
        processing_class=tok,
        class_weights=torch.tensor(weights, dtype=torch.float),
    )
    trainer.train()

    trainer.save_model(args.out)
    tok.save_pretrained(args.out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
