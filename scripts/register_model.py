"""Register the best teacher (published on Hugging Face) in the MLflow registry.

Run from the project root while the docker compose stack is up:
    uv run python scripts/register_model.py

What it does:
1. Finds the best teacher run in MLflow (highest f1_macro, tag stage=teacher).
2. Downloads the model from Hugging Face and checks its SHA-256.
3. Logs it as an MLflow model under that run and registers it as "ArabicSentiment".
4. Moves the alias "Production" to the new version.
"""

from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from importlib.metadata import version
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download
from mlflow import MlflowClient

HF_REPO = "DoaaMoh/arabert-sentiment-v2"
EXPECTED_SHA256 = "bd184466583c47f18a30985f8fdf32f520bdb87da4eafb21249c3904b8fd3157"
EXPECTED_RUN_NAME = "lr3e-05_bs16"  # the run that produced the weights in HF_REPO
EXPERIMENT = "arabic-sentiment"
MODEL_NAME = "ArabicSentiment"
ALIAS = "Production"
REQUIRED_ENV = (
    "SENTIMENT_MLFLOW_TRACKING_URI",
    "MLFLOW_S3_ENDPOINT_URL",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("register_model")


class SentimentModel(mlflow.pyfunc.PythonModel):
    """Loads the Hugging Face files stored as an artifact and predicts sentiment."""

    def load_context(self, context):
        from transformers import pipeline

        self.pipe = pipeline("text-classification", model=context.artifacts["hf_model"])

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            texts = model_input["text"].tolist()
        else:
            texts = list(model_input)
        outputs = self.pipe(texts, truncation=True, max_length=128)
        return [{"label": o["label"], "confidence": float(o["score"])} for o in outputs]


def sha256_of(path: Path, chunk: int = 1024 * 1024) -> str:
    """SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    load_dotenv()
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {missing}")

    mlflow.set_tracking_uri(os.environ["SENTIMENT_MLFLOW_TRACKING_URI"])
    client = MlflowClient()

    # 1. Best teacher run
    experiment = client.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        raise RuntimeError(f"Experiment {EXPERIMENT!r} not found")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="tags.stage = 'teacher'",
        order_by=["metrics.f1_macro DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No teacher runs found")
    best = runs[0]
    log.info(
        "Best run: %s (%s) f1_macro=%.4f",
        best.info.run_name,
        best.info.run_id,
        best.data.metrics["f1_macro"],
    )
    if best.info.run_name != EXPECTED_RUN_NAME:
        raise RuntimeError(
            f"Best run is {best.info.run_name!r}, but the weights in {HF_REPO} "
            f"came from {EXPECTED_RUN_NAME!r}. Not registering."
        )

    # 2. Download from Hugging Face and verify
    hf_commit = HfApi().model_info(HF_REPO).sha
    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(
            snapshot_download(
                repo_id=HF_REPO,
                local_dir=tmp,
                ignore_patterns=["training_args.bin"],
            )
        )
        actual = sha256_of(model_dir / "model.safetensors")
        if actual != EXPECTED_SHA256:
            raise RuntimeError(f"SHA-256 mismatch: {actual} != {EXPECTED_SHA256}")
        log.info("SHA-256 verified")

        # 3. Log under the best run and register
        with mlflow.start_run(run_id=best.info.run_id):
            mlflow.pyfunc.log_model(
                name="model",
                python_model=SentimentModel(),
                artifacts={"hf_model": str(model_dir)},
                pip_requirements=[
                    f"mlflow=={version('mlflow')}",
                    f"transformers=={version('transformers')}",
                    f"torch=={version('torch')}",
                ],
                registered_model_name=MODEL_NAME,
            )

    # 4. Tag the new version and move the alias
    new_version = str(
        max(int(v.version) for v in client.search_model_versions(f"name='{MODEL_NAME}'"))
    )
    for key, value in {
        "hf_repo": HF_REPO,
        "hf_commit": hf_commit,
        "weights_sha256": EXPECTED_SHA256,
        "source_run": best.info.run_name,
    }.items():
        client.set_model_version_tag(MODEL_NAME, new_version, key, value)
    client.set_registered_model_alias(MODEL_NAME, ALIAS, new_version)

    check = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    log.info("Alias %s -> %s version %s (run %s)", ALIAS, MODEL_NAME, check.version, check.run_id)


if __name__ == "__main__":
    main()
