"""Finish registering ArabicSentiment version 1: add tags and the Production alias."""

from __future__ import annotations

import logging
import os

import mlflow
from dotenv import load_dotenv
from mlflow import MlflowClient

MODEL_NAME = "ArabicSentiment"
VERSION = "1"
ALIAS = "Production"
TAGS = {
    "hf_repo": "DoaaMoh/arabert-sentiment-v2",
    "hf_commit": "ca94e48b392d0af4b1add28a3bd3161edb0b1930",
    "weights_sha256": "bd184466583c47f18a30985f8fdf32f520bdb87da4eafb21249c3904b8fd3157",
    "source_run": "lr3e-05_bs16",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("finish_registration")


def main() -> None:
    load_dotenv()
    mlflow.set_tracking_uri(os.environ["SENTIMENT_MLFLOW_TRACKING_URI"])
    client = MlflowClient()

    # Safety check: version 1 must come from the run whose weights are on Hugging Face
    mv = client.get_model_version(MODEL_NAME, VERSION)
    run_name = client.get_run(mv.run_id).info.run_name
    if run_name != TAGS["source_run"]:
        raise RuntimeError(
            f"Version {VERSION} comes from run {run_name!r}, not {TAGS['source_run']!r}"
        )

    for key, value in TAGS.items():
        client.set_model_version_tag(MODEL_NAME, VERSION, key, value)
    client.set_registered_model_alias(MODEL_NAME, ALIAS, VERSION)

    check = client.get_model_version_by_alias(MODEL_NAME, ALIAS)
    log.info("Alias %s -> %s version %s (run %s)", ALIAS, MODEL_NAME, check.version, run_name)


if __name__ == "__main__":
    main()
