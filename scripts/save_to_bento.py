"""Save the INT8 student (ONNX) and its tokenizer into the BentoML model store."""

import hashlib
import shutil
import sys
from pathlib import Path

import bentoml
from transformers import AutoTokenizer

ONNX_PATH = Path("models/onnx/student_int8.onnx")
TOKENIZER_REPO = "DoaaMoh/arabert-sentiment-student"
MODEL_NAME = "arabic_sentiment_student_int8"
MAX_LENGTH = 128


def sha256_of(path: Path) -> str:
    """Stream the file so the whole model is never held in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if not ONNX_PATH.exists():
        sys.stderr.write(f"Missing {ONNX_PATH}\n")
        return 1

    sha = sha256_of(ONNX_PATH)
    size = ONNX_PATH.stat().st_size

    with bentoml.models.create(
        MODEL_NAME,
        labels={"stage": "int8", "framework": "onnxruntime", "task": "arabic-sentiment"},
        metadata={
            "source_file": str(ONNX_PATH),
            "onnx_sha256": sha,
            "onnx_size_bytes": size,
            "tokenizer_repo": TOKENIZER_REPO,
            "max_length": MAX_LENGTH,
            "labels": {"0": "negative", "1": "neutral", "2": "positive"},
        },
    ) as model:
        target = Path(model.path)
        shutil.copy2(ONNX_PATH, target / "model.onnx")
        AutoTokenizer.from_pretrained(TOKENIZER_REPO).save_pretrained(target / "tokenizer")
        sys.stdout.write(f"Saved {model.tag} at {target}\n")
        sys.stdout.write(f"sha256={sha} size={size}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
