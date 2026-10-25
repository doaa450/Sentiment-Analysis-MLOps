"""Build the 4-variant benchmark table from the saved eval and latency reports."""

import csv
import json
import sys
from pathlib import Path

REPORTS = Path("reports")
VARIANTS = [
    ("teacher_pytorch", "Teacher (12 layers)", "PyTorch CPU FP32"),
    ("student_pytorch", "Student (6 layers)", "PyTorch CPU FP32"),
    ("student_onnx_fp32", "Student", "ONNX Runtime CPU FP32"),
    ("student_onnx_int8", "Student", "ONNX Runtime CPU INT8"),
]
HEADER = [
    "Model",
    "Backend",
    "Accuracy",
    "F1 macro",
    "F1 neutral",
    "Size (MB)",
    "p50 (ms)",
    "p95 (ms)",
]


def load(prefix: str, key: str) -> dict:
    return json.loads((REPORTS / f"{prefix}_{key}.json").read_text())


def main() -> int:
    rows = []
    for key, model, backend in VARIANTS:
        ev, lat = load("eval", key), load("latency", key)
        rows.append(
            [
                model,
                backend,
                f"{ev['accuracy']:.4f}",
                f"{ev['f1_macro']:.4f}",
                f"{ev['f1_neutral']:.4f}",
                f"{lat['size_bytes'] / 1e6:.1f}",
                f"{lat['p50_ms']:.1f}",
                f"{lat['p95_ms']:.1f}",
            ]
        )

    with (REPORTS / "benchmark_table.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(rows)

    lines = ["| " + " | ".join(HEADER) + " |", "|" + "---|" * len(HEADER)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    table = "\n".join(lines) + "\n"
    (REPORTS / "benchmark_table.md").write_text(table)
    sys.stdout.write(table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
