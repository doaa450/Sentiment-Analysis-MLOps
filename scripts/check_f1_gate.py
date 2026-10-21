"""Fail (exit 1) if the current f1_macro is below the baseline."""

import json
import sys
from pathlib import Path

BASELINE = Path("metrics/baseline.json")
CURRENT = Path("metrics/metrics.json")


def main() -> int:
    baseline = json.loads(BASELINE.read_text())["f1_macro"]
    current = json.loads(CURRENT.read_text())["f1_macro"]
    sys.stdout.write(f"baseline f1_macro={baseline:.4f} current f1_macro={current:.4f}\n")
    if current < baseline:
        sys.stdout.write("GATE FAILED: f1_macro dropped below baseline\n")
        return 1
    sys.stdout.write("GATE PASSED\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
