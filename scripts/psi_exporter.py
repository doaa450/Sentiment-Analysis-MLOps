"""Expose the latest PSI values (reports/psi_latest.json) to Prometheus at /metrics."""

import json
import sys
import time
from pathlib import Path

from prometheus_client import Gauge, start_http_server

PSI_FILE = Path("reports/psi_latest.json")
PORT = 9100
REFRESH_SECONDS = 5

psi_gauge = Gauge("psi", "Population Stability Index of a feature", ["run", "feature"])


def refresh() -> None:
    if not PSI_FILE.exists():
        return
    for run, features in json.loads(PSI_FILE.read_text(encoding="utf-8")).items():
        for feature, value in features.items():
            psi_gauge.labels(run=run, feature=feature).set(value)


def main() -> int:
    start_http_server(PORT)
    sys.stdout.write(f"PSI exporter on :{PORT}/metrics\n")
    while True:
        refresh()
        time.sleep(REFRESH_SECONDS)


if __name__ == "__main__":
    sys.exit(main())
