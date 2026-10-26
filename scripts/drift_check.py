"""PSI and Evidently drift reports on the batch-scored validation split."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

SCORED = Path("data/scoring/output/val_scored.csv")
FEATURES = ["text_length", "confidence"]
PSI_LATEST = Path("reports/psi_latest.json")
N_BINS = 10
EPS = 1e-6


def psi(reference: pd.Series, current: pd.Series, bins: int = N_BINS) -> float:
    """Population Stability Index with quantile bins taken from the reference."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, edges)[0] / len(reference) + EPS
    cur_pct = np.histogram(current, edges)[0] / len(current) + EPS
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def evidently_report(current: pd.DataFrame, reference: pd.DataFrame, out: Path) -> None:
    definition = DataDefinition(numerical_columns=FEATURES)
    cur_ds = Dataset.from_pandas(current[FEATURES], data_definition=definition)
    ref_ds = Dataset.from_pandas(reference[FEATURES], data_definition=definition)
    snapshot = Report([DataDriftPreset(columns=FEATURES)]).run(cur_ds, ref_ds)
    snapshot.save_html(str(out))


def main() -> int:
    df = pd.read_csv(SCORED).sample(frac=1.0, random_state=42).reset_index(drop=True)
    half = len(df) // 2
    reference, current = df.iloc[:half], df.iloc[half:]
    long_reviews = df[df["text_length"] > df["text_length"].median()]

    runs = {
        "val_halves": (current, "same distribution, expected: no drift"),
        "long_reviews": (long_reviews, "artificial drift: reviews above the median length"),
    }
    latest: dict[str, dict[str, float]] = {}
    for name, (cur, note) in runs.items():
        sys.stdout.write(f"[{name}] {note} (reference={len(reference)}, current={len(cur)})\n")
        for col in FEATURES:
            value = psi(reference[col], cur[col])
            latest.setdefault(name, {})[col] = value
            sys.stdout.write(f"  PSI {col}: {value:.4f}\n")
        out = Path(f"reports/drift_report_{name}.html")
        evidently_report(cur, reference, out)
        sys.stdout.write(f"  saved {out}\n")
    PSI_LATEST.write_text(json.dumps(latest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
