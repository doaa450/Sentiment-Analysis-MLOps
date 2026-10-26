# locust/locustfile.py
import random
from pathlib import Path

from locust import HttpUser, between, task

# Real Arabic reviews sampled from data/Final_Data.csv
SAMPLES = [
    line.strip()
    for line in (Path(__file__).parent / "samples.txt").read_text(encoding="utf-8").splitlines()
    if line.strip()
]


class SentimentUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(10)
    def predict(self) -> None:
        payload = {"texts": [random.choice(SAMPLES)]}
        with self.client.post("/predict", json=payload, catch_response=True) as resp:
            if resp.status_code != 200:
                resp.failure(f"Expected 200, got {resp.status_code}")
                return
            body = resp.json()[0]
            if not body.get("label"):
                resp.failure("Missing label")
            elif not 0.0 <= body.get("confidence", -1.0) <= 1.0:
                resp.failure("Confidence out of range")

    @task(1)
    def health(self) -> None:
        self.client.get("/livez")
