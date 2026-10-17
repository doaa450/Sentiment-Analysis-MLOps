from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient


@dataclass
class FakePrediction:
    # Field names must match PredictResponse in schemas.py
    label: str = "positive"
    confidence: float = 0.99
    model_version: str = "test"


class FakeModel:
    def predict(self, text: str):
        return FakePrediction()


@pytest.fixture
def client(monkeypatch):
    from arabic_sentiment import api

    # Replace the real model loader before lifespan runs
    monkeypatch.setattr(api, "SentimentModel", lambda *a, **k: FakeModel())
    with TestClient(api.app) as c:
        yield c
