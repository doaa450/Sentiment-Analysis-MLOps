# src/arabic_sentiment/api.py
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, Request

from arabic_sentiment.config import Settings
from arabic_sentiment.model import SentimentModel
from arabic_sentiment.schemas import PredictRequest, PredictResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at startup, shared by all requests
    s = Settings()
    app.state.model = SentimentModel(s.model_name, s.model_version, s.max_length)
    yield


app = FastAPI(title="Arabic Sentiment", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest, request: Request) -> PredictResponse:
    return PredictResponse(**asdict(request.app.state.model.predict(body.text)))
