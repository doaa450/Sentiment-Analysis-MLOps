# src/arabic_sentiment/schemas.py
from typing import Annotated

from pydantic import BaseModel, StringConstraints

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class PredictRequest(BaseModel):
    text: NonEmptyText  # empty or whitespace-only -> 422


class PredictResponse(BaseModel):
    label: str
    confidence: float
    model_version: str
