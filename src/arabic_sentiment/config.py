# src/arabic_sentiment/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SENTIMENT_", env_file=".env", extra="ignore")
    # Can be a HF model id or a relative path to a fine-tuned folder
    model_name: str = "aubmindlab/bert-base-arabertv2"
    model_version: str = "v0.1.0"
    max_length: int = 128
