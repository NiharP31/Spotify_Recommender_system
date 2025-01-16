from pydantic_settings import BaseSettings
from functools import lru_cache
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    APP_NAME: str = "Spotify Recommender API"
    DEBUG_MODE: bool = True
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./spotify_recommender.db")
    
    # Spotify API
    SPOTIFY_CLIENT_ID: str = os.getenv("SPOTIFY_CLIENT_ID")
    SPOTIFY_CLIENT_SECRET: str = os.getenv("SPOTIFY_CLIENT_SECRET")
    
    # JWT
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Redis
    # REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Model parameters
    EMBEDDING_DIM: int = 32
    MODEL_PATH: str = "models"

    # Model storage
    MODEL_PATH: str = "models_store"
    
    class Config:
        case_sensitive = True

@lru_cache()
def get_settings():
    return Settings()