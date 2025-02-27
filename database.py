from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import get_settings

# DATABASE_URL = "sqlite:///./spotify_recommender.db"

settings = get_settings()

engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    from models.database_models import Base
    Base.metadata.create_all(bind=engine)