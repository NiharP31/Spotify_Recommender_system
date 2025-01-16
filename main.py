from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import auth, recommendations
from database import init_db
from config import get_settings
import logging

settings = get_settings()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

init_db()

app = FastAPI(
    title=settings.APP_NAME,
    description="A machine learning-powered music recommendation system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development only
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Initialize database
@app.on_event("startup")
async def startup_event():
    logger.info("Initializing database...")
    from database import init_db
    init_db()
    logger.info("Database initialized successfully")

# Include routers
app.include_router(
    auth.router,
    prefix="/auth",
    tags=["Authentication"]
)
app.include_router(
    recommendations.router,
    prefix="/api",
    tags=["Recommendations"]
)

@app.get("/")
async def root():
    return {
        "message": "Welcome to the Spotify Recommender API",
        "version": "1.0.0",
        "documentation": "/docs",
        "status": "operational"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "environment": "development" if settings.DEBUG_MODE else "production"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG_MODE
    )