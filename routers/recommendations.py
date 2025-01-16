from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional
from database import get_db
from models.database_models import User, Track, UserPreference
from services.recommendation_service import RecommendationService
from routers.auth import get_current_user
from pydantic import BaseModel
import logging

router = APIRouter()
logger = logging.getLogger(__name__)

class TrackBase(BaseModel):
    id: str
    name: str
    artists: str
    album: str
    popularity: Optional[int]

    class Config:
        orm_mode = True

class RecommendationRequest(BaseModel):
    seed_tracks: List[str]
    n_recommendations: Optional[int] = 10

class UserPreferenceUpdate(BaseModel):
    track_id: str
    rating: float
    listen_count: Optional[int] = 1

@router.post("/recommendations", response_model=List[TrackBase])
async def get_recommendations(
    request: RecommendationRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get personalized music recommendations"""
    logger.info(f"Processing recommendation request: {request.dict()}")
    
    recommender = RecommendationService(db)
    recommendations = recommender.get_recommendations(
        user_id=current_user.id,
        seed_tracks=request.seed_tracks,
        n_recommendations=request.n_recommendations
    )
    
    logger.info(f"Found {len(recommendations)} recommendations")
    
    if not recommendations:
        logger.warning("No recommendations found")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No recommendations found"
        )
    
    return recommendations

@router.post("/preferences/update")
async def update_preferences(
    preference: UserPreferenceUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user preferences for a track"""
    recommender = RecommendationService(db)
    success = recommender.update_user_preferences(
        user_id=current_user.id,
        track_id=preference.track_id,
        rating=preference.rating,
        listen_count=preference.listen_count
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to update preferences"
        )
    
    return {"status": "success"}

@router.get("/tracks/{track_id}/features")
async def get_track_features(
    track_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get audio features for a track"""
    recommender = RecommendationService(db)
    features = recommender.get_track_features(track_id)
    
    if not features:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Track features not found"
        )
    
    return features

@router.get("/test-spotify-auth")
async def test_spotify_auth(db: Session = Depends(get_db)):
    """Test Spotify authentication"""
    try:
        recommender = RecommendationService(db)
        # Try to get a random popular track
        results = recommender.spotify.search(q='genre:pop', type='track', limit=1)
        
        if results and results['tracks']['items']:
            track = results['tracks']['items'][0]
            return {
                "status": "success",
                "spotify_connected": True,
                "test_track": {
                    "name": track["name"],
                    "artist": track["artists"][0]["name"],
                    "id": track["id"]
                }
            }
    except Exception as e:
        return {
            "status": "error",
            "spotify_connected": False,
            "error": str(e)
        }