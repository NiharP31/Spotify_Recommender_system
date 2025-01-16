import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, ndcg_score
import tensorflow as tf
import xgboost as xgb
from datetime import datetime
import logging
from typing import List, Dict, Optional
from pathlib import Path
from config import get_settings
from models.database_models import Track, UserPreference
from sqlalchemy.orm import Session
import json

settings = get_settings()

# Create model directory if it doesn't exist
MODEL_DIR = Path(settings.MODEL_PATH)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

class RecommendationService:
    def __init__(self, db: Session):
        self.db = db
        self.logger = self._setup_logging()
        self.spotify = self._setup_spotify()
        self.scaler = self._load_or_create_scaler()
        self._load_models()
    
    def _setup_logging(self):
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _setup_spotify(self):
        try:
            client_credentials_manager = SpotifyClientCredentials(
                client_id=settings.SPOTIFY_CLIENT_ID,
                client_secret=settings.SPOTIFY_CLIENT_SECRET
            )
            return spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        except Exception as e:
            self.logger.error(f"Failed to setup Spotify client: {str(e)}")
            raise

    def _load_or_create_scaler(self):
        """Load existing scaler or create new one"""
        scaler_path = MODEL_DIR / "scaler.pkl"
        if scaler_path.exists():
            try:
                return pd.read_pickle(scaler_path)
            except Exception as e:
                self.logger.warning(f"Could not load scaler: {str(e)}")
        return StandardScaler()

    def _save_scaler(self):
        """Save the scaler to disk"""
        try:
            scaler_path = MODEL_DIR / "scaler.pkl"
            pd.to_pickle(self.scaler, scaler_path)
        except Exception as e:
            self.logger.error(f"Failed to save scaler: {str(e)}")

    def _load_models(self):
        """Load or initialize recommendation models"""
        try:
            xgb_path = MODEL_DIR / "xgb_model.json"
            collab_path = MODEL_DIR / "collaborative_model"

            self.xgb_model = None
            self.collaborative_model = None

            if xgb_path.exists():
                self.xgb_model = xgb.Booster()
                self.xgb_model.load_model(str(xgb_path))
                self.logger.info("XGBoost model loaded successfully")

            if collab_path.exists():
                self.collaborative_model = tf.keras.models.load_model(str(collab_path))
                self.logger.info("Collaborative model loaded successfully")

        except Exception as e:
            self.logger.warning(f"Could not load existing models: {str(e)}")
            self.xgb_model = None
            self.collaborative_model = None

    def _get_track_info(self, track_id: str) -> Optional[Dict]:
        """Get basic track information from Spotify"""
        try:
            self.logger.info(f"Fetching track info for ID: {track_id}")
            track = self.spotify.track(track_id)
            
            if track:
                track_info = {
                    "id": track["id"],
                    "name": track["name"],
                    "artists": ", ".join([artist["name"] for artist in track["artists"]]),
                    "album": track["album"]["name"],
                    "popularity": track["popularity"]
                }
                self.logger.info(f"Successfully fetched track info: {track_info['name']}")
                return track_info
            return None
        except Exception as e:
            self.logger.error(f"Error fetching track info: {str(e)}")
            return None

    def get_recommendations(
        self,
        user_id: int,
        seed_tracks: List[str],
        n_recommendations: int = 10
    ) -> List[Dict]:
        """Get music recommendations"""
        try:
            self.logger.info(f"Processing recommendation request - User: {user_id}, Seeds: {seed_tracks}")
            
            if not seed_tracks:
                self.logger.warning("No seed tracks provided")
                return []

            # Get the seed track
            seed_track = self._get_track_info(seed_tracks[0])
            if not seed_track:
                self.logger.warning(f"Could not find seed track: {seed_tracks[0]}")
                return []

            self.logger.info(f"Using seed track: {seed_track['name']}")
            
            try:
                # Get track details to extract artist and genre info
                track_details = self.spotify.track(seed_tracks[0])
                if not track_details or not track_details["artists"]:
                    return []

                artist_id = track_details["artists"][0]["id"]
                artist_details = self.spotify.artist(artist_id)
                
                # Get artist's genres
                genres = artist_details.get("genres", [])
                if not genres:
                    genres = ["pop"]  # Default genre if none found
                
                self.logger.info(f"Found genres: {genres}")

                # Search for tracks in the same genre
                recommendations = []
                for genre in genres[:2]:  # Use up to 2 genres
                    results = self.spotify.search(
                        q=f"genre:{genre}",
                        type="track",
                        limit=n_recommendations
                    )
                    
                    if results and 'tracks' in results and 'items' in results['tracks']:
                        for track in results['tracks']['items']:
                            # Don't include the seed track
                            if track['id'] != seed_tracks[0]:
                                recommendations.append({
                                    "id": track["id"],
                                    "name": track["name"],
                                    "artists": ", ".join([artist["name"] for artist in track["artists"]]),
                                    "album": track["album"]["name"],
                                    "popularity": track["popularity"]
                                })
                            
                            if len(recommendations) >= n_recommendations:
                                break
                    
                    if len(recommendations) >= n_recommendations:
                        break

                self.logger.info(f"Successfully found {len(recommendations)} recommendations")
                return recommendations[:n_recommendations]

            except Exception as e:
                self.logger.error(f"Error getting recommendations: {str(e)}")
                return []

        except Exception as e:
            self.logger.error(f"Error in get_recommendations: {str(e)}")
            return []

    def _get_artist_recommendations(self, artist_id: str, limit: int) -> List[Dict]:
        """Get recommendations based on an artist"""
        try:
            self.logger.info(f"Getting recommendations for artist ID: {artist_id}")
            
            # Get related artists
            related = self.spotify.artist_related_artists(artist_id)
            recommendations = []
            
            for artist in related["artists"][:limit]:
                try:
                    # Get top tracks for each related artist
                    top_tracks = self.spotify.artist_top_tracks(artist["id"])["tracks"]
                    if top_tracks:
                        track = top_tracks[0]  # Get the first top track
                        recommendations.append({
                            "id": track["id"],
                            "name": track["name"],
                            "artists": ", ".join([a["name"] for a in track["artists"]]),
                            "album": track["album"]["name"],
                            "popularity": track["popularity"]
                        })
                        
                        if len(recommendations) >= limit:
                            break
                except Exception as e:
                    self.logger.error(f"Error getting top tracks for artist {artist['id']}: {str(e)}")
                    continue
            
            self.logger.info(f"Found {len(recommendations)} artist recommendations")
            return recommendations[:limit]
            
        except Exception as e:
            self.logger.error(f"Error getting artist recommendations: {str(e)}")
            return []

    def get_track_features(self, track_id: str) -> Optional[Dict]:
        """Get track audio features from Spotify API"""
        try:
            self.logger.info(f"Fetching audio features for track: {track_id}")
            features = self.spotify.audio_features([track_id])[0]
            return features if features else None
        except Exception as e:
            self.logger.error(f"Error fetching track features: {str(e)}")
            return None

    def update_user_preferences(
        self,
        user_id: int,
        track_id: str,
        rating: float,
        listen_count: int = 1
    ) -> bool:
        """Update user preferences for a track"""
        try:
            track_info = self._get_track_info(track_id)
            if not track_info:
                return False

            # Update or create track in database
            track = self.db.query(Track).filter(Track.id == track_id).first()
            if not track:
                track = Track(
                    id=track_id,
                    name=track_info["name"],
                    artists=track_info["artists"],
                    album=track_info["album"],
                    popularity=track_info["popularity"]
                )
                self.db.add(track)

            # Update preference
            pref = self.db.query(UserPreference).filter(
                UserPreference.user_id == user_id,
                UserPreference.track_id == track_id
            ).first()
            
            if pref:
                pref.rating = rating
                pref.listen_count += listen_count
                pref.last_listened = datetime.utcnow()
            else:
                pref = UserPreference(
                    user_id=user_id,
                    track_id=track_id,
                    rating=rating,
                    listen_count=listen_count,
                    last_listened=datetime.utcnow()
                )
                self.db.add(pref)
            
            self.db.commit()
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating user preferences: {str(e)}")
            self.db.rollback()
            return False
    
    def train_models(self):
        """Train/retrain recommendation models"""
        try:
            # Get all user preferences
            preferences = self.db.query(UserPreference).all()
            tracks = self.db.query(Track).all()
            
            if not preferences or not tracks:
                return False
            
            # Prepare training data
            train_data = []
            for pref in preferences:
                track = next((t for t in tracks if t.id == pref.track_id), None)
                if track:
                    features = self.get_track_features(track.id)
                    if features:
                        train_data.append({
                            'user_id': pref.user_id,
                            'track_id': track.id,
                            'rating': pref.rating,
                            **features
                        })
            
            if not train_data:
                return False
                
            df = pd.DataFrame(train_data)
            
            # Train XGBoost model
            feature_cols = ['danceability', 'energy', 'key', 'loudness', 'mode',
                          'speechiness', 'acousticness', 'instrumentalness',
                          'liveness', 'valence', 'tempo']
            
            X = df[feature_cols]
            y = df['rating']
            
            X_scaled = self.scaler.fit_transform(X)
            
            self.xgb_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                tree_method='gpu_hist' if tf.test.is_built_with_cuda() else 'hist'
            )
            
            self.xgb_model.fit(X_scaled, y)
            
            # Save XGBoost model
            self.xgb_model.save_model(f"{settings.MODEL_PATH}/xgb_model.json")
            
            # Train collaborative filtering model
            n_users = df['user_id'].nunique()
            n_tracks = df['track_id'].nunique()
            
            # Build and train collaborative model
            self.collaborative_model = self._build_collaborative_model(n_users, n_tracks)
            
            user_ids = df['user_id'].values
            track_ids = df['track_id'].values
            ratings = df['rating'].values
            
            history = self.collaborative_model.fit(
                [user_ids, track_ids],
                ratings,
                epochs=10,
                batch_size=64,
                validation_split=0.2,
                verbose=1
            )
            
            # Save collaborative model
            self.collaborative_model.save(f"{settings.MODEL_PATH}/collaborative_model")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training models: {str(e)}")
            return False
    
    def _build_collaborative_model(self, n_users: int, n_tracks: int) -> tf.keras.Model:
        """Build neural collaborative filtering model"""
        # User input and embedding
        user_input = tf.keras.layers.Input(shape=(1,))
        user_embedding = tf.keras.layers.Embedding(
            n_users + 1,
            settings.EMBEDDING_DIM
        )(user_input)
        user_vec = tf.keras.layers.Flatten()(user_embedding)
        
        # Track input and embedding
        track_input = tf.keras.layers.Input(shape=(1,))
        track_embedding = tf.keras.layers.Embedding(
            n_tracks + 1,
            settings.EMBEDDING_DIM
        )(track_input)
        track_vec = tf.keras.layers.Flatten()(track_embedding)
        
        # Combine embeddings
        concat = tf.keras.layers.Concatenate()([user_vec, track_vec])
        
        # Dense layers
        dense1 = tf.keras.layers.Dense(64, activation='relu')(concat)
        dropout1 = tf.keras.layers.Dropout(0.2)(dense1)
        dense2 = tf.keras.layers.Dense(32, activation='relu')(dropout1)
        dropout2 = tf.keras.layers.Dropout(0.2)(dense2)
        
        # Output layer
        output = tf.keras.layers.Dense(1)(dropout2)
        
        model = tf.keras.Model(inputs=[user_input, track_input], outputs=output)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error'
        )
        
        return model
    
    def evaluate_models(self, test_data: pd.DataFrame) -> dict:
        """Evaluate model performance using multiple metrics"""
        try:
            # Prepare test data
            feature_cols = ['danceability', 'energy', 'key', 'loudness', 'mode',
                        'speechiness', 'acousticness', 'instrumentalness',
                        'liveness', 'valence', 'tempo']
            
            X_test = test_data[feature_cols]
            y_test = test_data['rating']
            X_test_scaled = self.scaler.transform(X_test)
            
            # XGBoost predictions
            xgb_preds = self.xgb_model.predict(xgb.DMatrix(X_test_scaled))
            
            # Collaborative model predictions
            if self.collaborative_model:
                cf_preds = self.collaborative_model.predict([
                    test_data['user_id'].values,
                    test_data['track_id'].values
                ])
                
                # Combined predictions
                final_preds = 0.7 * xgb_preds + 0.3 * cf_preds.flatten()
            else:
                final_preds = xgb_preds
            
            # Calculate metrics
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_test, final_preds)),
                'mae': mean_absolute_error(y_test, final_preds),
                'r2': r2_score(y_test, final_preds),
                'ndcg': ndcg_score([y_test.values], [final_preds]),
                'precision_at_k': self._precision_at_k(y_test, final_preds, k=10),
                'recall_at_k': self._recall_at_k(y_test, final_preds, k=10)
            }
            
            # Calculate prediction distribution
            metrics['prediction_stats'] = {
                'mean': np.mean(final_preds),
                'std': np.std(final_preds),
                'min': np.min(final_preds),
                'max': np.max(final_preds)
            }
            
            # Track feature importance
            if self.xgb_model:
                importance = self.xgb_model.feature_importances_
                metrics['feature_importance'] = dict(zip(feature_cols, importance))
            
            return metrics
        
        except Exception as e:
            self.logger.error(f"Error evaluating models: {str(e)}")
            return {}

    def _precision_at_k(self, y_true, y_pred, k=10):
        """Calculate precision@k for recommendations"""
        true_relevance = y_true >= 4.0  # Consider ratings >= 4 as relevant
        pred_indices = np.argsort(y_pred)[::-1][:k]
        return np.mean(true_relevance[pred_indices])

    def _recall_at_k(self, y_true, y_pred, k=10):
        """Calculate recall@k for recommendations"""
        true_relevance = y_true >= 4.0
        pred_indices = np.argsort(y_pred)[::-1][:k]
        return np.sum(true_relevance[pred_indices]) / np.sum(true_relevance)

    def _ndcg_score(self, y_true, y_pred):
        """Calculate Normalized Discounted Cumulative Gain"""
        return ndcg_score([y_true], [y_pred])