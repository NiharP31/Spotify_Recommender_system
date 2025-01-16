import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
from typing import Dict, List, Optional
import time
import logging
from urllib.parse import urljoin

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and Configuration
API_URL = "http://localhost:8000"
TIMEOUT = 10  # seconds
CACHE_TTL = 3600  # 1 hour

class APIClient:
    """Handle all API communications"""
    def __init__(self, base_url: str, timeout: int = TIMEOUT):
        self.base_url = base_url
        self.timeout = timeout
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request with error handling"""
        try:
            url = urljoin(self.base_url, endpoint)
            kwargs['timeout'] = self.timeout
            
            # Print request details for debugging
            print(f"Making request to: {url}")
            print(f"Method: {method}")
            print(f"Headers: {kwargs.get('headers', {})}")
            print(f"Data: {kwargs.get('data', {})}")
            
            response = requests.request(method, url, **kwargs)
            
            # Print response details for debugging
            print(f"Response status: {response.status_code}")
            print(f"Response headers: {response.headers}")
            try:
                print(f"Response body: {response.json()}")
            except:
                print(f"Response text: {response.text}")
                
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            logger.error(f"API request failed: {str(e)}")
            if hasattr(e.response, 'text'):
                logger.error(f"Error response: {e.response.text}")
            raise
    
    def login(self, username: str, password: str) -> Dict:
        """Authenticate user"""
        try:
            # Use form-encoded data for OAuth2 password flow
            response = self._make_request(
                'POST',
                '/auth/token',
                data={
                    "username": username,
                    "password": password,
                    "grant_type": "password"  # Required for OAuth2 password flow
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded"
                }
            )
            return response.json()
        except requests.RequestException as e:
            print(f"Login error: {str(e)}")  # Debug print
            raise ValueError("Authentication failed")
    
    def register_user(self, email: str, username: str, password: str) -> bool:
        """Register a new user"""
        try:
            response = self._make_request(
                'POST',
                '/auth/register',
                json={"email": email, "username": username, "password": password}
            )
            return True
        except requests.RequestException:
            return False
    
    def get_recommendations(self, token: str, seed_tracks: List[str], n_recommendations: int) -> List[Dict]:
        """Get track recommendations"""
        try:
            response = self._make_request(
                'POST',
                '/api/recommendations',
                headers={"Authorization": f"Bearer {token}"},
                json={
                    "seed_tracks": seed_tracks,
                    "n_recommendations": n_recommendations
                }
            )
            return response.json()
        except requests.RequestException:
            raise ValueError("Failed to get recommendations")
    
    def get_track_features(self, token: str, track_id: str) -> Dict:
        """Get audio features for a track"""
        try:
            response = self._make_request(
                'GET',
                f'/api/tracks/{track_id}/features',
                headers={"Authorization": f"Bearer {token}"}
            )
            return response.json()
        except requests.RequestException:
            raise ValueError("Failed to get track features")
    
    def update_preference(self, token: str, track_id: str, rating: float) -> bool:
        """Update user preference for a track"""
        try:
            response = self._make_request(
                'POST',
                '/api/preferences/update',
                headers={"Authorization": f"Bearer {token}"},
                json={"track_id": track_id, "rating": rating}
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

class SpotifyRecommenderApp:
    def __init__(self):
        self.api_client = APIClient(API_URL)
        self._initialize_session_state()
        self._setup_page()
    
    def _initialize_session_state(self):
        """Initialize session state variables"""
        if 'token' not in st.session_state:
            st.session_state.token = None
        if 'username' not in st.session_state:
            st.session_state.username = None
        if 'recommendations_history' not in st.session_state:
            st.session_state.recommendations_history = []
    
    def _setup_page(self):
        """Configure page settings"""
        st.set_page_config(
            page_title="Spotify Music Recommender",
            page_icon="🎵",
            layout="wide"
        )
        st.title("🎵 Spotify Music Recommender")
    
    def register_user(self, email: str, username: str, password: str) -> bool:
        """Register a new user"""
        try:
            response = self._make_request(
                'POST',
                '/auth/register',
                json={"email": email, "username": username, "password": password}
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    def render_login_form(self):
        """Render login form"""
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            with st.form("login_form"):
                username = st.text_input("Username")
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button("Login")
                
                if submit:
                    try:
                        with st.spinner("Logging in..."):
                            result = self.api_client.login(username, password)
                            st.session_state.token = result["access_token"]
                            st.session_state.username = username
                            st.success("Logged in successfully!")
                            time.sleep(1)  # Give time for success message
                            st.rerun()
                    except ValueError as e:
                        st.error(f"Login failed: {str(e)}")
        
        with tab2:
            with st.form("register_form"):
                reg_email = st.text_input("Email")
                reg_username = st.text_input("Username")
                reg_password = st.text_input("Password", type="password")
                register_submit = st.form_submit_button("Register")
                
                if register_submit:
                    if not all([reg_email, reg_username, reg_password]):
                        st.error("Please fill in all fields")
                    else:
                        try:
                            with st.spinner("Registering..."):
                                response = self.api_client.register_user(reg_email, reg_username, reg_password)
                                if response:
                                    st.success("Registration successful! Please login.")
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error("Registration failed. Username or email might already exist.")
                        except ValueError as e:
                            st.error(f"Registration failed: {str(e)}")
    
    def render_recommendations_tab(self):
        """Render recommendations interface"""
        st.header("Get Recommendations")
        
        # Input for seed tracks with example
        st.write("Enter Spotify Track IDs (one per line)")
        st.write("Example track ID: 2plbrEY59IikOBgBGLjaoe (Lady Gaga - Die With A Smile)")
        
        seed_tracks_input = st.text_area(
            "Track IDs",
            help="Enter Spotify track IDs to use as seeds for recommendations"
        )
        
        col1, col2 = st.columns([2, 1])
        with col1:
            n_recommendations = st.slider(
                "Number of recommendations",
                min_value=1,
                max_value=20,
                value=5
            )
        
        if st.button("Get Recommendations", type="primary"):
            if not seed_tracks_input.strip():
                st.warning("Please enter at least one track ID")
                return
            
            # Clean and validate input
            seed_tracks = [track.strip() for track in seed_tracks_input.split('\n') if track.strip()]
            
            try:
                with st.spinner("Getting recommendations..."):
                    recommendations = self.api_client.get_recommendations(
                        st.session_state.token,
                        seed_tracks,
                        n_recommendations
                    )
                    
                    if recommendations:
                        st.success(f"Found {len(recommendations)} recommendations!")
                        self._display_recommendations(recommendations)
                    else:
                        st.warning("No recommendations found")
            
            except Exception as e:
                st.error(f"Error getting recommendations: {str(e)}")
                st.info("Try using a different track ID or check if the ID is valid")
    
    def _display_recommendations(self, recommendations: List[Dict]):
        """Display recommendation results"""
        st.write("### Recommended Tracks")
        
        for track in recommendations:
            with st.expander(f"🎵 {track['name']} by {track['artists']}"):
                cols = st.columns([3, 2, 1])
                
                with cols[0]:
                    st.write(f"**Album:** {track['album']}")
                    st.write(f"**Popularity:** {track['popularity']}/100")
                
                with cols[1]:
                    rating = st.slider(
                        "Rate this recommendation",
                        1.0, 5.0, 3.0,
                        key=f"rating_{track['id']}"
                    )
                
                with cols[2]:
                    if st.button("Save Rating", key=f"save_{track['id']}"):
                        try:
                            if self.api_client.update_preference(
                                st.session_state.token,
                                track['id'],
                                rating
                            ):
                                st.success("Rating saved!")
                            else:
                                st.error("Failed to save rating")
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                
                # Get and display track features
                try:
                    features = self.api_client.get_track_features(
                        st.session_state.token,
                        track['id']
                    )
                    if features:
                        self._display_track_features(features)
                except ValueError:
                    st.warning("Could not load track features")
    
    def _display_track_features(self, features: Dict):
        """Display track audio features"""
        # Prepare radar chart data
        feature_names = [
            'danceability', 'energy', 'speechiness', 
            'acousticness', 'instrumentalness', 'liveness', 
            'valence'
        ]
        
        feature_values = [features[name] for name in feature_names]
        
        # Create radar chart
        fig = go.Figure(data=go.Scatterpolar(
            r=feature_values,
            theta=feature_names,
            fill='toself',
            name='Audio Features'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display additional metrics
        cols = st.columns(4)
        cols[0].metric("Tempo", f"{features['tempo']:.0f} BPM")
        cols[1].metric("Key", features['key'])
        cols[2].metric("Loudness", f"{features['loudness']:.1f} dB")
        cols[3].metric("Mode", "Major" if features['mode'] else "Minor")
    
    def render_history_tab(self):
        """Render recommendation history"""
        st.header("Recommendation History")
        
        if not st.session_state.recommendations_history:
            st.info("No recommendation history yet")
            return
        
        for i, entry in enumerate(reversed(st.session_state.recommendations_history)):
            with st.expander(
                f"Recommendations from {datetime.fromisoformat(entry['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}",
                expanded=(i == 0)  # Expand most recent by default
            ):
                self._display_recommendations(entry['recommendations'])
    
    def render_profile_tab(self):
        """Render user profile tab"""
        st.header("My Profile")
        
        st.write(f"**Logged in as:** {st.session_state.username}")
        
        if st.button("Logout", type="primary"):
            st.session_state.token = None
            st.session_state.username = None
            st.session_state.recommendations_history = []
            st.rerun()
    
    def run(self):
        """Main app execution"""
        if not st.session_state.token:
            self.render_login_form()
        else:
            # Main navigation
            tab1, tab2, tab3 = st.tabs([
                "Get Recommendations",
                "Recommendation History",
                "My Profile"
            ])
            
            with tab1:
                self.render_recommendations_tab()
            
            with tab2:
                self.render_history_tab()
            
            with tab3:
                self.render_profile_tab()
            
            # Footer
            st.markdown("---")
            st.markdown(
                "Made with ❤️ using FastAPI, Streamlit, and Machine Learning"
            )

if __name__ == "__main__":
    app = SpotifyRecommenderApp()
    app.run()