# Spotify Music Recommendation System

A machine learning-powered music recommendation system built with FastAPI and Streamlit, utilizing the Spotify API for music data and recommendations.

## Features

- User authentication and registration
- Personalized music recommendations based on seed tracks
- Track audio feature analysis and visualization
- User preference tracking
- Interactive web interface
- Real-time music recommendations

## Tech Stack

- **Backend**: FastAPI, SQLAlchemy, Pydantic
- **Frontend**: Streamlit
- **Database**: SQLite
- **Authentication**: JWT
- **APIs**: Spotify Web API
- **Data Science**: Pandas, NumPy
- **Visualization**: Plotly

## Project Structure

```
spotify_recommendation_system/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── database.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database_models.py
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── auth.py
│   │   ├── recommendations.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── recommendation_service.py
│   └── models_store/
│       └── .gitkeep
├── streamlit_app.py
└── requirements.txt
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/spotify-recommendation-system.git
cd spotify-recommendation-system
```

2. Create and activate a virtual environment:
```bash
conda create -n env python=3.8+
conda activate env 
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your Spotify API credentials:
```env
SPOTIFY_CLIENT_ID=your_client_id_here
SPOTIFY_CLIENT_SECRET=your_client_secret_here
DATABASE_URL=sqlite:///./spotify_recommender.db
SECRET_KEY=your_secret_key_here
```

## Running the Application

1. Start the FastAPI backend:
```bash
uvicorn app.main:app --reload
```

2. In a new terminal, start the Streamlit frontend:
```bash
streamlit run streamlit_app.py
```

3. Access the application:
- Frontend: http://localhost:8501
- API documentation: http://localhost:8000/docs

## Usage

1. Register a new account using the registration form
2. Log in with your credentials
3. Enter a Spotify track ID to get recommendations
   - You can get track IDs from Spotify by right-clicking a song and selecting "Share > Copy Spotify URI"
   - The track ID is the string after "spotify:track:"
4. View and Rate recommendations 

## Development

- API development: FastAPI endpoints are in `app/routers/`
- Database models: SQLAlchemy models in `app/models/`
- Business logic: Services in `app/services/`
- Frontend: Streamlit interface in `streamlit_app.py`

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Spotify Web API](https://developer.spotify.com/documentation/web-api/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Streamlit](https://streamlit.io/)

