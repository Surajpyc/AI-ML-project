import streamlit as st
import pandas as pd
import numpy as np
from recommendation_engine import MovieRecommendationEngine
from data_handler import DataHandler
from utils import display_movie_card, format_rating
import os
import time

# Configure page
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'engine' not in st.session_state:
    st.session_state.engine = None
if 'data_handler' not in st.session_state:
    st.session_state.data_handler = DataHandler()
if 'user_ratings' not in st.session_state:
    st.session_state.user_ratings = {}
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = pd.DataFrame()

def main():
    st.title("🎬 Movie Recommendation System")
    st.markdown("### Discover your next favorite movie with AI-powered recommendations")
    
    # Sidebar for data loading and user ratings
    with st.sidebar:
        st.header("📊 Data Management")
        
        # Data loading section
        st.subheader("Load Movie Data")
        
        # Option 1: Upload CSV file
        uploaded_file = st.file_uploader(
            "Upload Movies CSV",
            type=['csv'],
            help="Upload a CSV file with columns: movieId, title, genres, and optionally: rating, userId"
        )
        
        # Option 2: Use sample data URL
        st.markdown("**Or use online dataset:**")
        dataset_url = st.text_input(
            "Dataset URL",
            placeholder="https://example.com/movies.csv",
            help="Enter URL to a CSV file with movie data"
        )
        
        # Load data button
        if st.button("Load Data", type="primary"):
            with st.spinner("Loading movie data..."):
                try:
                    if uploaded_file is not None:
                        success, message = st.session_state.data_handler.load_data_from_file(uploaded_file)
                    elif dataset_url:
                        success, message = st.session_state.data_handler.load_data_from_url(dataset_url)
                    else:
                        success, message = False, "Please upload a file or provide a URL"
                    
                    if success:
                        st.success(message)
                        # Initialize recommendation engine
                        st.session_state.engine = MovieRecommendationEngine(
                            st.session_state.data_handler.movies_df,
                            st.session_state.data_handler.ratings_df
                        )
                        st.rerun()
                    else:
                        st.error(message)
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
        
        # Display data info if loaded
        if st.session_state.data_handler.movies_df is not None:
            st.success("✅ Data loaded successfully!")
            st.info(f"📽️ Movies: {len(st.session_state.data_handler.movies_df)}")
            if st.session_state.data_handler.ratings_df is not None:
                st.info(f"⭐ Ratings: {len(st.session_state.data_handler.ratings_df)}")
        
        st.divider()
        
        # User rating section
        st.subheader("⭐ Rate Movies")
        if st.session_state.data_handler.movies_df is not None:
            # Movie selection for rating
            movie_titles = st.session_state.data_handler.movies_df['title'].tolist()
            selected_movie = st.selectbox(
                "Select a movie to rate:",
                [""] + movie_titles,
                key="movie_selector"
            )
            
            if selected_movie:
                rating = st.slider(
                    f"Rate '{selected_movie}'",
                    min_value=0.5,
                    max_value=5.0,
                    step=0.5,
                    value=3.0
                )
                
                if st.button("Add Rating"):
                    st.session_state.user_ratings[selected_movie] = rating
                    st.success(f"Added rating: {rating}⭐ for '{selected_movie}'")
                    st.rerun()
            
            # Display user ratings
            if st.session_state.user_ratings:
                st.markdown("**Your Ratings:**")
                for movie, rating in st.session_state.user_ratings.items():
                    st.write(f"• {movie}: {format_rating(rating)}")
        else:
            st.info("Load movie data first to start rating movies")
    
    # Main content area
    if st.session_state.data_handler.movies_df is None:
        # Welcome screen when no data is loaded
        st.markdown("""
        ## Welcome to the Movie Recommendation System! 🍿
        
        To get started:
        1. **Load your movie data** using the sidebar:
           - Upload a CSV file with movie information
           - Or provide a URL to an online dataset
        2. **Rate some movies** to get personalized recommendations
        3. **Explore** recommendations and discover new favorites!
        
        ### Expected Data Format:
        Your CSV should contain these columns:
        - `movieId`: Unique identifier for each movie
        - `title`: Movie title
        - `genres`: Movie genres (pipe-separated for multiple genres)
        - `rating` (optional): Average rating
        - `userId` (optional): For collaborative filtering
        """)
        
        st.info("💡 **Tip**: You can find movie datasets on sites like Kaggle, MovieLens, or TMDB")
        
    else:
        # Main application tabs
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Recommendations", "🔍 Search Movies", "🏆 Top Rated", "📊 Analytics"])
        
        with tab1:
            st.header("Personalized Recommendations")
            
            if not st.session_state.user_ratings:
                st.info("Rate some movies in the sidebar to get personalized recommendations!")
            else:
                col1, col2 = st.columns([3, 1])
                
                with col2:
                    num_recommendations = st.selectbox(
                        "Number of recommendations:",
                        [5, 10, 15, 20],
                        index=1
                    )
                    
                    if st.button("Get Recommendations", type="primary"):
                        with st.spinner("Generating recommendations..."):
                            try:
                                if st.session_state.engine:
                                    recommendations = st.session_state.engine.get_user_recommendations(
                                        st.session_state.user_ratings,
                                        num_recommendations
                                    )
                                    st.session_state.recommendations = recommendations
                                else:
                                    st.error("Recommendation engine not initialized")
                            except Exception as e:
                                st.error(f"Error generating recommendations: {str(e)}")
                
                with col1:
                    if hasattr(st.session_state, 'recommendations') and not st.session_state.recommendations.empty:
                        st.subheader(f"🎬 Your Top {len(st.session_state.recommendations)} Recommendations")
                        
                        # Display recommendations
                        for idx, (_, movie) in enumerate(st.session_state.recommendations.iterrows()):
                            display_movie_card(movie, show_similarity=True)
                    else:
                        st.info("Click 'Get Recommendations' to see your personalized movie suggestions!")
        
        with tab2:
            st.header("Search Movies")
            
            # Search functionality
            col1, col2 = st.columns([3, 1])
            
            with col1:
                search_query = st.text_input(
                    "Search for movies:",
                    placeholder="Enter movie title, genre, or keyword...",
                    key="search_input"
                )
            
            with col2:
                search_type = st.selectbox(
                    "Search by:",
                    ["Title", "Genre", "Both"]
                )
            
            if search_query:
                with st.spinner("Searching..."):
                    results = st.session_state.data_handler.search_movies(search_query, search_type.lower())
                
                if not results.empty:
                    st.success(f"Found {len(results)} movies")
                    
                    # Display search results
                    for idx, (_, movie) in enumerate(results.iterrows()):
                        display_movie_card(movie)
                        
                        # Quick rating option
                        rating_key = f"quick_rate_{movie.get('movieId', idx)}"
                        quick_rating = st.slider(
                            f"Quick rate",
                            min_value=0.5,
                            max_value=5.0,
                            step=0.5,
                            value=3.0,
                            key=rating_key
                        )
                        
                        if st.button(f"Rate", key=f"rate_btn_{movie.get('movieId', idx)}"):
                            st.session_state.user_ratings[movie['title']] = quick_rating
                            st.success(f"Rated '{movie['title']}': {format_rating(quick_rating)}")
                            time.sleep(1)
                            st.rerun()
                else:
                    st.warning("No movies found matching your search criteria")
        
        with tab3:
            st.header("Top Rated Movies")
            
            # Top movies controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                num_top_movies = st.selectbox(
                    "Number of movies:",
                    [10, 20, 50, 100],
                    index=1
                )
            
            with col2:
                genre_filter = st.selectbox(
                    "Filter by genre:",
                    ["All Genres"] + st.session_state.data_handler.get_unique_genres()
                )
            
            with col3:
                min_rating = st.slider(
                    "Minimum rating:",
                    min_value=0.0,
                    max_value=5.0,
                    value=3.0,
                    step=0.1
                )
            
            # Get and display top movies
            top_movies = st.session_state.data_handler.get_top_movies(
                num_movies=num_top_movies,
                genre_filter=genre_filter if genre_filter != "All Genres" else None,
                min_rating=min_rating
            )
            
            if not top_movies.empty:
                st.subheader(f"🏆 Top {len(top_movies)} Movies")
                
                # Display top movies
                for idx, (_, movie) in enumerate(top_movies.iterrows()):
                    display_movie_card(movie, show_rank=idx + 1)
            else:
                st.warning("No movies found matching the specified criteria")
        
        with tab4:
            st.header("Analytics Dashboard")
            
            # Analytics section
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Dataset Overview")
                
                # Basic statistics
                stats = st.session_state.data_handler.get_dataset_stats()
                for key, value in stats.items():
                    st.metric(key.replace('_', ' ').title(), value)
            
            with col2:
                st.subheader("🎭 Genre Distribution")
                
                # Genre distribution
                genre_dist = st.session_state.data_handler.get_genre_distribution()
                if not genre_dist.empty:
                    st.bar_chart(genre_dist.head(10))
                else:
                    st.info("No genre data available")
            
            # User rating analysis
            if st.session_state.user_ratings:
                st.subheader("⭐ Your Rating Analysis")
                
                try:
                    ratings_list = []
                    for movie, rating in st.session_state.user_ratings.items():
                        ratings_list.append({'Movie': movie, 'Rating': rating})
                    
                    if ratings_list:
                        user_df = pd.DataFrame(ratings_list)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Movies Rated", len(user_df))
                        
                        with col2:
                            avg_rating = user_df['Rating'].mean()
                            st.metric("Average Rating", f"{avg_rating:.1f}")
                        
                        with col3:
                            mode_rating = user_df['Rating'].mode()
                            if not mode_rating.empty:
                                st.metric("Most Common Rating", f"{mode_rating.iloc[0]:.1f}")
                            else:
                                st.metric("Most Common Rating", "N/A")
                        
                        # Rating distribution
                        rating_chart_data = user_df.set_index('Movie')['Rating']
                        st.bar_chart(rating_chart_data)
                    else:
                        st.info("No ratings to analyze yet")
                except Exception as e:
                    st.error(f"Error displaying rating analysis: {str(e)}")

if __name__ == "__main__":
    main()
