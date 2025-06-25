import streamlit as st
import pandas as pd
import numpy as np

def display_movie_card(movie, show_similarity=False, show_rank=None):
    """
    Display a movie card with information
    
    Args:
        movie (pd.Series): Movie data
        show_similarity (bool): Whether to show similarity score
        show_rank (int, optional): Rank number to display
    """
    with st.container():
        # Movie title with rank if provided
        title = movie.get('title', 'Unknown Title')
        if show_rank:
            st.markdown(f"### #{show_rank} {title}")
        else:
            st.markdown(f"### {title}")
        
        # Rating
        rating = movie.get('rating', 0)
        if rating > 0:
            st.markdown(f"⭐ **Rating:** {format_rating(rating)}")
        else:
            st.markdown("⭐ **Rating:** Not available")
        
        # Genres
        genres = movie.get('genres', 'Unknown')
        if genres and genres != 'Unknown':
            # Format genres nicely
            if '|' in genres:
                genre_list = genres.split('|')
                formatted_genres = ', '.join(genre_list[:3])  # Show max 3 genres
                if len(genre_list) > 3:
                    formatted_genres += f" (+{len(genre_list) - 3} more)"
            else:
                formatted_genres = genres
            st.markdown(f"🎭 **Genres:** {formatted_genres}")
        else:
            st.markdown("🎭 **Genres:** Not specified")
        
        # Movie ID
        if 'movieId' in movie:
            st.markdown(f"🆔 **ID:** {movie['movieId']}")
        
        # Similarity score if provided
        if show_similarity and 'similarity_score' in movie:
            similarity = movie['similarity_score']
            st.markdown(f"🔍 **Match:** {similarity:.2%}")
        
        # Add separator
        st.divider()

def format_rating(rating):
    """
    Format rating with stars
    
    Args:
        rating (float): Rating value
        
    Returns:
        str: Formatted rating string
    """
    if pd.isna(rating) or rating == 0:
        return "Not rated"
    
    try:
        rating = float(rating)
        stars = "⭐" * int(rating)
        return f"{rating:.1f}/5.0 {stars}"
    except (ValueError, TypeError):
        return "Invalid rating"

def create_rating_histogram(ratings_data):
    """
    Create a histogram of ratings
    
    Args:
        ratings_data (pd.Series or list): Rating values
        
    Returns:
        dict: Histogram data for Streamlit
    """
    if isinstance(ratings_data, pd.Series):
        ratings_data = ratings_data.tolist()
    
    # Create bins for ratings (0.5 intervals)
    bins = np.arange(0, 5.5, 0.5)
    hist, _ = np.histogram(ratings_data, bins=bins)
    
    return {
        'bins': [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)],
        'counts': hist.tolist()
    }

def get_color_for_rating(rating):
    """
    Get color code based on rating
    
    Args:
        rating (float): Rating value
        
    Returns:
        str: Color code
    """
    try:
        rating = float(rating)
        if rating >= 4.5:
            return "#4CAF50"  # Green
        elif rating >= 4.0:
            return "#8BC34A"  # Light Green
        elif rating >= 3.5:
            return "#FFC107"  # Amber
        elif rating >= 3.0:
            return "#FF9800"  # Orange
        elif rating >= 2.0:
            return "#FF5722"  # Deep Orange
        else:
            return "#F44336"  # Red
    except (ValueError, TypeError):
        return "#9E9E9E"  # Grey for invalid ratings

def format_genre_tags(genres_str):
    """
    Format genres as tags
    
    Args:
        genres_str (str): Pipe-separated genres string
        
    Returns:
        str: HTML formatted genre tags
    """
    if not genres_str or pd.isna(genres_str) or genres_str == 'Unknown':
        return '<span style="background: #f0f0f0; padding: 2px 6px; border-radius: 12px; font-size: 0.8em;">Unknown</span>'
    
    try:
        if '|' in genres_str:
            genres = genres_str.split('|')
        else:
            genres = [genres_str]
        
        tags = []
        colors = ['#e3f2fd', '#f3e5f5', '#e8f5e8', '#fff3e0', '#fce4ec']
        
        for i, genre in enumerate(genres[:5]):  # Show max 5 genres
            color = colors[i % len(colors)]
            tag = f'<span style="background: {color}; padding: 2px 6px; border-radius: 12px; font-size: 0.8em; margin-right: 4px;">{genre.strip()}</span>'
            tags.append(tag)
        
        if len(genres) > 5:
            tags.append(f'<span style="background: #f5f5f5; padding: 2px 6px; border-radius: 12px; font-size: 0.8em;">+{len(genres) - 5} more</span>')
        
        return ''.join(tags)
    except:
        return '<span style="background: #f0f0f0; padding: 2px 6px; border-radius: 12px; font-size: 0.8em;">Unknown</span>'

def calculate_recommendation_confidence(similarity_score):
    """
    Calculate confidence level for recommendations
    
    Args:
        similarity_score (float): Similarity score between 0 and 1
        
    Returns:
        tuple: (confidence_level, confidence_text, confidence_color)
    """
    try:
        score = float(similarity_score)
        
        if score >= 0.8:
            return (5, "Very High", "#4CAF50")
        elif score >= 0.6:
            return (4, "High", "#8BC34A")
        elif score >= 0.4:
            return (3, "Medium", "#FFC107")
        elif score >= 0.2:
            return (2, "Low", "#FF9800")
        else:
            return (1, "Very Low", "#F44336")
    except (ValueError, TypeError):
        return (0, "Unknown", "#9E9E9E")

def safe_get_value(series_or_dict, key, default="N/A"):
    """
    Safely get value from pandas Series or dictionary
    
    Args:
        series_or_dict: pandas Series or dictionary
        key: key to access
        default: default value if key not found
        
    Returns:
        Value or default
    """
    try:
        if isinstance(series_or_dict, pd.Series):
            return series_or_dict.get(key, default)
        elif isinstance(series_or_dict, dict):
            return series_or_dict.get(key, default)
        else:
            return default
    except:
        return default

def truncate_text(text, max_length=50):
    """
    Truncate text to specified length
    
    Args:
        text (str): Text to truncate
        max_length (int): Maximum length
        
    Returns:
        str: Truncated text
    """
    if not text or pd.isna(text):
        return "N/A"
    
    text = str(text)
    if len(text) <= max_length:
        return text
    else:
        return text[:max_length-3] + "..."

def validate_rating_input(rating):
    """
    Validate rating input
    
    Args:
        rating: Rating value to validate
        
    Returns:
        tuple: (is_valid, validated_rating, error_message)
    """
    try:
        rating = float(rating)
        if 0 <= rating <= 5:
            return (True, rating, None)
        else:
            return (False, None, "Rating must be between 0 and 5")
    except (ValueError, TypeError):
        return (False, None, "Rating must be a number")

def create_progress_bar(value, max_value=5, color="#4CAF50"):
    """
    Create HTML progress bar
    
    Args:
        value (float): Current value
        max_value (float): Maximum value
        color (str): Progress bar color
        
    Returns:
        str: HTML progress bar
    """
    try:
        percentage = (float(value) / float(max_value)) * 100
        return f"""
        <div style="background-color: #f0f0f0; border-radius: 10px; padding: 2px;">
            <div style="background-color: {color}; width: {percentage}%; height: 20px; border-radius: 8px; text-align: center; color: white; font-size: 12px; line-height: 20px;">
                {value:.1f}/{max_value}
            </div>
        </div>
        """
    except:
        return f"<div>{value}/{max_value}</div>"
