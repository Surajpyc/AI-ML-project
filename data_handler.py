import pandas as pd
import numpy as np
import requests
import io
from typing import Tuple, Optional

class DataHandler:
    """
    Handles data loading, processing, and basic operations for the movie recommendation system
    """
    
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        self.original_movies_df = None
        self.original_ratings_df = None
    
    def load_data_from_file(self, uploaded_file) -> Tuple[bool, str]:
        """
        Load movie data from uploaded CSV file
        
        Args:
            uploaded_file: Streamlit uploaded file object
            
        Returns:
            Tuple[bool, str]: Success flag and message
        """
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Validate and process the data
            success, message = self._process_uploaded_data(df)
            
            if success:
                return True, f"Successfully loaded {len(self.movies_df)} movies"
            else:
                return False, message
                
        except Exception as e:
            return False, f"Error reading file: {str(e)}"
    
    def load_data_from_url(self, url: str) -> Tuple[bool, str]:
        """
        Load movie data from URL
        
        Args:
            url (str): URL to CSV file
            
        Returns:
            Tuple[bool, str]: Success flag and message
        """
        try:
            # Download the data
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Read CSV from response content
            df = pd.read_csv(io.StringIO(response.text))
            
            # Validate and process the data
            success, message = self._process_uploaded_data(df)
            
            if success:
                return True, f"Successfully loaded {len(self.movies_df)} movies from URL"
            else:
                return False, message
                
        except requests.exceptions.RequestException as e:
            return False, f"Error downloading data: {str(e)}"
        except Exception as e:
            return False, f"Error processing data: {str(e)}"
    
    def _process_uploaded_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """
        Process and validate uploaded data
        
        Args:
            df (pd.DataFrame): Raw data from CSV
            
        Returns:
            Tuple[bool, str]: Success flag and message
        """
        try:
            # Store original data
            self.original_movies_df = df.copy()
            
            # Check for required columns
            required_columns = ['title']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                return False, f"Missing required columns: {missing_columns}"
            
            # Process movies data
            self.movies_df = df.copy()
            
            # Ensure movieId exists
            if 'movieId' not in self.movies_df.columns:
                self.movies_df['movieId'] = range(len(self.movies_df))
            
            # Handle genres
            if 'genres' not in self.movies_df.columns:
                self.movies_df['genres'] = 'Unknown'
            
            # Handle ratings
            if 'rating' not in self.movies_df.columns:
                # Try to find rating-like columns
                rating_cols = [col for col in df.columns if 'rating' in col.lower() or 'score' in col.lower()]
                if rating_cols:
                    self.movies_df['rating'] = df[rating_cols[0]]
                else:
                    self.movies_df['rating'] = 3.0  # Default rating
            
            # Clean and validate ratings
            if 'rating' in self.movies_df.columns:
                self.movies_df['rating'] = pd.to_numeric(self.movies_df['rating'], errors='coerce')
                self.movies_df['rating'] = self.movies_df['rating'].fillna(3.0)
                # Normalize ratings to 0-5 scale
                max_rating = self.movies_df['rating'].max()
                if max_rating > 5:
                    self.movies_df['rating'] = (self.movies_df['rating'] / max_rating) * 5
            
            # Check for ratings data (separate from movies)
            if 'userId' in df.columns and len(df.columns) > 3:
                # This might be a ratings dataset
                self.ratings_df = df.copy()
                self.original_ratings_df = df.copy()
                
                # Ensure required columns for ratings
                if 'rating' not in self.ratings_df.columns:
                    rating_cols = [col for col in df.columns if 'rating' in col.lower() or 'score' in col.lower()]
                    if rating_cols:
                        self.ratings_df['rating'] = df[rating_cols[0]]
                    else:
                        self.ratings_df = None
            
            # Remove duplicates
            self.movies_df = self.movies_df.drop_duplicates(subset=['title'])
            
            # Reset index
            self.movies_df.reset_index(drop=True, inplace=True)
            if self.ratings_df is not None:
                self.ratings_df.reset_index(drop=True, inplace=True)
            
            return True, "Data processed successfully"
            
        except Exception as e:
            return False, f"Error processing data: {str(e)}"
    
    def search_movies(self, query: str, search_type: str = "both") -> pd.DataFrame:
        """
        Search for movies based on title or genre
        
        Args:
            query (str): Search query
            search_type (str): Type of search ("title", "genre", "both")
            
        Returns:
            pd.DataFrame: Matching movies
        """
        if self.movies_df is None or query.strip() == "":
            return pd.DataFrame()
        
        query_lower = query.lower()
        
        try:
            if search_type == "title":
                mask = self.movies_df['title'].str.lower().str.contains(query_lower, na=False, regex=False)
            elif search_type == "genre":
                mask = self.movies_df['genres'].str.lower().str.contains(query_lower, na=False, regex=False)
            else:  # both
                title_mask = self.movies_df['title'].str.lower().str.contains(query_lower, na=False, regex=False)
                genre_mask = self.movies_df['genres'].str.lower().str.contains(query_lower, na=False, regex=False)
                mask = title_mask | genre_mask
            
            results = self.movies_df[mask].copy()
            
            # Sort by rating (descending) and then by title
            if 'rating' in results.columns:
                results = results.sort_values(['rating', 'title'], ascending=[False, True])
            else:
                results = results.sort_values('title')
            
            return results
            
        except Exception as e:
            print(f"Error searching movies: {e}")
            return pd.DataFrame()
    
    def get_top_movies(self, num_movies: int = 20, genre_filter: Optional[str] = None, 
                      min_rating: float = 0.0) -> pd.DataFrame:
        """
        Get top-rated movies
        
        Args:
            num_movies (int): Number of movies to return
            genre_filter (str, optional): Filter by genre
            min_rating (float): Minimum rating threshold
            
        Returns:
            pd.DataFrame: Top-rated movies
        """
        if self.movies_df is None:
            return pd.DataFrame()
        
        try:
            df = self.movies_df.copy()
            
            # Apply genre filter
            if genre_filter:
                df = df[df['genres'].str.contains(genre_filter, case=False, na=False)]
            
            # Apply rating filter
            if 'rating' in df.columns:
                df = df[df['rating'] >= min_rating]
                # Sort by rating (descending)
                df = df.sort_values('rating', ascending=False)
            else:
                # Sort by title if no rating
                df = df.sort_values('title')
            
            return df.head(num_movies)
            
        except Exception as e:
            print(f"Error getting top movies: {e}")
            return pd.DataFrame()
    
    def get_unique_genres(self) -> list:
        """
        Get list of unique genres
        
        Returns:
            list: List of unique genres
        """
        if self.movies_df is None or 'genres' not in self.movies_df.columns:
            return []
        
        try:
            # Split genres and flatten the list
            all_genres = []
            for genres_str in self.movies_df['genres'].dropna():
                if '|' in str(genres_str):
                    genres = str(genres_str).split('|')
                else:
                    genres = [str(genres_str)]
                all_genres.extend([g.strip() for g in genres if g.strip()])
            
            # Return unique genres sorted
            unique_genres = sorted(list(set(all_genres)))
            return [g for g in unique_genres if g.lower() not in ['unknown', '', 'n/a']]
            
        except Exception as e:
            print(f"Error getting genres: {e}")
            return []
    
    def get_dataset_stats(self) -> dict:
        """
        Get basic statistics about the dataset
        
        Returns:
            dict: Dataset statistics
        """
        stats = {}
        
        if self.movies_df is not None:
            stats['Total Movies'] = len(self.movies_df)
            stats['Unique Genres'] = len(self.get_unique_genres())
            
            if 'rating' in self.movies_df.columns:
                stats['Average Rating'] = round(self.movies_df['rating'].mean(), 1)
                stats['Highest Rated'] = round(self.movies_df['rating'].max(), 1)
                stats['Lowest Rated'] = round(self.movies_df['rating'].min(), 1)
        
        if self.ratings_df is not None:
            stats['Total Ratings'] = len(self.ratings_df)
            if 'userId' in self.ratings_df.columns:
                stats['Unique Users'] = self.ratings_df['userId'].nunique()
        
        return stats
    
    def get_genre_distribution(self) -> pd.Series:
        """
        Get distribution of movies by genre
        
        Returns:
            pd.Series: Genre distribution
        """
        if self.movies_df is None or 'genres' not in self.movies_df.columns:
            return pd.Series()
        
        try:
            # Count movies for each genre
            genre_counts = {}
            
            for genres_str in self.movies_df['genres'].dropna():
                if '|' in str(genres_str):
                    genres = str(genres_str).split('|')
                else:
                    genres = [str(genres_str)]
                
                for genre in genres:
                    genre = genre.strip()
                    if genre and genre.lower() not in ['unknown', '', 'n/a']:
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1
            
            # Convert to Series and sort
            genre_series = pd.Series(genre_counts).sort_values(ascending=False)
            return genre_series
            
        except Exception as e:
            print(f"Error getting genre distribution: {e}")
            return pd.Series()
    
    def get_movie_by_id(self, movie_id: int) -> Optional[pd.Series]:
        """
        Get movie details by ID
        
        Args:
            movie_id (int): Movie ID
            
        Returns:
            pd.Series or None: Movie details
        """
        if self.movies_df is None:
            return None
        
        matches = self.movies_df[self.movies_df['movieId'] == movie_id]
        
        if not matches.empty:
            return matches.iloc[0]
        else:
            return None
    
    def get_random_movies(self, num_movies: int = 10) -> pd.DataFrame:
        """
        Get random movies for exploration
        
        Args:
            num_movies (int): Number of random movies to return
            
        Returns:
            pd.DataFrame: Random movies
        """
        if self.movies_df is None:
            return pd.DataFrame()
        
        try:
            sample_size = min(num_movies, len(self.movies_df))
            return self.movies_df.sample(n=sample_size)
        except Exception as e:
            print(f"Error getting random movies: {e}")
            return pd.DataFrame()
