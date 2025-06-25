import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

warnings.filterwarnings('ignore')


class MovieRecommendationEngine:
    """
    Movie recommendation engine using collaborative filtering and content-based filtering
    """

    def __init__(self, movies_df, ratings_df=None):
        """
        Initialize the recommendation engine
        
        Args:
            movies_df (pd.DataFrame): DataFrame with movie information
            ratings_df (pd.DataFrame, optional): DataFrame with user ratings
        """
        self.movies_df = movies_df.copy()
        self.ratings_df = ratings_df.copy() if ratings_df is not None else None
        self.similarity_matrix = None
        self.content_similarity_matrix = None
        self.user_item_matrix = None

        # Prepare the data
        self._prepare_data()
        self._build_content_similarity()
        if self.ratings_df is not None:
            self._build_collaborative_filtering()

    def _prepare_data(self):
        """Prepare and clean the movie data"""
        # Ensure required columns exist
        required_cols = ['movieId', 'title']
        for col in required_cols:
            if col not in self.movies_df.columns:
                if col == 'movieId':
                    self.movies_df['movieId'] = range(len(self.movies_df))
                else:
                    raise ValueError(
                        f"Required column '{col}' not found in movies data")

        # Handle genres
        if 'genres' not in self.movies_df.columns:
            self.movies_df['genres'] = 'Unknown'

        # Clean genres (replace pipe separators with spaces for TF-IDF)
        self.movies_df['genres_cleaned'] = self.movies_df[
            'genres'].str.replace('|', ' ')

        # Handle ratings if not present
        if 'rating' not in self.movies_df.columns:
            if self.ratings_df is not None and 'rating' in self.ratings_df.columns:
                # Calculate average ratings from ratings_df
                avg_ratings = self.ratings_df.groupby(
                    'movieId')['rating'].mean().reset_index()
                self.movies_df = self.movies_df.merge(avg_ratings,
                                                      on='movieId',
                                                      how='left')
                self.movies_df['rating'] = self.movies_df['rating'].fillna(3.0)
            else:
                self.movies_df['rating'] = 3.0  # Default rating

    def _build_content_similarity(self):
        """Build content-based similarity matrix using genres"""
        try:
            # Create TF-IDF matrix for genres
            tfidf = TfidfVectorizer(stop_words='english', lowercase=True)
            tfidf_matrix = tfidf.fit_transform(
                self.movies_df['genres_cleaned'].fillna(''))

            # Calculate cosine similarity
            self.content_similarity_matrix = cosine_similarity(tfidf_matrix)

        except Exception as e:
            print(f"Error building content similarity: {e}")
            # Fallback: create identity matrix
            n_movies = len(self.movies_df)
            self.content_similarity_matrix = np.eye(n_movies)

    def _build_collaborative_filtering(self):
        """Build collaborative filtering model"""
        if self.ratings_df is None or len(self.ratings_df) == 0:
            return

        try:
            # Create user-item matrix
            self.user_item_matrix = self.ratings_df.pivot_table(
                index='userId',
                columns='movieId',
                values='rating',
                fill_value=0)

            # Calculate item-item similarity
            # Transpose to get item-item relationships
            item_matrix = self.user_item_matrix.T.values

            # Handle case where there are no ratings
            if item_matrix.size > 0:
                self.similarity_matrix = cosine_similarity(item_matrix)
            else:
                self.similarity_matrix = None

        except Exception as e:
            print(f"Error building collaborative filtering model: {e}")
            self.similarity_matrix = None

    def get_content_based_recommendations(self,
                                          movie_title,
                                          num_recommendations=10):
        """
        Get content-based recommendations for a given movie
        
        Args:
            movie_title (str): Title of the movie
            num_recommendations (int): Number of recommendations to return
            
        Returns:
            pd.DataFrame: Recommended movies
        """
        try:
            # Find the movie index
            movie_idx = self.movies_df[self.movies_df['title'].str.contains(
                movie_title, case=False, na=False)].index

            if len(movie_idx) == 0:
                return pd.DataFrame()

            movie_idx = movie_idx[0]

            # Get similarity scores
            similarity_scores = list(
                enumerate(self.content_similarity_matrix[movie_idx]))

            # Sort by similarity (excluding the movie itself)
            similarity_scores = sorted(similarity_scores,
                                       key=lambda x: x[1],
                                       reverse=True)[1:]

            # Get top recommendations
            movie_indices = [
                i[0] for i in similarity_scores[:num_recommendations]
            ]

            recommendations = self.movies_df.iloc[movie_indices].copy()
            recommendations['similarity_score'] = [
                i[1] for i in similarity_scores[:num_recommendations]
            ]

            return recommendations

        except Exception as e:
            print(f"Error getting content-based recommendations: {e}")
            return pd.DataFrame()

    def get_collaborative_recommendations(self,
                                          user_ratings,
                                          num_recommendations=10):
        """
        Get collaborative filtering recommendations based on user ratings
        
        Args:
            user_ratings (dict): Dictionary of {movie_title: rating}
            num_recommendations (int): Number of recommendations to return
            
        Returns:
            pd.DataFrame: Recommended movies
        """
        if self.similarity_matrix is None or len(user_ratings) == 0:
            return pd.DataFrame()

        try:
            # Convert movie titles to movie IDs
            user_movie_ids = {}
            for title, rating in user_ratings.items():
                movie_matches = self.movies_df[
                    self.movies_df['title'].str.contains(title,
                                                         case=False,
                                                         na=False)]
                if not movie_matches.empty:
                    movie_id = movie_matches.iloc[0]['movieId']
                    user_movie_ids[movie_id] = rating

            if not user_movie_ids:
                return pd.DataFrame()

            # Create user profile vector
            movie_ids = self.movies_df['movieId'].tolist()
            user_vector = np.zeros(len(movie_ids))

            for movie_id, rating in user_movie_ids.items():
                if movie_id in movie_ids:
                    idx = movie_ids.index(movie_id)
                    user_vector[idx] = rating

            # Calculate weighted scores for all movies
            scores = np.dot(self.similarity_matrix, user_vector)

            # Create movie scores dataframe
            movie_scores = pd.DataFrame({
                'movieId': movie_ids,
                'score': scores
            })

            # Merge with movie information
            recommendations = movie_scores.merge(self.movies_df, on='movieId')

            # Filter out movies the user has already rated
            rated_movie_ids = list(user_movie_ids.keys())
            recommendations = recommendations[~recommendations['movieId'].
                                              isin(rated_movie_ids)]

            # Sort by score and return top recommendations
            recommendations = recommendations.sort_values(
                'score', ascending=False).head(num_recommendations)
            recommendations['similarity_score'] = recommendations['score']

            return recommendations

        except Exception as e:
            print(f"Error getting collaborative recommendations: {e}")
            return pd.DataFrame()

    def get_user_recommendations(self, user_ratings, num_recommendations=10):
        """
        Get hybrid recommendations combining content-based and collaborative filtering
        
        Args:
            user_ratings (dict): Dictionary of {movie_title: rating}
            num_recommendations (int): Number of recommendations to return
            
        Returns:
            pd.DataFrame: Recommended movies
        """
        if not user_ratings:
            # If no user ratings, return top-rated movies
            return self.movies_df.nlargest(num_recommendations, 'rating')

        # Get collaborative filtering recommendations
        collab_recs = self.get_collaborative_recommendations(
            user_ratings, num_recommendations * 2)

        # If collaborative filtering fails or returns few results, use content-based
        if collab_recs.empty or len(collab_recs) < num_recommendations:
            # Get content-based recommendations for the highest-rated movie
            highest_rated_movie = max(user_ratings.items(),
                                      key=lambda x: x[1])[0]
            content_recs = self.get_content_based_recommendations(
                highest_rated_movie, num_recommendations)

            if not content_recs.empty:
                return content_recs.head(num_recommendations)
            else:
                # Fallback to top-rated movies
                return self.movies_df.nlargest(num_recommendations, 'rating')

        return collab_recs.head(num_recommendations)

    def get_similar_movies(self, movie_title, num_similar=5):
        """
        Get movies similar to a given movie
        
        Args:
            movie_title (str): Title of the movie
            num_similar (int): Number of similar movies to return
            
        Returns:
            pd.DataFrame: Similar movies
        """
        return self.get_content_based_recommendations(movie_title, num_similar)

    def get_movie_details(self, movie_title):
        """
        Get detailed information about a specific movie
        
        Args:
            movie_title (str): Title of the movie
            
        Returns:
            pd.Series or None: Movie details
        """
        matches = self.movies_df[self.movies_df['title'].str.contains(
            movie_title, case=False, na=False)]

        if not matches.empty:
            return matches.iloc[0]
        else:
            return None

    def __init__(self, movies_df=None, ratings_df=None):
        self.movies_df = movies_df
        self.ratings_df = ratings_df

    def get_user_recommendations(self, user_ratings, num_recommendations=10):
        # Dummy implementation: return top N movies
        if self.movies_df is not None:
            return self.movies_df.head(num_recommendations)
        import pandas as pd
        return pd.DataFrame()
