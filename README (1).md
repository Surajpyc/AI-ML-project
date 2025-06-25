# Movie Recommendation System

A Python-based machine learning movie recommendation system built with Streamlit that provides personalized movie recommendations using collaborative filtering and content-based filtering algorithms.

## Features

- **Interactive Web Interface**: Clean, intuitive Streamlit web application
- **Multiple Data Sources**: Support for CSV file uploads and URL-based datasets
- **Hybrid Recommendation Engine**: Combines collaborative filtering and content-based filtering
- **Real-time Movie Search**: Search movies by title, genre, or both
- **User Rating System**: Rate movies to get personalized recommendations
- **Analytics Dashboard**: View dataset statistics and genre distributions
- **Top Movies Display**: Browse top-rated movies with filtering options

## System Architecture

The application follows a modular architecture with clear separation of concerns:

- **Frontend**: Streamlit web application with interactive UI components
- **Backend Logic**: Python modules handling data processing and recommendation algorithms
- **Data Layer**: CSV-based data handling with support for file uploads and URL datasets
- **ML Engine**: Scikit-learn powered recommendation algorithms using TF-IDF and cosine similarity

## Installation

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Dependencies

```bash
pip install streamlit pandas numpy scikit-learn requests
```

### Quick Start

1. Clone or download the project files
2. Navigate to the project directory
3. Run the application:

```bash
streamlit run app.py --server.port 5000
```

4. Open your browser and go to `http://localhost:5000`

## File Structure

```
movie-recommendation-system/
├── app.py                    # Main Streamlit application
├── recommendation_engine.py  # ML recommendation algorithms
├── data_handler.py          # Data loading and processing
├── utils.py                 # Utility functions for UI components
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── replit.md               # Project documentation
└── README.md               # This file
```

## Usage

### Getting Started

1. **Load Movie Data**:
   - Upload a CSV file with movie information, or
   - Provide a URL to an online CSV dataset

2. **Rate Movies**:
   - Select movies from the dropdown in the sidebar
   - Rate them on a scale of 0.5 to 5.0 stars

3. **Get Recommendations**:
   - Navigate to the "Recommendations" tab
   - Click "Get Recommendations" to receive personalized suggestions

### Data Format

Your CSV file should contain these columns:

#### Required Columns
- `title`: Movie title (string)

#### Optional Columns
- `movieId`: Unique identifier for each movie (integer)
- `genres`: Movie genres, pipe-separated for multiple genres (e.g., "Action|Comedy|Drama")
- `rating`: Average rating (0.0 to 5.0)
- `userId`: User identifier for collaborative filtering

#### Example CSV Format
```csv
movieId,title,genres,rating
1,"Toy Story (1995)","Animation|Children|Comedy",4.2
2,"Jumanji (1995)","Adventure|Children|Fantasy",3.8
3,"Grumpier Old Men (1995)","Comedy|Romance",3.1
```

## Features Overview

### 1. Recommendations Tab
- **Personalized Suggestions**: Get movie recommendations based on your ratings
- **Hybrid Algorithm**: Combines collaborative filtering and content-based filtering
- **Adjustable Count**: Choose between 5, 10, 15, or 20 recommendations

### 2. Search Movies Tab
- **Flexible Search**: Search by movie title, genre, or both
- **Quick Rating**: Rate movies directly from search results
- **Real-time Results**: Instant search as you type

### 3. Top Rated Tab
- **Popular Movies**: Browse highest-rated movies in the dataset
- **Genre Filtering**: Filter by specific genres
- **Rating Threshold**: Set minimum rating requirements
- **Customizable Count**: Display 10, 20, 50, or 100 movies

### 4. Analytics Dashboard
- **Dataset Overview**: View total movies, genres, and rating statistics
- **Genre Distribution**: Visual chart of movie distribution by genre
- **Personal Analytics**: Track your rating patterns and preferences

## Recommendation Algorithms

### Content-Based Filtering
- Uses TF-IDF vectorization on movie genres
- Calculates cosine similarity between movies
- Recommends movies similar to those you've rated highly

### Collaborative Filtering
- Builds user-item rating matrices
- Uses item-item collaborative filtering
- Finds movies liked by users with similar preferences

### Hybrid Approach
- Combines both algorithms for better recommendations
- Falls back to content-based when collaborative data is insufficient
- Provides diverse and accurate suggestions

## Configuration

### Streamlit Configuration (.streamlit/config.toml)
```toml
[server]
headless = true
address = "0.0.0.0"
port = 5000
enableCORS = false
enableXsrfProtection = false

[theme]
base = "light"

[browser]
gatherUsageStats = false
```

## API Reference

### DataHandler Class
- `load_data_from_file(uploaded_file)`: Load data from uploaded CSV
- `load_data_from_url(url)`: Load data from URL
- `search_movies(query, search_type)`: Search movies by title/genre
- `get_top_movies(num_movies, genre_filter, min_rating)`: Get top-rated movies
- `get_unique_genres()`: Get list of available genres

### MovieRecommendationEngine Class
- `get_user_recommendations(user_ratings, num_recommendations)`: Get personalized recommendations
- `get_content_based_recommendations(movie_title, num_recommendations)`: Content-based suggestions
- `get_collaborative_recommendations(user_ratings, num_recommendations)`: Collaborative filtering
- `get_similar_movies(movie_title, num_similar)`: Find similar movies

## Sample Datasets

You can test the system with these publicly available movie datasets:

- **MovieLens**: https://grouplens.org/datasets/movielens/
- **TMDB**: https://www.themoviedb.org/documentation/api
- **IMDb**: https://developer.imdb.com/

## Troubleshooting

### Common Issues

1. **Empty Recommendations**:
   - Ensure you've rated at least 2-3 movies
   - Check that your dataset has genre information
   - Verify the CSV format is correct

2. **Data Loading Errors**:
   - Confirm the CSV has a 'title' column
   - Check for proper encoding (UTF-8 recommended)
   - Ensure the URL is accessible and returns CSV data

3. **Performance Issues**:
   - Large datasets (>10,000 movies) may take longer to process
   - Consider filtering your dataset for better performance

### Error Messages

- **"Missing required columns"**: Your CSV must have a 'title' column
- **"No movies found"**: Check your search criteria or dataset
- **"Recommendation engine not initialized"**: Load movie data first

## Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## Technical Requirements

- **Python**: 3.11+
- **Streamlit**: 1.45.1+
- **Pandas**: 2.3.0+
- **NumPy**: 2.3.0+
- **Scikit-learn**: 1.7.0+
- **Requests**: 2.32.4+

## Performance Considerations

- **Dataset Size**: Optimized for datasets up to 50,000 movies
- **Memory Usage**: Approximately 100-500MB depending on dataset size
- **Processing Time**: Initial loading takes 5-30 seconds for large datasets
- **Recommendation Speed**: 1-3 seconds for generating recommendations

## Security Notes

- Only CSV files are accepted for uploads
- URL downloads have a 30-second timeout
- No user data is stored permanently
- All processing happens locally in your browser session

## Future Enhancements

- Support for movie posters and metadata
- Integration with external movie APIs
- User profile persistence
- Advanced filtering options
- Export functionality for recommendations
- Mobile-responsive design improvements

## License

This project is open source and available under the MIT License.

## Support

For issues, questions, or contributions:

1. Check the troubleshooting section
2. Review the code documentation
3. Test with sample datasets
4. Verify your Python environment

## Changelog

- **June 17, 2025**: Initial release with hybrid recommendation system
- Core features: Data loading, recommendation engine, search, analytics
- Streamlit web interface with responsive design
- Support for CSV uploads and URL-based datasets

---

**Note**: This system is designed for educational and demonstration purposes. For production use, consider implementing user authentication, data persistence, and additional security measures.