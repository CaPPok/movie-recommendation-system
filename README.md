# MOVIE RECOMMENDATION SYSTEM

> [!NOTE]
> This project uses [**The Movies Dataset**](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) from Kaggle. Download the dataset and place it in the `data/movies_dataset_raw` folder before running the code.

This project implements a hybrid Netflix-style movie recommendation system. It combines Collaborative Filtering and Content-Based Filtering in a batch-first architecture, designed to serve recommendations rapidly via DynamoDB without requiring continuous real-time model inference.

## Content
1. [Folder Structure](#folder-structure)
2. [Features](#features)
3. [Data for training and pretraining](#data-for-training-and-pretraining)
4. [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
5. [Interaction Weights](#interaction-weights)
6. [Algorithm and Ranking Mechanisms](#algorithm-and-ranking-mechanisms)

## Folder Structure
The project is organized into the following directories:

``` text
MOVIE-RECOMMENDATION-SYSTEM/
├── artifacts/                  # Trained model
├── configs/                    # YAML config
├── data/
│   ├── movies_dataset_raw/     # The Movies Dataset from Kaggle
│   ├── interim/                # Data processing is incomplete, the rejected file
│   ├── processed/              # Cleaned data
│   ├── features/               # Feature matrix (TF-IDF, interactions)
│   ├── splits/                 # Train/Val/Test
│   ├── serving/                # Data for DynamoDB (Popular movies...)
│   └── samples/
├── docs/                       # Design documents 
├── scripts/                    # Scripts for Data Pipeline
├── src/                        # SOURCE CODE
│   ├── data/                   # Preprocessing, split, export
│   ├── features/               # Creating features (content, interactions)
│   ├── models/                 # ALS model and Hybrid RRF
│   ├── recommenders/           # Inference
│   └── utils/
├── tests/
├── evaluate.py                 # Evaluate model performance (offline)
├── inference.py                # Inference script (online)
├── train.py                    # Training model
├── requirements.txt            # Python dependencies
└── README.md
```

## Features
1. **Top Rated Movies** - **Guest Mode**: recommend top-rated movies.

> [!NOTE]
> Recommended for users without an account or history. The system uses a precomputed IMDb-style weighted rating to rank movies globally or by specific genres.

2. **First-Login** - **Onboarding**: Solves the user cold-start problem. When a new user selects their favorite genres or movies, the system uses a Content-Based module to find movies with the most similar content profiles.

3. **Top Picks for You** - **Returning User**: For users with $\ge 5$ valid interactions. The system applies a Hybrid Ranking layer that fuses Collaborative Filtering and Content-Based recommendations.

4. **Because you watched**: recommend movies similar to those the user has watched.

> [!NOTE]
> When user clicks on a movie, the system use metadata (e.g., genre, cast, director, keywords, ...) of the movie to find similar movies to recommend.
5. **Real-time Feedback Loop**: update hobbies and preferences of users continuously. Stored in the database, the system can use this information to improve recommendations over time.

## Data for training and pretraining

### Movies Metadata

> [!NOTE]
> Used for Content-Based Filtering and solving the Cold-start problem (New Movies).

- Information: movie_id, title, release_year.
- Categories: genre, original_language.
- Cast and Crew: director, cast, writer, producer, production_company.
- NLP Features: keywords, overview, tagline.
- Parameters: vote_average, vote_count, budget/revenue.

> [!TIP]
> _Source: movies_metadata.csv, credits.csv, keywords.csv from [The Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)._

### User Profile

> [!NOTE]
> Used for solving the Cold-start problem (New Users).

- Information: user_id, age, gender.
- Location: country, city.
- Onboarding data: save 3-5 categories or movies user chooses when first using the system.

> [!TIP]
> _Source: user registration data._

### User Interactions

- Explicit Feedback: rating_score, like_dislike, add_to_watchlist, share_movie.
- Implicit Feedback: watch_percentage, click_detail, hover_trailer, search_query.

> [!TIP]
> _Source: user interactions data from the system._

## Data Preprocessing Pipeline

> [!IMPORTANT]
> Read at: [DATA_PIPELINE.md](DATA_PIPELINE.md)

## Interaction Weights
Converting Interactions into Scores

> [!IMPORTANT]
> To recommend movies to users correctly, the system needs **User Profile Vectorization + Behavioral Decay**. For each user, they have a profile vector. Instead of storing which movies a user likes, they store which movie characteristics the user prefers.

Before adding or subtracting points, the system must define a scoring scale for each interaction. The lower the cost of the user signal, the greater the point adjustment.

Example:

| Interaction | Weight | Note |
| ------------- | ------------- | ------------- |
| Search Impression | -1 | Saw it in the search results but scrolled past - signal of mild dislike. |
| Click detail | +2 | Curious about the movie. |
| Add to Watchlist | +5 | User added the movie to their watchlist - medium signal of interest. |
| Watch > 50% | +10 | User watched more than 50% of the movie - strong signal of interest. |
| Share Movie | +15 | User shared the movie with others - very strong signal of interest. |
| Like/Rate 5 stars | +15 | User explicitly liked the movie. |
| Dislike/Rate 1 star | -15 | User explicitly disliked the movie. |

Each user will be stored as a JSON document containing vectors.

## Algorithm and Ranking Mechanisms

The system employs a 4-layer architecture, combining non-personalized baselines, content similarity, and collaborative filtering, ultimately fused by a dynamic ranking layer.

### 1. Popularity Ranker - Guest

> [!IMPORTANT]
> Serves as the default recommender for Guests and acts as the fallback to guarantee the system always returns results.

* **Algorithm:** Uses an IMDb-style weighted rating formula: $$score(m) = \frac{v}{v + m_0} \cdot R + \frac{m_0}{v + m_0} \cdot C$$.
* **Mechanism:** Pulls the score of movies with very few ratings towards the global average (C) to prevent skewed rankings.
* **Data-Driven Parameters:** The threshold $m_0$ is dynamically calculated from the 90th percentile of the rating count distribution (which is 704 votes).

### 2. Content-Based Recommender - Onboarding

* **Feature Representation:** Aggregates all movie metadata (title, overview, genres, keywords, cast, and directors) into a single unified text document per movie.
* **Algorithm:** Applies TF-IDF vectorization across the entire catalog to create a 30,000-dimensional sparse matrix.
* **Mechanism:** Uses Cosine Similarity to find movies closest to a user's onboarding choices (selected movies/genres) or to precompute the top 50 similar items for the "Because you watched" feature.

### 3. Collaborative Filtering - Implicit Alternative Least Squares

* **Algorithm:** Alternating Least Squares Matrix Factorization.
* **Signal Conversion:** Transforms explicit rating data into implicit feedback (preference and confidence).
    * **Positive Signals** (Rating $\ge 4.0$): Fed into the training matrix, with confidence scaling linearly with the rating.
    * **Neutral Signals** (Rating $3.0 - 3.5$): Excluded from the training matrix by default (treated as unknown). 
    * **Negative Signals** (Rating $\le 2.5$): Completely bypass the ALS model and are reserved strictly for filtering out bad recommendations later.

> [!NOTE]
> Academic Baseline vs Production Model: The project includes a implementation of the Explicit ALS algorithm in `src/models/als_model/als.py` - [als.py](src/models/als_model/als.py) and [ALS.md](src/models/als_model/ALS.md). However, this implementation assumes a dense matrix representation. Converting the actual dataset's sparse matrix ($270,883 \times 44,577$) into a dense np.ndarray would require approximately 97 GB of memory, and running basic Python loops across hundreds of thousands of users would take dozens of hours. Therefore, `als.py` is strictly maintained as an academic baseline to run on small subsets (e.g., `ratings_small.csv`), while the production system utilizes the highly optimized implicit Python library to handle large-scale sparse matrices efficiently. 

### 4. Hybrid Ranking Layer and Business Rules

**Problem:** Raw scores from [Popularity Ranker](#1-popularity-ranker---guest) (0-5 scale), [Content-Base Recommender](#2-content-based-recommender---onboarding) (0-1 scale), and [Collaborative Filtering](#3-collaborative-filtering---implicit-alternative-least-squares) (unbounded dot product) cannot be directly added together.

**Solution:** The system discards raw scores and merges the candidates using Weighted Reciprocal Rank Fusion. $$rrf\_score(m) = \sum_{s} w_s \cdot \frac{1}{k + rank_s(m)}, k = 60$$

_Where:_
* $rank_s(m)$ is the rank of movie $m$ in source $s$ (Content-based, Collaborative, or Popularity).
* $w_s$ is the dynamic weight. For instance, if a user has highly reliable history (e.g., $> 20$ interactions), the collaborative weight $w_{cf}$ approaches $1.0$. If they have limited history, the content-based weight $w_{cb}$ leadsleads.

**Dynamic Weighting:** The weights for Collaborative Filtering and Content-Based adjust dynamically based on the user's history. Users with a short history lean heavily on content-based scores, while users with $\ge 20$ valid interactions rely entirely on collaborative filtering. 

**Business Rules and Filtration:** Before returning the final top 20 list, the system strictly applies the following filters:

* Removes movies the user has interacted with in the training set or recent history.
* Removes explicitly disliked movies (rating $\le 2.5$).
* Diversifies results - Enforces genre diversity by allowing a maximum of 4 movies from the same primary genre in the final top 20.
* If the final list contains fewer than the requested limit, it backfills sequentially from Content-Based $\rightarrow$ Top-Rated Genre $\rightarrow$ Top-Rated Global to guarantee a full response.