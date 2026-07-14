# MOVIE RECOMMENDATION SYSTEM

> [!NOTE]
> This project uses [**The Movies Dataset**](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) from Kaggle. Download the dataset and place it in the `movies_dataset` folder before running the code.

This project implements a movie recommendation system using collaborative filtering techniques. The system is designed to recommend movies to users based on their preferences and the preferences of similar users.

## Content
1. [Folder Structure](#folder-structure)
2. [Features](#features)
3. [Data for training and pretraining](#data-for-training-and-pretraining)
4. [Interaction Weights](#interaction-weights)

## Folder Structure
The project is organized into the following directories:

``` text
├── movie-recommendation-system
|   ├── movie_dataset     # Dataset
|   ├── config.py         # File directory
|   ├── preprocess.py     # Preprocess data
|   ├── train.py          # Train Model, save to file model
|   ├── evaluate.py       # Predict on test data and evaluate model
|   └── inference.py      # Make predictions on new data - Backend call to model
└── README.md
```

## Features
1. **Trending Movies** or **Top Rated Movies**: recommend trending or top-rated movies.

> [!NOTE]
> A user has not history actions (e.g., ratings, comment, search, ...), the system can not use Machine Learning to recommend movies. In this case, Backend will query trending or top-rated movies to recommend.

2. **Top Picks for You**: recommend movies based on the user's past interactions and preferences.

3. **Because you watched**: recommend movies similar to those the user has watched.

> [!NOTE]
> When user clicks on a movie, the system use metadata (e.g., genre, cast, director, keywords, ...) of the movie to find similar movies to recommend.

4. **Real-time Feedback Loop**: update hobbies and preferences of users continuously. Stored in the database, the system can use this information to improve recommendations over time.

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

> [!WARNING]
> Grade Inflation and Oblivion - Time Decay

**Problem:** If a user has been using system for three years, their score for the "Action" genre could reach +50000 points, whereas a newly released movie starts with a score of zero. The model would break down and only recommend older content. Furthermore, people's preferences change over time, for instance, someone might have liked Action movies last year but prefers Romance this year.

**Solution:** Exponential Decay. Instead of simple accumulation (New Score = Old Score + Action), apply the following formula whenever a new interaction occurs: 

$$ New\_Score = (Old\_Score \times \alpha) + Interaction\_Weight) $$

- $\alpha:$ Time Decay Factor 

**Dot Product Calculation**: Example
To calculate the affinity score between User 101 and Movie M (Action, Sci-Fi), there is no need to invoke the computationally expensive ALS algorithm. We can simply use a basic dot product calculation:
- User Vector (Genre only): [Action: 25.5, Sci-Fi: 10.0, Romance: -5.0]
- Movie M Vector (Genre only): [Action: 1, Sci-Fi: 1, Romance: 0] (1 if the genre is present, 0 otherwise).

$$ Match\_Score = (25.5 \times 1) + (10.0 \times 1) + (-5.0 \times 0) = \mathbf{35.5} $$

> _Movies with the highest Match Scores are then returned to the Frontend and displayed in recommendation rows such as "Because you watched..." or "Top Picks for You"._