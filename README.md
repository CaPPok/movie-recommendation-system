# MOVIE RECOMMENDATION SYSTEM

> [!NOTE]
> This project uses [**The Movies Dataset**](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset) from Kaggle. Download the dataset and place it in the `movies_dataset` folder before running the code.

This project implements a movie recommendation system using collaborative filtering techniques. The system is designed to recommend movies to users based on their preferences and the preferences of similar users.

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