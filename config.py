import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 

# Data paths 
DATA = os.path.join(BASE_DIR, "movies_dataset")
PROCESSED_DATA = os.path.join(BASE_DIR, "processed_data") 
MODEL = os.path.join(BASE_DIR, "models") 
RATING = os.path.join(DATA, "ratings.csv") 
MOVIES = os.path.join(DATA, "movies.csv")

# Output paths 
TRAIN_MATRIX = os.path.join(PROCESSED_DATA, "train_matrix.npz") 
TEST_MATRIX = os.path.join(PROCESSED_DATA, "test_matrix.npz") 
MAPPING = os.path.join(MODEL, "mapping.pkl") 
MODEL_FILE = os.path.join(MODEL, "model.pkl") 

# Preprocessing parameters 
MIN_USER_RATING = 5         # Minimum number of movies ratings- a user has given 
MIN_MOVIE_RATING = 10       # Minimum number of ratings - a movie has received 
TEST_SIZE_RATIO = 0.2 
RANDOM_STATE = 42 
FACTORS = 50 
REGULARIZATION = 0.1 
ITERATIONS = 20 

def init_directories(): 
    for d in [PROCESSED_DATA, MODEL]: 
        os.makedirs(d, exist_ok=True) 
        
init_directories()