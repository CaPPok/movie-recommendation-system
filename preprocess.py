import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, save_npz
import pickle 
from config import RATING, TRAIN_MATRIX, TEST_MATRIX, MAPPING, MIN_USER_RATING, MIN_MOVIE_RATING, TEST_SIZE_RATIO, RANDOM_STATE 

def preprocess(): 
    dtypes = { 'userId': np.int32, 'movieId': np.int32, 'rating': np.float32, } 
    df = pd.read_csv(RATING, usecols=['userId', 'movieId', 'rating'], dtype=dtypes) 
    print(f"Original dataset shape: {df.shape}") 
    
    # Filter movies 
    movie_counts = df['movieId'].value_counts() 
    valid_movies = movie_counts[movie_counts >= MIN_MOVIE_RATING].index 
    df = df[df['movieId'].isin(valid_movies)] 
    
    # Filter users 
    user_counts = df['userId'].value_counts() 
    valid_users = user_counts[user_counts >= MIN_USER_RATING].index 
    df = df[df['userId'].isin(valid_users)] 
    
    print(f"Filtered dataset shape: {df.shape}") 
    
    # Mapping IDs to indices 
    df['user_idx'], unique_users = pd.factorize(df['userId']) 
    df['movie_idx'], unique_movies = pd.factorize(df['movieId']) 
    n_users = len(unique_users) 
    n_movies = len(unique_movies) 
    
    print(f"Number of users: {n_users}, Number of movies: {n_movies}") 
    
    mappings = { 
        'userId_to_idx': {u: i for i, u in enumerate(unique_users)}, 
        'idx_to_userId': {i: u for i, u in enumerate(unique_users)}, 
        'movieId_to_idx': {m: i for i, m in enumerate(unique_movies)}, 
        'idx_to_movieId': {i: m for i, m in enumerate(unique_movies)}, 
    } 
    
    # Split into train and test sets 
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True) 
    split_index = int(len(df) * (1 - TEST_SIZE_RATIO)) 
    train_df = df.iloc[:split_index] 
    test_df = df.iloc[split_index:] 
    
    # Create sparse matrices 
    train_matrix = csr_matrix((train_df['rating'], (train_df['user_idx'], train_df['movie_idx'])), shape=(n_users, n_movies)) 
    test_matrix = csr_matrix((test_df['rating'], (test_df['user_idx'], test_df['movie_idx'])), shape=(n_users, n_movies)) 
    
    # Save matrices and mappings 
    save_npz(TRAIN_MATRIX, train_matrix) 
    save_npz(TEST_MATRIX, test_matrix) 
    with open(MAPPING, 'wb') as f: 
        pickle.dump(mappings, f) 
    print(f"Preprocessing done!") 

if __name__ == "__main__": 
    preprocess()