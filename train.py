import os
import pickle
import implicit
from scipy.sparse import load_npz

from config import (
    TRAIN_MATRIX, MODEL_FILE, FACTORS, 
    REGULARIZATION, ITERATIONS, RANDOM_STATE
)

def train():
    train_matrix = load_npz(TRAIN_MATRIX)
    
    model = implicit.als.AlternatingLeastSquares(
        factors = FACTORS,
        regularization = REGULARIZATION,
        iterations = ITERATIONS,
        random_state = RANDOM_STATE,
        calculate_training_loss = True
    )
    
    print(f"Training model in progress...")
    
    model.fit(train_matrix)
    
    print(f"Training completed. Saving model to: {MODEL_FILE}")
    
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    
    
if __name__ == "__main__":
    train()