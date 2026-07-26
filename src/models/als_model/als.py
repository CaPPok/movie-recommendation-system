import numpy as np
import pickle

class ALS:
    def __init__(self, n_factors=50, regularization=0.1, iterations=10):
        self.n_factors = n_factors
        self.regularization = regularization
        self.iterations = iterations
        self.user_factors = None
        self.item_factors = None
        
    def fit(self, R):
        n_users, n_items = R.shape
        
        '''
            self.user_factors: Matrix U has shape (n_users, n_factors - User preferences.
            self.item_factors: Matrix V has shape (n_items, n_factors - Item features.
        '''
        
        self.user_factors = np.random.normal(scale=1./self.n_factors, size=(n_users, self.n_factors))
        self.item_factors = np.random.normal(scale=1./self.n_factors, size=(n_items, self.n_factors))
        
        lambda_I = self.regularization * np.eye(self.n_factors)
        
        for step in range(self.iterations):
            # Phase 1: update User factors
            for u in range(n_users):
                idx = np.where (R[u, :] > 0)[0]
                if len(idx) > 0:
                    Vu = self.item_factors[idx, :]
                    Ru = R[u, idx]
                    self.user_factors[u, :] = np.linalg.solve(Vu.T.dot(Vu) + lambda_I, Vu.T.dot(Ru))
                    
            # Phase 2: update Item factors
            for i in range(n_items):
                idx = np.where(R[:, i] > 0)[0]
                if len(idx) > 0:
                    Ui = self.user_factors[idx, :]
                    Ri = R[idx, i]
                    self.item_factors[i, :] = np.linalg.solve(Ui.T.dot(Ui) + lambda_I, Ui.T.dot(Ri))
                
    def predict(self, user_id, item_id):
        return self.user_factors[user_id, :].dot(self.item_factors[item_id, :].T)
    
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump({'U': self.user_factors, 'V': self.item_factors}, f)